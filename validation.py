"""Chronological threshold search and trading-date bootstrap."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from metrics import classification, diagnostics
from model import decision, logit

SEED = 20261126
PRECISION_TARGETS = (0.65, 0.68, 0.70, 0.72, 0.75)
REGIME_CUT = 1.7
DEFAULT_POLICY = {"threshold": 0.74, "regime_cut": REGIME_CUT, "calm_add": 0.0}


def _ratios(tp, fp, total):
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / np.maximum(total, 1e-12)
    score = np.sqrt(4 * precision * recall / np.maximum(3 * recall + precision, 1e-12))
    return precision, recall, score


def _date_indicators(dates):
    _, inverse = np.unique(dates, return_inverse=True)
    return (inverse == np.arange(inverse.max() + 1)[:, None]).astype(float)


def _intervals(tp, fp, positives, draws):
    rng = np.random.default_rng(SEED)
    days = len(positives)
    counts = rng.multinomial(days, np.ones(days) / days, size=draws)
    p, r, score = _ratios(counts @ tp, counts @ fp, counts @ positives[:, None])
    return {name: np.quantile(values, [0.05, 0.95], axis=0)
            for name, values in (("precision", p), ("recall", r), ("score", score))}


def intervals(frame, predicted, draws=1000):
    by_date = _date_indicators(frame.index.get_level_values("date"))
    y = frame["y"].to_numpy(bool)
    result = _intervals(by_date @ (y & predicted)[:, None],
                        by_date @ (~y & predicted)[:, None], by_date @ y, draws)
    return {key: limits[:, 0].tolist() for key, limits in result.items()}


def threshold_search(frame, probability, calm_add=0.0, targets=PRECISION_TARGETS, thresholds=None):
    """Predict with each threshold and resample complete dates jointly."""
    if frame.empty:
        return {str(target): None for target in targets}
    thresholds = (np.linspace(0.22, 0.94, 73) if thresholds is None else np.asarray(thresholds))
    y = frame["y"].to_numpy(bool)
    calm = frame["market_iv"].to_numpy() < REGIME_CUT
    candidates = probability[:, None] >= thresholds + calm[:, None] * calm_add
    by_date = _date_indicators(frame.index.get_level_values("date"))
    tp = by_date @ (y[:, None] & candidates)
    fp = by_date @ (~y[:, None] & candidates)
    positives = by_date @ y
    precision, recall, scores = _ratios(tp.sum(axis=0), fp.sum(axis=0), positives.sum())
    bands = _intervals(tp, fp, positives, 250)
    # Shift the mix toward calmer dates as a stress scenario using observable IV.
    day_weight = np.where(by_date @ calm > 0, 1.5, 0.75)
    p_stress, _, s_stress = _ratios(day_weight @ tp, day_weight @ fp, day_weight @ positives)
    fold_scores, fold_precision = [], []
    for fold in frame["fold"].unique():
        mask = frame["fold"].to_numpy() == fold
        p, _, s = _ratios((y[mask, None] & candidates[mask]).sum(axis=0),
                          (~y[mask, None] & candidates[mask]).sum(axis=0), y[mask].sum())
        fold_scores.append(s)
        fold_precision.append(p)
    variability = np.std(fold_scores, axis=0)
    utility = 0.65 * scores + 0.35 * s_stress - 0.12 * variability
    result = {}
    for target in targets:
        good = ((precision >= target) & (p_stress >= target - 0.045)
                & (tp.sum(axis=0) + fp.sum(axis=0) >= 12) & ((tp + fp > 0).sum(axis=0) >= 4))
        if not good.any():
            result[str(target)] = None
            continue
        idx = int(np.argmax(np.where(good, utility, -np.inf)))
        result[str(target)] = {
            "threshold": float(thresholds[idx]), "calm_add": calm_add, "regime_cut": REGIME_CUT,
            "precision": float(precision[idx]), "recall": float(recall[idx]), "score": float(scores[idx]),
            "precision_interval": bands["precision"][:, idx].tolist(), "calm_stress_precision": float(p_stress[idx]),
            "calm_stress_score": float(s_stress[idx]), "fold_score_sd": float(variability[idx]),
            "fold_scores": np.asarray(fold_scores)[:, idx].tolist(),
            "fold_precision": np.asarray(fold_precision)[:, idx].tolist(), "utility": float(utility[idx]),
        }
    return result


def policy_from_past(frame, probability, target, calm_add):
    if frame.index.get_level_values("date").nunique() < 8:
        return {**DEFAULT_POLICY, "calm_add": calm_add}
    bounds = np.linspace(0.32, 0.94, 63)
    targets = (target, *(value for value in reversed(PRECISION_TARGETS) if value < target))
    choices = threshold_search(frame, probability, calm_add, targets, bounds)
    policy = next((choice for choice in choices.values() if choice is not None), None)
    if policy is None:
        return {**DEFAULT_POLICY, "threshold": 0.84, "calm_add": calm_add}
    return {key: policy[key] for key in DEFAULT_POLICY}


def fit_calibration(past_probability, y):
    if len(y) < 400 or np.unique(y).size < 2:
        return None
    fit = LogisticRegression(C=0.2).fit(logit(past_probability).reshape(-1, 1), y)
    return {"slope": float(fit.coef_[0, 0]), "intercept": float(fit.intercept_[0])}


def evaluate(frame, probability, predicted):
    result = diagnostics(frame["y"].to_numpy(), probability, predicted)
    result["folds"] = {}
    for fold in frame["fold"].unique():
        mask = frame["fold"].to_numpy() == fold
        result["folds"][str(fold)] = diagnostics(frame.loc[mask, "y"].to_numpy(),
                                                 probability[mask], predicted[mask])
    result["fold_score_sd"] = float(np.std([fold["score"] for fold in result["folds"].values()]))
    return result


def stress_checks(frame, probability, policy):
    flags = decision(probability, frame["market_iv"], policy)
    calm = frame["market_iv"].to_numpy() < REGIME_CUT
    result = {"calm" if is_calm else "active": classification(frame.loc[mask, "y"], flags[mask])
              for is_calm, mask in ((True, calm), (False, ~calm))}
    y = frame["y"].to_numpy(bool)
    tpr = flags[y].mean()
    fpr = flags[~y].mean()
    result["prior_shift_precision"] = {
        str(rate): float((rate * tpr) / max(rate * tpr + (1 - rate) * fpr, 1e-12))
        for rate in (0.25, 0.30, 0.35, 0.40)
    }
    result["threshold_sensitivity"] = {
        str(delta): classification(y, decision(probability, frame["market_iv"],
                                               {**policy, "threshold": policy["threshold"] + delta}))
        for delta in (-0.06, -0.03, 0.0, 0.03, 0.06)
    }
    names = frame.index.get_level_values("name")
    result["stocks"] = {name: classification(y[names == name], flags[names == name])
                        for name in sorted(names.unique())}
    return result
