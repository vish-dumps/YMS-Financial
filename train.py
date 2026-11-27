#!/usr/bin/env python3
"""Run training-only model selection, lock a policy, and build portable snapshots."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from threadpoolctl import threadpool_limits

from features import WINDOW, build_features, read_history, read_supplement
from metrics import classification
from model import FALLBACK_CUTOFF, apply_calibration, decision, fallback_score, snapshot_probability
from research_models import ABLATIONS, ENSEMBLES, SPECS, run_fit, train_components
from validation import (
    PRECISION_TARGETS, SEED, evaluate, fit_calibration, intervals,
    policy_from_past, stress_checks, threshold_search,
)

ROOT = Path(__file__).resolve().parent
WARMUP, REFIT, HOLDOUT = 16, 5, 18
ADJUSTMENTS = (0.0, 0.045, -0.045, -0.10)
RULES = ("template", "implied_threshold", "range_threshold", "combined_rule")
CALIBRATED = {"log_core_sigmoid": "log_core", "log_simple_sigmoid": "log_simple"}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(".tmp")
    scratch.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    scratch.replace(path)


def load_panel(include_holdout=False):
    paths = ROOT / "data"
    files = sorted((paths / "historical_info").glob("*_n50.info.csv"))
    dates = [path.name[:8] for path in files]
    cutoff = dates[-HOLDOUT - 1]
    last = None if include_holdout else cutoff
    raw = read_history(paths / "historical_info", cutoff=last)
    extra = read_supplement(ROOT / "user_historical_info.csv", cutoff=last)
    features = build_features(raw, extra)
    # Read labels in increasing date blocks and stop at the development boundary.
    with (paths / "train_labels.csv").open(newline="", encoding="utf-8") as file:
        labels = []
        for row in csv.DictReader(file):
            if last is not None and row["date"] > last:
                break
            labels.append((row["date"], row["name"], row["moved"] == "True"))
    labels = pd.DataFrame(labels, columns=["date", "name", "y"]).set_index(["date", "name"])
    if not labels.index.is_unique or set(labels.index) != set(features.index):
        raise ValueError("Training label keys must match the historical inputs exactly")
    panel = features.join(labels)
    panel["fold"] = 0
    for index, block in enumerate(np.array_split(dates[WARMUP:-HOLDOUT], 4), start=1):
        panel.loc[panel.index.get_level_values("date").isin(block), "fold"] = index
    panel.loc[panel.index.get_level_values("date") > cutoff, "fold"] = 5
    return panel, dates[:len(files) - (0 if include_holdout else HOLDOUT)], cutoff


def eligible(panel, dates):
    return panel.loc[panel.index.get_level_values("date").isin(dates) & panel["usable"]]


def schedule(dates):
    for i in range(WARMUP, len(dates), REFIT):
        yield dates[:i], dates[i:i + REFIT]


def rule_scores(frame, rule):
    iv = frame["daily_iv"].to_numpy()
    ranges = frame["prev_range"].to_numpy()
    if rule == "template":
        return np.fmax(iv / 2, ranges / 3)
    if rule == "implied_threshold":
        return iv / 4
    if rule == "range_threshold":
        return ranges / 7
    return fallback_score(frame) / 4


def rule_prediction(fit_rows, validation, rule):
    probabilities = np.nan_to_num(rule_scores(validation, rule), nan=0.0)
    if rule == "template":
        return probabilities, probabilities > 1.0, 1.0
    fitted_score = np.nan_to_num(rule_scores(fit_rows, rule), nan=0.0)
    search = threshold_search(fit_rows, fitted_score, targets=(0.70, 0.68, 0.65))
    choice = next((value for value in search.values() if value is not None), None)
    threshold = choice["threshold"] if choice else FALLBACK_CUTOFF / 4
    return probabilities, probabilities >= threshold, threshold


def out_of_fold(panel, dates, specs, rules=True):
    work = eligible(panel, dates[WARMUP:]).copy()
    timing = {key: {"fit_seconds": 0.0, "predict_seconds": 0.0} for key in specs}
    for fit_dates, val_dates in schedule(dates):
        fit, val = eligible(panel, fit_dates), eligible(panel, val_dates)
        if val.empty:
            continue
        for key, config in specs.items():
            probability, fit_time, predict_time = run_fit(fit, val, fit_dates[-1], config)
            work.loc[val.index, key] = probability
            timing[key]["fit_seconds"] += fit_time
            timing[key]["predict_seconds"] += predict_time
        for rule in RULES if rules else ():
            probability, predicted, cutoff = rule_prediction(fit, val, rule)
            work.loc[val.index, rule] = probability
            work.loc[val.index, rule + "_flags"] = predicted.astype(float)
            work.loc[val.index, rule + "_threshold"] = cutoff
    return work, timing


def sequential_flags(oof, dates, key, target, calm_add):
    predicted = np.zeros(len(oof), dtype=bool)
    row_dates = oof.index.get_level_values("date")
    thresholds = []
    for fit_dates, val_dates in schedule(dates):
        past = oof.loc[row_dates <= fit_dates[-1]]
        mask = row_dates.isin(val_dates)
        policy = policy_from_past(past, past[key].to_numpy(), target, calm_add)
        predicted[mask] = decision(oof.loc[mask, key], oof.loc[mask, "market_iv"], policy)
        thresholds.append({"through": fit_dates[-1], "threshold": policy["threshold"]})
    return predicted, thresholds


def evaluate_config(oof, dates, key):
    score = oof[key].to_numpy()
    table = {}
    for calm_add in ADJUSTMENTS:
        options = threshold_search(oof, score, calm_add)
        for target in PRECISION_TARGETS:
            fixed = options[str(target)]
            flags, thresholds = sequential_flags(oof, dates, key, target, calm_add)
            causal = evaluate(oof, score, flags)
            table[f"{target}/{calm_add}"] = {
                "target": target, "calm_add": calm_add, "fixed_diagnostic": fixed, "causal": causal,
                "causal_thresholds": thresholds,
            }
    return table


def choose_candidate(results):
    choices = []
    for model, table in results.items():
        for record in table.values():
            fixed, causal = record["fixed_diagnostic"], record["causal"]
            if fixed is None or record["target"] < 0.68:
                continue
            deficit = max(0.0, 0.65 - causal["precision"])
            lower_bound_penalty = max(0.0, 0.60 - fixed["precision_interval"][0])
            utility = (0.72 * fixed["utility"] + 0.28 * causal["score"]
                       - 0.60 * deficit - 0.45 * lower_bound_penalty)
            utility -= (0.006 * (model in ENSEMBLES) + 0.007 * (record["calm_add"] != 0)
                        + 0.003 * (model in CALIBRATED))
            choices.append((utility, model, record))
    if not choices:
        raise ValueError("No candidate met the offline screening constraints")
    choices.sort(key=lambda item: item[0], reverse=True)
    return choices


def model_specs(key):
    key = CALIBRATED.get(key, key)
    weights = ENSEMBLES.get(key, {key: 1.0})
    return [SPECS[model] for model in weights], list(weights.values())


def combined_oof(oof):
    for name, weights in ENSEMBLES.items():
        oof[name] = sum(oof[key] * value for key, value in weights.items())


def add_calibration(oof, dates, name, source):
    row_dates = oof.index.get_level_values("date")
    for fit_dates, val_dates in schedule(dates):
        past, current = row_dates <= fit_dates[-1], row_dates.isin(val_dates)
        params = fit_calibration(oof.loc[past, source], oof.loc[past, "y"])
        oof.loc[current, name] = apply_calibration(oof.loc[current, source].to_numpy(), params)


def ablations(panel, dates):
    base = SPECS["log_extended"]
    configs = {
        f"without_{key}": {**base, "features": [f for f in base["features"] if f not in drop]}
        for key, drop in ABLATIONS.items()
    }
    oof, timing = out_of_fold(panel, dates, configs, rules=False)
    results = {}
    for key, cost in timing.items():
        probability = oof[key].to_numpy()
        flags, _ = sequential_flags(oof, dates, key, 0.70, 0.0)
        results[key] = {"threshold_search": threshold_search(oof, probability),
                        "causal": evaluate(oof, probability, flags), **cost}
    return results


def permutation(panel, dates, name):
    val_dates = [date for date in dates if panel.xs(date)["fold"].iat[0] == 4]
    fit_dates = [date for date in dates if date < val_dates[0]]
    fit = eligible(panel, fit_dates)
    val = eligible(panel, val_dates)
    configs, weights = model_specs(name)
    snapshot, estimators = train_components(fit, fit_dates[-1], configs, weights)
    base = average_precision_score(val["y"], snapshot_probability(val, snapshot, estimators=estimators))
    features = set().union(*(spec["features"] for spec in configs))
    groups = {key: sorted(features & columns & set(val.columns)) for key, columns in ABLATIONS.items()}
    rng = np.random.default_rng(SEED)
    results = {}
    row_dates = val.index.get_level_values("date")
    for group, columns in groups.items():
        deltas = []
        if not columns:
            continue
        for _ in range(12):
            altered = val.copy()
            # Permute whole equal-sized daily cross sections and stock rows jointly.
            source = rng.permutation(row_dates.unique())
            old = dict(zip(row_dates.unique(), source))
            index = pd.MultiIndex.from_arrays([row_dates.map(old), val.index.get_level_values("name")],
                                              names=val.index.names)
            altered.loc[:, columns] = val[columns].reindex(index).to_numpy()
            probability = snapshot_probability(altered, snapshot, estimators=estimators)
            deltas.append(base - average_precision_score(val["y"], probability))
        results[group] = {"mean_pr_auc_drop": float(np.mean(deltas)), "sd": float(np.std(deltas))}
    fitted_importance = []
    for component, estimator in zip(snapshot["components"], estimators):
        if component["kind"] == "logistic":
            values = np.abs(component["coef"])
            measure = "absolute coefficient per fitted IQR"
        elif component["kind"] == "lightgbm":
            values = estimator.booster_.feature_importance(importance_type="gain")
            measure = "split gain"
        else:
            values = estimator.get_feature_importance()
            measure = "prediction value change"
        importance = sorted(zip(component["features"], values), key=lambda item: item[1], reverse=True)
        fitted_importance.append({"kind": component["kind"], "measure": measure,
                                  "features": {key: float(value) for key, value in importance}})
    return {"block_permutation": results, "model_importance": fitted_importance}


def develop():
    panel, dates, cutoff = load_panel()
    oof, timing = out_of_fold(panel, dates, SPECS)
    combined_oof(oof)
    for name, source in CALIBRATED.items():
        add_calibration(oof, dates, name, source)
    results = {}
    for key in (*SPECS, *ENSEMBLES, *CALIBRATED):
        print(f"Screening {key}", flush=True)
        results[key] = evaluate_config(oof, dates, key)
    ranking = choose_candidate(results)
    utility, winner, selection = ranking[0]
    specs, weights = model_specs(winner)
    chosen = {"model": winner, "target": selection["target"], "calm_add": selection["calm_add"],
              "development_end": cutoff, "holdout_dates": HOLDOUT, "warmup_dates": WARMUP,
              "refit_dates": REFIT, "seed": SEED, "components": specs, "component_weights": weights,
              "calibration": "sigmoid" if winner in CALIBRATED else "none"}
    flags, thresholds = sequential_flags(oof, dates, winner, chosen["target"], chosen["calm_add"])
    causal = evaluate(oof, oof[winner].to_numpy(), flags)
    causal["date_bootstrap_90"] = intervals(oof, flags)
    uncalibrated, calibrated, labels = [], [], []
    raw_winner = CALIBRATED.get(winner, winner)
    for fit_dates, val_dates in schedule(dates):
        row_dates = oof.index.get_level_values("date")
        prior, current = oof.loc[row_dates <= fit_dates[-1]], oof.loc[row_dates.isin(val_dates)]
        if len(prior) < 400 or current.empty:
            continue
        probability = current[raw_winner].to_numpy()
        calibrated.extend(apply_calibration(probability, fit_calibration(prior[raw_winner], prior["y"])))
        uncalibrated.extend(probability)
        labels.extend(current["y"])
    frozen_policy = policy_from_past(oof, oof[winner].to_numpy(), chosen["target"], chosen["calm_add"])
    comparison = {}
    for key, table in results.items():
        preferred = next((record for _, model, record in ranking if model == key), None)
        preferred = preferred or table["0.7/0.0"]
        if key in ENSEMBLES:
            cost = {field: sum(timing[model][field] for model in ENSEMBLES[key])
                    for field in ("fit_seconds", "predict_seconds")}
        else:
            cost = timing[CALIBRATED.get(key, key)]
        comparison[key] = {**preferred, **cost, "configuration": ENSEMBLES.get(key, SPECS.get(CALIBRATED.get(key, key))),
                           "disposition": "selected" if key == winner else "lower screening utility"}
    for rule in RULES:
        comparison[rule] = {
            "causal": evaluate(oof, oof[rule].to_numpy(), oof[rule + "_flags"].to_numpy(bool)),
            "thresholds": sorted(oof[rule + "_threshold"].unique().tolist()),
        }
    report = {
        "configuration": chosen, "screening_utility": utility, "winner": causal,
        "causal_thresholds": thresholds, "holdout_policy": frozen_policy,
        "development_rows": len(oof), "first_oof": oof.index.get_level_values("date").min(),
        "comparison": comparison,
        "threshold_study": results[winner],
        "selected_diagnostic_stress": stress_checks(oof, oof[winner].to_numpy(), frozen_policy),
        "sigmoid_calibration": {"rows": len(labels), "raw_brier": brier_score_loss(labels, uncalibrated),
                                "sigmoid_brier": brier_score_loss(labels, calibrated)},
        "ablations": ablations(panel, dates),
        "permutation": permutation(panel, dates, winner),
        "screening_rank": [{"model": key, "target": record["target"], "calm_add": record["calm_add"],
                            "utility": value} for value, key, record in ranking[:25]],
    }
    write_json(ROOT / "reports" / "development.json", report)
    write_json(ROOT / "model_config.json", chosen)
    print(f"Selected {chosen}\nCausal development {causal['precision']:.4f} precision, "
          f"{causal['recall']:.4f} recall, {causal['score']:.4f} score", flush=True)


def thin_history(dates):
    """Keep each opening input and remove alternating earlier calendar sessions."""
    raw = read_history(ROOT / "data" / "historical_info")
    extra = read_supplement(ROOT / "user_historical_info.csv")
    calendar = sorted(extra["date"].unique())
    frames = []
    for date in dates:
        recent = [session for session in calendar if session <= date][-WINDOW - 1:]
        keep = {date, *recent[:-1:2]}
        features = build_features(raw.loc[raw["date"].isin(keep)], extra.loc[extra["date"].isin(recent)])
        frames.append(features.xs(date, level="date", drop_level=False))
    return pd.concat(frames)


def validate_holdout(panel, snapshot, estimators, winner):
    cut = snapshot["cutoff"]
    row_dates = panel.index.get_level_values("date")
    hold = panel.loc[row_dates > cut]
    fit, available = panel.loc[(row_dates <= cut) & panel["usable"]], hold.loc[hold["usable"]]
    probability = snapshot_probability(available, snapshot, estimators=estimators)
    result = evaluate(available, probability, decision(probability, available["market_iv"], snapshot["policy"]))
    result["date_bootstrap_90"] = intervals(available, decision(probability, available["market_iv"], snapshot["policy"]))
    flags = np.zeros(len(hold), dtype=bool)
    flags[hold["usable"].to_numpy()] = decision(probability, available["market_iv"], snapshot["policy"])
    baselines = {}
    for rule in RULES:
        prob, prediction, threshold = rule_prediction(fit, available, rule)
        baselines[rule] = {**evaluate(available, prob, prediction), "threshold": threshold}
    configs, _ = model_specs(winner)
    history_features = set(ABLATIONS["rolling"])
    sparse = thin_history(available.index.unique("date")).reindex(available.index)
    sparse = sparse.join(available[["fold", "y"]])
    sparse_probability = snapshot_probability(sparse, snapshot, estimators=estimators)
    scarce = evaluate(sparse, sparse_probability, decision(sparse_probability, sparse["market_iv"], snapshot["policy"]))
    partial = available[["daily_iv", "prev_range"]].isna().any(axis=1).to_numpy()
    selected_flags = flags[hold["usable"].to_numpy()]
    return {
        "dates": sorted(hold.index.unique("date").tolist()),
        "policy": snapshot["policy"], "eligible": result, "all_rows": classification(hold["y"], flags),
        "fully_unavailable": classification(hold.loc[~hold["usable"], "y"], flags[~hold["usable"].to_numpy()]),
        "baseline_comparison": baselines,
        "stress": stress_checks(available, probability, snapshot["policy"]),
        "partially_available": classification(available.loc[partial, "y"], selected_flags[partial]),
        "reduced_history": {**scarce, "thinned_dates": int(sparse.index.unique("date").size),
                            "uses_rolling": any(history_features & set(c["features"]) for c in configs)},
    }


def save_snapshot(snapshot, estimators, directory, probe):
    expected = snapshot_probability(probe, snapshot, estimators=estimators)
    for i, (component, estimator) in enumerate(zip(snapshot["components"], estimators)):
        kind = component["kind"]
        if kind == "lightgbm":
            component["file"] = f"{snapshot['cutoff']}_{i}.txt"
            estimator.booster_.save_model(str(directory / component["file"]))
        if kind == "catboost":
            component["file"] = f"{snapshot['cutoff']}_{i}.cbm"
            estimator.save_model(str(directory / component["file"]))
    restored = json.loads(json.dumps(snapshot, allow_nan=False))
    actual = snapshot_probability(probe, restored, directory)
    np.testing.assert_allclose(expected, actual, rtol=1e-11, atol=1e-12)
    return float(np.max(np.abs(expected - actual)))


def finalize():
    config = json.loads((ROOT / "model_config.json").read_text(encoding="utf-8"))
    if (config["seed"], config["warmup_dates"], config["refit_dates"], config["holdout_dates"]) != (
            SEED, WARMUP, REFIT, HOLDOUT):
        raise ValueError("Frozen model config does not match the validation schedule")
    dev, dev_dates, cutoff = load_panel()
    if config["development_end"] != cutoff:
        raise ValueError("Frozen model config does not match the development boundary")
    panel, dates, _ = load_panel(include_holdout=True)
    start = perf_counter()
    key, target, adjustment = config["model"], config["target"], config["calm_add"]
    specs, weights = model_specs(key)
    if config["components"] != specs or config["component_weights"] != weights:
        raise ValueError("Frozen component settings do not match training settings")
    source = CALIBRATED.get(key, key)
    constituents = ENSEMBLES.get(source, {source: 1.0})
    dev_oof, _ = out_of_fold(dev, dev_dates, {name: SPECS[name] for name in constituents}, rules=False)
    if key in ENSEMBLES:
        dev_oof[key] = sum(dev_oof[name] * value for name, value in constituents.items())
    if key in CALIBRATED:
        add_calibration(dev_oof, dev_dates, key, source)
    policy = policy_from_past(dev_oof, dev_oof[key].to_numpy(), target, adjustment)
    fit_rows = eligible(dev, dev_dates)
    hold_snapshot, hold_estimators = train_components(fit_rows, cutoff, specs, weights)
    hold_snapshot["policy"] = policy
    if key in CALIBRATED:
        hold_snapshot["calibration"] = fit_calibration(dev_oof[source], dev_oof["y"])
    holdout = validate_holdout(panel, hold_snapshot, hold_estimators, key)
    directory = ROOT / "models"
    directory.mkdir(exist_ok=True)
    max_delta, snapshots = 0.0, []
    for i in sorted(set(range(WARMUP, len(dates) + 1, REFIT)) | {len(dev_dates), len(dates)}):
        snapshot_dates = dates[:i]
        fit = eligible(panel, snapshot_dates)
        snapshot_date = snapshot_dates[-1]
        past = dev_oof.loc[dev_oof.index.get_level_values("date") <= snapshot_date]
        local_policy = policy_from_past(past, past[key].to_numpy(), target, adjustment)
        snapshot, fit_models = train_components(fit, snapshot_date, specs, weights)
        snapshot["policy"] = local_policy
        if key in CALIBRATED:
            snapshot["calibration"] = fit_calibration(past[source], past["y"])
        snapshot["fit_sessions"] = int(fit.index.unique("date").size)
        max_delta = max(max_delta, save_snapshot(snapshot, fit_models, directory, fit.tail(75)))
        snapshots.append(snapshot)
    write_json(directory / "snapshots.json", {"version": 1, "snapshots": snapshots})
    holdout.update(snapshot_count=len(snapshots), serialization_max_error=max_delta,
                   finalize_seconds=perf_counter() - start)
    write_json(ROOT / "reports" / "holdout.json", holdout)
    v = holdout["eligible"]
    print(f"Locked holdout {v['precision']:.4f} precision, {v['recall']:.4f} recall, "
          f"{v['score']:.4f} score on {v['rows']} usable rows. {len(snapshots)} snapshots.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("develop", "finalize", "all"), default="all")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.stage in ("develop", "all"):
            develop()
        if args.stage in ("finalize", "all"):
            finalize()


if __name__ == "__main__":
    main()
