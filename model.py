"""Portable snapshot inference and the fixed volatility fallback."""

from bisect import bisect_left
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(x):
    x = np.clip(x, -35, 35)
    return 1 / (1 + np.exp(-x))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def apply_calibration(probability, params):
    if params is None:
        return probability
    return sigmoid(params["slope"] * logit(probability) + params["intercept"])


def fallback_score(features):
    """A 2.45 cutoff blends daily implied SD and a scaled prior range."""
    iv = features["daily_iv"].to_numpy()
    realized = features["prev_range"].to_numpy() / 1.55
    combined = 0.70 * iv + 0.30 * realized
    return np.where(np.isfinite(combined), combined, np.fmax(iv, realized))


FALLBACK_CUTOFF = 2.45


def select_snapshot(snapshots, date):
    position = bisect_left([snapshot["cutoff"] for snapshot in snapshots], date) - 1
    return snapshots[position] if position >= 0 else None


def with_stock_prior(features, stock_rates, global_rate):
    values = features.index.get_level_values("name").map(stock_rates)
    return features.assign(stock_prior=np.where(pd.isna(values), global_rate, values))


def design(frame, spec):
    values = frame.reindex(columns=spec["features"]).to_numpy(float, copy=True)
    for index in spec["log"]:
        values[:, index] = np.log1p(np.maximum(0.0, values[:, index]))
    return values


def transform(values, params):
    values = np.where(np.isfinite(values), values, params["median"])
    values = np.clip(values, params["lower"], params["upper"])
    return (values - params["median"]) / params["scale"]


@lru_cache(maxsize=32)
def _load_tree(path, kind):
    if kind == "lightgbm":
        from lightgbm import Booster

        return Booster(model_file=str(path))
    from catboost import CatBoostClassifier

    tree = CatBoostClassifier()
    tree.load_model(str(path))
    return tree


def component_probability(values, params, directory=None, estimator=None):
    values = transform(values, params["preprocess"])
    kind = params["kind"]
    if kind == "logistic":
        return sigmoid(values @ params["coef"] + params["intercept"])
    if estimator is None:
        estimator = _load_tree(Path(directory) / params["file"], kind)
    if kind == "lightgbm":
        booster = estimator.booster_ if hasattr(estimator, "booster_") else estimator
        return booster.predict(values, num_threads=1)
    return estimator.predict_proba(values, thread_count=1)[:, 1]


def snapshot_probability(frame, snapshot, directory=None, estimators=None):
    frame = with_stock_prior(frame, snapshot.get("stock_rates", {}), snapshot["global_rate"])
    result = np.zeros(len(frame))
    for index, component in enumerate(snapshot["components"]):
        values = design(frame, component)
        fitted = estimators[index] if estimators is not None else None
        result += component["weight"] * component_probability(values, component, directory, fitted)
    return apply_calibration(result, snapshot.get("calibration"))


def decision(probability, market_iv, policy):
    adjustment = np.where(market_iv < policy.get("regime_cut", 1.7), policy.get("calm_add", 0), 0)
    return probability >= policy["threshold"] + adjustment
