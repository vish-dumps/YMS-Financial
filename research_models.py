"""Train compact regularized candidates on expanding as-of windows."""

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from features import GROUPS
from model import design, snapshot_probability, transform
from validation import SEED

CORE = ["daily_iv", "iv_1m", "iv_to_1m", "iv_z_smooth", "prev_range"]
MARKET = ["market_iv", "market_range", "market_iv_iqr", "iv_market_relative", "range_market_relative"]
HISTORY = ["range_mean_5", "range_ewm_10", "range_mean_20", "iv_mean_20", "range_short_long"]
ACTIVITY = ["log_median_value", "log_rel_value", "liq_rank"]
STOCK = ["stock_prior"]
EXTRAS = ["iv_to_3m", "iv_z_diff", "close_position", "range_to_iv", "abs_return",
          "range_std_10", "range_max_10", "return_vol_10", "range_rank", "market_high_iv"]
WIDE = CORE + MARKET + HISTORY + ACTIVITY + STOCK + EXTRAS + GROUPS["calendar"]
LOG_KEYS = {"daily_iv", "iv_1m", "prev_range", "market_iv", "market_range",
            "range_mean_5", "range_ewm_10", "range_mean_20", "iv_mean_20", "abs_return",
            "range_max_10", "range_std_10", "return_vol_10", "market_iv_iqr"}


def spec(kind, features, half_life=35, **params):
    return {"kind": kind, "features": features, "half_life": half_life, "params": params}


SPECS = {
    "log_simple": spec("logistic", ["daily_iv", "prev_range"], C=0.085),
    "log_core": spec("logistic", CORE, C=0.15),
    "log_shape": spec("logistic", CORE + ["iv_z_diff", "prev_up", "prev_down", "close_position",
                                        "range_ewm_10", "market_range", "log_rel_value"], C=0.09),
    "log_pulse": spec("logistic", ["daily_iv", "iv_1m", "iv_to_1m", "prev_range", "market_range",
                                  "range_ewm_10", "iv_rank", "iv_z_diff", "log_rel_value"],
                      half_life=22, C=0.065),
    "log_market": spec("logistic", CORE + MARKET + STOCK, C=0.10),
    "log_recent": spec("logistic", CORE + MARKET + HISTORY + STOCK, half_life=15, C=0.075),
    "log_extended": spec("logistic", WIDE, C=0.055),
    "lgb_shallow": spec("lightgbm", CORE + MARKET + HISTORY + STOCK, max_depth=2, num_leaves=4,
                        n_estimators=120, learning_rate=0.045, min_child_samples=50, reg_lambda=10),
    "lgb_wide": spec("lightgbm", WIDE, max_depth=3, num_leaves=7, n_estimators=145,
                     learning_rate=0.04, min_child_samples=50, reg_lambda=12, feature_fraction=0.85),
    "cat_shallow": spec("catboost", CORE + MARKET + HISTORY + STOCK, depth=3,
                        iterations=140, learning_rate=0.045, l2_leaf_reg=8),
    "cat_wide": spec("catboost", WIDE, depth=5, iterations=170, learning_rate=0.035, l2_leaf_reg=12),
}
ENSEMBLES = {
    "blend_log_lgb": {"log_market": 0.65, "lgb_shallow": 0.35},
    "blend_lgb_cat": {"lgb_shallow": 0.5, "cat_shallow": 0.5},
    "blend_log_cat": {"log_market": 0.65, "cat_shallow": 0.35},
}
ABLATIONS = {
    "implied": set(GROUPS["implied"]), "range": set(GROUPS["range"]),
    "rolling": set(GROUPS["rolling"]), "liquidity": set(GROUPS["liquidity"]) | {"liq_rank"},
    "cross": set(GROUPS["cross"]) - {"liq_rank"}, "stock": set(STOCK), "calendar": set(GROUPS["calendar"]),
}


def stock_priors(panel, strength=40):
    """Each fitting row's prior uses only labels from strictly earlier dates."""
    label = panel["y"].astype(float).unstack("name").sort_index()
    n = label.notna().cumsum().shift(fill_value=0)
    wins = label.fillna(0).cumsum().shift(fill_value=0)
    global_by_date = (wins.sum(axis=1) + 20) / (n.sum(axis=1) + 50)
    rates = (wins.add(strength * global_by_date, axis=0) / (n + strength)).stack(future_stack=True)
    global_end = float((label.sum().sum() + 20) / (label.notna().sum().sum() + 50))
    stock_end = (label.sum() + strength * global_end) / (label.notna().sum() + strength)
    return panel.assign(stock_prior=rates.reindex(panel.index)), stock_end.to_dict(), global_end


def _preprocess(values):
    columns = [values[np.isfinite(values[:, i]), i] for i in range(values.shape[1])]
    bounds = np.array([np.quantile(col, [0.01, 0.25, 0.5, 0.75, 0.99])
                       if len(col) else np.zeros(5) for col in columns])
    lower, q1, median, q3, upper = bounds.T
    scale = np.where(q3 > q1, q3 - q1, 1.0)
    return {key: value.tolist() for key, value in
            zip(("median", "lower", "upper", "scale"), (median, lower, upper, scale))}


def train_components(panel, cutoff, specs, weights=None):
    if not panel.index.get_level_values("date").max() <= cutoff:
        raise ValueError("Training dates exceed snapshot cutoff")
    panel, stock_rates, global_rate = stock_priors(panel)
    dates = panel.index.get_level_values("date")
    age = len(dates.unique()) - 1 - dates.unique().get_indexer(dates)
    snapshot = {"cutoff": cutoff, "global_rate": global_rate, "stock_rates": stock_rates, "components": []}
    fitted = []
    weights = weights or [1.0] * len(specs)
    for config, weight in zip(specs, weights):
        kind = config["kind"]
        component = {"kind": kind, "features": config["features"],
                     "log": [i for i, key in enumerate(config["features"]) if key in LOG_KEYS],
                     "weight": weight}
        x = design(panel, component)
        prep = _preprocess(x)
        data = transform(x, prep)
        sample_weight = np.exp2(-age / config["half_life"])
        if kind == "lightgbm":
            from lightgbm import LGBMClassifier

            fit = LGBMClassifier(**config["params"], verbosity=-1, random_state=SEED,
                                 deterministic=True, force_col_wise=True, n_jobs=1)
        elif kind == "catboost":
            from catboost import CatBoostClassifier

            fit = CatBoostClassifier(**config["params"], random_seed=SEED, thread_count=1,
                                     verbose=False, allow_writing_files=False, has_time=True)
        else:
            fit = LogisticRegression(**config["params"], max_iter=500)
        with threadpool_limits(limits=1):
            fit.fit(data, panel["y"], sample_weight=sample_weight)
        component["preprocess"] = prep
        if kind == "logistic":
            component.update(coef=fit.coef_[0].tolist(), intercept=float(fit.intercept_[0]))
            fit = None
        fitted.append(fit)
        snapshot["components"].append(component)
    if not any("stock_prior" in config["features"] for config in specs):
        snapshot.pop("stock_rates")
    return snapshot, fitted


def run_fit(fit_rows, validation, cutoff, config):
    start = perf_counter()
    snapshot, fitted = train_components(fit_rows, cutoff, [config])
    fit_seconds = perf_counter() - start
    start = perf_counter()
    probabilities = snapshot_probability(validation, snapshot, estimators=fitted)
    return probabilities, fit_seconds, perf_counter() - start
