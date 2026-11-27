#!/usr/bin/env python3
"""Summarize the supplied training inputs and create two development plots."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from features import PRICE, VALUE, VIX, Z, read_history
from train import ROOT, load_panel


def _quantiles(series):
    return {str(key): float(value) for key, value in series.quantile([0.01, 0.25, 0.5, 0.75, 0.99]).items()}


def main():
    panel, _, _ = load_panel(include_holdout=True)
    raw = read_history(ROOT / "data" / "historical_info")
    numeric = raw[PRICE + VALUE + VIX + Z].apply(pd.to_numeric, errors="coerce")
    good = panel.loc[panel["usable"]]
    day = panel.groupby(level="date").agg(event_rate=("y", "mean"), market_iv=("market_iv", "first"),
                                         paired_rows=("usable", "sum"))
    stock = panel.groupby(level="name")["y"].agg(["mean", "size"])
    keys = ("daily_iv", "prev_range", "market_iv", "market_range", "log_rel_value", "log_median_value")
    bins = {}
    for key in keys:
        ordered = good[[key, "y"]].dropna()
        quantile = pd.qcut(ordered[key], 5, duplicates="drop")
        grouped = ordered.groupby(quantile, observed=True).agg(
            lower=(key, "min"), upper=(key, "max"), event_rate=("y", "mean"), rows=("y", "size"))
        bins[key] = grouped.to_dict(orient="records")
    result = {
        "rows": len(panel), "date_count": len(day), "stock_count": len(stock),
        "schema": {key: ("float" if key in numeric.columns else "string") for key in raw.columns},
        "duplicates": int(raw.duplicated(["date", "name"]).sum()),
        "empty_by_column": raw.isna().sum().astype(int).to_dict(),
        "infinite_by_column": np.isinf(numeric).sum().astype(int).to_dict(),
        "unparseable_by_column": (numeric.isna() & raw[numeric.columns].notna()).sum().astype(int).to_dict(),
        "positive_rate": float(panel["y"].mean()),
        "positive_rate_usable": float(good["y"].mean()),
        "fully_masked_dates": int(day["paired_rows"].eq(0).sum()),
        "partial_primary_rows_on_usable_dates": int(good[["daily_iv", "prev_range"]].isna().any(axis=1).sum()),
        "primary_rows_available": len(good),
        "by_stock": stock.rename(columns={"mean": "event_rate", "size": "rows"}).to_dict(orient="index"),
        "by_date": day.astype(object).where(day.notna(), None).to_dict(orient="index"),
        "feature_quantiles": {key: _quantiles(good[key]) for key in keys + ("market_iv_iqr", "market_range_iqr")},
        "raw_value_quantiles_rupees": {key: _quantiles(numeric[key]) for key in VALUE},
        "iv_correlation": numeric[VIX + Z].corr(method="spearman").to_dict(),
        "range_iv_spearman": float(good[["prev_range", "daily_iv"]].corr(method="spearman").iloc[0, 1]),
        "equal_6m_1y_fraction": float(numeric["vix_6m"].eq(numeric["vix_1y"]).loc[numeric["vix"].notna()].mean()),
        "target_rate_by_feature_quintiles": bins,
    }
    directory = ROOT / "reports"
    directory.mkdir(exist_ok=True)
    (directory / "eda.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    colors = ["#146c75", "#b26339"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    for key, color, label in zip(keys[:2], colors, ("Stock implied daily volatility", "Previous-session range")):
        ax.plot(np.arange(1, len(bins[key]) + 1), [row["event_rate"] for row in bins[key]],
                marker="o", linewidth=2, color=color, label=label)
    ax.set(xticks=range(1, 6), xlabel="Feature quintile, lowest to highest", ylabel="Move frequency",
           title="More volatile stocks crossed the 2% band more often", ylim=(0, 0.85))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(directory / "volatility_quintiles.png", dpi=145)
    plt.close(fig)

    fig, (upper, lower) = plt.subplots(2, 1, sharex=True, figsize=(8, 4.8), layout="constrained")
    dates = pd.to_datetime(day.index, format="%Y%m%d")
    upper.scatter(dates, day["event_rate"], alpha=0.38, s=14, color=colors[0])
    upper.plot(dates, day["event_rate"].rolling(7, min_periods=1).mean(), lw=2, color=colors[0])
    upper.set(ylabel="Move frequency", title="Volatility regimes across supplied training sessions")
    upper.yaxis.set_major_formatter(PercentFormatter(1.0))
    lower.scatter(dates, day["market_iv"], s=15, color=colors[1], label="Available market median")
    lower.set(ylabel="Median daily IV", xlabel="Trading date (2026)")
    lower.yaxis.set_major_formatter(PercentFormatter(100.0))
    lower.xaxis.set_major_formatter(DateFormatter("%b"))
    for ax in (upper, lower):
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(directory / "market_regimes.png", dpi=145)
    plt.close(fig)
    print(f"Wrote training-only EDA to {Path(directory).name}")


if __name__ == "__main__":
    main()
