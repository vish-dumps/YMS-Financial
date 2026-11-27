#!/usr/bin/env python3
"""Run the frozen point-in-time predictor for one trading date."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from features import SUFFIX, WINDOW, build_features, day, read_history, read_supplement
from model import FALLBACK_CUTOFF, decision, fallback_score, select_snapshot, sigmoid, snapshot_probability

MODELS = Path(__file__).resolve().parent / "models"


def forecast(frame, snapshot, directory=MODELS):
    fallback = fallback_score(frame)
    usable = frame["usable"].to_numpy(bool)
    if snapshot is None:
        probability = np.where(np.isfinite(fallback), sigmoid((fallback - FALLBACK_CUTOFF) / 0.65), 0.0)
        return probability, fallback >= FALLBACK_CUTOFF
    probabilities = np.zeros(len(frame))
    predicted = np.zeros(len(frame), dtype=bool)
    if usable.any():
        available = frame.loc[usable]
        probabilities[usable] = snapshot_probability(available, snapshot, directory)
        predicted[usable] = decision(probabilities[usable], available["market_iv"], snapshot["policy"])
    return probabilities, predicted


def inputs_for_date(date, historical_info, supplement_path):
    day(date)
    if not (Path(historical_info) / f"{date}{SUFFIX}").is_file():
        raise FileNotFoundError(f"No historical information for {date}")
    supplement = read_supplement(supplement_path, cutoff=date)
    sessions = sorted(supplement["date"].unique())
    minimum = None
    if date in sessions and len(sessions) > WINDOW:
        minimum = sessions[-WINDOW - 1]
        supplement = supplement.loc[supplement["date"] >= minimum]
    raw = read_history(historical_info, cutoff=date, minimum=minimum)
    features = build_features(raw, supplement).xs(date, level="date", drop_level=False)
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--historical-info", required=True, type=Path)
    parser.add_argument("--user-historical-info", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    frame = inputs_for_date(args.date, args.historical_info, args.user_historical_info)
    manifest = json.loads((MODELS / "snapshots.json").read_text(encoding="utf-8"))
    snapshot = select_snapshot(manifest["snapshots"], args.date)
    _, predicted = forecast(frame, snapshot)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / f"{args.date}.predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["name", "prediction"])
        writer.writerows((name, bool(flag)) for (_, name), flag in zip(frame.index, predicted))


if __name__ == "__main__":
    main()
