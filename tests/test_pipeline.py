"""Leakage, scorer parity and evaluator-style inference checks."""

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from evaluator.run_evaluation import SKIP_EXT
from features import SUFFIX, build_features, read_history, read_supplement, safe_divide
from metrics import classification, final_score
from model import select_snapshot, snapshot_probability
from predict import forecast, inputs_for_date
from research_models import SPECS, stock_priors, train_components
from test_predictions import final_score as official_score
from test_predictions import score as official_counts
from train import save_snapshot

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "data" / "historical_info"
EXTRA = ROOT / "user_historical_info.csv"
MODELS = ROOT / "models"


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.supplement = read_supplement(EXTRA)
        cls.raw = read_history(HISTORY)
        cls.features = build_features(cls.raw, cls.supplement)
        cls.manifest = json.loads((MODELS / "snapshots.json").read_text(encoding="utf-8"))["snapshots"]

    def test_metric_and_strict_precision(self):
        y = np.array([True, False, True, True, False, True, False])
        p = np.array([True, True, True, False, True, True, False])
        keys = list(zip(["20260101"] * len(y), map(str, range(len(y)))))
        expected = official_counts(dict(zip(keys, y)), dict(zip(keys, p)))
        actual = classification(y, p)
        for key in ("precision", "recall", "score", "tp", "fp", "tn", "fn"):
            self.assertAlmostEqual(expected[key], actual[key])
        self.assertFalse(actual["passed"])
        self.assertEqual(final_score(0, 0.7), official_score(0, 0.7))
        self.assertAlmostEqual(final_score(0.656, 0.297), official_score(0.656, 0.297))

    def test_safe_parsing_and_asof(self):
        result = safe_divide([4, 0, np.inf, 4], [2, 0, 3, np.nan])
        np.testing.assert_allclose(result, [2, np.nan, np.nan, np.nan], equal_nan=True)
        raw = self.raw.iloc[:2].copy()
        raw.loc[:, "vix_asof"] = raw["date"]
        raw.loc[:, "pclose"] = ["", "0"]
        raw.loc[:, "phi"] = ["inf", "broken"]
        f = build_features(raw)
        self.assertTrue(f["daily_iv"].isna().all())
        self.assertTrue(f["prev_range"].isna().all())
        self.assertFalse(f["usable"].any())

    def test_batch_equals_inference_and_dates_are_sorted(self):
        days = self.features.index.unique("date")
        self.assertTrue(days.is_monotonic_increasing)
        for date in (days[0], days[39], days[-1]):
            single = inputs_for_date(date, HISTORY, EXTRA)
            batch = self.features.xs(date, level="date", drop_level=False)
            pd.testing.assert_frame_equal(single, batch, check_exact=False, rtol=1e-10, atol=1e-10)
        absent_extra = inputs_for_date(days[-1], HISTORY, ROOT / "missing_optional.csv")
        snapshot = select_snapshot(self.manifest, days[-1])
        self.assertEqual(len(forecast(absent_extra, snapshot)[1]), 50)

    def test_range_not_shifted_and_no_return_across_missing_session(self):
        raw = pd.DataFrame([
            dict(date="20260105", name="A", pclose="100", phi="105", plo="99"),
            dict(date="20260107", name="A", pclose="120", phi="122", plo="116"),
            dict(date="20260108", name="A", pclose="132", phi="134", plo="131"),
        ])
        supplementary = pd.DataFrame(
            {"date": ["20260105", "20260106", "20260107", "20260108"], "name": "A"})
        features = build_features(raw, supplementary)
        self.assertAlmostEqual(features.loc[("20260107", "A"), "prev_range"], 5.0)
        self.assertTrue(np.isnan(features.loc[("20260107", "A"), "abs_return"]))
        self.assertAlmostEqual(features.loc[("20260108", "A"), "abs_return"], 10.0)
        self.assertAlmostEqual(features.loc[("20260108", "A"), "range_mean_3"], (5 + 300 / 132) / 2)

    def test_prior_and_snapshot_never_use_same_day_labels(self):
        panel = self.features.iloc[:150].copy()
        panel["y"] = True
        initial, _, _ = stock_priors(panel)
        panel.loc[panel.index.get_level_values("date") == panel.index[-1][0], "y"] = False
        replaced, _, _ = stock_priors(panel)
        pd.testing.assert_series_equal(initial["stock_prior"], replaced["stock_prior"])
        first, second = self.manifest[:2]
        self.assertIsNone(select_snapshot(self.manifest, first["cutoff"]))
        self.assertEqual(select_snapshot(self.manifest, second["cutoff"]), first)
        self.assertEqual(select_snapshot(self.manifest, "20990101"), self.manifest[-1])

    def test_future_invariance_and_determinism(self):
        mid = self.features.index.unique("date")[45:55]
        date = self.features.loc[mid].loc[lambda frame: frame["usable"]].index[0][0]
        snapshot = select_snapshot(self.manifest, date)
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            hist = dest / "historical_info"
            shutil.copytree(HISTORY, hist)
            supplemental = dest / "user_historical_info.csv"
            self.supplement.loc[self.supplement["date"] <= date].to_csv(supplemental, index=False)
            original = inputs_for_date(date, hist, supplemental)
            p0, y0 = forecast(original, snapshot)
            for path in hist.glob(f"*{SUFFIX}"):
                if path.name[:8] > date:
                    path.write_text("this file must not be read", encoding="utf-8")
            (hist / f"20990101{SUFFIX}").write_text("future observation", encoding="utf-8")
            with supplemental.open("a", encoding="utf-8") as file:
                file.write("20990101,ADANIENT,1,2,1\n")
            again = inputs_for_date(date, hist, supplemental)
            p1, y1 = forecast(again, snapshot)
            pd.testing.assert_frame_equal(original, again)
            np.testing.assert_array_equal(p0, p1)
            np.testing.assert_array_equal(y0, y1)

    def test_serialization_parity(self):
        fit = self.features.iloc[:600].copy()
        fit = fit.loc[fit["usable"]].copy()
        fit["y"] = fit["daily_iv"] > fit["daily_iv"].median()
        config = [SPECS["log_market"], SPECS["lgb_shallow"], SPECS["cat_shallow"]]
        snapshot, estimators = train_components(fit, fit.index[-1][0], config, [0.4, 0.3, 0.3])
        with tempfile.TemporaryDirectory() as directory:
            maximum = save_snapshot(snapshot, estimators, Path(directory), fit)
        self.assertLess(maximum, 1e-11)

    def test_all_calendar_dates_with_synthetic_complete_inputs(self):
        dates = (ROOT / "data" / "all_dates.txt").read_text(encoding="utf-8").split()
        names = (ROOT / "data" / "n50_names.txt").read_text(encoding="utf-8").split()
        expected = pd.MultiIndex.from_product([dates, names], names=["date", "name"]).sort_values()
        raw_supplement = pd.read_csv(EXTRA, dtype={"date": str, "name": str})
        self.assertEqual(raw_supplement.columns[:2].tolist(), ["date", "name"])
        pd.testing.assert_index_equal(raw_supplement.set_index(["date", "name"]).sort_index().index, expected)
        mock = self.supplement[["date", "name"]].copy()
        mock["pclose"] = 500
        mock["phi"], mock["plo"] = 511, 493
        mock["vix_asof"] = (pd.to_datetime(mock["date"]) - pd.Timedelta(days=1)).dt.strftime("%Y%m%d")
        mock["vix"] = 28
        features = build_features(mock, self.supplement)
        for date in dates:
            snapshot = select_snapshot(self.manifest, date)
            rows = features.xs(date, level="date", drop_level=False)
            probabilities, flags = forecast(rows, snapshot)
            self.assertEqual(len(flags), len(names))
            self.assertTrue(np.isfinite(probabilities).all())

    def test_isolated_cli_and_stock_coverage(self):
        day = self.features.index.unique("date")[-1]
        with tempfile.TemporaryDirectory() as directory:
            box = Path(directory)
            code, history, output = (box / name for name in ("code", "historical_info", "predictions"))
            code.mkdir()
            history.mkdir()
            for name in ("predict.py", "features.py", "model.py"):
                shutil.copyfile(ROOT / name, code / name)
            shutil.copytree(MODELS, code / "models", ignore=lambda _, names: [
                name for name in names if name.lower().endswith(SKIP_EXT)])
            for path in HISTORY.glob(f"*{SUFFIX}"):
                if path.name[:8] <= day:
                    shutil.copyfile(path, history / path.name)
            extra = box / EXTRA.name
            self.supplement.loc[self.supplement["date"] <= day].to_csv(extra, index=False)
            args = [sys.executable, "predict.py", "--date", day, "--historical-info", str(history),
                    "--user-historical-info", str(extra), "--out-dir", str(output)]
            self.assertFalse(list(box.rglob("train_labels.csv")))
            subprocess.run(args, cwd=code, check=True, capture_output=True, timeout=15)
            path = output / f"{day}.predictions.csv"
            original = path.read_bytes()
            subprocess.run(args, cwd=code, check=True, capture_output=True, timeout=15)
            self.assertEqual(path.read_bytes(), original)
            with path.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                self.assertEqual(reader.fieldnames, ["name", "prediction"])
                rows = list(reader)
            expected = self.features.xs(day).index
            self.assertEqual(len(rows), len(expected))
            self.assertEqual({row["name"] for row in rows}, set(expected))
            self.assertTrue(all(row["prediction"] in {"True", "False"} for row in rows))
            current = inputs_for_date(day, HISTORY, EXTRA)
            probability = snapshot_probability(current, select_snapshot(self.manifest, day), MODELS)
            self.assertTrue(np.isfinite(probability).all())


if __name__ == "__main__":
    unittest.main()
