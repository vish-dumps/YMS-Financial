"""Point-in-time features from the evaluator's already lagged daily records."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SUFFIX = "_n50.info.csv"
WINDOW = 20
PRICE = ["pclose", "phi", "plo"]
VIX = ["vix", "vix_1m", "vix_3m", "vix_6m", "vix_1y"]
Z = ["vix_zscore_smooth", "vix_zscore_raw"]
VALUE = ["pvolume", "mddv_20"]
RAW = ["date", "name", "vix_asof", *PRICE, *VIX, *Z, *VALUE]
GROUPS = {
    "implied": ["daily_iv", "iv_band_ratio", "iv_1m", "iv_3m", "iv_1y",
                "iv_minus_1m", "iv_to_1m", "iv_to_3m", "iv_to_1y", "iv_term_slope",
                "iv_z_smooth", "iv_z_diff", "iv_age", "iv_acceleration", "iv_change_1"],
    "range": ["prev_range", "prev_up", "prev_down", "close_position",
              "parkinson", "abs_return", "range_to_iv", "range_iv_interaction"],
    "rolling": ["range_mean_3", "range_mean_5", "range_mean_10", "range_mean_20",
                "range_median_5", "range_max_10", "range_std_10", "range_ewm_10",
                "return_vol_10", "return_vol_20", "large_range_freq_10",
                "iv_mean_5", "iv_mean_20", "range_short_long", "range_persistence"],
    "liquidity": ["log_value", "log_median_value", "log_rel_value", "activity_iv"],
    "cross": ["iv_rank", "range_rank", "liq_rank", "market_iv", "market_range",
              "market_iv_iqr", "market_range_iqr", "iv_market_relative",
              "range_market_relative", "market_high_iv"],
    "calendar": ["weekday", "gap_days", "month_end"],
    "stock": ["stock_prior"],
}


def safe_divide(numerator, denominator):
    """Divide arrays while leaving invalid inputs as NaN."""
    a, b = np.broadcast_arrays(np.asarray(numerator, float), np.asarray(denominator, float))
    invalid = ~np.isfinite(a) | ~np.isfinite(b) | (b == 0)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = np.divide(a, b, out=np.full(a.shape, np.nan), where=~invalid)
    return np.where(np.isfinite(out), out, np.nan)


def day(value):
    return datetime.strptime(str(value), "%Y%m%d").date()


def calendar_values(date, previous=None):
    current = day(date)
    return (current.weekday(), (current - day(previous)).days if previous else 1,
            int((pd.Timestamp(current) + pd.offsets.BMonthEnd(0) - pd.Timestamp(current)).days <= 3))


def read_supplement(path, cutoff=None):
    empty = pd.DataFrame(columns=["date", "name"])
    if path is None or not Path(path).is_file():
        return empty
    try:
        supplement = pd.read_csv(path, dtype={"date": str, "name": str},
                                 usecols=lambda key: key in {"date", "name", *GROUPS["calendar"]})
    except pd.errors.EmptyDataError:
        return empty
    if not {"date", "name"} <= set(supplement.columns):
        return empty
    valid = (supplement["date"].str.fullmatch(r"[0-9]{8}", na=False)
             & pd.to_datetime(supplement["date"], format="%Y%m%d", errors="coerce").notna())
    supplement = supplement.loc[valid & supplement["name"].notna()]
    if cutoff is not None:
        supplement = supplement.loc[supplement["date"] <= cutoff]
    return supplement.drop_duplicates(["date", "name"], keep="first")


def read_history(directory, cutoff=None, minimum=None):
    frames = []
    for path in sorted(Path(directory).glob(f"*{SUFFIX}")):
        session = path.name.removesuffix(SUFFIX)
        if len(session) != 8 or not session.isascii() or not session.isdigit():
            continue
        if (cutoff is not None and session > cutoff) or (minimum is not None and session < minimum):
            continue
        frame = pd.read_csv(path, dtype=str, usecols=lambda key: key in RAW)
        frame["date"] = session
        frames.append(frame.reindex(columns=RAW))
    if not frames:
        raise ValueError("No eligible historical info files")
    return pd.concat(frames, ignore_index=True)


def _calendar(dates, supplement):
    supplied = sorted(set(supplement["date"].dropna().astype(str)))
    # Without a trading calendar, retain weekday gaps and avoid uncertain returns.
    if not supplied:
        weekdays = pd.bdate_range(day(dates[0]), day(dates[-1])).strftime("%Y%m%d")
        return sorted(set(dates) | set(weekdays)), False
    return sorted(set(dates) | set(supplied)), True


def _range_features(wide, dated, trusted_calendar):
    close, high, low = (wide[k] for k in PRICE)
    valid = (high >= low) & (high > 0) & (low > 0) & (close > 0)
    high, low = high.where(valid), low.where(valid)
    prev_range = safe_divide(high - low, close) * 100
    previous_close = close.shift()
    pct_return = pd.DataFrame((safe_divide(close, previous_close) - 1) * 100,
                              index=close.index, columns=close.columns)
    if not trusted_calendar:
        consecutive = dated.to_series().diff().dt.days.eq(1).to_numpy()
        pct_return.loc[~consecutive] = np.nan
    return {
        "prev_range": pd.DataFrame(prev_range, index=close.index, columns=close.columns),
        "prev_up": safe_divide(high - close, close) * 100,
        "prev_down": safe_divide(close - low, close) * 100,
        "close_position": safe_divide(close - low, high - low),
        "parkinson": np.log(high / low) * (100 / np.sqrt(4 * np.log(2))),
        "abs_return": pct_return.abs(),
    }, pct_return


def _rolling(ranges, iv, returns):
    result = {}
    for window in (3, 5, 10, 20):
        avg = ranges.rolling(window, min_periods=2).mean()
        result[f"range_mean_{window}"] = avg.combine_first(ranges)
    window5 = ranges.rolling(5, min_periods=2)
    window10 = ranges.rolling(10, min_periods=2)
    result["range_median_5"] = window5.median().combine_first(ranges)
    result["range_max_10"] = window10.max().combine_first(ranges)
    result["range_std_10"] = window10.std()
    decay = 0.8
    # A bounded exponential window gives identical results with truncated input.
    weights = decay ** np.arange(9, -1, -1)
    values = sum(ranges.shift(lag).fillna(0) * weight for lag, weight in enumerate(weights[::-1]))
    presence = sum(ranges.shift(lag).notna() * weight for lag, weight in enumerate(weights[::-1]))
    result["range_ewm_10"] = safe_divide(values, presence)
    for window in (10, 20):
        result[f"return_vol_{window}"] = returns.rolling(window, min_periods=3).std()
    result["large_range_freq_10"] = ((ranges > 3.0).where(ranges.notna())
                                     .rolling(10, min_periods=2).mean())
    for window in (5, 20):
        result[f"iv_mean_{window}"] = iv.rolling(window, min_periods=2).mean().combine_first(iv)
    result["range_short_long"] = safe_divide(result["range_mean_3"], result["range_mean_20"])
    result["range_persistence"] = safe_divide(ranges, result["range_mean_10"])
    return result


def _cross_section(iv, ranges, liquidity):
    market_iv, market_range = iv.median(axis=1), ranges.median(axis=1)
    return {
        "iv_rank": iv.rank(axis=1, pct=True), "range_rank": ranges.rank(axis=1, pct=True),
        "liq_rank": liquidity.rank(axis=1, pct=True), "market_iv": market_iv,
        "market_range": market_range, "market_iv_iqr": iv.quantile(0.75, axis=1) - iv.quantile(0.25, axis=1),
        "market_range_iqr": ranges.quantile(0.75, axis=1) - ranges.quantile(0.25, axis=1),
        "iv_market_relative": iv.div(market_iv, axis=0),
        "range_market_relative": ranges.div(market_range, axis=0),
        "market_high_iv": (iv > 2).where(iv.notna()).mean(axis=1),
    }


def build_features(raw, supplement=None):
    """Build a date/name-indexed table. Current rows contain previous-session prices."""
    if supplement is None:
        supplement = pd.DataFrame(columns=["date", "name"])
    raw = raw.loc[:, raw.columns.intersection(RAW)].reindex(columns=RAW)
    raw = raw.dropna(subset=["date", "name"]).drop_duplicates(["date", "name"], keep="first")
    raw = raw.set_index(["date", "name"]).sort_index()
    calendar, trusted_calendar = _calendar(raw.index.unique("date"), supplement)
    dated = pd.to_datetime(calendar, format="%Y%m%d")
    asof = pd.to_datetime(raw["vix_asof"], format="%Y%m%d", errors="coerce")
    trade_day = pd.to_datetime(raw.index.get_level_values("date"), format="%Y%m%d")
    age = (trade_day - asof).dt.days.where(lambda value: value > 0)
    for key in PRICE + VIX + VALUE + Z:
        column = pd.to_numeric(raw[key], errors="coerce").replace([np.inf, -np.inf], np.nan)
        raw[key] = column if key in Z else column.where(column > 0)
        if key in VIX + Z:
            raw[key] = raw[key].where(age.notna())
    wide = {k: raw[k].unstack("name").reindex(calendar) for k in PRICE + VIX + VALUE + Z}
    iv = wide["vix"] / np.sqrt(252)
    iv_1m, iv_3m, iv_1y = [wide[key] / np.sqrt(252) for key in ("vix_1m", "vix_3m", "vix_1y")]
    # 6m and 1y have identical support for most of the sample. Keep the longest.
    features = {
        "daily_iv": iv, "iv_band_ratio": iv / 2, "iv_1m": iv_1m, "iv_3m": iv_3m,
        "iv_1y": iv_1y, "iv_minus_1m": iv - iv_1m, "iv_to_1m": safe_divide(iv, iv_1m),
        "iv_to_3m": safe_divide(iv, iv_3m), "iv_to_1y": safe_divide(iv, iv_1y),
        "iv_term_slope": safe_divide(iv_1m, iv_1y), "iv_z_smooth": wide[Z[0]],
        "iv_z_diff": wide[Z[1]] - wide[Z[0]], "iv_age": age.unstack("name").reindex(calendar),
        "iv_acceleration": iv.diff().diff(), "iv_change_1": iv.diff(),
    }
    price_features, returns = _range_features(wide, dated, trusted_calendar)
    features.update(price_features)
    ranges = features["prev_range"]
    features.update(_rolling(ranges, iv, returns))
    log_relative = np.log(wide["pvolume"]) - np.log(wide["mddv_20"])
    features.update(range_to_iv=safe_divide(ranges, iv), range_iv_interaction=ranges * iv,
                    log_value=np.log(wide["pvolume"]), log_median_value=np.log(wide["mddv_20"]),
                    log_rel_value=log_relative, activity_iv=log_relative * iv)
    features.update(_cross_section(iv, ranges, wide["mddv_20"]))

    rows = [calendar_values(date, calendar[index - 1] if index else None)
            for index, date in enumerate(calendar)]
    calendar_frame = pd.DataFrame(rows, index=calendar, columns=GROUPS["calendar"])
    features.update({key: calendar_frame[key] for key in GROUPS["calendar"]})
    date_pos = pd.Index(calendar).get_indexer(raw.index.get_level_values("date"))
    name_pos = iv.columns.get_indexer(raw.index.get_level_values("name"))
    result = pd.DataFrame({
        key: (np.asarray(value, dtype=float)[date_pos] if np.ndim(value) == 1
              else np.asarray(value, dtype=float)[date_pos, name_pos])
        for key, value in features.items()
    }, index=raw.index)
    if not supplement.empty:
        known = supplement.set_index(["date", "name"]).reindex(raw.index)
        bounds = {"weekday": (0, 6), "gap_days": (1, 14), "month_end": (0, 1)}
        for key in GROUPS["calendar"]:
            if key in known:
                value = pd.to_numeric(known[key], errors="coerce")
                result[key] = result[key].mask(value.between(*bounds[key]), value)
    result["usable"] = result[["daily_iv", "prev_range"]].notna().any(axis=1)
    return result.replace([np.inf, -np.inf], np.nan)
