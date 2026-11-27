#!/usr/bin/env python3
"""Generate public calendar features for each required date and stock."""

import argparse
import csv
from pathlib import Path

from features import calendar_values


def prepare(dates_path, names_path, out):
    dates = sorted(Path(dates_path).read_text(encoding="utf-8").split())
    names = Path(names_path).read_text(encoding="utf-8").split()
    if not dates or not names or len(set(dates)) != len(dates) or len(set(names)) != len(names):
        raise ValueError("The calendar and stock list must be nonempty and unique")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["date", "name", "weekday", "gap_days", "month_end"])
        previous = None
        for date in dates:
            values = calendar_values(date, previous)
            writer.writerows((date, name, *values) for name in names)
            previous = date
    return len(dates), len(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", default="data/all_dates.txt")
    parser.add_argument("--names", default="data/n50_names.txt")
    parser.add_argument("--out", default="user_historical_info.csv")
    args = parser.parse_args()
    dates, names = prepare(args.dates, args.names, args.out)
    print(f"Wrote {args.out} with {dates:,} dates and {names:,} stocks ({dates * names:,} rows)")


if __name__ == "__main__":
    main()
