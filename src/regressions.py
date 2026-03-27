"""Reproduce the paper's main OLS regression: PVI ~ after.

Usage:
    python regressions.py --before data/before_PVI.json --after data/after_PVI.json
"""

import argparse

import pandas as pd
import statsmodels.formula.api as smf


FORMULA = 'PVI ~ C(time_period, Treatment(reference="before"))'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run main OLS regression on PVI data.")
    parser.add_argument("--before", required=True, help="Path to before-period PVI JSONL.")
    parser.add_argument("--after", required=True, help="Path to after-period PVI JSONL.")
    args = parser.parse_args()

    before = pd.read_json(args.before, lines=True)
    after = pd.read_json(args.after, lines=True)
    before["time_period"] = "before"
    after["time_period"] = "after"
    df = pd.concat([before, after], ignore_index=True)

    print(f"Loaded {len(before):,} before + {len(after):,} after = {len(df):,} total\n")
    print(smf.ols(formula=FORMULA, data=df).fit().summary())
