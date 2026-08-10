"""Emit the model-ready dataset, train, and write win probabilities.

Produces four files:
    ml_train.csv        11,051 closed deals -- 11 features + is_won label. Feed to any classifier.
    ml_score.csv        1,912 open deals    -- same 11 features, no label. What you predict on.
    ml_predictions.csv  1,912 open deals    -- win_probability + health band + acv
    ml_schema.csv       column dictionary   -- name, type, meaning

CSV rather than parquet so this opens in Excel/Sheets and loads anywhere.
Categoricals are left as strings; encode them however your model wants
(one-hot for linear, native categorical for LightGBM/CatBoost).

Usage:
    python build_features.py && python make_dataset.py
"""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from build_features import CATEGORICALS, NO_AMOUNT_FEATURES

FEATURES = NO_AMOUNT_FEATURES
LABEL = "is_won"
# join key + revenue weight + split key; carried in the CSVs but NEVER fed to the model.
# created_date is here so a downstream notebook can do the time split itself -- CSV row
# order is dim_cache order, NOT chronological, and a positional split on it silently
# becomes a quasi-random split (78 test wins instead of 49, PR-AUC 0.75 instead of 0.56).
PASSTHROUGH = ["display_id", "acv", "created_date"]

# Green/Yellow/Red cut on CALIBRATED probability. Placeholder thresholds -- sales
# owns these numbers, and they should be set from the reliability table, not taste.
HEALTH_EDGES = [-0.01, 0.10, 0.30, 1.0]
HEALTH_LABELS = ["Red", "Yellow", "Green"]

SCHEMA = [
    ("days_in_current_stage", "number", "Days since the stage last changed"),
    ("days_in_sales_cycle", "number", "Days since the deal was created"),
    ("stall_ratio", "number 0-1", "days_in_current_stage / days_in_sales_cycle -- share of the deal's life parked in one stage"),
    ("contract_months", "number", "Contract term in months"),
    ("num_stakeholders", "count", "Number of contacts on the deal"),
    ("has_champion", "0/1", "A champion is named on the deal"),
    ("poc", "0/1", "POC recorded (checkbox -- see caveat in build_features.py)"),
    ("opportunity_type", "category", "new / renewal / upsell / amendment"),
    ("source", "category", "outbound / inbound / events / partner / advisor / referral / unknown"),
    ("geo", "category", "amer / apj / emea / unknown"),
    ("segment", "category", "enterprise / mid-market / smb / start-up / 2k / unknown"),
    ("is_won", "0/1 LABEL", "1 = closed won, 0 = closed lost. Present in ml_train.csv only"),
    ("display_id", "key", "Deal ID -- join key, not a feature"),
    ("acv", "number", "Annual contract value. NOT a feature -- used only to weight the revenue roll-up"),
    ("created_date", "date", "Deal creation timestamp. NOT a feature (cohort censoring) -- sort by it to build the time split"),
]


def encode(train: pd.DataFrame, score: pd.DataFrame):
    """One-hot the categoricals, aligning score columns to train's."""
    num = [f for f in FEATURES if f not in CATEGORICALS]
    cats = [f for f in FEATURES if f in CATEGORICALS]

    def build(df):
        out = df[num].astype(float).copy()
        for c in cats:
            out = pd.concat([out, pd.get_dummies(df[c], prefix=c, dtype=float)], axis=1)
        return out

    xtr, xsc = build(train), build(score)

    # reindex below silently zeroes any level the model never trained on, so such a deal
    # gets scored as if the field were blank. Surface it instead of swallowing it --
    # geo="public sector" appears on open deals and never on closed ones.
    for c in cats:
        unseen = set(score[c].unique()) - set(train[c].unique())
        for lvl in sorted(unseen):
            n = (score[c] == lvl).sum()
            print(f"  WARNING unseen level {c}={lvl!r} on {n} open deals -- "
                  "scored as all-zeros, no training support")

    # Levels the model has almost no positives for. Predictions on these are not
    # trustworthy no matter what the calibration curve says.
    for c in cats:
        share_score = score[c].value_counts(normalize=True)
        wins = train.groupby(c)[LABEL].agg(["size", "sum"])
        for lvl, row in wins.iterrows():
            if row["sum"] <= 1 and share_score.get(lvl, 0) > 0.02:
                print(f"  WARNING thin level {c}={lvl!r}: {int(row['sum'])} win(s) in "
                      f"{int(row['size'])} closed deals, but {share_score[lvl]:.0%} of open pipeline")

    xsc = xsc.reindex(columns=xtr.columns, fill_value=0.0)
    return xtr.to_numpy(), xsc.to_numpy()


def main():
    # Fixed shuffle before any fitting. HistGB's early-stopping holdout is the LAST 15%
    # of rows as given, so on chronologically-ordered data it validates on one era only.
    # Deterministic seed, so the output is still reproducible run to run.
    train = pd.read_parquet("train.parquet").sample(frac=1, random_state=0).reset_index(drop=True)
    score = pd.read_parquet("score.parquet")

    train[PASSTHROUGH + FEATURES + [LABEL]].to_csv("ml_train.csv", index=False)
    score[PASSTHROUGH + FEATURES].to_csv("ml_score.csv", index=False)
    pd.DataFrame(SCHEMA, columns=["column", "type", "meaning"]).to_csv("ml_schema.csv", index=False)

    print(f"ml_train.csv   {len(train):>6,} closed deals  {train[LABEL].sum()} won "
          f"({train[LABEL].mean():.1%})  {len(FEATURES)} features + label")
    print(f"ml_score.csv   {len(score):>6,} open deals    {len(FEATURES)} features, no label")
    print(f"ml_schema.csv  {len(SCHEMA):>6} columns documented")

    # Sigmoid (Platt) not isotonic: isotonic needs several hundred positives to behave
    # and we have 541. 3 folds keeps each large enough to hold real positives.
    #
    # shuffle=True matters. cv=3 alone uses UNSHUFFLED StratifiedKFold, so on
    # chronologically-sorted rows each calibration fold is a different era of the
    # business and the fitted sigmoid depends on row order: expected wins swung 71.8
    # (file order) / 79.2 (by created_date) / 72.9 (reversed) -- a 10% move in the
    # headline revenue number from nothing but sort order. Shuffling cuts that spread
    # from 7.4 wins to 3.7. The residual is HistGB's own early-stopping holdout, which
    # is also a positional slice -- hence shuffle_train() below.
    x_train, x_score = encode(train, score)
    model = CalibratedClassifierCV(
        HistGradientBoostingClassifier(
            max_depth=3, min_samples_leaf=50, max_iter=200, learning_rate=0.05,
            early_stopping=True, validation_fraction=0.15, random_state=0),
        method="sigmoid", cv=StratifiedKFold(3, shuffle=True, random_state=0),
    ).fit(x_train, train[LABEL])

    out = score[PASSTHROUGH + ["stage_name"] + FEATURES].copy()
    out["win_probability"] = model.predict_proba(x_score)[:, 1]
    out["health"] = pd.cut(out["win_probability"], HEALTH_EDGES, labels=HEALTH_LABELS)
    out["expected_revenue"] = out["win_probability"] * out["acv"]
    out = out.sort_values("win_probability", ascending=False)
    out.to_csv("ml_predictions.csv", index=False)

    print(f"\nml_predictions.csv  {len(out):,} open deals scored")
    print(f"  win_probability   min {out.win_probability.min():.3f}  "
          f"median {out.win_probability.median():.3f}  max {out.win_probability.max():.3f}")
    print(f"  expected wins     {out.win_probability.sum():.0f} of {len(out):,}")
    print(f"  expected revenue  ${out.expected_revenue.sum() / 1e6:.1f}M")

    real_amount = (score["acv_missing"] == 0).mean() if "acv_missing" in score else np.nan
    print(f"  ^ amount is real on {real_amount:.0%} of these deals -- the rest is "
          "imputed (shrunk geometric mean by segment x type), so quote revenue as a "
          "range with coverage stated")

    print("\nhealth bands:")
    print(out["health"].value_counts().reindex(HEALTH_LABELS).to_string())
    print("\ntop 5 by win probability:")
    print(out[["display_id", "stage_name", "win_probability", "acv", "health"]]
          .head(5).round(3).to_string(index=False))

    # Self-check: calibrated probabilities must be usable as a revenue weight.
    assert out.win_probability.between(0, 1).all(), "probability outside [0,1]"
    assert out.health.notna().all(), "deal fell outside every health band"
    base = train[LABEL].mean()
    assert 0.2 * base < out.win_probability.mean() < 5 * base, (
        f"mean prediction {out.win_probability.mean():.4f} implausible vs base rate {base:.4f}"
    )
    print("\nchecks passed")


if __name__ == "__main__":
    main()
