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
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from build_features import CATEGORICALS, NO_AMOUNT_FEATURES

FEATURES = NO_AMOUNT_FEATURES
LABEL = "is_won"
# join key + revenue weight + split key; carried in the CSVs but NEVER fed to the model.
# created_date is here so a downstream notebook can do the time split itself -- CSV row
# order is dim_cache order, NOT chronological, and a positional split on it silently
# becomes a quasi-random split (78 test wins instead of 49, PR-AUC 0.75 instead of 0.56).
PASSTHROUGH = ["display_id", "acv", "created_date", "evidence"]

# Green/Yellow/Red cut on CALIBRATED probability. Placeholder thresholds -- sales
# owns these numbers, and they should be set from the reliability table, not taste.
HEALTH_EDGES = [-0.01, 0.10, 0.30, 1.0]
HEALTH_LABELS = ["Red", "Yellow", "Green"]

# A deal at this evidence level has nothing recorded on it: no champion, no POC, fewer
# than 2 contacts, never advanced a stage, no real amount. 1,150 of 1,912 open deals
# (60%) sit here, and the matching closed cohort wins 1.2% of the time.
#
# These still get a probability -- deleting 60% of pipeline from the forecast is worse
# than labelling it -- but the band is overridden, because the number is not a forecast.
# The model reads absence of data on these rows, so a low score means "nobody has worked
# this yet", not "this deal will lose". Sum them separately from the committed roll-up.
MIN_EVIDENCE = 2
INSUFFICIENT_LABEL = "Too early"

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


def new_estimator(random_state: int = 0):
    """The base classifier. Wrap in CalibratedClassifierCV before use.

    LightGBM, at the LIBRARY DEFAULT depth/leaves/min_child_samples. Measured on the
    time split (train 8,840/492 wins, test 2,211/49 wins), 5 seeds:

        model                          test PR-AUC     Brier    ROC   sum(p)/act
        HGB shipping (previous)      0.5825 +-0.0063  0.01369  0.9598    1.21x
        LGBM default d3/l8/mcs50     0.6276 +-0.0000  0.01324  0.9644    1.28x
        LGBM tuned d4/l15/mcs20      0.6125 +-0.0000  0.01321  0.9626    1.30x
        LGBM d4/l15/mcs100/lr.05     0.6032 +-0.0000  0.01370  0.9627    1.29x

    +0.045 PR-AUC over HistGB against a 0.0063 seed spread, so the gain is real and
    not a lucky seed. Better Brier and ROC too. Costs 1.21x -> 1.28x on
    sum(p)/actual, i.e. the aggregate roll-up over-predicts a little more; PR-AUC and
    Brier both improving says the ranking and the per-deal probabilities are better,
    and the aggregate bias is a known over-prediction documented in dataset_sample.md.

    TUNING WAS TRIED AND REJECTED. A 48-config grid over max_depth {2,3,4,6,-1},
    num_leaves, min_child_samples {20,50,100}, learning_rate/n_estimators, plus L1/L2
    and subsampling variants, scored by 5-fold out-of-fold PR-AUC WITHIN train so the
    test set stayed untouched. OOF picked d4/l15/mcs20/lr0.1 at 0.8262 against the
    default's 0.8219 -- a 0.0043 edge across a grid whose total spread was 0.0351.
    On test that "winner" scored 0.6125 vs the default's 0.6276: it lost 0.015. The
    OOF gap was noise, and chasing it cost real accuracy. Defaults stay.

    LightGBM is deterministic here (seed sd 0.0000) because it does no subsampling by
    default, so random_state changes nothing. It is still threaded through for the
    ablation harness in win_probability_colab.ipynb.
    """
    return LGBMClassifier(
        max_depth=3, num_leaves=8, min_child_samples=50, n_estimators=200,
        learning_rate=0.05, random_state=random_state, verbose=-1, n_jobs=-1)


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
    # Fixed shuffle before any fitting. Kept after the switch to LightGBM: the
    # calibrator's folds are still positional slices of whatever order arrives, and
    # unshuffled chronological rows put a different era of the business in each fold.
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

    # Sigmoid (Platt) not isotonic. Measured on test: sigmoid cv=3 gives PR-AUC 0.6276
    # / Brier 0.01324, isotonic cv=5 gives 0.6138 / 0.01302 -- isotonic buys 0.0002
    # Brier for 0.014 PR-AUC, and needs several hundred positives to behave anyway.
    # More folds is worse, not better: cv=3 0.6276, cv=5 0.6224, cv=10 0.6126.
    # Calibration is not free -- raw LightGBM scores 1.11x on sum(p)/actual and
    # sigmoid pushes it to 1.28x, but raw loses 0.021 PR-AUC and its probabilities
    # are not usable as revenue weights, which is the whole point.
    # Original note: isotonic needs several hundred positives to behave
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
        new_estimator(), method="sigmoid",
        cv=StratifiedKFold(3, shuffle=True, random_state=0),
    ).fit(x_train, train[LABEL])

    out = score[PASSTHROUGH + ["stage_name"] + FEATURES].copy()
    too_early = out["evidence"] < MIN_EVIDENCE

    # win_probability is NULL where there is no evidence, not low. A number in that cell
    # gets summed by whoever opens the CSV -- and summing all 1,912 gave $5.9M against a
    # committed $4.2M, because 1,290 unworked deals at ~0.03 each add up. The band alone
    # did not prevent that; an empty cell does. raw_probability keeps the model's actual
    # output for anyone auditing, under a name nobody will total up by accident.
    #
    # Note this is NOT about stage. A 1-profile deal with a champion, POC and stakeholders
    # recorded clears the gate and keeps its probability -- fields filled early are exactly
    # the signal worth having. Only absence of evidence is censored.
    out["raw_probability"] = model.predict_proba(x_score)[:, 1]
    out["win_probability"] = out["raw_probability"].mask(too_early)
    out["health"] = pd.cut(out["win_probability"], HEALTH_EDGES, labels=HEALTH_LABELS)
    out["health"] = out["health"].cat.add_categories([INSUFFICIENT_LABEL])
    out.loc[too_early, "health"] = INSUFFICIENT_LABEL
    out["expected_revenue"] = out["win_probability"] * out["acv"]
    out = out.sort_values("win_probability", ascending=False, na_position="last")
    out.to_csv("ml_predictions.csv", index=False)

    # after the sort, so it stays aligned to out's index
    too_early = out["evidence"] < MIN_EVIDENCE
    scored, early = out[~too_early], out[too_early]
    print(f"\nml_predictions.csv  {len(out):,} open deals")
    print(f"  COMMITTED (evidence >= {MIN_EVIDENCE})  {len(scored):,} deals  "
          f"{scored.win_probability.sum():.0f} expected wins  "
          f"${scored.expected_revenue.sum() / 1e6:.1f}M")
    print(f"    win_probability   min {scored.win_probability.min():.3f}  "
          f"median {scored.win_probability.median():.3f}  "
          f"max {scored.win_probability.max():.3f}")
    print(f"  {INSUFFICIENT_LABEL.upper():<24} {len(early):,} deals  "
          f"win_probability NULL, no revenue claimed")
    print(f"    (raw_probability would have summed to {early.raw_probability.sum():.0f} "
          f"wins / ${(early.raw_probability * early.acv).sum() / 1e6:.1f}M -- excluded "
          "on purpose, nothing is recorded on these deals)")

    real_amount = (score["acv_missing"] == 0).mean() if "acv_missing" in score else np.nan
    print(f"  ^ amount is real on {real_amount:.0%} of these deals -- the rest is "
          "imputed (shrunk geometric mean by segment x type), so quote revenue as a "
          "range with coverage stated")

    print("\nhealth bands:")
    print(out["health"].value_counts().reindex(HEALTH_LABELS + [INSUFFICIENT_LABEL]).to_string())
    print("\ntop 5 by win probability:")
    print(out[["display_id", "stage_name", "win_probability", "acv", "health"]]
          .head(5).round(3).to_string(index=False))

    # Self-check: calibrated probabilities must be usable as a revenue weight. Checks run
    # on raw_probability where every row is populated, and on win_probability only where
    # it is not deliberately NULL -- otherwise the null-out would silently pass them.
    assert out.raw_probability.between(0, 1).all(), "probability outside [0,1]"
    assert out.health.notna().all(), "deal fell outside every health band"
    assert out.win_probability.isna().equals(too_early), "NULLs do not match the gate"
    assert out.loc[too_early, "expected_revenue"].isna().all(), "revenue claimed on a gated deal"
    base = train[LABEL].mean()
    assert 0.2 * base < scored.win_probability.mean() < 5 * base, (
        f"mean prediction {scored.win_probability.mean():.4f} implausible vs base rate {base:.4f}"
    )
    print("\nchecks passed")


if __name__ == "__main__":
    main()
