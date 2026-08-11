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
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

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

    XGBoost at depth 3, min_child_weight 1. Measured by pick_shipping.py on the time
    split (train 2,858/462 wins, test 715/79 wins, base rate 11.0%), 5 seeds, each
    candidate run through the FULL shipping path -- same encode, same calibrator, same
    evidence gate -- because the deliverable is a calibrated probability that gets
    multiplied by ACV, not a test-set PR-AUC:

        model                    PR-AUC     sd    ROC   Brier   ECE    rec@20  sum(p)/act
        XGBoost min_child_wt1    0.6304 0.0000  0.8790  0.0668  0.0228  0.785    1.10x
        LightGBM (previous)      0.6206 0.0000  0.8763  0.0682  0.0272  0.747    1.11x
        CatBoost d3 it600 lr.03  0.6201 0.0025  0.8770  0.0677  0.0260  0.762    1.14x
        CatBoost d3              0.6164 0.0040  0.8766  0.0688  0.0279  0.722    1.15x
        CatBoost raw (uncal)     0.6159 0.0033  0.8764  0.0667  0.0260  0.724    1.15x

    XGBoost wins on ALL SEVEN columns -- ranking, discrimination, per-deal probability,
    calibration error, top-quintile recall and aggregate ratio. No trade-off to weigh,
    which is why the switch is unambiguous. Both it and LightGBM have seed sd 0.0000
    (neither subsamples by default), so the +0.0099 gap is not seed luck.

    The forecast barely moves: 56.9 expected wins / $4.04M vs LightGBM's 56.2 / $3.97M
    across the 622 gated open deals. A $0.06M difference, i.e. this is free accuracy,
    not a revenue restatement.

    CALIBRATION IS NOT OPTIONAL, and CatBoost raw shows why. Its ranking is mid-pack
    (0.6159) yet it ships 62.9 wins / $4.61M -- $0.6M above the calibrated models --
    and its max probability is 0.880 against XGBoost's 0.954. Uncalibrated output
    cannot be used as a revenue weight however good its ordering looks.

    TUNING WAS TRIED AND REJECTED on the previous dataset: a 48-config grid scored by
    5-fold out-of-fold PR-AUC within train picked a config that then LOST 0.015 on
    test. Not re-run after the DROP_LOST_WITHOUT_AMOUNT filter changed the base rate
    from 2.2% to 11.0%; the grid's conclusion may no longer hold, and 79 test
    positives is too few to chase 0.01 differences anyway.
    """
    return XGBClassifier(
        max_depth=3, n_estimators=200, learning_rate=0.05, min_child_weight=1,
        random_state=random_state, n_jobs=-1, eval_metric="logloss")


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
    # Fixed shuffle before any fitting. Kept across the HistGB -> LightGBM -> XGBoost
    # switches: the calibrator's folds are still positional slices of whatever order
    # arrives, and unshuffled chronological rows put a different era of the business in
    # each fold. Deterministic seed, so the output is still reproducible run to run.
    train = pd.read_parquet("train.parquet").sample(frac=1, random_state=0).reset_index(drop=True)
    score = pd.read_parquet("score.parquet")

    train[PASSTHROUGH + FEATURES + [LABEL]].to_csv("ml_train.csv", index=False)
    score[PASSTHROUGH + FEATURES].to_csv("ml_score.csv", index=False)
    pd.DataFrame(SCHEMA, columns=["column", "type", "meaning"]).to_csv("ml_schema.csv", index=False)

    print(f"ml_train.csv   {len(train):>6,} closed deals  {train[LABEL].sum()} won "
          f"({train[LABEL].mean():.1%})  {len(FEATURES)} features + label")
    print(f"ml_score.csv   {len(score):>6,} open deals    {len(FEATURES)} features, no label")
    print(f"ml_schema.csv  {len(SCHEMA):>6} columns documented")

    # Sigmoid (Platt), 3 folds, shuffled. Isotonic needs several hundred positives to
    # behave and we have 541; 3 folds keeps each large enough to hold real ones.
    #
    # Calibration is not free accuracy, it is a units fix. pick_shipping.py measured
    # CatBoost raw uncalibrated at 1.15x on sum(p)/actual against 1.10x calibrated --
    # $0.6M of phantom pipeline -- with max probability 0.880 vs 0.954. Ranking can look
    # fine while the probabilities are unusable as revenue weights, which is the point.
    #
    # STALE, from the 11,051-row dataset at a 2.2% base rate, kept because the ORDERING
    # is what mattered and it was consistent: sigmoid cv=3 0.6276 > cv=5 0.6224 > cv=10
    # 0.6126, and isotonic cv=5 bought 0.0002 Brier for 0.014 PR-AUC. Not re-measured
    # since DROP_LOST_WITHOUT_AMOUNT moved the base rate to 11.0%.
    #
    # shuffle=True matters. cv=3 alone uses UNSHUFFLED StratifiedKFold, so on
    # chronologically-sorted rows each calibration fold is a different era of the
    # business and the fitted sigmoid depends on row order: expected wins swung 71.8
    # (file order) / 79.2 (by created_date) / 72.9 (reversed) -- a 10% move in the
    # headline revenue number from nothing but sort order. Shuffling cuts that spread
    # from 7.4 wins to 3.7.
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
