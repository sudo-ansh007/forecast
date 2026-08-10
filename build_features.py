"""Feature builder for deal win-probability model.

Reads the raw dim_opportunity pull (dim_cache.parquet) and produces:
    train.parquet  -- closed deals, has is_won label
    score.parquet  -- open + in_progress deals, inference targets

dim_opportunity is the CURRENT-STATE table: one row per deal, already deduped.
fact_opportunity is the change-log, but its coverage only starts 2026-05-01 while
dim deals go back to 2023-01, so it is not used here.

Usage:
    python fetch_dim.py            # writes dim_cache.parquet
    python build_features.py
"""
import json

import numpy as np
import pandas as pd

DIM_CACHE = "dim_cache.parquet"

WON_STAGES = {"8-closed_won", "Closed Won"}

# MEDDPICC free-text fields. Used ONLY as a filled-count, never embedded.
# WARNING: leaks on this org's data (won 82-86% filled vs lost 4-33%) because reps
# document deals they're already winning. Kept for measurement, excluded from
# LEAK_SAFE_FEATURES until point-in-time reconstruction lands.
MEDDPICC = [
    "tnt__why_now", "tnt__why_anything", "tnt__why_devrev", "tnt__identified_pain",
    "tnt__decision_criteria", "tnt__decision_process", "tnt__metrics",
]

# Features safe to train on today. Deliberately excludes anything whose fill-rate
# diverges by outcome -- see field_analysis.md section 1.
# NOTE: stage_num is deliberately ABSENT. On closed deals it is 8 (won) or 9 (lost),
# i.e. the label itself -- including it gives PR-AUC 1.0. It is the strongest signal
# for scoring OPEN deals (stages 1-7) but can only be trained on via point-in-time
# reconstruction: the stage a deal sat at N days before it closed.
LEAK_SAFE_FEATURES = [
    "log_acv", "acv_missing", "days_in_current_stage", "days_in_sales_cycle",
    "stall_ratio", "contract_months", "num_stakeholders", "has_champion", "poc",
    "opportunity_type", "source", "geo", "segment",
]

# nda is DROPPED -- it made the model WORSE. 9.1x raw lift (40.6% win rate on the 138
# closed deals that have it vs 4.4% without) but only 56 wins behind it, so a shallow
# tree carves a near-pure leaf and overfits. Ablation over 5 seeds: removing nda moves
# PR-AUC 0.5631 -> 0.5781, a gap 2.5x the seed spread.
#
# poc is KEPT: removing it costs PR-AUC 0.5631 -> 0.5289 (~5x the seed spread), and
# permutation importance ranks it 5th of 11. Read it as "POC recorded", NOT "a POC
# ran" -- tnt__poc is a checkbox with no unset state (362 True / 12,601 False / 0
# null), and 165 deals have tnt__poc_notes written with the box still False. The
# True side carries real signal (41% actual win rate on the 56 flagged test deals);
# the False side conflates "no POC" with "nobody filled it in".

# technical_voc is DROPPED -- train/serve collapse, not weak signal. 8.1% of closed
# deals carry it (896 positives) but 0.1% of open ones (2 deals). The field stopped
# being maintained, so the model would learn from 896 examples and then never see it
# again at scoring time. AUC 0.5176 anyway. Removing it costs nothing measurable
# (PR-AUC 0.5455 -> 0.5426 before has_champion was added).
#
# has_champion is INCLUDED despite fill-rate divergence (won 81.7% vs lost 39.3%).
# It is the strongest single field available: 9.68% win rate with a champion vs 1.53%
# without -- 6.3x lift, AUC 0.7122 standalone -- and only 0.341 correlated with
# num_stakeholders, so it is not a restatement of the contact count (66% of champion
# deals have exactly one contact). The divergence is partly real (deals with a
# champion do win more) and partly reverse causality (reps backfill the champion
# field on deals they are closing). Point-in-time reconstruction is what separates
# the two; until then treat any driver attribution to has_champion as provisional.
# See CHAMPION_CAVEAT below.

# History features (pushes_pre, acv_revs_pre, stage transitions) are NOT included:
# fact_opportunity's log only starts 2026-05-01 while dim deals go back to 2023-01,
# so it covers ~20% of closed deals and 0.2% of open ones. Nothing to backfill --
# revisit once the log has accumulated a year of coverage.

# Available but leak-prone: only usable once evaluated at a point in time strictly
# before the outcome. Held out of the default feature set.
#   meddpicc_count  won 82-86% filled vs lost 4-33% -- almost pure rep-diligence
#                   proxy, and the gap is too wide to be anything but retrospective
#   has_se          won 54.7% vs lost 23.2%, AUC 0.6573. Adds +0.003 PR-AUC on top
#                   of has_champion -- not worth a second leak-prone field
POINT_IN_TIME_REQUIRED = ["meddpicc_count", "has_se"]

# has_champion is in the shipping set, but its coefficient is NOT a causal claim.
# The original hypothesis -- "champion absent N days after creation is a negative
# signal" -- needs the DATE the field was filled, which dim_opportunity does not
# retain. Until snapshots exist we know only whether a champion is recorded now,
# not whether one was identified early or added at the finish line.
CHAMPION_CAVEAT = (
    "has_champion: 6.3x win-rate lift, but fill rate diverges by outcome "
    "(won 81.7% / lost 39.3%). Direction is real, magnitude is inflated by "
    "reps backfilling the field on deals they are already closing."
)

# acv_missing is retrospective: 71.2% of losses vs 6.8% of wins, because reps stop
# maintaining amount once a deal is written off. Not a forward-looking signal.
#
# days_in_current_stage was ALSO excluded here on the theory that it keeps counting
# while a deal sits in closed_lost. It does not -- the clock stops at close. Median
# days_in_current_stage on lost deals is 53/53/48/64 for deals closed in
# 2023/2024/2025/2026 and 13 on won deals in every one of those years. A post-close
# clock would grow with age (2023 losses would read ~1000d). The 53-vs-13 gap is real
# pre-close stall, and open deals sit between the two (median 50) as they should.
# 89% of open deals (1,709 of 1,912) were created in 2026, while most training labels are
# 2024-2025 -- a resolved cohort cannot be recent, by definition. So calibration always
# extrapolates forward across whatever changed in between, whatever window is chosen.
# This is why the model is scored on Brier and a reliability curve, not PR-AUC alone.
FORWARD_EXTRAPOLATION_CAVEAT = (
    "most training labels are 2024-2025 cohorts; 89% of the scored pipeline is 2026. "
    "Calibration extrapolates across that drift -- report it with the revenue number."
)

ABANDONMENT_PRONE = ["acv_missing"]
CONSERVATIVE_FEATURES = [f for f in LEAK_SAFE_FEATURES if f not in ABANDONMENT_PRONE]

# Amount is largely a CONSEQUENCE of closing, not a predictor of it: 93.2% of won
# deals carry an amount vs 28.8% of lost and 40.8% of open, and among deals with a
# real (non-imputed) amount log_acv scores AUC 0.334 -- i.e. bigger deals win LESS,
# which is the small-renewal mix showing through rather than deal-size signal.
# So amount is excluded from the probability model and used only downstream, to
# weight calibrated probabilities into the revenue roll-up.
AMOUNT_FEATURES = ["log_acv", "acv_missing"]
NO_AMOUNT_FEATURES = [f for f in CONSERVATIVE_FEATURES if f not in AMOUNT_FEATURES]

# DATE FIELDS -- deliberately not features, each for a different reason:
#
# created_date (raw)   COHORT CENSORING. Old cohorts are fully resolved and their
#                      survivors are wins; recent cohorts are mostly still open and
#                      the ones that HAVE closed are the fast losses. Win rate of
#                      closed deals runs 94.6% (2023Q2) -> 2.5% (2026Q2) while
#                      pct-closed runs 100% -> 61%. Raw created_date scores AUC 0.299
#                      by learning "old = won". Adding it drops PR-AUC 0.513 -> 0.337
#                      and inflates sum(p)/actual to 5.2x, because every deal being
#                      scored is recent. Deal AGE is legitimate and already enters via
#                      days_in_sales_cycle; the absolute timestamp is not.
#
# target_close_date    OVERWRITTEN AT CLOSE. Equals actual_close_date exactly on
#                      99.3% of closed deals (99.5% within a day), so target-created
#                      is the realised cycle length, not the estimate. Tests well
#                      (PR-AUC 0.545) purely because it leaks. Also absent on 44.7%
#                      of open deals. The original estimates survive only in
#                      fact_opportunity, whose log starts 2026-05-01 -- so close-date
#                      PUSH COUNT, the feature we actually want, is unrecoverable
#                      today. Revisit once the log has a year of coverage.
#
# actual_close_date    The label's timestamp.

# Share of the sales cycle spent parked in one stage. Lost deals sit at median 0.935
# (never advanced), open 1.000, won 0.166 -- winners move through stages. More
# portable than the raw day count because it normalises out cycle length.
# CAVEAT: days_in_sales_cycle has train/serve skew -- on a closed deal it is the
# FINAL cycle length (corr 0.95 with actual_close-created), on an open deal it is
# age-so-far. Same column, different meaning. Watch it under point-in-time.
STALL_RATIO_NUM = "days_in_current_stage"
STALL_RATIO_DEN = "days_in_sales_cycle"

CATEGORICALS = ["opportunity_type", "source", "geo", "segment"]

# TRAINING-SET FILTERS -- both measured, both OFF. Kept as flags because the arguments
# for them are sound; they just do not survive contact with the data. Ablation over 5
# seeds, time split, calibrated HGB on NO_AMOUNT_FEATURES:
#
#   filter                        train/wins    test wins  PR-AUC            sum(p)/actual
#   none (shipping)               11,051/541    49         0.5781 +-0.0058   1.36x
#   2024Q1-2025Q4 window          8,031/392     35         0.5910 +-0.0193   1.54x
#   new business only             10,439/395    37         0.5536 +-0.0120   1.41x
#   window + new only             7,541/266     25         0.5241 +-0.0053   1.54x
#
# The cohort window looks like a small gain but its spread is 3x wider and it overlaps
# the baseline, while calibration degrades 1.36x -> 1.54x -- and calibration is the
# product metric. Costs 149 wins to buy noise.
#
# Scoping to new business is measurably WORSE (-0.025 PR-AUC, 4x the seed spread), which
# contradicts the split-populations argument. Reason: opportunity_type is already a
# feature, so a tree can carve the renewal branch itself when it pays -- and pooling
# keeps the 126 renewal/upsell wins contributing to every OTHER split in the tree.
# Splitting throws away a quarter of the label supply to isolate a distinction the model
# had already priced. Revisit if renewal wins pass ~300, enough to fit standalone.
TRAIN_COHORT = None   # ("2024-01-01", "2026-01-01") to re-enable, [start, end)
TRAIN_TYPES = None    # ["new"] to re-enable


def _json_get(raw, key):
    """dim_opportunity struct columns arrive as JSON strings (or dicts)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    obj = json.loads(raw) if isinstance(raw, str) else raw
    return obj.get(key) if isinstance(obj, dict) else None


def _filled(s: pd.Series) -> pd.Series:
    """Truthy-and-present mask. Empty string and empty list count as missing."""
    return s.apply(
        lambda v: v is not None
        and not (isinstance(v, float) and pd.isna(v))
        and v != ""
        and not (hasattr(v, "__len__") and not isinstance(v, str) and len(v) == 0)
    )


def parse_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Dates arrive as epoch MILLISECONDS. pd.to_datetime without unit="ms" reads them as
    # nanoseconds and silently returns 1970-01-01, which looks like data rather than a
    # bug. Convert once here so nothing downstream has to remember.
    for c in ["created_date", "target_close_date", "actual_close_date", "modified_date"]:
        if c in df and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(pd.to_numeric(df[c], errors="coerce"), unit="ms")

    df["stage_name"] = df["stage_json"].apply(lambda v: _json_get(v, "name"))

    # stage_json.ordinal is a stage IDENTIFIER, not a funnel position: won=6200,
    # lost=7200, and legacy "Closed Won"=953 sorts below "1-profile"=1175. The
    # numeric name prefix is the only orderable funnel signal.
    df["stage_num"] = pd.to_numeric(
        df["stage_name"].str.extract(r"^(\d)", expand=False), errors="coerce"
    )

    # value.amount / annual_contract_value.amount are strings inside a JSON struct.
    # Every populated row is USD (verified), so no FX conversion is needed.
    acv = pd.to_numeric(
        df["annual_contract_value"].apply(lambda v: _json_get(v, "amount")), errors="coerce"
    )
    val = pd.to_numeric(df["value"].apply(lambda v: _json_get(v, "amount")), errors="coerce")
    df["acv"] = acv.fillna(val)
    df.loc[df["acv"] <= 0, "acv"] = np.nan

    df["rep_id"] = df["owned_by_ids"].apply(
        lambda v: v[0] if hasattr(v, "__len__") and not isinstance(v, str) and len(v) else None
    )
    df["num_stakeholders"] = df["contact_ids"].apply(
        lambda v: len(v) if hasattr(v, "__len__") and not isinstance(v, str) else 0
    )

    cf = df["custom_fields"].apply(
        lambda v: json.loads(v) if isinstance(v, str) else (v if isinstance(v, dict) else {})
    )
    flat = pd.DataFrame(list(cf), index=df.index)

    def cf_num(key):
        return pd.to_numeric(flat[key], errors="coerce") if key in flat else pd.Series(np.nan, index=df.index)

    def cf_raw(key):
        return flat[key] if key in flat else pd.Series(None, index=df.index, dtype=object)

    df["days_in_current_stage"] = cf_num("tnt__days_in_current_stage")
    df["days_in_sales_cycle"] = cf_num("tnt__days_in_sales_cycle")
    df["contract_months"] = cf_num("tnt__contract_months")
    df["segment"] = cf_raw("tnt__segment")
    df["geo"] = cf_raw("tnt__geo")
    df["source"] = cf_raw("tnt__source")
    df["opportunity_type"] = cf_raw("tnt__opportunity_type")

    for out, key in [("poc", "tnt__poc"), ("technical_voc", "tnt__technical_voc"), ("nda", "tnt__nda")]:
        df[out] = (cf_raw(key) == True).astype(int)  # noqa: E712 -- JSON bools, not truthiness

    df["has_champion"] = _filled(cf_raw("tnt__champion")).astype(int)
    df["has_se"] = _filled(cf_raw("tnt__sales_engineer")).astype(int)

    present = [c for c in MEDDPICC if c in flat]
    df["meddpicc_count"] = (
        sum(_filled(flat[c]).astype(int) for c in present) if present
        else pd.Series(0, index=df.index)
    )

    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (train, score) and impute. Imputation params come from train only."""
    before = len(df)
    df = df[df["stage_num"].notna()].copy()
    if before != len(df):
        # legacy SFDC stages (Prospecting, Needs Analysis, ...) -- different pipeline
        # schema, no numeric prefix, unmappable to the funnel
        print(f"dropped {before - len(df)} rows with unmappable legacy stage names")

    df["is_won"] = df["stage_name"].isin(WON_STAGES).astype(int)

    # in_progress = stages 5-validate/6-negotiate/7-submit_for_close: live late-funnel
    # deals, so they belong in the scoring population, not excluded
    train = df[df["state"] == "closed"].copy()
    score = df[df["state"].isin(["open", "in_progress"])].copy()

    # Filters apply to TRAIN only -- the whole open pipeline still gets scored, so any
    # narrowing here becomes a train/serve gap that must be reported, never assumed away.
    if TRAIN_COHORT:
        lo, hi = pd.Timestamp(TRAIN_COHORT[0]), pd.Timestamp(TRAIN_COHORT[1])
        keep = train["created_date"].between(lo, hi, inclusive="left")
        print(f"cohort window {TRAIN_COHORT[0]}..{TRAIN_COHORT[1]}: dropped "
              f"{(~keep).sum()} closed deals ({train.loc[~keep, 'is_won'].sum()} wins) "
              "outside it -- 2023 survivors are all wins, 2026 closures are fast losses")
        train = train[keep].copy()

    if TRAIN_TYPES:
        keep = train["opportunity_type"].astype(str).str.strip().str.lower().isin(TRAIN_TYPES)
        print(f"scope to {TRAIN_TYPES}: dropped {(~keep).sum()} closed deals "
              f"({train.loc[~keep, 'is_won'].sum()} wins). Open pipeline still scored in "
              "full, so non-new deals are scored by a model not trained on them -- see "
              "OUT_OF_SCOPE_CAVEAT")
        train = train[keep].copy()

    # amount is only ~31% populated. Dropping those rows biases hard toward wins, so
    # impute and flag instead -- the flag lets the model learn "no amount recorded".
    #
    # ACV spans $1 to $50M+, so average in LOG space -- the geometric mean, which is the
    # Gaussian mean of log1p(acv). Raw-space filling is what produces the mismatch: one
    # $50M deal drags a whole segment's mean, and a raw median in a 3-deal group just
    # copies whichever of the three is middle. Then shrink each group toward the global
    # value by n/(n+K) so a thin group barely moves off global while a 500-deal group
    # gets its own number. K=20 is a judgement call, not a fitted value.
    SHRINK_K = 20
    log_acv = np.log1p(train["acv"])
    grp = log_acv.groupby([train["segment"], train["opportunity_type"]], dropna=False).agg(["mean", "count"])
    global_log = log_acv.mean()
    log_by_group = (
        (grp["mean"].fillna(global_log) * grp["count"] + global_log * SHRINK_K)
        / (grp["count"] + SHRINK_K)
    )

    stage_medians = {
        c: train.groupby("stage_num")[c].median()
        for c in ["days_in_current_stage", "days_in_sales_cycle"]
    }
    contract_med = train["contract_months"].median()

    for part in (train, score):
        part["acv_missing"] = part["acv"].isna().astype(int)
        keys = pd.MultiIndex.from_arrays([part["segment"], part["opportunity_type"]])
        filled = pd.Series(log_by_group.reindex(keys).to_numpy(), index=part.index)
        part["acv"] = part["acv"].fillna(np.expm1(filled.fillna(global_log)))
        part["log_acv"] = np.log1p(part["acv"])

        for c in ["days_in_current_stage", "days_in_sales_cycle"]:
            # fillna(0) would fabricate "brand new deal" for a missing stall counter
            part[c] = part[c].fillna(part["stage_num"].map(stage_medians[c]))
            part[c] = part[c].fillna(train[c].median())
        part["contract_months"] = part["contract_months"].fillna(contract_med)

        # after imputation so it is never NaN; clip(1) guards day-zero deals
        part["stall_ratio"] = (
            part[STALL_RATIO_NUM] / part[STALL_RATIO_DEN].clip(lower=1)
        )

        for c in CATEGORICALS:
            # missingness is informative here, so it gets its own level
            part[c] = part[c].astype(object).where(_filled(part[c]), "unknown")
            part[c] = part[c].astype(str).str.strip().str.lower()

    return train, score


def main():
    raw = pd.read_parquet(DIM_CACHE)
    print(f"loaded {len(raw)} deals from {DIM_CACHE}")

    df = parse_raw(raw)
    train, score = clean(df)

    keep = LEAK_SAFE_FEATURES + POINT_IN_TIME_REQUIRED + ["id", "display_id", "stage_name",
                                                          "rep_id", "account_id", "created_date",
                                                          "target_close_date", "actual_close_date",
                                                          "acv", "state"]
    train[keep + ["is_won"]].to_parquet("train.parquet", index=False)
    score[keep].to_parquet("score.parquet", index=False)

    print(f"train: {len(train)} closed  ({train.is_won.sum()} won, "
          f"win rate {train.is_won.mean():.3f})")
    print(f"score: {len(score)} open/in_progress")
    print(f"amount imputed: {train.acv_missing.mean():.1%} train, {score.acv_missing.mean():.1%} score")
    print(f"stall_ratio median: won {train.loc[train.is_won == 1, 'stall_ratio'].median():.3f} "
          f"lost {train.loc[train.is_won == 0, 'stall_ratio'].median():.3f} "
          f"open {score.stall_ratio.median():.3f}")


if __name__ == "__main__":
    main()
