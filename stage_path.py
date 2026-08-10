"""Where do lost deals die? Recover the stage a deal sat in before it went closed-lost.

fact_opportunity is a SNAPSHOT table: one row per (id, object_version), ~44 versions per
deal. extract_pipeline.py keeps only rn=1 (the latest), which is correct for features but
throws the stage PATH away. This pulls the path back.

Method: for each deal, order versions by object_version and take the last DISTINCT stage
before the terminal one. Not "the previous version" -- most of those 44 revisions change
an unrelated field and leave stage untouched, so version N-1 is usually the same stage as
N. lag() over distinct-stage-changes only.

Caveat that limits what this can say: object_version ordering is reliable, but there is no
timestamp per version in this table, so this gives the stage SEQUENCE, not how long the
deal sat in each stage. Dwell-time-per-stage needs the fact log, which only starts
2026-05-01 while deals go back to 2023-01.

Usage:
    export DEVREV_TOKEN="<bearer token>"   # never hardcode it
    python stage_path.py
"""
import os

import pandas as pd

from extract_pipeline import run_query

CACHE = "stage_path.parquet"

# Collapse consecutive duplicate stages first (the ~44 revisions per deal mostly touch
# other fields), then read off the last two entries of what remains. row_number over the
# deduped sequence, descending, so rn=1 is terminal and rn=2 is the stage it fell from.
SQL = """
SELECT id, stage_name, rn, n_stages FROM (
  SELECT
    id, stage_name,
    row_number() OVER (PARTITION BY id ORDER BY first_version DESC) AS rn,
    count(*) OVER (PARTITION BY id) AS n_stages
  FROM (
    SELECT id, stage_name, min(object_version) AS first_version
    FROM (
      SELECT
        id,
        object_version,
        stage_json->>'name' AS stage_name,
        lag(stage_json->>'name') OVER (PARTITION BY id ORDER BY object_version) AS prev_stage
      FROM devrev.fact_opportunity
      WHERE is_deleted = false
    )
    WHERE prev_stage IS NULL OR stage_name <> prev_stage
    GROUP BY id, stage_name
  )
)
WHERE rn <= 2
ORDER BY id, rn
"""


def main():
    token = os.environ.get("DEVREV_TOKEN")
    if not token and not os.path.exists(CACHE):
        raise SystemExit("set DEVREV_TOKEN (never hardcode it), or pre-populate " + CACHE)

    raw = run_query(SQL, token, cache_path=CACHE)
    raw["rn"] = pd.to_numeric(raw["rn"])

    wide = raw.pivot(index="id", columns="rn", values="stage_name")
    wide.columns = ["final_stage", "from_stage"][: len(wide.columns)]
    wide["n_stages"] = raw.groupby("id")["n_stages"].first()

    lost = wide[wide["final_stage"] == "9-closed_lost"]
    won = wide[wide["final_stage"] == "8-closed_won"]
    print(f"{len(wide):,} deals   {len(lost):,} closed_lost   {len(won):,} closed_won\n")

    for name, sub in [("CLOSED-LOST", lost), ("CLOSED-WON", won)]:
        c = sub["from_stage"].fillna("(no earlier stage recorded)").value_counts()
        print(f"--- {name}: stage it came FROM ---")
        print(pd.DataFrame({"deals": c, "share": (c / len(sub)).round(3)}).to_string())
        print()

    # A deal whose whole recorded life is one stage never advanced at all. That is the
    # 45% stall_ratio==1.0 cohort from build_features.py, seen from the other side.
    single = (wide["n_stages"] == 1)
    print(f"deals with only ONE stage ever recorded: {single.sum():,} "
          f"({single.mean():.1%}) -- never advanced")
    print(wide.loc[single, "final_stage"].value_counts().to_string())

    wide.to_parquet("stage_transitions.parquet")
    print("\nwrote stage_transitions.parquet")

    # The point of the exercise: does from_stage predict the outcome? If lost and won
    # deals fall out of the same places in the same proportions, this column is useless
    # as a feature and the answer is just "everything dies in stage 1".
    both = wide[wide["final_stage"].isin(["8-closed_won", "9-closed_lost"])].copy()
    both["is_won"] = (both["final_stage"] == "8-closed_won").astype(int)
    rate = both.groupby(both["from_stage"].fillna("(none)"))["is_won"].agg(["size", "sum", "mean"])
    print("\n--- win rate by the stage the deal came from ---")
    print(rate.rename(columns={"size": "deals", "sum": "won", "mean": "win_rate"})
          .sort_values("deals", ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
