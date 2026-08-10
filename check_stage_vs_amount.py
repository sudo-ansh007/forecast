"""Does missing amount correlate with early stage, or is it independent of stage?"""
import os
from extract_pipeline import EXTRACTION_SQL, run_query, parse_raw_columns

token = os.environ["DEVREV_TOKEN"]
raw = run_query(EXTRACTION_SQL, token)
raw = parse_raw_columns(raw)

closed = raw[raw["state"] == "closed"].copy()
closed["is_won"] = (closed["stage_name"] == "8-closed_won").astype(int)
closed["has_amount"] = closed["amount"].notna() & (closed["amount"] > 0)

# if your theory holds: missing-amount deals should cluster at LOW stage_ordinal
# (died early, before amount was ever sized) regardless of won/lost
print("--- stage_ordinal by has_amount ---")
print(closed.groupby("has_amount")["stage_ordinal"].describe())

print("\n--- win rate BY STAGE, split by has_amount ---")
print(closed.groupby(["stage_name", "has_amount"])["is_won"].agg(["count", "mean"]))
