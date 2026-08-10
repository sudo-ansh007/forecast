"""Check whether missing-amount closed deals skew toward wins or losses."""
import os
from extract_pipeline import EXTRACTION_SQL, run_query, parse_raw_columns

token = os.environ["DEVREV_TOKEN"]
raw = run_query(EXTRACTION_SQL, token)
raw = parse_raw_columns(raw)

closed = raw[raw["state"] == "closed"].copy()
closed["is_won"] = (closed["stage_name"] == "8-closed_won").astype(int)
closed["has_amount"] = closed["amount"].notna() & (closed["amount"] > 0)

print(closed.groupby("has_amount")["is_won"].agg(["count", "mean"]))
