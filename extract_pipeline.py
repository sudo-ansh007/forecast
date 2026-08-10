"""
V0 data extraction + cleaning pipeline for deal win-probability model.
Pulls from DevRev's oasis.data.query API instead of a direct DB connection.

Usage:
    export DEVREV_TOKEN="<bearer token>"   # never hardcode it
    python extract_pipeline.py --out features

Produces feature tables ready for training (see train.py).
"""
import argparse
import base64
import json
import os

import pandas as pd
import requests

DEVREV_QUERY_URL = "https://api.devrev.ai/internal/oasis.data.query"
RAW_CACHE_PATH = "raw_cache.parquet"

FEATURES = [
    "amount", "stage_ordinal", "days_in_current_stage", "days_in_sales_cycle",
    "num_stakeholders", "segment", "region", "rep_win_rate", "close_date_push_days",
    "priority", "partner_sourced",
]

# oasis.data.query rejects Postgres-style ->>'x' / len() / array indexing in SQL for
# top-level struct columns like stage_json/value — those get pulled raw and parsed in
# pandas. But tnt__ custom fields live INSIDE the custom_fields JSON blob, and ->>'x'
# DOES work directly on that column (confirmed) — so pull those pre-extracted here.
# fact_opportunity is a SNAPSHOT table: one row per (id, object_version). 303,265 rows =
# only 6,900 real opportunities, ~44 revisions each. Querying it flat double-counts every
# deal and silently biases any aggregate. row_number() picks the latest version per id.
EXTRACTION_SQL = """
SELECT * FROM (
SELECT
  row_number() OVER (PARTITION BY id ORDER BY object_version DESC) AS rn,
  id,
  account_id,
  value,
  annual_contract_value,
  stage_json,
  state,
  probability,
  priority,
  forecast_category,
  created_date,
  target_close_date,
  actual_close_date,
  modified_date,
  owned_by_ids,
  contact_ids,
  custom_fields->>'tnt__segment' AS tnt_segment,
  custom_fields->>'tnt__region' AS tnt_region,
  custom_fields->>'tnt__days_in_current_stage' AS tnt_days_in_current_stage,
  custom_fields->>'tnt__days_in_sales_cycle' AS tnt_days_in_sales_cycle,
  custom_fields->>'tnt__competition' AS tnt_competition,
  custom_fields->>'tnt__partner_sourced' AS tnt_partner_sourced,
  custom_fields->>'tnt__ramped_arr' AS tnt_ramped_arr,
  custom_fields->>'tnt__probability_by_forecast' AS tnt_probability_by_forecast
FROM devrev.fact_opportunity
WHERE is_deleted = false
) WHERE rn = 1
ORDER BY id
"""


def _run_single_query(sql: str, token: str) -> list[dict]:
    resp = requests.post(
        DEVREV_QUERY_URL,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        },
        json={"sql_query": sql},
        timeout=120,
    )
    if not resp.ok:
        raise RuntimeError(f"oasis query failed ({resp.status_code}): {resp.text[:2000]}")

    payload = resp.json()
    decoded_bytes = base64.b64decode(payload["data"])
    rows_by_index = json.loads(decoded_bytes)  # {"0": {...}, "1": {...}, ...}
    return list(rows_by_index.values())


def run_query(
    base_sql: str, token: str, page_size: int = 1000, cache_path: str = RAW_CACHE_PATH,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Calls oasis.data.query in OFFSET-paginated batches until a short page signals the end.

    base_sql must NOT already contain a LIMIT/OFFSET clause — this appends them.
    Caches the full raw pull to `cache_path`; subsequent calls load from cache instead
    of re-running the (slow) pagination loop, unless use_cache=False forces a refresh.
    """
    if use_cache and os.path.exists(cache_path):
        print(f"loading cached raw pull from {cache_path} (delete it or pass --refresh to re-fetch)")
        return pd.read_parquet(cache_path)

    all_rows: list[dict] = []
    offset = 0
    while True:
        page_sql = f"{base_sql.strip().rstrip(';')} LIMIT {page_size} OFFSET {offset}"
        rows = _run_single_query(page_sql, token)
        all_rows.extend(rows)
        print(f"fetched {len(rows)} rows at offset {offset} (total so far: {len(all_rows)})")
        if len(rows) < page_size:
            break
        offset += page_size

    df = pd.DataFrame(all_rows)
    df.to_parquet(cache_path, index=False)
    print(f"cached {len(df)} raw rows to {cache_path}")
    return df


def parse_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Unpacks nested/JSON-string columns from the raw oasis response into flat fields."""
    df = df.copy()

    def _json_field(raw, key):
        if not raw:
            return None
        obj = json.loads(raw) if isinstance(raw, str) else raw
        return obj.get(key) if isinstance(obj, dict) else None

    df["stage_name"] = df["stage_json"].apply(lambda v: _json_field(v, "name"))
    df["stage_ordinal"] = df["stage_json"].apply(lambda v: _json_field(v, "ordinal"))
    df["rep_id"] = df["owned_by_ids"].apply(lambda v: v[0] if isinstance(v, list) and v else None)
    df["num_stakeholders"] = df["contact_ids"].apply(lambda v: len(v) if isinstance(v, list) else 0)

    # amount column is unpopulated in this org — value.amount is the real deal
    # value (confirmed live in UI); annual_contract_value.amount as fallback.
    df["amount"] = pd.to_numeric(
        df["value"].apply(lambda v: _json_field(v, "amount")), errors="coerce"
    )
    df["acv"] = pd.to_numeric(
        df["annual_contract_value"].apply(lambda v: _json_field(v, "amount")), errors="coerce"
    )
    df["amount"] = df["amount"].fillna(df["acv"])

    # tnt__ fields pulled pre-extracted via custom_fields->>'x' in EXTRACTION_SQL
    df["segment"] = df["tnt_segment"]
    df["region"] = df["tnt_region"]
    df["days_in_current_stage"] = pd.to_numeric(df["tnt_days_in_current_stage"], errors="coerce")
    df["days_in_sales_cycle"] = pd.to_numeric(df["tnt_days_in_sales_cycle"], errors="coerce")
    df["ramped_arr"] = pd.to_numeric(df["tnt_ramped_arr"], errors="coerce")
    df["partner_sourced"] = df["tnt_partner_sourced"].fillna("false") == "true"

    return df


def clean_and_engineer(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (train_df, open_df). train_df has is_won label; open_df is inference-ready."""
    df = df.dropna(subset=["stage_name", "state"]).copy()

    # two label variants seen in the data ("8-closed_won" and "Closed Won") — catch both
    won_labels = {"8-closed_won", "Closed Won"}
    df["is_won"] = df["stage_name"].isin(won_labels).astype(int)
    train_df = df[df["state"] == "closed"].copy()
    open_df = df[df["state"] == "open"].copy()

    for part in (train_df, open_df):
        before = len(part)
        part.drop(part[part["amount"].isna() | (part["amount"] <= 0)].index, inplace=True)
        dropped = before - len(part)
        if dropped:
            print(f"dropped {dropped}/{before} rows with missing/zero amount")

        for col in ["days_in_current_stage", "days_in_sales_cycle", "num_stakeholders"]:
            part[col] = part[col].fillna(0)
        for col in ["segment", "region", "priority"]:
            part[col] = part[col].fillna("unknown")
        part["partner_sourced"] = part["partner_sourced"].fillna(False)

        # dates come back as epoch milliseconds from the oasis API
        part["close_date_push_days"] = (
            pd.to_datetime(part["actual_close_date"], unit="ms")
            - pd.to_datetime(part["target_close_date"], unit="ms")
        ).dt.days.fillna(0)

    # rep_win_rate: leave-one-out on closed deals only, applied to both splits by rep_id
    grp = train_df.groupby("rep_id")["is_won"]
    train_df["rep_win_rate"] = (grp.transform("sum") - train_df["is_won"]) / (
        grp.transform("count") - 1
    ).clip(lower=1)

    rep_rate_lookup = train_df.groupby("rep_id")["is_won"].mean()
    open_df["rep_win_rate"] = open_df["rep_id"].map(rep_rate_lookup).fillna(
        train_df["is_won"].mean()
    )

    return train_df, open_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="features", help="output file prefix")
    parser.add_argument("--refresh", action="store_true", help="force a fresh pull, ignore raw cache")
    args = parser.parse_args()

    token = os.environ.get("DEVREV_TOKEN")
    if not token:
        raise SystemExit("Set DEVREV_TOKEN env var before running (never hardcode the bearer token).")

    raw = run_query(EXTRACTION_SQL, token, use_cache=not args.refresh)
    raw = parse_raw_columns(raw)
    train_df, open_df = clean_and_engineer(raw)

    train_df[FEATURES + ["is_won", "id"]].to_parquet(f"{args.out}_train.parquet", index=False)
    open_df[FEATURES + ["id"]].to_parquet(f"{args.out}_open.parquet", index=False)

    print(f"train rows: {len(train_df)}, open rows: {len(open_df)}")
    print(f"win rate: {train_df['is_won'].mean():.3f}")


if __name__ == "__main__":
    main()
