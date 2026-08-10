"""custom_fields is a nested struct on fact_opportunity, not a separate table. Test access."""
import os
from extract_pipeline import _run_single_query

token = os.environ["DEVREV_TOKEN"]

queries = {
    "json_extract_segment": "SELECT custom_fields->>'tnt__segment' AS segment FROM devrev.fact_opportunity LIMIT 5;",
    "raw_custom_fields": "SELECT custom_fields FROM devrev.fact_opportunity LIMIT 3;",
}

for label, sql in queries.items():
    print(f"\n--- {label} ---")
    try:
        rows = _run_single_query(sql, token)
        print(rows[:5])
    except Exception as e:
        print("failed:", e)
