import os, requests

token = os.environ["DEVREV_TOKEN"]
sql = "Select * from devrev.fact_opportunity limit 10;"

resp = requests.post(
    "https://api.devrev.ai/internal/oasis.data.query",
    headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    },
    json={"sql_query": sql},
    timeout=60,
)
print("status:", resp.status_code)
print("body:", resp.text[:2000])

# --- amount visibility check ---
import base64, json
payload = resp.json()
rows = json.loads(base64.b64decode(payload["data"]))
sample = list(rows.values())[:5]
for r in sample:
    print("id:", r.get("id"), "| amount:", repr(r.get("amount")), "| all keys:", list(r.keys()))

print("\n--- checking alternate money fields ---")
for r in sample:
    print("id:", r.get("id"),
          "| amount:", r.get("amount"),
          "| value:", r.get("value"),
          "| budget:", r.get("budget"),
          "| customer_budget:", r.get("customer_budget"),
          "| arr:", r.get("annual_recurring_revenue"),
          "| acv:", r.get("annual_contract_value"))
