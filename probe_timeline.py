"""Find where stage-change history lives. Read-only probe -- writes nothing, changes nothing.

Why: dim_opportunity is current-state (one row per deal, verified: 12,963 rows / 12,963
unique ids), so it cannot answer "which stage did this deal die in". fact_opportunity is
versioned and can, but only covers 4,623 of 10,527 closed-lost deals (44%) -- it stops
dead at 2026-04-30, so its coverage is the recent cohort only.

Timeline events would be strictly better than either: they carry a TIMESTAMP per change,
which gives dwell-time-per-stage, not just the sequence. This script finds out whether
they are reachable and whether stage transitions are actually in them.

Two access paths, both probed:
  1. oasis.data.query SQL against candidate devrev.* tables
  2. the public timeline-entries REST API, per deal

Usage:
    export DEVREV_TOKEN="<bearer token>"   # never hardcode it
    python probe_timeline.py
"""
import base64
import json
import os

import requests

OASIS_URL = "https://api.devrev.ai/internal/oasis.data.query"
REST_BASE = "https://api.devrev.ai"

# A deal that IS in fact_opportunity, so anything found here can be cross-checked against
# the version history we already have.
SAMPLE_DEAL = "don:core:dvrv-us-1:devo/0:opportunity/10000"

# Guesses, cheapest first. LIMIT 1 on each -- a name that exists returns a row shape, a
# name that does not returns an error, and either answer is what we want.
CANDIDATE_TABLES = [
    "dim_timeline_entry",
    "fact_timeline_entry",
    "dim_timeline_event",
    "fact_timeline_event",
    "dim_event",
    "fact_event",
    "dim_work_timeline_entry",
    "dim_audit_log",
    "fact_audit_log",
    "dim_stage_transition",
    "fact_stage_transition",
    "dim_work_history",
]


def sql(query: str, token: str, quiet: bool = False):
    r = requests.post(
        OASIS_URL,
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 "Authorization": f"Bearer {token}"},
        json={"sql_query": query},
        timeout=60,
    )
    if not r.ok:
        if not quiet:
            print(f"    HTTP {r.status_code}: {r.text[:200]}")
        return None
    return list(json.loads(base64.b64decode(r.json()["data"])).values())


def main():
    token = os.environ.get("DEVREV_TOKEN")
    if not token:
        raise SystemExit("set DEVREV_TOKEN in the environment (never hardcode it)")

    # 1. Ask the catalog directly. If this works, the guessing below is unnecessary.
    print("=== 1. table catalog ===")
    for probe in ("SELECT table_name FROM information_schema.tables ORDER BY table_name",
                  "SHOW TABLES"):
        rows = sql(probe, token, quiet=True)
        if rows:
            names = sorted({str(v) for r in rows for v in r.values()})
            print(f"  {probe.split()[0]} worked -- {len(names)} tables")
            hits = [n for n in names if any(k in n.lower() for k in
                    ("timeline", "event", "audit", "history", "transition", "stage"))]
            print("  candidates:", hits or "(none matched)")
            break
        print(f"  {probe.split()[0]} rejected")
    else:
        print("  no catalog access -- falling back to name guessing")

    # 2. Guess table names.
    print("\n=== 2. candidate tables ===")
    found = []
    for t in CANDIDATE_TABLES:
        rows = sql(f"SELECT * FROM devrev.{t} LIMIT 1", token, quiet=True)
        if rows is None:
            print(f"  {t:<26} no")
            continue
        cols = list(rows[0].keys()) if rows else []
        print(f"  {t:<26} YES  {len(cols)} cols")
        print(f"    {cols}")
        found.append(t)

    # 3. REST timeline API. Separate service from oasis, so it can work when SQL does not.
    print("\n=== 3. timeline-entries REST API ===")
    r = requests.get(f"{REST_BASE}/timeline-entries.list",
                     headers={"Authorization": f"Bearer {token}"},
                     params={"object": SAMPLE_DEAL}, timeout=60)
    print(f"  HTTP {r.status_code}")
    if r.ok:
        entries = r.json().get("timeline_entries", [])
        print(f"  {len(entries)} entries for {SAMPLE_DEAL.split('/')[-1]}")
        kinds = {}
        for e in entries:
            kinds[e.get("type")] = kinds.get(e.get("type"), 0) + 1
        print("  types:", kinds)
        # The question that decides everything: is a stage change in here at all?
        blob = json.dumps(entries)
        for probe in ("stage", "9-closed_lost", "1-profile"):
            print(f"  mentions {probe!r}: {probe in blob}")
        print("\n  first entry verbatim:")
        print("   ", json.dumps(entries[0], indent=2)[:1200] if entries else "(none)")
    else:
        print(f"  {r.text[:400]}")

    print("\n--- verdict ---")
    if found:
        print("SQL tables reachable:", found)
        print("Next: check date coverage on them the same way fact_opportunity was checked")
        print("  -- min/max timestamp, and how many of the 10,527 closed-lost deals appear.")
    if r.ok:
        print("REST works. Caveat: it is PER-DEAL, so full history = 12,963 calls.")
        print("  Fine as a one-off backfill, too slow to sit in the feature pipeline.")
    if not found and not r.ok:
        print("Neither path open with this token. Stage path stays limited to the")
        print("fact_opportunity 44%, or the forward-only daily snapshot.")


if __name__ == "__main__":
    main()
