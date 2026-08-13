#!/usr/bin/env python3
"""
Fetch DevRev oasis data and download the resulting file.

Usage:
    python fetch_oasis.py <devrev_token> [dataset_id]

Default dataset: devrev.fact_opportunity
"""

import sys
import os
import requests


OASIS_URL = "https://api.devrev.ai/internal/oasis.data.fetch"


def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_oasis.py <devrev_token> [dataset_id]")
        sys.exit(1)

    token = sys.argv[1]
    dataset_id = sys.argv[2] if len(sys.argv) > 2 else "devrev.fact_opportunity"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }

    print(f"Fetching dataset: {dataset_id}")
    resp = requests.post(OASIS_URL, headers=headers, json={"dataset_id": dataset_id})
    resp.raise_for_status()

    body = resp.json()

    data = body.get("data", {})
    # data is a list of dataset objects; each has a nested "sources" list
    if isinstance(data, list):
        sources = [s for item in data for s in (item.get("sources") or [])]
    else:
        sources = data.get("sources", [])

    if not sources:
        print("No sources in response. Full response:")
        import json; print(json.dumps(body, indent=2))
        sys.exit(1)

    if isinstance(sources, dict):
        sources = [sources]

    for i, source in enumerate(sources):
        url = source.get("url") or source.get("uri")
        if not url:
            print(f"Source {i} has no url, skipping. Keys: {list(source.keys())}")
            continue

        print(f"Downloading source {i}: {url}")
        dl_resp = requests.get(url, headers={"Authorization": f"Bearer {token}"})
        dl_resp.raise_for_status()

        content_type = dl_resp.headers.get("Content-Type", "")
        data_bytes = dl_resp.content
        if "json" in content_type:
            ext = ".json"
        elif "csv" in content_type:
            ext = ".csv"
        elif "parquet" in content_type or data_bytes[:4] == b"PAR1":
            ext = ".parquet"
        elif data_bytes[:2] == b"\x1f\x8b":
            ext = ".gz"
        else:
            ext = ".bin"
        filename = f"{dataset_id.replace('.', '_')}_{i}{ext}"

        with open(filename, "wb") as f:
            f.write(dl_resp.content)

        print(f"Saved → {filename} ({len(dl_resp.content):,} bytes)")


if __name__ == "__main__":
    main()
