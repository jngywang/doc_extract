"""
Build a ground-truth CSV of 2018 financials from SEC XBRL data.

Reads the edgar-corpus records (raw text filings), takes each file's CIK,
and looks up the "official" numbers from SEC's structured XBRL API:
  - total_revenue    <- us-gaap Revenue tags
  - net_income_loss  <- us-gaap:NetIncomeLoss  (negative == loss)

All SEC API responses are cached on disk, so re-runs are free / offline.

Usage:
    cd src
    python build_ground_truth.py --user-agent "Your Name your.email@example.com"
    python build_ground_truth.py --fill-missing

SEC requires a descriptive User-Agent with contact info; without one the API
may return HTTP 403. Pass it via --user-agent.
"""

import os
import csv
import sys
import json
import glob
import time
import hashlib
import argparse
import urllib.request
import urllib.error

YEAR = 2018
OUT_PATH = "../ground_truth_2018.csv"

FRAMES_URL = "https://data.sec.gov/api/xbrl/frames/us-gaap/{tag}/USD/CY{year}.json"
CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{tag}.json"

# Tried in order; first tag that has a value for the CIK wins.
REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
]
LOSS_TAG = "NetIncomeLoss"


def load_records(data_path, year):
    """data_path may be an actual .jsonl file, a glob, a directory, or a
    template-listing file (lines containing '{year}', like ../data)."""
    paths = []
    if os.path.isfile(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            head = f.read(400)
        is_template = "{year}" in head or head.lstrip().startswith("#")
        if is_template:
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    paths.extend(glob.glob(line.format(year=year)))
        else:
            paths = [data_path]
    elif os.path.isdir(data_path):
        paths = glob.glob(os.path.join(data_path, "*.jsonl"))
    else:
        paths = glob.glob(data_path)

    records = []
    for p in sorted(paths):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records, paths


def fetch_json(url, cache_dir, user_agent, sleep=0.15):
    """GET url as JSON, caching the response (including 404 -> null) on disk."""
    key = hashlib.md5(url.encode("utf-8")).hexdigest() + ".json"
    cache_path = os.path.join(cache_dir, key)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            data = None
        elif e.code == 403:
            sys.exit("SEC returned 403. Set a real contact via --user-agent "
                     '(e.g. "Jane Doe jane@example.com").')
        else:
            raise
    time.sleep(sleep)  # SEC allows <= 10 req/s; stay well under

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def build_frame_map(tag, year, cache_dir, user_agent):
    """cik (int) -> reported USD value, for one concept across all filers."""
    url = FRAMES_URL.format(tag=tag, year=year)
    data = fetch_json(url, cache_dir, user_agent)
    out = {}
    if data and "data" in data:
        for row in data["data"]:
            out[int(row["cik"])] = row["val"]
    return out


def concept_value(cik, tag, year, cache_dir, user_agent):
    """Fallback: pull one company's FY 10-K value for a concept."""
    url = CONCEPT_URL.format(cik=cik, tag=tag)
    data = fetch_json(url, cache_dir, user_agent)
    if not data:
        return None
    best = None
    for u in data.get("units", {}).get("USD", []):
        if u.get("form") != "10-K" or u.get("fy") != year or u.get("fp") != "FY":
            continue
        if best is None or u.get("end", "") > best.get("end", ""):
            best = u
    return best["val"] if best else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="../data",
                    help="jsonl file / glob / dir / template-listing (default: ../data)")
    ap.add_argument("--cache-dir", default="../.sec_cache")
    ap.add_argument("--user-agent", default="doc_extract ground-truth script",
                    help='SEC requires contact info, e.g. "Jane Doe jane@example.com"')
    ap.add_argument("--fill-missing", action="store_true",
                    help="for CIKs absent from the CY frame, query companyconcept per-company "
                         "(slower first run, catches non-calendar fiscal years)")
    args = ap.parse_args()

    out_path = OUT_PATH
    os.makedirs(args.cache_dir, exist_ok=True)

    records, paths = load_records(args.data, YEAR)
    if not records:
        sys.exit(f"No records found from {args.data!r} (resolved: {paths})")
    print(f"Loaded {len(records)} records from {len(paths)} file(s)")

    print("Fetching SEC XBRL frames...")
    loss_map = build_frame_map(LOSS_TAG, YEAR, args.cache_dir, args.user_agent)
    rev_maps = [(t, build_frame_map(t, YEAR, args.cache_dir, args.user_agent))
                for t in REVENUE_TAGS]
    print(f"  frame coverage: NetIncomeLoss={len(loss_map)}, "
          + ", ".join(f"{t}={len(m)}" for t, m in rev_maps))

    seen = set()
    rows = []
    n_rev = n_loss = 0
    for rec in records:
        raw_cik = rec.get("cik")
        filename = rec.get("filename", "")
        if raw_cik is None:
            continue
        try:
            cik = int(str(raw_cik).strip())
        except ValueError:
            continue
        if (cik, filename) in seen:
            continue
        seen.add((cik, filename))

        revenue, revenue_tag = "", ""
        for tag, m in rev_maps:
            if cik in m:
                revenue, revenue_tag = m[cik], tag
                break
        net_income_loss = loss_map.get(cik, "")

        if args.fill_missing and revenue == "":
            for tag in REVENUE_TAGS:
                v = concept_value(cik, tag, YEAR, args.cache_dir, args.user_agent)
                if v is not None:
                    revenue, revenue_tag = v, tag + " (companyconcept)"
                    break
        if args.fill_missing and net_income_loss == "":
            v = concept_value(cik, LOSS_TAG, YEAR, args.cache_dir, args.user_agent)
            if v is not None:
                net_income_loss = v

        n_rev += revenue != ""
        n_loss += net_income_loss != ""
        rows.append([cik, filename, revenue, net_income_loss, revenue_tag])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cik", "filename", "total_revenue", "net_income_loss", "revenue_tag"])
        w.writerows(rows)

    print(f"\nWrote {len(rows)} rows -> {out_path}")
    print(f"  matched revenue:        {n_rev}/{len(rows)}")
    print(f"  matched net_income_loss: {n_loss}/{len(rows)}")
    print(f"  cache: {args.cache_dir}")


if __name__ == "__main__":
    main()