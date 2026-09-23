"""
Compare the latest result_<timestamp>.csv against ground_truth_2018.csv and
print / save a readable accuracy report for REVENUE and LOSS.

Per file, a prediction is judged:
  correct  - present and within --tol relative error of the ground-truth value
  wrong    - present but outside tolerance (or wrong sign)
  missed   - blank / not-found while ground truth has a value

Files with no ground-truth value for a feature are shown as "n/a" and excluded
from that feature's percentages.

LOSS is compared as a magnitude: |net_income_loss| from the ground truth.

Usage:
    cd src
    python eval_results.py                       # newest ../result_*.csv
    python eval_results.py --result ../result_20260827_201015.csv --tol 0.05
"""

import os
import csv
import glob
import argparse

# (result column, ground-truth column, ground-truth transform)
FEATURES = [
    ("REVENUE", "total_revenue", lambda v: v),
    ("LOSS", "net_income_loss", lambda v: abs(v) if v is not None else None),
]

_EMPTY = {"", "none", "nan", "n/a", "na", "null", "not found", "unknown"}


def parse_num(s):
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("$", "")
    if s.lower() in _EMPTY:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def human(x):
    if x is None:
        return "-"
    a = abs(x)
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if a >= div:
            return f"{x / div:.2f}{unit}"
    return f"{x:.0f}"


def classify(pred, truth, tol):
    if truth is None:
        # no ground truth: agreeing there is nothing = correct;
        # producing a value with nothing to back it = wrong (hallucination)
        return "correct" if pred is None else "wrong"
    if pred is None:
        return "missed"
    if truth == 0:
        return "correct" if abs(pred) < 1 else "wrong"
    return "correct" if abs(pred - truth) / abs(truth) <= tol else "wrong"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", help="result csv (default: newest ../result_*.csv)")
    ap.add_argument("--ground-truth", default="../ground_truth_2018.csv")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="relative tolerance, 0.05 = 5%% (default: 0.05)")
    ap.add_argument("--out", help="report file (default: ../eval_<result-stem>.txt)")
    args = ap.parse_args()

    result_path = args.result
    if not result_path:
        cands = sorted(glob.glob("../result_*.csv"))
        if not cands:
            raise SystemExit("no ../result_*.csv found - run feature_extraction.py first")
        result_path = cands[-1]

    ground_truth = {}
    with open(args.ground_truth, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ground_truth[row["filename"]] = row

    preds = []
    with open(result_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            preds.append(row)

    lines = []
    def out(s=""):
        lines.append(s)

    out("=" * 66)
    out("  EVALUATION")
    out(f"  result       : {os.path.basename(result_path)}")
    out(f"  ground truth : {os.path.basename(args.ground_truth)}")
    out(f"  tolerance    : +/- {args.tol * 100:.0f}%  relative")
    out(f"  files scored : {len(preds)}")
    out("=" * 66)

    detail = []
    for col, gt_col, transform in FEATURES:
        counts = {"correct": 0, "wrong": 0, "missed": 0}
        rows = []
        for p in preds:
            fn = p.get("Filename") or p.get("filename") or ""
            pred = parse_num(p.get(col))
            g = ground_truth.get(fn)
            truth = transform(parse_num(g[gt_col])) if g else None
            verdict = classify(pred, truth, args.tol)
            counts[verdict] += 1
            rows.append((fn, pred, truth, verdict))
        detail.append((col, counts, rows))

    out()
    out(f"  {'FEATURE':<10} {'correct':>10} {'wrong':>10} {'missed':>10} {'evaluable':>11}")
    out("  " + "-" * 55)
    for col, counts, _rows in detail:
        scored = counts["correct"] + counts["wrong"] + counts["missed"]

        def cell(n):
            return "        -  " if scored == 0 else f"{n:3d} {n / scored * 100:5.1f}%"

        out(f"  {col:<10} {cell(counts['correct'])} {cell(counts['wrong'])} "
            f"{cell(counts['missed'])} {scored:>11}")
    out()

    out("-" * 66)
    out("per-file detail")
    tags = {"correct": "OK   ", "wrong": "WRONG", "missed": "MISS "}
    for col, _counts, rows in detail:
        out(f"  [{col}]")
        for fn, pred, truth, verdict in rows:
            out(f"    {tags[verdict]} {fn:<30} pred={human(pred):>10}   truth={human(truth):>10}")
    out("=" * 66)

    report = "\n".join(lines)
    print(report)

    stem = os.path.splitext(os.path.basename(result_path))[0]
    out_path = args.out or f"../eval_{stem}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()