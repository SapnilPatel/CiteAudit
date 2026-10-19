#!/usr/bin/env python3
"""Evaluate sentence retrieval when gold and retrieved sentence IDs are available.

Expected JSONL format, one citation/query per line, by default:
{
  "gold": [2, 7],
  "retrieved": [9, 7, 3, 2, ...]
}

Use --gold-key and --retrieved-key to adapt to exported files from the notebook.
Recall@k here means fraction of queries for which at least one gold evidence sentence
is found in the top k. MRR uses the rank of the first gold sentence.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--gold-key", default="gold")
    ap.add_argument("--retrieved-key", default="retrieved")
    ap.add_argument("--k", nargs="+", type=int, default=[1, 5, 10, 20])
    args = ap.parse_args()

    hits = {k: 0 for k in args.k}
    rr_sum = 0.0
    n = 0

    with args.input.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            gold = {str(x) for x in rec[args.gold_key]}
            retrieved = [str(x) for x in rec[args.retrieved_key]]
            if not gold:
                continue
            n += 1
            for k in args.k:
                if gold.intersection(retrieved[:k]):
                    hits[k] += 1
            first = next((i for i, x in enumerate(retrieved, 1) if x in gold), None)
            if first:
                rr_sum += 1.0 / first

    if n == 0:
        raise SystemExit("No evaluable rows found")
    for k in args.k:
        print(f"Recall@{k}: {hits[k] / n:.4f}")
    print(f"MRR: {rr_sum / n:.4f}")
    print(f"queries: {n}")


if __name__ == "__main__":
    main()
