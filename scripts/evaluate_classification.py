#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score, confusion_matrix

DEFAULT_MAP = {
    "ACCURATE": "ACCURATE",
    "INDIRECT": "ACCURATE",
    "SUPPORT": "ACCURATE",
    "SUPPORTED": "ACCURATE",
    "CONTRADICT": "NOT_ACCURATE",
    "NOT_SUBSTANTIATE": "NOT_ACCURATE",
    "OVERSIMPLIFY": "NOT_ACCURATE",
    "MISQUOTE": "NOT_ACCURATE",
    "ETIQUETTE": "NOT_ACCURATE",
    "REFUTE": "NOT_ACCURATE",
    "REFUTED": "NOT_ACCURATE",
    "IRRELEVANT": "IRRELEVANT",
    "NOT ENOUGH INFO": "IRRELEVANT",
    "NOT_ENOUGH_INFO": "IRRELEVANT",
    "NEI": "IRRELEVANT",
}
ORDER = ["ACCURATE", "NOT_ACCURATE", "IRRELEVANT"]


def norm(x):
    s = str(x).strip().upper().replace("-", "_")
    return DEFAULT_MAP.get(s, DEFAULT_MAP.get(s.replace("_", " "), s))


def read_table(path: Path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    raise ValueError("Input must be CSV, JSON, or JSONL")


def main():
    ap = argparse.ArgumentParser(description="Evaluate 3-way citation classification.")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--gold-column", default="gold")
    ap.add_argument("--pred-column", default="prediction")
    args = ap.parse_args()

    df = read_table(args.input)
    missing = [c for c in (args.gold_column, args.pred_column) if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns {missing}. Found: {list(df.columns)}")

    y_true = [norm(v) for v in df[args.gold_column]]
    y_pred = [norm(v) for v in df[args.pred_column]]

    unknown = sorted((set(y_true) | set(y_pred)) - set(ORDER))
    if unknown:
        print("WARNING: unrecognized labels:", unknown)

    labels = [x for x in ORDER if x in set(y_true) | set(y_pred)]
    print("n =", len(y_true))
    print("micro-F1:", round(f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0), 4))
    print("macro-F1:", round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4))
    print("\nPer-class report:\n")
    print(classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0))
    print("Confusion matrix (rows=gold, cols=pred):")
    print("labels:", labels)
    print(confusion_matrix(y_true, y_pred, labels=labels))


if __name__ == "__main__":
    main()
