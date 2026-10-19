#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

KNOWN_LABELS = {
    "ACCURATE", "CONTRADICT", "NOT_SUBSTANTIATE", "IRRELEVANT",
    "OVERSIMPLIFY", "MISQUOTE", "INDIRECT", "ETIQUETTE",
    "SUPPORT", "SUPPORTED", "REFUTE", "REFUTED", "NOT ENOUGH INFO",
    "NOT_ENOUGH_INFO", "NEI"
}


def read_records(path: Path) -> Iterable[Any]:
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    elif path.suffix.lower() == ".json":
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if isinstance(obj, list):
            yield from obj
        else:
            yield obj


def collect_labels(obj: Any, out: Counter):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                norm = v.strip().upper()
                if norm in KNOWN_LABELS or "label" in k.lower():
                    if norm in KNOWN_LABELS:
                        out[norm] += 1
            else:
                collect_labels(v, out)
    elif isinstance(obj, list):
        for v in obj:
            collect_labels(v, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="JSON/JSONL file or directory")
    ap.add_argument("--max-files", type=int, default=100)
    args = ap.parse_args()

    paths = [args.path] if args.path.is_file() else sorted(
        [*args.path.rglob("*.jsonl"), *args.path.rglob("*.json")]
    )[: args.max_files]

    if not paths:
        raise SystemExit(f"No JSON/JSONL files found under {args.path}")

    grand_labels = Counter()
    print(f"Found {len(paths)} JSON/JSONL file(s).")

    for path in paths:
        n = 0
        keys = Counter()
        labels = Counter()
        for rec in read_records(path):
            n += 1
            if isinstance(rec, dict):
                keys.update(rec.keys())
            collect_labels(rec, labels)
        grand_labels.update(labels)
        print(f"\n{path}")
        print(f"  records: {n}")
        if keys:
            print("  top-level keys:", ", ".join(k for k, _ in keys.most_common(12)))
        if labels:
            print("  labels:", dict(labels))

    if grand_labels:
        print("\nCombined discovered labels:")
        for label, count in grand_labels.most_common():
            print(f"  {label:20s} {count}")


if __name__ == "__main__":
    main()
