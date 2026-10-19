#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"
PATTERNS = [
    "bm25t5.ipynb",
    "train_target.py",
    "get_checkpoint.py",
    "claims_train.jsonl",
    "claims_dev.jsonl",
    "claims_test.jsonl",
    "corpus.jsonl",
    "*checkpoint*",
]

print(f"Searching: {UPSTREAM}")
if not UPSTREAM.exists():
    raise SystemExit("upstream/ not found. Run: bash bootstrap.sh")

found_any = False
for pattern in PATTERNS:
    matches = sorted(UPSTREAM.rglob(pattern))
    if matches:
        found_any = True
        print(f"\n[{pattern}]")
        for p in matches[:30]:
            print(" -", p.relative_to(ROOT))
        if len(matches) > 30:
            print(f" ... {len(matches)-30} more")

if not found_any:
    print("No expected files found. Did bootstrap.sh complete successfully?")
