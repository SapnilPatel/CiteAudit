#!/usr/bin/env python3
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "results" / "_smoke_predictions.csv"
rows = [
    ("ACCURATE", "SUPPORT"),
    ("INDIRECT", "ACCURATE"),
    ("IRRELEVANT", "NEI"),
    ("MISQUOTE", "REFUTE"),
    ("ETIQUETTE", "NOT_ACCURATE"),
]
with TMP.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["gold", "prediction"])
    w.writerows(rows)

cmd = [sys.executable, str(ROOT / "scripts" / "evaluate_classification.py"), "--input", str(TMP)]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
TMP.unlink(missing_ok=True)
print("\nSmoke test passed.")
