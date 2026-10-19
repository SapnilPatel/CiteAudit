#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "results" / "ours.csv"
out = ROOT / "results" / "week1_comparison.md"

df = pd.read_csv(path)
lines = [
    "# Week 1 Reproduction Results",
    "",
    "| Component | Setting | Metric | Paper | Ours | Difference | Notes |",
    "|---|---|---|---:|---:|---:|---|",
]
for _, r in df.iterrows():
    paper = r.get("paper_value")
    ours = r.get("ours")
    try:
        diff = float(ours) - float(paper) if pd.notna(ours) else None
    except Exception:
        diff = None
    lines.append(
        "| {component} | {setting} | {metric} | {paper} | {ours} | {diff} | {notes} |".format(
            component=r.get("component", ""),
            setting=r.get("setting", ""),
            metric=r.get("metric", ""),
            paper="" if pd.isna(paper) else f"{float(paper):.3f}",
            ours="" if pd.isna(ours) else str(ours),
            diff="" if diff is None else f"{diff:+.3f}",
            notes="" if pd.isna(r.get("notes")) else str(r.get("notes")),
        )
    )
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out)
