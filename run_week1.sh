#!/usr/bin/env bash
set -euo pipefail

echo "CSCI 544 Group 23 — Week 1 baseline helper"
echo ""
echo "1) Install local helpers: pip install -r requirements-local.txt"
echo "2) Clone official repos: bash bootstrap.sh"
echo "3) Locate files: python scripts/locate_upstream.py"
echo "4) Inspect data: python scripts/inspect_dataset.py upstream/Citation-Integrity/Data"
echo "5) Run authors' bm25t5.ipynb and record retrieval metrics"
echo "6) Run authors' MultiVerS baseline and record classification metrics"
echo "7) Edit results/ours.csv"
echo "8) Build comparison: python scripts/make_results_table.py"
