#!/usr/bin/env bash
# The paper's own pipeline measured on the SAME 70 articles the cross-encoder was cross-validated
# on, so the total improvement can be stated on 70 articles instead of dev's 10.
#
# Inference only: MonoT5 is zero-shot, so there is no training and no fold structure to respect —
# one pass over the whole train split is the right comparison for the concatenated CV rankings.
set -u
PY=${PY:-python3}
m=results/retrieval/train_monot5_paper.metrics.json
if [ -f "$m" ]; then
  echo "== paper pipeline on train already scored, skipping"
else
  echo "== paper pipeline (BM25+MonoT5, no context) on the 70 train articles  $(date +%H:%M)"
  $PY scripts/retrieval_harness.py --split train --ranker bm25-monot5 --rerank-depth 60 \
      --rankings-out results/retrieval/train_monot5_paper.jsonl --metrics-out "$m" --resume
fi
echo; echo "=== total improvement on 70 articles (delta = ours - paper) ==="
for k in Recall@20 AllGold@20 MRR Recall@1; do
  $PY scripts/compare_runs.py results/retrieval/train_cv_pilot.jsonl \
      results/retrieval/train_monot5_paper.jsonl --metric $k \
      | grep -E "delta" | sed "s/^/  ours - paper  ${k}:  /"
done
