# Week 1 Reproduction Results

| Component | Setting | Metric | Paper | Ours | Difference | Notes |
|---|---|---|---:|---:|---:|---|
| retrieval | citance | Recall@1 | 0.090 | 0.1 | +0.010 |  |
| retrieval | citance | Recall@5 | 0.280 | 0.298 | +0.018 |  |
| retrieval | citance | Recall@10 | 0.400 | 0.422 | +0.022 |  |
| retrieval | citance | Recall@20 | 0.530 | 0.529 | -0.001 | paper's pipeline reproduced: BM25 top-60 (k1=0.9 b=0.4) + monot5-base-med-msmarco; metrics in the paper's definitions |
| retrieval | citance | Recall@50 |  | 0.693 |  |  |
| retrieval | citance | Recall@100 |  | 0.818 |  |  |
| retrieval | citance | MRR | 0.320 | 0.323 | +0.003 |  |
| retrieval | citance | AllGold@20 |  | 0.372 |  |  |
| retrieval | citance_bm25_only | Recall@1 |  | 0.08 |  |  |
| retrieval | citance_bm25_only | Recall@5 |  | 0.22 |  |  |
| retrieval | citance_bm25_only | Recall@10 |  | 0.319 |  |  |
| retrieval | citance_bm25_only | Recall@20 |  | 0.444 |  | ablation: BM25 only, no rerank |
| retrieval | citance_bm25_only | Recall@50 |  | 0.651 |  |  |
| retrieval | citance_bm25_only | Recall@100 |  | 0.818 |  |  |
| retrieval | citance_bm25_only | MRR |  | 0.257 |  |  |
| retrieval | citance_bm25_only | AllGold@20 |  | 0.29 |  |  |
| retrieval | citance_monot5_fullpool | Recall@1 |  | 0.098 |  |  |
| retrieval | citance_monot5_fullpool | Recall@5 |  | 0.3 |  |  |
| retrieval | citance_monot5_fullpool | Recall@10 |  | 0.428 |  |  |
| retrieval | citance_monot5_fullpool | Recall@20 |  | 0.541 |  | ablation: MonoT5 over the whole pool, no BM25 cutoff |
| retrieval | citance_monot5_fullpool | Recall@50 |  | 0.719 |  |  |
| retrieval | citance_monot5_fullpool | Recall@100 |  | 0.865 |  |  |
| retrieval | citance_monot5_fullpool | MRR |  | 0.322 |  |  |
| retrieval | citance_monot5_fullpool | AllGold@20 |  | 0.368 |  |  |
| retrieval | citance_monot5_top60_ctx | Recall@1 |  | 0.063 |  |  |
| retrieval | citance_monot5_top60_ctx | Recall@5 |  | 0.295 |  |  |
| retrieval | citance_monot5_top60_ctx | Recall@10 |  | 0.441 |  |  |
| retrieval | citance_monot5_top60_ctx | Recall@20 |  | 0.575 |  | ablation: paper depth + prev/next sentence context - best cost/benefit, 23 min |
| retrieval | citance_monot5_top60_ctx | Recall@50 |  | 0.703 |  |  |
| retrieval | citance_monot5_top60_ctx | Recall@100 |  | 0.818 |  |  |
| retrieval | citance_monot5_top60_ctx | MRR |  | 0.253 |  |  |
| retrieval | citance_monot5_top60_ctx | AllGold@20 |  | 0.46 |  |  |
| retrieval | citance_monot5_fullpool_ctx | Recall@1 |  | 0.052 |  |  |
| retrieval | citance_monot5_fullpool_ctx | Recall@5 |  | 0.271 |  |  |
| retrieval | citance_monot5_fullpool_ctx | Recall@10 |  | 0.422 |  |  |
| retrieval | citance_monot5_fullpool_ctx | Recall@20 |  | 0.598 |  | ablation: full pool + context - best R@20, 115 min |
| retrieval | citance_monot5_fullpool_ctx | Recall@50 |  | 0.806 |  |  |
| retrieval | citance_monot5_fullpool_ctx | Recall@100 |  | 0.923 |  |  |
| retrieval | citance_monot5_fullpool_ctx | MRR |  | 0.226 |  |  |
| retrieval | citance_monot5_fullpool_ctx | AllGold@20 |  | 0.46 |  |  |
| retrieval | citance_ce_crossencoder | Recall@1 |  | 0.119 |  |  |
| retrieval | citance_ce_crossencoder | Recall@5 |  | 0.385 |  |  |
| retrieval | citance_ce_crossencoder | Recall@10 |  | 0.523 |  |  |
| retrieval | citance_ce_crossencoder | Recall@20 |  | 0.638 |  | ours: BM25 top-60 + fine-tuned PubMedBERT cross-encoder with neighbour-sentence context; +0.109 R@20 over the paper's pipeline, 17/20 articles (findings S11) |
| retrieval | citance_ce_crossencoder | Recall@50 |  | 0.708 |  |  |
| retrieval | citance_ce_crossencoder | Recall@100 |  | 0.818 |  |  |
| retrieval | citance_ce_crossencoder | MRR |  | 0.365 |  |  |
| retrieval | citance_ce_crossencoder | AllGold@20 |  | 0.493 |  |  |
| classification | top20 | Micro-F1 | 0.590 | 0.584 | -0.006 | test split; released checkpoint run frozen on top-20 evidence from the paper's BM25+MonoT5 pipeline |
| classification | top20 | Macro-F1 | 0.520 | 0.524 | +0.004 | test split; released checkpoint run frozen on top-20 evidence from the paper's BM25+MonoT5 pipeline |
| classification | top20 | ACCURATE-F1 | 0.690 | 0.692 | +0.002 | test split; released checkpoint run frozen on top-20 evidence from the paper's BM25+MonoT5 pipeline |
| classification | top20 | NOT_ACCURATE-F1 | 0.430 | 0.426 | -0.004 | test split; released checkpoint run frozen on top-20 evidence from the paper's BM25+MonoT5 pipeline |
| classification | top20 | IRRELEVANT-F1 | 0.420 | 0.455 | +0.035 | test split; released checkpoint run frozen on top-20 evidence from the paper's BM25+MonoT5 pipeline |
| classification | gold_evidence | Micro-F1 | 0.730 |  |  | paper's test-split figure; not run here (needs gold-evidence and gold-context inputs on test) |
| classification | gold_evidence | Macro-F1 | 0.750 |  |  | paper's test-split figure; not run here (needs gold-evidence and gold-context inputs on test) |
| classification | gold_context_plus_evidence | Micro-F1 | 0.750 |  |  | paper's test-split figure; not run here (needs gold-evidence and gold-context inputs on test) |
| classification | gold_context_plus_evidence | Macro-F1 | 0.780 |  |  | paper's test-split figure; not run here (needs gold-evidence and gold-context inputs on test) |
