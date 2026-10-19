# Week 2 report — evidence retrieval

## What we did

1. Wrote `scripts/retrieval_harness.py`. The old `week2_pipeline.py` could not read the official
   data: it expects one abstract per claim, but claims list `cited_doc_ids`, a range of corpus
   chunks that together form the cited article. The new harness pools every sentence of that
   article and reports Recall@k, MRR and AllGold@k in both the paper's pooled definitions and a
   per-query average, with 95% CIs from a bootstrap over reference articles — test has only 20
   articles, so resampling queries would understate the error.
2. Ran five retrieval settings on the test split: BM25 alone, BM25 top-60 + MonoT5 (the authors'
   depth and checkpoint), MonoT5 over the full candidate pool, and both depths again with each
   candidate scored together with its previous and next sentence.
3. Diagnosed the differences by tracking individual gold sentences across runs, and read the
   paper's Table 3 to fix the metric definitions, which turned out to matter more than any setting.
4. Aligned `Data/annotations.zip` to claim ids (`scripts/align_annotations.py`).

MonoT5 is `castorini/monot5-base-med-msmarco`, zero-shot — no in-domain training yet.

## Results (test, 606 citances, 487 with evidence, 20 articles)

Metrics are the paper's own definitions: Recall@k pooled over the 1215 gold evidence sentences, MRR
over all 606 citances truncated at rank 20. Per-query averaging runs ~0.07 higher at k=20 and is not
comparable to published numbers.

| setting | R@1 | R@5 | R@10 | R@20 | R@100 | MRR | AllGold@20 |
|---|---|---|---|---|---|---|---|
| paper, citance row | 0.09 | 0.28 | 0.40 | 0.53 | – | 0.32 | – |
| BM25 only | 0.080 | 0.220 | 0.319 | 0.444 | 0.818 | 0.257 | 0.290 |
| **BM25 top-60 + MonoT5 — the paper's pipeline** | 0.100 | 0.298 | 0.422 | **0.529** | 0.818 | **0.323** | 0.372 |
| full-pool MonoT5 | 0.098 | 0.300 | 0.428 | 0.541 | 0.865 | 0.322 | 0.368 |
| top-60 + MonoT5 + context | 0.063 | 0.295 | 0.441 | 0.575 | 0.818 | 0.253 | 0.460 |
| full-pool MonoT5 + context | 0.052 | 0.271 | 0.422 | **0.598** | **0.923** | 0.226 | 0.460 |

## Findings

**The paper's citance row reproduces exactly.** Their pipeline, their checkpoint, their BM25 depth:
R@20 0.529 and MRR 0.323 against the published 0.53 and 0.32, with R@1/5/10 within 0.02 throughout.

**An earlier version of this report claimed we were 7 points above the paper. That was a metric
artifact and it is withdrawn.** Nothing about the retrieval changed, only the averaging: the paper
pools Recall@k over evidence sentences ("the proportion of evidence sentences within the top k")
while we had averaged each query's own recall, which downweights the queries carrying the most gold
sentences; and the paper averages MRR over all 606 citances with the rank truncated at 20, while we
divided by the 487 that have evidence. Together those shifted R@20 from 0.529 to 0.600 and MRR from
0.323 to 0.405. Any number compared against this paper must state its averaging.

**Our one real gain over the published pipeline is neighbour context: +0.046 R@20 for 9 extra
minutes.** Scoring each candidate together with its previous and next sentence takes R@20 from 0.529
to 0.575 and AllGold@20 — the complete gold set inside the top 20, which an accuracy judgement needs
— from 0.372 to 0.460. It costs top-rank precision: R@1 0.100 → 0.063, MRR 0.323 → 0.253, because a
neighbour's window contains the gold sentence's own text and often outranks it (top-1 is a gold
sentence 24.4% → 12.9% of queries, top-1 is an immediate neighbour of one 2.3% → 14.2%). For a
pipeline whose classifier consumes top-20 evidence, that is the right trade.

**Removing the BM25 cutoff is worth little, and how little depends on the averaging.** Full-pool
scoring gives +0.012 R@20 on its own and +0.023 on top of context in the paper's metric, for five
times the compute (115 against 23 minutes); per query it is zero to slightly negative. Tracking gold
sentences shows the mechanism: it lifts 52 golds into the top 20 that top-60 reranking cannot reach
and loses 38 that top-60 reranking had, displaced by the extra distractors — net +14 of 1215. Of the
golds BM25 ranked 60 or deeper, zero-shot MonoT5 rescues only 15%. A bigger pool is necessary but
not sufficient; it pays off only with a more precise scorer, so it is worth re-measuring once an
in-domain model exists. Full pool does win in the deep range (R@100 0.923 against 0.818), which is
the setting for a two-stage design.

**Half the queries are structurally noisy.** Citances marked `<|multi_cit|>` cite several papers at
once, so the query carries text belonging to other references: R@20 0.498 against 0.564 for
single-cite queries under the paper's pipeline. The gap has no query-side fix (see below); context
scoring narrows it to 0.040, the best of the four settings.

**`Data/annotations.zip` is now aligned to claim ids, and it ruled out three ideas rather than
confirming them.** `scripts/align_annotations.py` ties all 606/316/2141 records 1:1 to claim ids
(exactly, via `doc_id = reference * 1000 + paragraph`, then text keys within each reference group;
480 of 487 checkable test rows confirmed through an independent evidence-span path). What the
aligned data then showed: (1) the authors' query equals the citance for 590/606 test rows, so query
construction was never the explanation for the apparent gap (the metric was); (2) `marker_span` adds nothing, because it marks our own citation position, which
the claims file already encodes, and 279 of 340 multi_cit rows put every co-cited reference at one
position, so the multi_cit gap has no query-side fix; (3) expanding the query with the citing
paragraph hurts monotonically — BM25 R@20 0.498 citance, 0.463 with neighbour sentences, 0.405 with
the whole paragraph. The alignment remains necessary for the classification stage, which needs the
per-claim original labels and evidence spans.

**The data has 9 original labels, not 8.** `INDIRECT_NOT_REVIEW` (25 train / 9 test / 0 dev) is the
extra one; it is presumably folded into `INDIRECT` in the paper. Whoever owns the 8→3 mapping needs
to confirm which collapse was used.

## Next

No cheap query-side win is left, so the in-domain cross-encoder is the only remaining lever on
retrieval. Develop it at BM25 top-60 + context (23 min/run), not full pool (115 min/run), and
re-measure full pool only once a trained model exists, since the pool effect depends on scorer
precision. Dev baselines are in place for tuning (BM25 0.462, paper pipeline 0.534, +context 0.585
R@20), and both effects replicate there with the same sign, so tuning happens on dev and test gets a
single final measurement. No classification numbers yet — those wait on the released MultiVerS
checkpoint.

Details and repro commands: `results/week2_retrieval_findings.md`. Per-run metrics and rankings:
`results/retrieval/`.
