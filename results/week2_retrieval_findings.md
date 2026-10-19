# Week 2 — retrieval findings (test split)

All numbers from `scripts/retrieval_harness.py` on `Data/multivers-format/`, citance as query,
606 test citances over 20 reference articles (487 carry gold evidence; 1215 gold sentences in all).

**Metrics follow the paper's definitions**, which matters more than it sounds (section 1):
Recall@k is pooled over evidence sentences — Σ(gold inside top k) / Σ(gold) — not averaged per
query, and MRR is averaged over *all* 606 citances with the rank truncated at 20, so a citance with
no evidence or no gold inside the top 20 contributes 0. The harness prints the per-query (macro)
averaging alongside; it runs about 0.07 higher at k=20 and is not comparable to published numbers.
Brackets are 95% CIs from an article-level bootstrap — the unit of resampling is the reference
article, because test holds only 20 of them.

| setting | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | MRR |
|---|---|---|---|---|---|---|---|
| paper (Sarol et al., Table 3, citance row) | 0.09 | 0.28 | 0.40 | 0.53 | – | – | 0.32 |
| BM25 only | 0.080 | 0.220 | 0.319 | 0.444 [0.40,0.50] | 0.651 | 0.818 | 0.257 |
| BM25 top-60 → MonoT5 — **the paper's pipeline** | 0.100 | 0.298 | 0.422 | 0.529 [0.47,0.59] | 0.693 | 0.818 | 0.323 |
| full-pool MonoT5 (no BM25 cutoff) | 0.098 | 0.300 | 0.428 | 0.541 [0.49,0.60] | 0.719 | 0.865 | 0.322 |
| BM25 top-60 → MonoT5 + prev/next sentence | 0.063 | 0.295 | 0.441 | 0.575 [0.51,0.65] | 0.703 | 0.818 | 0.253 |
| full-pool MonoT5 + prev/next sentence | 0.052 | 0.271 | 0.422 | 0.598 [0.54,0.66] | 0.806 | 0.923 | 0.226 |

MonoT5 = `castorini/monot5-base-med-msmarco`, zero-shot, the authors' own checkpoint. Mean candidate
pool 239 sentences/query.

## 1. The paper's citance row reproduces exactly — the earlier "we beat it" was a metric artifact

Running the authors' pipeline (their checkpoint, their BM25 depth of 60) gives **R@20 0.529 and
MRR 0.323** against their published **0.53 and 0.32**, with R@1 0.100 / R@5 0.298 / R@10 0.422
against 0.09 / 0.28 / 0.40. That is a reproduction, not a discrepancy.

An earlier version of this document reported the same run at 0.600 and concluded we were seven
points above the paper, and that our *unreranked* BM25 matched their reranked row. Both claims were
wrong, and nothing about the retrieval changed — only the averaging:

- **Recall@k.** The paper defines it as "the proportion of evidence sentences within the top k
  retrieved sentences", i.e. pooled over all 1215 gold sentences. We had averaged each query's own
  recall, which downweights queries carrying many gold sentences — and those are the harder ones.
  The gap is +0.071 at k=20.
- **MRR.** The paper averages over every citance and evaluates "on top 20 sentences". We had divided
  by the 487 citances that have evidence and used the untruncated rank: 0.405 against 0.323.
  487/606 accounts for almost all of it.

Lesson for the group: a metric definition shifted a headline number by 13% relative and inverted a
conclusion. Any number compared against this paper must state its averaging, and
`results/ours.csv` now carries the pooled values only.

What this also settles: our pure-Python BM25 with Lucene's default parameters (k1=0.9, b=0.4, no
stemming, no stopword list) reaches 0.444 where the authors' pyserini index plus MonoT5 reaches
0.529, so the reranker accounts for +0.085 and the remaining implementation differences are small.
Running `bm25t5.ipynb` under pyserini is no longer a blocker for trusting these numbers, though it
stays on the Week-1 checklist.

## 2. Full-pool scoring is worth little, and how little depends on the averaging

Candidate depth crossed with neighbour context, R@20 in both averagings, with wall clock on M2 MPS:

| | no context | + context |
|---|---|---|
| **BM25 top-60** | 0.529 micro / 0.600 macro — 14 min | 0.575 / 0.632 — 23 min |
| **full pool (239/query)** | 0.541 / 0.605 — 50 min | 0.598 / 0.626 — 115 min |
| pool effect | +0.012 micro / +0.005 macro | **+0.023 micro / −0.006 macro** |

The two averagings disagree in *sign* on the bottom-right cell, so the earlier flat claim that full
pool "buys nothing" was too strong. In the paper's pooled metric the pool effect is consistently
positive but small (+0.012 alone, +0.023 on top of context) for five times the compute; per query it
is zero or slightly negative. The direction of the disagreement is itself informative: pooling
weights queries with many gold sentences more, and those are exactly the queries a deeper candidate
list helps, because their evidence is spread further down the ranking.

Mechanism, measured by tracking individual gold sentences between the two no-context runs:

- full-pool scoring lifted **52** gold sentences into the top 20 that the top-60 rerank could not reach
- it **lost 38** gold sentences that the top-60 rerank had in the top 20, displaced by the ~180 extra distractors
- net **+14** of 1215 gold sentences, which is the +0.012 pooled difference

Of the 350 gold sentences BM25 ranked at 60 or deeper, zero-shot MonoT5 rescued only 14.9% into the
top 20. So a larger pool is a *necessary but not sufficient* condition: it only pays off with a
scorer precise enough to exploit it. Full pool does win clearly in the deep range — R@50 0.806 vs
0.703, R@100 0.923 vs 0.818 — so it is the right setting for a two-stage design that recalls 50-100
sentences before a finer pass.

**Implication for the planned PubMedBERT/BioLinkBERT cross-encoder:** develop and tune at top-60 +
context (23 min/run) and verify the final model at full pool (115 min/run), where its extra
precision should widen the +0.023. Do not conclude full pool is useless: the
52-rescued-against-38-displaced balance depends on scorer precision and must be re-measured once an
in-domain model exists.

## 3. Scoring with neighbour sentences trades top-rank precision for recall

Giving each candidate its previous and next sentence (`--context`, windows cross chunk boundaries),
at the authors' top-60 depth, in the paper's metrics:

| metric | no context | + context | delta |
|---|---|---|---|
| R@20 | 0.529 | 0.575 | +0.046 |
| R@10 | 0.422 | 0.441 | +0.019 |
| AllGold@20 (complete gold set in top 20, per query) | 0.372 | 0.460 | +0.088 |
| R@1 | 0.100 | 0.063 | −0.037 |
| MRR | 0.323 | 0.253 | −0.070 |

This is the largest single gain we have over the published pipeline: **+0.046 R@20 on top of an
exact reproduction**, and it costs 9 extra minutes.

The top-rank loss is a blur effect, measured directly: the fraction of queries whose top-1 is a gold
sentence falls 24.4% → 12.9%, while the fraction whose top-1 is an *immediate neighbour* of a gold
sentence rises 2.3% → 14.2%. A neighbour's window contains the gold sentence's own text, so it
scores nearly as high and often outranks it. Neighbour-of-gold slots inside the top 20 double,
0.52 → 1.08 per query — cheap at k=20, fatal at k=1.

**Right trade for this pipeline.** The classifier consumes top-20 evidence, so R@20 and AllGold@20
are the operative metrics, and AllGold@20 is where the gain is largest: an accuracy judgement needs
the complete evidence set, not one sentence of it. Report the MRR regression as a known cost. A
reranker trained in-domain could get both by scoring windows and attributing the score to the centre
sentence (max-pool per sentence over the windows containing it) instead of ranking windows.

## 3b. Dev baselines, and both effects replicate there

Dev (316 citances, 255 with evidence, 10 reference articles), same metrics:

| setting | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | MRR | AllGold@20 |
|---|---|---|---|---|---|---|---|---|
| BM25 only | 0.107 | 0.256 | 0.350 | 0.462 [0.40,0.57] | 0.653 | 0.812 | 0.293 | 0.282 |
| paper pipeline (top-60 + MonoT5) | 0.137 | 0.325 | 0.425 | 0.534 [0.46,0.63] | 0.685 | 0.812 | 0.364 | 0.388 |
| top-60 + MonoT5 + context | 0.095 | 0.343 | 0.459 | 0.585 [0.52,0.66] | 0.683 | 0.812 | 0.300 | 0.447 |

Both effects carry across splits with the same sign and nearly the same size:

| delta in R@20 | dev | test |
|---|---|---|
| MonoT5 over BM25 | +0.072 | +0.086 |
| neighbour context over the paper's pipeline | +0.050 | +0.046 |

So the context gain is not a test-split artifact, and the MRR cost replicates too (−0.064 dev,
−0.070 test). Dev sits 0.01-0.02 above test at every setting, which is split difficulty and does not
affect the comparisons. **Tune on dev, and leave test for a single final measurement.**

Full-pool variants were not run on dev: top-60 + context is the development configuration, and the
pool effect needs re-measuring against a trained model anyway.

## 4. multi_cit queries are 5-7 points worse, and the cause is intrinsic

R@20 in the paper's pooled metric, split by whether the citance cites one reference or several:

| setting | `<\|multi_cit\|>` (n=241, 633 gold) | `<\|cit\|>` (n=246, 582 gold) | gap |
|---|---|---|---|
| BM25 only | 0.417 | 0.473 | 0.055 |
| paper pipeline (top-60 + MonoT5) | 0.498 | 0.564 | 0.066 |
| top-60 + context | 0.556 | 0.596 | 0.040 |
| full pool + context | 0.572 | 0.627 | 0.055 |

Half the test queries cite several papers at once, so the query text carries content belonging to
other references. (Per-query averaging puts this gap at 11 points; pooled it is 5-7, because
multi_cit citances carry more gold sentences each — 2.6 against 2.4 — and the two averagings weight
that differently. The pooled figure is the one to quote.)

Section 5 shows the gap cannot be closed on the query side: in 279 of the 340 multi_cit rows every
co-cited reference sits at one marker position, so the citance carries no signal separating them.
Only a model trained to ignore the irrelevant clauses can recover it. Context scoring already
narrows the gap to 0.040, the smallest of the four settings.

## 5. `Data/annotations.zip` is aligned to claim ids — and it closes off three ideas

Extracted to `upstream/annotations_extracted/` (gitignored) and aligned by
`scripts/align_annotations.py` into `results/annotations/{train,dev,test}.jsonl`:
2141/316/606 records, one per citance-reference pair, carrying `citing_paragraph`,
`citation_context` spans, `evidence_segments` spans, `marker_span`, and the original label.

**Alignment.** Structural first: corpus `doc_id = <reference number> * 1000 + <paragraph index>`,
verified by `references_sentence/*.json`, whose `par_id` reproduces the corpus sentence lists
exactly (89/89 on reference 010). No claim spans two references, and every one of the 100
reference groups holds equally many claims and annotation files, so the group assignment is exact.
Within a group, claims are matched to files in strict key tiers — K1 the citance equals a
`citation_context` sentence, K2 a gold evidence sentence equals an `evidence_segment`, K3 label
agreement as a hard constraint — then maximum bipartite matching. K1 must outrank K2: several
citances of one reference can rest on the same evidence sentence, so a K2 edge will otherwise be
chosen over the correct K1 pair (that bug produced visibly wrong pairs on the first pass).

Result: 606/606, 316/316, 2141/2141 aligned 1:1, none unresolved; test is 542 by K1, 47 by K2 only,
17 by label and elimination. Verified through an independent path — mapping each
`evidence_segment`'s character span onto `(par_id, sentence index)` and comparing with the claim's
own evidence indices — 480 agree, 7 disagree (all off-by-one sentence boundaries, all K1 pairs),
119 not checkable because those claims carry no evidence. 29 test rows could have been permuted
with another row at their tier and are flagged `ambiguous: true` in the output.

**What the alignment then ruled out.** All three were worth testing and none survived. Note that
the BM25 R@20 figures quoted in item 3 are per-query averages, from before the metric was corrected;
the comparison between them is unaffected:

1. *The authors query with a wider citation-context span.* They do not. Once aligned, their query
   (`citing_paragraph` sliced from the min start to the max end of the `citation_context` spans, as
   `bm25t5.ipynb` builds it) is identical to the citance for 590/606 test and 316/316 dev rows. This
   was the leading explanation for the apparent gap against the paper's 0.53 — which section 1 then
   showed to be a metric artifact, so the question it was asked to answer no longer exists.
2. *`marker_span` can disambiguate a multi_cit query.* It cannot add anything: it points at
   `<|cit|>` or `<|multi_cit|>` — our own marker, never `<|other_cit|>` — in 606/606 rows, and the
   claims file already uses that same convention (99/99 agreement on rows carrying both). The only
   extra information is the original citation string ("501", "Koeth et al., 2013"), which is
   worthless for ranking sentences inside an already-fixed reference article. Worse, 279 of the 340
   multi_cit rows carry `<|multi_cit|>` as their only marker, meaning every co-cited reference sits
   at one position and the sentence genuinely does not distinguish them. The multi_cit gap has no
   query-side fix.
3. *Expanding the query with the citing paragraph.* It hurts, monotonically in how much is added
   (BM25 R@20): citance 0.498, citance + prev/next sentence 0.463, whole paragraph 0.405. The
   paragraph is seven times longer than the citance (median 1177 vs 171 chars) and dilutes the
   signal, equally for multi_cit and single-cite queries.

**What it is still good for.** Per-claim-id original labels and `evidence_segments` character spans,
both of which the classification stage and the 8→3 mapping audit need, and `citing_paragraph` if a
trained model can use selectively what BM25 cannot.

Written before section 1 was understood, this section was aimed at explaining a gap that turned out
not to exist. The three negative results stand on their own, and the alignment is still required for
the classification stage.

## 6. The dataset has 9 original labels, not 8

Across the annotations: `ACCURATE`, `ETIQUETTE`, `NOT_SUBSTANTIATE`, `IRRELEVANT`, `CONTRADICT`,
`OVERSIMPLIFY`, `INDIRECT`, `MISQUOTE`, and `INDIRECT_NOT_REVIEW`. Train and test carry all nine;
dev has eight (no `INDIRECT_NOT_REVIEW`, which has 25 train / 9 test instances and none in dev).
The paper and `TEAM_TASKS.md` refer to 8 labels, so `INDIRECT_NOT_REVIEW` is presumably folded into
`INDIRECT`. Whoever owns the 8→3 mapping must confirm this and record which collapse was used.

## Reproducing these numbers

```bash
python scripts/retrieval_harness.py --split test                        # BM25 only, 14 s
python scripts/retrieval_harness.py --split test --ranker bm25-monot5   # the paper's pipeline, 14 min
python scripts/retrieval_harness.py --split test --ranker bm25-monot5 --context            # 23 min
python scripts/retrieval_harness.py --split test --ranker bm25-monot5 --rerank-depth 1000  # 50 min
python scripts/retrieval_harness.py --split test --ranker bm25-monot5 --rerank-depth 1000 --context
```

Every run prints both averagings; quote the micro column against published numbers. To re-score an
existing rankings file without rerunning the model:

```bash
python scripts/retrieval_harness.py --split test --ranker file \
  --rankings-in results/retrieval/test_bm25_monot5.jsonl \
  --metrics-out results/retrieval/test_bm25_monot5.metrics.json
```

Add `--resume` with `--rankings-out` to continue an interrupted run; rankings are flushed per query.
Per-run metrics and rankings are in `results/retrieval/`. MonoT5 runs need
`torch transformers sentencepiece protobuf`.
