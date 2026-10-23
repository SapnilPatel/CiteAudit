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
The paper and the original Week 1 plan refer to 8 labels, so `INDIRECT_NOT_REVIEW` is presumably folded into
`INDIRECT`. Whoever owns the 8→3 mapping must confirm this and record which collapse was used.

## 7. What actually produces the cross-encoder gain: two of six variables

The fine-tuned cross-encoder beats the published pipeline on dev by +0.109 pooled R@20
(0.534 → 0.643). Six candidate causes were ablated one variable at a time, each scored by the same
harness and compared with a **paired** article bootstrap. Only two survive.

Chain, where each delta is the increment that step adds (dev, pooled R@20):

| step | R@20 | delta | 95% CI | articles won | verdict |
|---|---|---|---|---|---|
| paper pipeline, BM25 + MonoT5 top-60 | 0.534 | — | — | — | baseline |
| + neighbour-sentence context window | 0.585 | **+0.050** | [+0.026,+0.073] | — | real |
| + fine-tuned cross-encoder replacing MonoT5 (generic `bert-base-uncased`) | 0.605 | **+0.020** | [-0.007,+0.043] | 8/10 | weak, consistent |
| + swap the encoder for PubMedBERT | 0.643 | **+0.039** | [+0.012,+0.066] | 9/10 | real |
| end to end (pilot − MonoT5+context) | | +0.059 | [+0.037,+0.080] | 9/10 | |

The three increments sum to 0.109, matching the end-to-end 0.534 → 0.643, so no interaction term is
hiding in the decomposition.

Four knobs do nothing. Each row is the variant minus the default recipe:

| variant | R@20 delta | 95% CI | AllGold@20 delta | verdict |
|---|---|---|---|---|
| `marked` — `[C]`/`[/C]` around the centre sentence | +0.003 | [-0.010,+0.023] | +0.006 | no effect |
| `bce` — pointwise BCE instead of listwise in-group softmax | +0.000 | [-0.017,+0.018] | -0.006 | no effect |
| `neg0` — negatives from BM25 rank 0 instead of rank 5 | -0.007 | [-0.020,+0.005] | -0.009 | no effect |
| `full` — 2 epochs and 7 negatives instead of 1 and 4 | -0.007 | [-0.029,+0.009] | -0.009 | no effect |

`bce` and `neg0` lose a hair of AllGold@20 on 10/10 articles, so the sign is probably real, but at
0.006-0.009 the magnitude does not justify mentioning either as a method component. In particular
the false-negative argument for skipping BM25 ranks 0-5 — plausible, and argued for twice — buys
essentially nothing.

`bce`, `neg0` and `full` each lose a hair of AllGold@20 on 10/10 articles, so the sign is probably
real, but at 0.006-0.009 the magnitude does not justify mentioning any of them as a method
component.

The `full` row is worth stating separately, because the pilot's 1 epoch / 4 negatives was chosen for
speed and was expected to be the under-trained setting. It is not: training loss halves between the
epochs (1.104 → 0.437) while dev gets slightly *worse*, so the model is already overfitting 4138
training groups after one pass. The cheap configuration is the final one — there is no larger run
waiting to be cashed in.

**Dev headline numbers for every run** (pooled, the paper's definitions):

| run | R@1 | R@5 | R@20 | AllGold@20 | MRR |
|---|---|---|---|---|---|
| pilot (context + PubMedBERT + listwise, negatives 5-100) | 0.142 | 0.420 | **0.643** | 0.525 | 0.405 |
| full (pilot recipe, 2 epochs, 7 negatives) | 0.134 | 0.441 | 0.637 | 0.514 | 0.390 |
| marked | 0.136 | 0.444 | 0.647 | 0.533 | 0.398 |
| noctx | 0.154 | 0.360 | 0.573 | 0.435 | 0.406 |
| neg0 | 0.141 | 0.422 | 0.637 | 0.514 | 0.398 |
| bce | 0.147 | 0.415 | 0.643 | 0.518 | 0.404 |
| bertbase | 0.147 | 0.369 | 0.605 | 0.478 | 0.380 |

### The context window matters more after fine-tuning, not less

Removing the neighbours from the trained model costs **-0.070** R@20 [-0.105,-0.047], worse on 10 of
10 articles — *larger* than the +0.050 the same window buys zero-shot MonoT5. Fine-tuning amplifies
the window rather than absorbing it. This was predicted the other way round here before it was
measured.

### The window helps retrieve evidence completely, not at all

The three k=20 metrics separate the mechanism:

| metric | noctx | with context | delta |
|---|---|---|---|
| Hit@20 (≥1 gold retrieved) | 0.703 | 0.718 | +0.016 [-0.000,+0.036] |
| AllGold@20 (every gold retrieved) | 0.544 | 0.617 | +0.073 [+0.040,+0.106] |
| Recall@20 pooled | 0.573 | 0.643 | +0.070 [+0.047,+0.105] |

Which citances find *something* barely changes; what the window adds is the second, third and fourth
sentence of the same piece of evidence. The dev gold-count distribution explains why this dominates
pooled recall: of 255 citances with evidence only 67 have a single gold sentence (2 golds: 107,
3: 40, 4: 27, up to one with 15), so **89% of all gold sentences live in multi-evidence citances**.

### The MRR cost of the window is a property of zero-shot scoring, not of the window

Section 3 reported that the context window trades MRR for recall (dev 0.364 → 0.300). Fine-tuning
reverses that completely, and *any* fine-tuned cross-encoder does it — the biomedical encoder is
almost irrelevant here:

| system | MRR (paper definition) |
|---|---|
| paper pipeline | 0.364 |
| + context, zero-shot MonoT5 | 0.300 |
| + context, fine-tuned generic BERT | 0.380 |
| + context, fine-tuned PubMedBERT | 0.405 |

Of the +0.131 MRR gain over MonoT5+context [+0.054,+0.184], **+0.099 comes from fine-tuning alone**
[+0.018,+0.182] and only +0.025 from the biomedical pretraining. So the recall/top-rank trade-off in
section 3 is not inherent to scoring blurred windows: it is what an untrained scorer does with them.

**Amended on 70 articles (§8).** Two parts of the paragraph above are too strong. Against the paper
pipeline on the 70 training articles the window still *costs* MRR after fine-tuning (+0.062 with the
window vs +0.069 without), so fine-tuning shrinks the cost rather than reversing it. And the
recovery is not "any fine-tuned cross-encoder": a fine-tuned generic BERT lands at MRR +0.001
[-0.023,+0.027] over the paper, i.e. exactly the paper's level, so the whole of the MRR gain is the
biomedical encoder, not fine-tuning. Dev's 10 articles had it the other way round. Prefer §8.

### Two bugs in `compare_runs.py` that produced plausible wrong numbers

Both were found while assembling this section, and both would have survived unnoticed:

1. Only `MRR` was handled explicitly; **every other metric name fell through to pooled Recall@k**.
   `--metric AllGold@20` therefore returned Recall@20's value, which was nearly taken for a genuine
   tie between two metrics. All four metrics are now implemented separately and an unknown name
   raises instead of silently answering a different question.
2. The script prints `delta = A - B`, while `run_ablations.sh` labelled it `B - A`. The variant now
   goes first, so `delta` reads *variant − baseline* and `A better on n/10` counts the variant's wins.

`run_ablations.sh` also referenced the pre-rename baseline `dev_ce_nomark.jsonl` and crashed after
all five trainings had finished; it now skips variants whose rankings file is missing.

## 8. The whole retrieval gain, validated on 70 articles

Sections 3b and 7 were measured on dev's 10 articles. The paired bootstrap resamples *articles*, so
its intervals are set by the article count, not the citance count — 70 articles buys roughly 2.6x
tighter intervals in theory and about 2.0x in practice. Everything below is therefore the number to
quote; where it disagrees with sections 3-7, section 8 wins.

**Design.** Five folds cut **by reference article**, never by citance (`fold_articles()` in
`train_reranker.py`): every citance of one reference ranks sentences of that same reference, so a
citance in train and another in eval would share the text being ranked. Each fold trains on 56
articles and scores the 14 held out; the five fold outputs are concatenated into one rankings file
covering all 1699 evaluable citances of the training split, and scored by the same harness code as
every other run. Verified per fold: the scored articles are exactly the 14 held out, zero overlap
with the fold's training articles, no duplicate ids, and citances with no gold evidence are retained
so `MRR_paper`'s denominator stays correct.

The paper's own pipeline (BM25 + MonoT5 top-60, no context) was then run over the same 70 articles
so the comparison is like-for-like. Note the asymmetry and that it is the *right* asymmetry for the
question "would replacing the MonoT5 stage improve the published pipeline": our arms never see the
article they are scored on, and the paper's arm is zero-shot so it sees no article at all.

| pipeline (70 articles, 1699 citances) | R@20 | R@10 | AllGold@20 | MRR |
|---|---|---|---|---|
| paper: BM25 + MonoT5 top-60 | 0.487 | 0.386 | 0.351 | 0.320 |
| **ours: + context window, fine-tuned PubMedBERT** | **0.560** | **0.472** | **0.429** | **0.369** |
| ours without the context window | 0.527 | 0.431 | 0.395 | 0.375 |
| ours with generic bert-base | 0.529 | 0.421 | 0.404 | 0.321 |

Paired against the paper's pipeline, variant − paper:

| arm | R@20 | R@10 | AllGold@20 | MRR |
|---|---|---|---|---|
| ours | **+0.073** [+0.055,+0.091] 62/70 | **+0.086** [+0.064,+0.108] 56/70 | **+0.078** [+0.060,+0.095] 54/70 | **+0.062** [+0.037,+0.087] 56/70 |
| no context window | +0.040 [+0.024,+0.055] 55/70 | +0.045 [+0.028,+0.061] 50/70 | +0.044 [+0.028,+0.060] 41/70 | +0.069 [+0.046,+0.092] 57/70 |
| generic bert-base | +0.042 [+0.026,+0.057] 49/70 | +0.035 [+0.015,+0.054] 45/70 | +0.053 [+0.036,+0.071] 46/70 | +0.001 [-0.023,+0.027] 40/70 |

All three R@20 intervals exclude zero with P(delta>0) = 1.000. Leaving each ingredient out gives its
contribution by subtraction: the context window is worth +0.033 R@20 and the biomedical encoder
+0.031, which sum to +0.064 against a measured total of +0.073 — the residual is the interaction,
and the two ingredients are close to additive.

**MRR behaves differently from recall, and that is the one finding dev got backwards.** The whole of
the MRR gain is the biomedical encoder: a fine-tuned *generic* BERT scores +0.001 [-0.023,+0.027],
statistically identical to the paper. And the context window still costs MRR after fine-tuning
(+0.062 with it, +0.069 without), so fine-tuning shrinks the trade-off of section 3 rather than
reversing it. Dev's 10 articles reported the opposite on both counts.

### Withdrawn: "fine-tuned cross-encoder replacing MonoT5, +0.020"

Section 7's attribution table credited +0.020 R@20 to *replacing the MonoT5 architecture with a
cross-encoder*, "weak but consistent, 8/10 articles". **That attribution was wrong**, and the error
was a confounded ablation, not a noisy measurement: the `bertbase` arm changed the architecture and
the training regime at the same time, because the MonoT5 baseline it was compared against had never
been fine-tuned. The 8/10 articles were real; what they measured was fine-tuning.

The control that settles it fine-tunes MonoT5 itself with our recipe, so architecture is the only
thing that differs. `Scorer` in `train_reranker.py` is deliberately the single scoring path for both
training and inference — for MonoT5 it is `log_softmax` over the {true, false} logits at the first
decoder step, verified bit-exact against the zero-shot scores `retrieval_harness.py` produces (max
absolute difference 0.00e+00), so fine-tuning moves exactly the quantity the published pipeline
ranks by. Both arms use the same context window, the same depth and `--max-len 192`:

| dev, length-matched and context-matched | R@20 | MRR |
|---|---|---|
| MonoT5 + context, zero-shot | 0.563 | 0.286 |
| MonoT5 + context, fine-tuned with our recipe | 0.611 | 0.395 |
| bert-base + context, fine-tuned | 0.605 | 0.380 |
| PubMedBERT + context, fine-tuned | 0.643 | 0.405 |

So **fine-tuning is worth +0.049** R@20 [+0.020,+0.070] (MRR +0.135, 10/10 articles) and the
T5 → BERT architecture swap is worth **-0.007** [-0.025,+0.010], i.e. nothing. The corrected chain
is: context window, then fine-tuning, then the biomedical encoder; the encoder *architecture* is not
an ingredient. This correction came from the objection that a fine-tuned model cannot be compared
against a zero-shot one — it was right, and nothing in section 7's table had controlled for it.

### Why the CV number is smaller than dev's — not a winner's curse

| effect | dev (10 articles) | 70 articles, 5-fold CV | test (20 articles) |
|---|---|---|---|
| context window | +0.050 | +0.033 | — |
| biomedical encoder | +0.039 | +0.031 | — |
| total over the paper | +0.109 | +0.073 | **+0.109** |

This section first read the dev/CV gap as dev inflating every effect by 1.3-1.5x, with the usual
winner's-curse explanation: dev is 10 articles, and the arms were selected on it. **The test
measurement (§11) contradicts that** — test reproduces dev's +0.109 exactly, on 20 articles that
were never used for any decision. Dev did not overestimate.

The explanation is the caveat this section raised in the same breath, now confirmed: **each CV arm
trains on 56 articles while the dev and test models train on all 70.** The cross-validated +0.073
is a conservative lower bound on the recipe's effect, not a correction to dev's +0.109. The CV
design is what makes the *attribution* trustworthy — every arm is scored on articles it never saw,
on 70 articles rather than 10 — and the per-ingredient splits (+0.033 window, +0.031 encoder) carry
the same 56-vs-70 discount, so they too are lower bounds.

## 9. Running the released MultiVerS checkpoint, and three things that block a strict reproduction

The retrieval gain in section 8 is a retrieval number. Whether it reaches the task's actual metric
has to go through the classification stage, so the released fine-tuned checkpoint
(`checkpoints/bestModel.ckpt`, 5.25 GB, from the authors' Google Drive link) was made to run and
both retrieval arms were pushed through it. Three obstacles are worth recording, because each one
is a limit on how strictly the published numbers can be reproduced rather than a local problem.

**1. The authors' `predict.py` cannot run as shipped.** It opens `../train_configs.json`, which is
not in the repository, and it does not build its dataloader from `--input_file`/`--corpus_file` at
all — those are only read at the end, to recover claim ids, while the data comes from
`data_train.ConcatDataModule.test_dataloader()` with the directory hardcoded to
`data_train/target/scifact_10`. Its pinned dependencies (`torch==1.7.1`,
`pytorch-lightning==1.2.1`, plus `bertviz` and `focal_loss`) are also far behind anything current.
`scripts/predict_multivers.py` therefore reimplements the label path directly: the pickled
checkpoint is read behind a `importlib.abc.MetaPathFinder` that manufactures empty *packages* for
the missing modules (a plain module stub fails, because `import pytorch_lightning.callbacks` needs
`__path__`), the encoder and head are rebuilt and the weights loaded with a strict key check. The
rationale head is skipped: `rationale_weight` is 0.0 in the checkpoint's hparams and rationales do
not enter the label prediction.

**2. The training-time input construction cannot be reconstructed from the released data.** The
checkpoint's hparams record `num_training_instances = 2454`, and no construction derivable from
`Data/multivers-format/claims-train.jsonl` produces that number:

| candidate construction | instances |
|---|---|
| one per (claim, gold evidence chunk) | 2568 |
| one per claim | 2141 |
| one per (claim, gold evidence chunk) plus one per no-evidence claim | 3010 |
| **recorded in the checkpoint** | **2454** |

The authors' over-4096-token skip rule does not explain the 114-instance gap either: of the 2568
(claim, evidence chunk) pairs, **zero** exceed 4096 tokens. So whatever is in their
`data_train/target/scifact_10/claims_train.jsonl` was derived by an unpublished step. The
consequence is narrow but real: **absolute F1 from this checkpoint is not comparable to the
paper's**, and only the A/B between two retrieval arms through the same checkpoint is.

**3. The 3-way gold label cannot be derived from `claims-*.jsonl`, and deriving it there mislabels
12% of the split.** The claims files carry the nine fine-grained labels of section 6 while the
model's head has three classes, and the collapse the paper uses is recorded in this repository's
own `docs/notes/week1_plan.md`:

- `ACCURATE`, `INDIRECT` → ACCURATE
- `CONTRADICT`, `NOT_SUBSTANTIATE`, `OVERSIMPLIFY`, `MISQUOTE`, `ETIQUETTE` → NOT_ACCURATE
- `IRRELEVANT` → NOT ENOUGH INFO

The trap is that `claims-*.jsonl` stores a label **only inside `evidence[doc][*]["label"]`**, so a
citance with no gold evidence sentences carries no label at all — and those citances are not one
class. `export_for_multivers.py` first assumed they were NEI, which is how the authors' *reader*
derives NEI for a non-evidence cited document, and that was wrong. Checking the aligned annotations
(§5) shows dev's 61 label-less citances are **37 ETIQUETTE** (which collapse to NOT_ACCURATE) and
**24 IRRELEVANT** (NEI). Together with `INDIRECT` → ACCURATE, the first attempt mislabelled 43 of
316 dev citances, 13.6%:

| dev | first attempt | correct |
|---|---|---|
| ACCURATE | 185 | 191 |
| NOT_ACCURATE | 70 | 101 |
| NOT ENOUGH INFO | 61 | 24 |

The checkpoint's own behaviour confirms the correction rather than just the README: under the wrong
mapping the NEI class had 61 instances and recall 0.377, and under the correct one it has 24 and
recall **0.958**. The model recovers 23 of the 24 `IRRELEVANT` citances, so NEI as it was trained
means `IRRELEVANT`, not "no evidence retrieved". The exporter now takes the fine-grained label from
`results/annotations/<split>.jsonl` and refuses to run without it; `LABEL_MAP` lists all nine
labels explicitly and raises on anything unmapped. `INDIRECT_NOT_REVIEW` — the ninth label the
paper does not list — is treated as a variant of `INDIRECT`; it never occurs in dev (25 train, 9
test), so nothing here depends on that call.

Predictions depend only on the document, so the corrected labels needed no re-running of the model:
`score_multivers.py --gold-from <export dir>` re-reads gold through one path for every arm. Absolute
F1 moves a lot — the paper's retrieval arm goes from accuracy 0.604 / macro-F1 0.548 to 0.677 /
0.684 — while every A/B conclusion in §10 survives unchanged.

### How the classification input is built, and why not the obvious way

`scripts/export_for_multivers.py` gives each claim **one synthetic document** holding the top-k
retrieved sentences, so there is exactly one (claim, document) instance per claim. The obvious
alternative — keep the original corpus and just rewrite each claim's `evidence` — does not work
here, and the reason is structural rather than cosmetic. The authors' reader builds one instance
per (claim, cited_doc) pair and `metrics.py` scores per instance; in this dataset a claim's
`cited_doc_ids` is the whole reference article split into paragraph chunks, **median 79 of them**,
median 2 sentences each. Reusing the original corpus therefore yields **24797 instances for dev's
316 claims**, almost all of them NEI, and about 78 predictions per claim with no defined
aggregation. One document per claim makes per-instance F1 equal per-citation F1 — which is what
the task asks for — and makes `--topk` the knob it is meant to be. The authors' README confirms the
construction: "Retrieve the top x sentences from the reference article using BM25+T5reranker ...
Ensure that the training, validation, and test data are placed under ./data_train/target/scifact_10/".

Two cross-checks that the export is faithful:

- the fraction of gold evidence sentences inside the exported documents reproduces the retrieval
  metric exactly — 384/597 = **0.643** for our arm at k=20, which is the pooled Recall@20 the
  harness independently reports for `dev_ce_pilot`, and 319/597 = **0.534** for the paper's arm,
  its recorded Recall@20. The classification input and the retrieval metric are demonstrably
  talking about the same sentences.
- across the four exports the claim ids, claim texts and gold labels are identical, and all 316
  documents differ between arms. The arms differ in exactly one thing.

**The rankings file must cover every claim, including the no-gold ones.** The harness skips those
61 dev citances because they cannot be scored, so `dev_monot5_paper.jsonl` had 255 rows while a
reranker file has all 316. Exporting from it as-is would have given the 61 NEI claims an *empty*
document in the paper's arm and a 20-sentence one in ours — the label would then be readable off
the input length, in the arm-dependent way most likely to be mistaken for a real effect.
`retrieval_harness.py --rank-all` ranks them too, contributing no metrics; with `--resume` only the
61 missing rows had to be computed, and every metric of the existing file stayed identical
(R@20 0.534 before and after).

| k | claims whose retrieved evidence contains no gold sentence at all |
|---|---|
| 10, paper's retrieval | 72 of 255 |
| 10, ours | 46 of 255 |
| 20, paper's retrieval | 56 of 255 |
| 20, ours | 28 of 255 |

That halving is the mechanism by which better retrieval could help classification: at k=20 our
reranker leaves half as many citances with nothing for the classifier to read.

## 10. Swapping the retriever under a frozen classifier buys nothing — and why that is not the same as "retrieval does not matter"

Section 8's gain is a retrieval number; the task's metric is 3-way citation-label F1. Both retrieval
arms were pushed through the released checkpoint unchanged — same claims, same gold labels, same
document length budget, differing only in which sentences retrieval put in front of the model — and
two further arms bracket the evidence-quality range: an **oracle** (every gold sentence present, the
remaining k slots filled from the rankings, so recall is 1.0 at the same document length) and a
**random** arm (k sentences drawn uniformly from the same candidate pool). All numbers use the
corrected label mapping of §9.

| dev, 316 claims | evidence recall | accuracy | macro-F1 | ACCURATE F1 | NOT_ACCURATE F1 | NEI F1 |
|---|---|---|---|---|---|---|
| random, k=20 | 0.119 | 0.487 | 0.473 | 0.524 | 0.473 | 0.422 |
| **paper's retrieval, k=20** | 0.534 | **0.677** | **0.684** | 0.743 | 0.529 | 0.780 |
| ours, k=20 | 0.643 | 0.674 | 0.674 | 0.747 | 0.520 | 0.755 |
| oracle, k=20 | 1.000 | 0.655 | 0.656 | 0.734 | 0.480 | 0.755 |
| random, k=10 | 0.054 | 0.386 | 0.369 | — | — | — |
| paper's retrieval, k=10 | 0.425 | 0.633 | 0.615 | — | — | — |
| ours, k=10 | 0.551 | 0.620 | 0.593 | — | — | — |
| oracle, k=10 | 1.000 | 0.601 | 0.582 | — | — | — |

Paired, with McNemar's exact test on the claims where exactly one arm is right:

| comparison | Δ accuracy | Δ macro-F1 | only A right / only B right | p |
|---|---|---|---|---|
| ours − paper's, k=20 | -0.003 | -0.010 | 30 / 31 | **1.000** |
| ours − paper's, k=10 | -0.013 | -0.021 | 32 / 36 | 0.716 |
| oracle − paper's, k=20 | -0.022 | -0.028 | 24 / 31 | 0.419 |
| oracle − paper's, k=10 | -0.032 | -0.032 | 27 / 37 | 0.260 |
| oracle − ours, k=20 | -0.019 | -0.018 | 4 / 10 | 0.180 |
| oracle − ours, k=10 | -0.019 | -0.011 | 6 / 12 | 0.238 |
| **random − paper's, k=20** | **-0.190** | **-0.211** | 20 / 80 | **0.000** |
| **random − paper's, k=10** | **-0.247** | **-0.246** | 25 / 103 | **0.000** |

Two results:

1. **Evidence matters enormously.** Replacing it with random sentences from the same pool costs
   19-25 points of accuracy at p < 0.001. The classifier reads its evidence; this is not a wiring
   problem, and the 21-23% of claims on which the two retrieval arms disagree confirms the document
   reaches the output.
2. **Evidence *better than the paper's own retrieval* buys nothing.** Across recall 0.425 → 1.000
   macro-F1 runs 0.684 → 0.656, slightly the wrong way, with all six paired tests insignificant and
   the headline ours-vs-paper comparison a near-exact tie (p = 1.000, 30 wins against 31).

Not a length artifact: at k=20 our documents average 973 tokens against the paper's 1014 (paired
median difference -35), the oracle matches the same k, and no instance in any arm exceeds 4096
tokens, so nothing was skipped.

### A plausible mechanism, and the control that killed it

Within either arm, claims whose evidence was retrieved are classified far better than claims whose
evidence was missed. Decomposing the overall delta by which arm found evidence makes that look
causal — and gives the extra evidence sentences a verdict of their own:

| k=20, ours − paper's | n | ours | paper's | Δ | contribution |
|---|---|---|---|---|---|
| only ours retrieved evidence | 30 | 0.667 | 0.467 | **+0.200** | +0.019 |
| both retrieved evidence | 197 | 0.690 | 0.690 | **+0.000** | +0.000 |
| only the paper's retrieved evidence | 2 | 1.000 | 1.000 | +0.000 | +0.000 |
| neither retrieved evidence | 26 | 0.423 | 0.577 | -0.154 | -0.013 |
| no gold sentences exist | 61 | 0.721 | 0.770 | -0.049 | -0.009 |
| total | 316 | 0.674 | 0.677 | | **-0.003** |

The `both retrieved evidence` row is exactly **0.000** over 197 claims: our arm puts more gold
sentences in those documents and it changes nothing. Section 8's gain is mostly a *completeness*
gain (AllGold@20 +0.078), and completeness is worth precisely zero here.

That leaves the +0.200 on the 30 rescued claims, which reads as a clean story: the improvement works
where it should but reaches only 9.5% of claims, so its +0.019 is cancelled elsewhere. **The oracle
refutes it.** The oracle rescues 56 claims — a larger analogous group, with perfect rather than
merely better evidence — and on exactly those claims it *loses*:

| k=20, oracle − paper's | n | oracle | paper's | Δ | contribution |
|---|---|---|---|---|---|
| only the oracle retrieved evidence | 56 | 0.464 | 0.518 | **-0.054** | -0.009 |
| both retrieved evidence | 199 | 0.688 | 0.693 | -0.005 | -0.003 |
| no gold sentences exist | 61 | 0.721 | 0.770 | -0.049 | -0.009 |
| total | 316 | 0.655 | 0.677 | | **-0.022** |

The intervention that should help most helps least. So the +0.200 does not survive — it was a
subgroup result at p = 0.070 — and the association between "evidence retrieved" and "classified
correctly" is **confounded**: claims whose evidence is easy to retrieve are also easier to classify.
Forcing the gold evidence into the documents of claims that lacked it does not transfer.

### What this does and does not license

The paper's own reported figures (as recorded in this repository's `docs/notes/week1_plan.md`) settle the
interpretation, and they rule out the conclusion this section first drew:

| the paper's classification settings | micro-F1 | macro-F1 |
|---|---|---|
| top-20 retrieved evidence + MultiVerS | 0.59 | 0.52 |
| **gold evidence** | **0.73** | **0.75** |
| gold citation context + gold evidence | 0.75 | 0.78 |

The paper measures **+0.23 macro-F1 from perfect evidence**. This section's oracle measures zero.
Both can be true, because they differ in one thing: the paper's gold-evidence model is *trained* on
gold evidence, while the oracle here feeds gold evidence to a checkpoint trained on retrieved
evidence. Evidence quality is worth a great deal to this architecture — but only to a classifier
fitted to the evidence distribution it will be given.

So the defensible conclusion is narrower than "retrieval is not the bottleneck", which an earlier
draft of this section asserted:

> **A better retriever cannot be dropped in under the frozen released checkpoint.** Swapping the
> retrieval stage alone changes nothing at any quality level up to perfect evidence, because the
> classifier is fitted to the evidence distribution it was trained on. Whether §8's retrieval gain
> is worth anything to the task is therefore **still open**, and it turns on one experiment that was
> not run here: retraining MultiVerS on our retrieval output and comparing against retraining on
> the paper's. The paper's own gold-evidence row is the reason to expect that experiment to show
> something — it is the only evidence either way, and it points towards evidence quality mattering.

That retraining is Longformer-large fine-tuning and needs a CUDA GPU; §9's constraint applies too —
the authors' training-time input construction cannot be reconstructed from the released data, so a
retrained model would be our recipe, not theirs.

One caveat on the absolute figures: the paper's numbers above are its test-split results and
everything in this section is dev, so only the *relative* 0.52 → 0.75 jump is informative here, not
the gap between 0.52 and our 0.684.

## 11. The single test measurement

Test was held untouched for every decision in sections 1-10: the recipe, all eight ablations, the
window and depth settings, the CV design and the classification arms were chosen on dev and on
cross-validated training articles. This is the one measurement, run with the default flags — which
are the pilot recipe, verified item by item against it — on the checkpoint trained on all 70
training articles.

```bash
python scripts/train_reranker.py --no-train --init-from checkpoints/ce_pilot \
  --predict test --rankings-out results/retrieval/test_ce_pilot.jsonl
python scripts/retrieval_harness.py --split test --ranker file \
  --rankings-in results/retrieval/test_ce_pilot.jsonl
```

| test, 20 articles, 487 evaluable citances | R@20 | R@10 | AllGold@20 | MRR |
|---|---|---|---|---|
| BM25 only | 0.444 | 0.319 | 0.290 | 0.257 |
| paper pipeline, BM25 + MonoT5 top-60 | 0.529 | 0.422 | 0.372 | 0.323 |
| *the paper's published figures* | *0.53* | — | — | *0.32* |
| **ours** | **0.638** | **0.523** | **0.493** | **0.365** |

Paired over the 20 test articles, ours − the paper's pipeline:

| metric | delta | 95% CI | articles won |
|---|---|---|---|
| Recall@20 | **+0.109** | [+0.084,+0.131] | 17/20 |
| Recall@10 | +0.101 | [+0.058,+0.138] | 17/20 |
| AllGold@20 | +0.121 | [+0.080,+0.169] | 16/20 |
| MRR | +0.053 | [+0.008,+0.096] | 13/20 |

All four intervals exclude zero. MRR is positive and significant here, where the 70-article CV had
it at +0.062 and dev's zero-shot comparison had the window costing MRR — §8's amended reading holds:
the cost is a zero-shot artifact, and a fine-tuned biomedical reranker beats the paper on every
metric including MRR.

Test reproduces dev's +0.109 exactly, which is what retires the winner's-curse reading in §8: the
cross-validated +0.073 is lower because each fold trains on 56 articles rather than 70, not because
dev was optimistic.

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
