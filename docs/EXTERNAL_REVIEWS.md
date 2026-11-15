# External reviews of the handoff (2026-11-15), checked against the data

Two outside reviews of [`HANDOFF.md`](HANDOFF.md), one written with ChatGPT and one with Gemini,
proposed next steps. This records every proposal, what the data says about it, and the order we
are running them in. The numbers here were measured in this repository.

## What both reviews proposed

| # | Proposal | Who | Verdict after checking |
|---|---|---|---|
| 1 | Keep citation markers in the hypothesis (`[CITED]`, `[CITED, with others]`, `[other]`) and fine-tune end to end | both, first priority | **Done: CV accuracy 0.676 → 0.695 (p = 0.015), macro-F1 unchanged, gate not passed;** same as current after recalibration ([`markers_v1.md`](../results/markers_v1.md)) |
| 2 | Oracle: hypothesis = gold `citation_context` span, then a span extractor if it helps | both | **Little room: see "Citation scope" below.** The claim already equals the gold span for 91% of citations |
| 3 | Oracle: gold evidence with the current classifier, same budget, to see if retrieval still limits | both (Gemini: do it first) | **Done: perfect evidence changes nothing (test 20 fixed, 20 broken).** Retrieval is not the bottleneck |
| 4 | Decompose into relevance / attribution / faithfulness heads on a shared encoder, combined by an out-of-fold stacker | both (ChatGPT: most important architecture) | Next, after v9. Not the failed 9-way head: three binary heads with masked losses |
| 5 | Attribution cue: does the matching evidence sentence itself carry a citation ("[3]", "(12)")? | Gemini | Cheap feature for #4; to measure |
| 6 | NLI mismatch: overgeneralised claims are "neutral" to NLI but NOT_ACCURATE here | ChatGPT | Valid. Fine-tuning relearns the head, but it argues for #4's separate faithfulness head |
| 7 | Report single vs grouped citations, ETIQUETTE vs substantive-error recall, not just totals | ChatGPT | **Built into the v9 notebook** |
| 8 | Do not let INDIRECT (counted ACCURATE) become NOT_ACCURATE in an attribution head | ChatGPT | Adopted for #4 |
| 9 | Audit confident disagreements for label noise before claiming a win | ChatGPT | **Done** ([`label_audit_v1.md`](../results/label_audit_v1.md)): of 30 confident errors, 13 model errors, 13 debatable, 4 likely mislabelled |
| 10 | Stop: bigger encoders, LLM judges, same-model ensembles, unfiltered synthetic data | both | Agreed; already the plan |
| 11 | Targets: accuracy > 0.71, macro-F1 > 0.60, NOT ACCURATE F1 > 0.48 | ChatGPT | Adopted |

## Caveat on markers (measured before running v9)

The current classifier already labels 65% of ETIQUETTE citations NOT ACCURATE in cross-validation
(198 of 304) without seeing any marker, presumably from how ETIQUETTE sentences read (background
statements). And it wrongly calls 159 accurate grouped citations NOT ACCURATE. So the room for
markers is the remaining 35% of ETIQUETTE plus those false alarms, not all of ETIQUETTE. The v9
gate (CV macro-F1 >= 0.580 and NOT ACCURATE F1 not lower) is set in advance.

## Citation scope (proposal 2)

Comparing each citation's text with its gold `citation_context` (both with markers removed):

| | Same text | Gold span much shorter (< 90%) | Longer |
|---|---|---|---|
| train (2141) | 1955 (91%) | 111 (5%) | 11 |
| test (606) | 556 (92%) | 32 (5%) | 3 |

The dataset's citing sentence *is* the annotators' citation span for 9 citations in 10. A scope
oracle can change at most the 32 test citations where they differ, and 13 of those are ACCURATE.
Even if every one became correct, that is about 3 accuracy points at most. A scope extractor is
not worth building for this benchmark. In a deployed tool, where users paste a whole sentence, it
may still matter.

## Oracle evidence (proposal 3)

`scripts/oracle_evidence.py` scored the released classifier (`Sapnil/citeaudit-nli`) on the same
claims with three premises: our retrieved top 20; the gold evidence sentences first, then retrieved
ones up to 20 (same budget); and the gold sentences only. Predictions are in
[`results/classification/oracle_evidence/`](../results/classification/oracle_evidence/).

| Premise | Dev acc | Dev macro-F1 | Test acc | Test macro-F1 | Test NOT ACC F1 | Test vs retrieved (right only here / only retrieved, McNemar p) |
|---|---|---|---|---|---|---|
| Retrieved top 20 | 0.703 | 0.616 | 0.662 | 0.546 | 0.394 | — |
| Gold first, same budget | 0.696 | 0.608 | 0.662 | 0.547 | 0.383 | 20 / 20, p = 1.00 |
| Gold only | 0.516 | 0.453 | 0.500 | 0.428 | 0.435 | 33 / 131, p < 0.001 |

**Perfect evidence changes nothing.** With gold evidence guaranteed at the front of the premise,
the classifier fixes 20 test citations and breaks 20 others. Retrieval is not what limits this
classifier, as it was not for the paper's MultiVerS (`repro/`). Gold-only premises are much worse,
because the model was trained on 20-sentence premises and loses most of its NOT ENOUGH INFO
calibration on short ones. So focused-evidence aggregation would need retraining to test, and
nothing here suggests it would help.

Where the classifier fails, by fine label (test, share recalled as NOT ACCURATE, or as NEI for
IRRELEVANT; identical with gold evidence except CONTRADICT 0.50 → 0.42):

| ETIQUETTE (73) | NOT_SUBSTANTIATE (29) | OVERSIMPLIFY (47) | CONTRADICT (12) | MISQUOTE (9) | IRRELEVANT (50) |
|---|---|---|---|---|---|
| 0.48 | 0.28 | **0.13** | 0.50 | **0.11** | 0.54 |

The weakest errors are oversimplification and misquotation: subtle wording differences between a
claim and evidence that the model *already has*. That is a judgement problem, which supports the
faithfulness head of the three-head design (#4) and ChatGPT's point #6, and argues against more
retrieval work.

## Order

1. ~~v9 markers~~: done, no macro-F1 gain (`results/markers_v1.md`).
2. The three-head verifier (#4, with #5, #6, #8), same folds, with the plain input (v9 did not pass).
3. Label-noise audit of confident disagreements (#9) before any claim.
