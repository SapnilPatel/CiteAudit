# External reviews of the handoff (2026-11-15), checked against the data

Two outside reviews of [`HANDOFF.md`](HANDOFF.md), one written with ChatGPT and one with Gemini,
proposed next steps. This records every proposal, what the data says about it, and the order we
are running them in. The numbers here were measured in this repository.

## What both reviews proposed

| # | Proposal | Who | Verdict after checking |
|---|---|---|---|
| 1 | Keep citation markers in the hypothesis (`[CITED]`, `[CITED, with others]`, `[other]`) and fine-tune end to end | both, first priority | **Running now** (`notebooks/nli_markers_colab.ipynb`, v9). See the caveat below |
| 2 | Oracle: hypothesis = gold `citation_context` span, then a span extractor if it helps | both | **Little room: see "Citation scope" below.** The claim already equals the gold span for 91% of citations |
| 3 | Oracle: gold evidence with the current classifier, same budget, to see if retrieval still limits | both (Gemini: do it first) | **Done**, see "Oracle evidence" below |
| 4 | Decompose into relevance / attribution / faithfulness heads on a shared encoder, combined by an out-of-fold stacker | both (ChatGPT: most important architecture) | Next, after v9. Not the failed 9-way head: three binary heads with masked losses |
| 5 | Attribution cue: does the matching evidence sentence itself carry a citation ("[3]", "(12)")? | Gemini | Cheap feature for #4; to measure |
| 6 | NLI mismatch: overgeneralised claims are "neutral" to NLI but NOT_ACCURATE here | ChatGPT | Valid. Fine-tuning relearns the head, but it argues for #4's separate faithfulness head |
| 7 | Report single vs grouped citations, ETIQUETTE vs substantive-error recall, not just totals | ChatGPT | **Built into the v9 notebook** |
| 8 | Do not let INDIRECT (counted ACCURATE) become NOT_ACCURATE in an attribution head | ChatGPT | Adopted for #4 |
| 9 | Audit confident disagreements for label noise before claiming a win | ChatGPT | Planned after v9 |
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

Pending: `scripts/oracle_evidence.py` (local, about 45 minutes on Apple Silicon). Partial run: dev with retrieved evidence gives macro-F1 0.616, matching the release run.

## Order

1. v9 markers (running): one CV, gated, then 3 seeds on test.
2. The three-head verifier (#4, with #5, #6, #8), same folds, with the v9 input if v9 passes.
3. Label-noise audit of confident disagreements (#9) before any claim.
