# Synthetic misrepresentations v1: they teach the right thing, and also a shortcut

`notebooks/nli_synthetic_colab.ipynb` on a Colab T4. `scripts/generate_misrep.py` had
Qwen2.5-7B-Instruct (4-bit) rewrite the 1107 accurate training citations whose gold evidence our
retrieval keeps. Each rewrite misrepresents the same evidence in one way: unsupported (double
share), overstate, causal, generalise, detail, contradict. 1016 rewrites passed the output filters
(`data_train/synthetic/misrep_raw.jsonl`). The NLI classifier (`nli_classifier_v1.md`) was trained
on real + synthetic data (`train_nli.py --synthetic`, about 680 synthetic rows per fold, taken only
from training-fold citations). The CV protocol is the same as v1. Logs are in
`classification/v6/`.

## Cross-validation (2141 out-of-fold predictions)

| Model | Accuracy | Macro-F1 | ACCURATE F1 | NOT ACCURATE F1 | NEI F1 |
|---|---|---|---|---|---|
| **NLI, real data only** | **0.676** | **0.570** | **0.783** | 0.513 | **0.415** |
| NLI, real + raw synthetic | 0.629 | 0.542 | 0.724 | 0.518 | 0.386 |
| same, prior-corrected (p(NA) × sqrt(real/train count), no tuning) | 0.645 | 0.544 | 0.754 | 0.486 | 0.393 |

Phase 2 was skipped, as the rule required. The aggregate hides a split result. Accuracy per fine label:

| Fine label | n | Real only | + synthetic |
|---|---|---|---|
| ACCURATE | 1308 | 0.767 | 0.660 |
| INDIRECT | 35 | 0.743 | 0.457 |
| INDIRECT_NOT_REVIEW | 25 | 0.840 | 0.640 |
| NOT_SUBSTANTIATE | 189 | 0.434 | **0.571** |
| CONTRADICT | 58 | 0.466 | **0.638** |
| OVERSIMPLIFY | 56 | 0.482 | **0.625** |
| MISQUOTE | 23 | 0.304 | **0.696** |
| ETIQUETTE | 304 | 0.651 | 0.674 |
| IRRELEVANT | 143 | 0.392 | 0.350 |

The synthetic data did what it was built for: every real misrepresentation label gains 13 to 39
points. But accurate citations lose 11, and they are 61% of the data. Correcting for the shifted
class balance does not recover them, so this is not only a prior effect.

## Why: two artefacts in the rewrites

1. **Style shortcut.** The LLM tidies what it rewrites. 91% of rewrites end in a full stop against
   59% of the real citing sentences, which carry truncations and leftover reference fragments
   (",7", "(Wang et al.,."). "Tidy sentence -> NOT ACCURATE" penalises tidy accurate citations,
   which fits a loss concentrated on ACCURATE and INDIRECT.
2. **Non-edits labelled NOT ACCURATE.** Some rewrites change only a reference number ("-43" to
   "-45"), a year, a word fragment ("overn" to "overnigh"), or cut the sentence short at the
   generation limit.

## Fix for v2

`scripts/clean_misrep.py` rebuilds each rewrite from its original: it takes only the edits that
change words, keeps the original's punctuation, case and both ends, and drops non-edits. 975 of
1016 are kept (33 word fragments, 4 number-only edits outside the evidence, 4 truncations). The
full-stop rate is now 0.59, the same as the originals (`data_train/synthetic/misrep_clean.jsonl`).
`generate_misrep.py` now also rejects outputs that never reached an end token. v2 retrains on the
cleaned data, at the full amount and at half (to limit the class-balance shift), and chooses
between them by CV.
