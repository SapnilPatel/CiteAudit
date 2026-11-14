# Synthetic misrepresentations v2: cleaned, still no gain

`notebooks/nli_synthetic_v2_colab.ipynb` on a Colab T4. Same protocol as `synthetic_v1.md`, but
trained on the 975 cleaned rewrites (`data_train/synthetic/misrep_clean.jsonl`, from
`clean_misrep.py`), at the full amount and at half. Logs are in `classification/v7/`.

## Cross-validation (2141 out-of-fold predictions)

| Model | Accuracy | Macro-F1 | Per fold | ACCURATE F1 | NOT ACCURATE F1 | NEI F1 |
|---|---|---|---|---|---|---|
| **NLI, real data only** | **0.676** | **0.570** | 0.562, 0.543, 0.588 | **0.783** | 0.513 | **0.415** |
| + raw synthetic (v1) | 0.629 | 0.542 | | 0.724 | 0.518 | 0.386 |
| + cleaned synthetic, full | 0.660 | 0.557 | 0.565, 0.511, 0.587 | 0.766 | 0.503 | 0.403 |
| + cleaned synthetic, half | 0.660 | 0.550 | 0.543, 0.531, 0.572 | 0.766 | **0.523** | 0.361 |

Neither variant beats real data only, so Phase 2 was skipped. Full against real only: −0.013
macro-F1, paired bootstrap 95% CI [−0.040, +0.012]. That is within noise, but no gain.

Accuracy per fine label:

| Fine label | n | Real only | Raw synthetic | Cleaned, full | Cleaned, half |
|---|---|---|---|---|---|
| ACCURATE | 1308 | 0.767 | 0.660 | 0.751 | 0.725 |
| INDIRECT | 35 | 0.743 | 0.457 | 0.571 | 0.657 |
| INDIRECT_NOT_REVIEW | 25 | 0.840 | 0.640 | 0.760 | 0.720 |
| NOT_SUBSTANTIATE | 189 | 0.434 | 0.571 | 0.471 | 0.508 |
| CONTRADICT | 58 | 0.466 | 0.638 | 0.603 | 0.621 |
| OVERSIMPLIFY | 56 | 0.482 | 0.625 | 0.482 | 0.518 |
| MISQUOTE | 23 | 0.304 | 0.696 | 0.522 | 0.652 |
| ETIQUETTE | 304 | 0.651 | 0.674 | 0.579 | 0.655 |
| IRRELEVANT | 143 | 0.392 | 0.350 | 0.378 | 0.336 |

Cleaning removed most of the damage to accurate citations (0.660 to 0.751), which confirms the
style shortcut. It also gave back much of the gain on misrepresentations: the gain on those labels
was partly the same shortcut. What remains is a trade-off of one class against another, not an
improvement.

Averaging the real-only and cleaned-full models' out-of-fold probabilities gives 0.684 / 0.582,
above either alone. But a 2-model average of the same model with different seeds is expected to
gain about as much (3 seeds gained +0.02 macro-F1 on test in `nli_classifier_v1.md`), so this is
not evidence for the synthetic data.

## Conclusion

Synthetic rewrites of accurate citations do not improve the NLI classifier: raw, they teach a
style shortcut, and cleaned, they shift errors from one class to another. LLM rewrites are an easy
contrast (one edit, explicit), while real misrepresentations are mostly NOT_SUBSTANTIATE: on topic
but not stated in the paper, which a minimal edit does not imitate well. The model to beat is still
NLI on real data (`nli_classifier_v1.md`).
