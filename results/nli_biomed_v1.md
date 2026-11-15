# Biomedical intermediate training and 1024-token evidence: no reliable gain

`notebooks/nli_biomed_colab.ipynb` on a Colab T4. Two changes to the NLI classifier
(`nli_classifier_v1.md`), judged by the same 3-fold CV over train:

1. **Intermediate training.** One epoch on 15,589 human-labelled biomedical claim-verification
   pairs (HealthVer + SciFact, `build_pretrain_data.py`, `pretrain_nli.py`) before fine-tuning on
   citations.
2. **1024 tokens instead of 512.** The classifier reads all 20 evidence sentences for 80% of
   citations, rather than about 12.

Logs and configs are in `classification/v8/`.

## Cross-validation (2141 out-of-fold predictions)

| Model | Accuracy | Macro-F1 | ACCURATE F1 | NOT ACCURATE F1 | NEI F1 |
|---|---|---|---|---|---|
| NLI, 512 (`nli_classifier_v1.md`) | 0.676 | **0.5702** | 0.783 | 0.513 | **0.415** |
| + biomedical, 512 | 0.674 | 0.5621 | 0.780 | 0.501 | 0.405 |
| + biomedical, 1024 | **0.689** | 0.5701 | **0.791** | **0.522** | 0.396 |

Intermediate training alone does not help. After it, zero-shot dev macro-F1 rose from 0.361 to
0.390, but the gain is gone once the model is fine-tuned on citations. Adding 1024 tokens recovers
the loss and adds 1.3 points of CV accuracy, but macro-F1 ties the baseline (0.5701 against
0.5702).

**Protocol note.** Phase 2 was meant to run only if a variant beat the baseline's CV macro-F1. It
ran because the notebook compared against the rounded 0.570. In fact it is a tie. The test numbers
below are reported, but they did not decide anything.

## Test (606 citations)

| Model | Accuracy | Macro-F1 | ACCURATE F1 | NOT ACCURATE F1 | NEI F1 |
|---|---|---|---|---|---|
| Paper system, reproduced | 0.584 | 0.524 | 0.692 | 0.426 | 0.455 |
| NLI 512, single seed (mean ± sd of 3) | 0.668 ± 0.015 | 0.556 ± 0.015 | | | |
| NLI 512, 3 seeds averaged | **0.680** | 0.575 | **0.796** | **0.439** | 0.490 |
| Biomedical + 1024, single seed (mean ± sd of 3) | 0.671 ± 0.019 | **0.583 ± 0.018** | | | |
| Biomedical + 1024, 3 seeds averaged | 0.673 | 0.580 | 0.788 | 0.436 | **0.515** |

Averaged over seeds, the two models are the same: 48 citations are right only under the new one
and 52 only under the old (McNemar p = 0.76), and the macro-F1 difference is +0.005 (paired
bootstrap 95% CI [−0.039, +0.052]). Single seeds of the new model score higher macro-F1 on
average (+0.027), but that is within seed variation and is not what CV shows. Against the paper's
system, the new model is +0.089 accuracy (McNemar p = 0.00015) and +0.056 macro-F1 (CI [−0.002,
+0.113]).

CV-tuned decision biases (`tune_bias.py`) chose +0.3 / +1.2 here and lowered test macro-F1 from
0.580 to 0.570. On the 512 model they raised it from 0.575 to 0.580. The procedure is not reliable
across models, so the released classifier does not use biases.

## Conclusion

Neither more relevant pretraining data nor reading all the evidence moves this classifier beyond
seed-to-seed noise. Along with the synthetic-data results (`synthetic_v1.md`, `synthetic_v2.md`),
this suggests the NLI model is at the limit of what 2141 training citations support, around 0.67
to 0.69 accuracy and 0.57 to 0.58 macro-F1. The release keeps the simpler and 2.5x faster
512-token model without intermediate training.
