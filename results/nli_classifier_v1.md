# NLI classifier v1: an NLI-pretrained model that keeps its NLI head

`notebooks/nli_classifier_colab.ipynb` on a Colab T4, `scripts/train_nli.py`.
`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, fine-tuned with its entailment /
contradiction / neutral head mapped to ACCURATE / NOT ACCURATE / NOT ENOUGH INFO, so training
starts from a working faithfulness judge and does not have to learn one from scratch. The premise
is our reranker's top-20 sentences in rank order, and the hypothesis is the citing sentence with its
markers removed. 3 epochs, lr 1e-5, effective batch 8, sqrt class weights, fp16 autocast. The
epoch is chosen on dev, and epoch 0 (zero-shot) is a candidate. Predictions and logs are in
`classification/v5/`.

Why: `diagnosis_v1.md` found that 76% of the remaining errors are faithfulness judgements on
evidence that was retrieved, and that a zero-shot NLI model already ranks those citations as well
as MultiVerS does (AUC 0.689 against 0.683).

## Phase 1: cross-validation over train (2141 out-of-fold predictions, cut by article)

| Model | Accuracy | Macro-F1 | ACCURATE F1 | NOT ACCURATE F1 | NEI F1 |
|---|---|---|---|---|---|
| BiomedBERT, no context (`classifier_v4.md`) | 0.643 | 0.428 | 0.758 | 0.454 | 0.072 |
| **NLI DeBERTa-v3-large** | **0.676** | **0.570** | **0.783** | **0.513** | **0.415** |

The +0.142 macro-F1 is the largest single gain in this project, and it holds on all three
classes. NEI goes from almost never predicted (0.07) to 0.42: the NLI "neutral" class carries
over directly to "cited paper is irrelevant". Zero-shot (epoch 0) scores 0.361 / 0.573 on dev, and
fine-tuning is what makes the model work.

## Phase 2: all of train, 3 seeds, test (1037 citations)

| Test | Accuracy | Macro-F1 | ACCURATE F1 | NOT ACCURATE F1 | NEI F1 |
|---|---|---|---|---|---|
| Paper system, reproduced (MultiVerS, paper retrieval) | 0.584 | 0.524 | 0.692 | 0.426 | 0.455 |
| Previous best: MultiVerS + E9 ensemble (`classifier_v3.md`) | 0.642 | 0.538 | 0.764 | 0.426 | 0.424 |
| BiomedBERT E3 (`classifier_v3.md`) | 0.662 ± 0.018 | 0.489 ± 0.013 | 0.789 | 0.438 | 0.239 |
| NLI, single seed (mean ± sd of 3) | 0.668 ± 0.015 | 0.556 ± 0.015 | — | — | — |
| **NLI, 3 seeds averaged** | **0.680** | **0.575** | **0.796** | **0.439** | **0.490** |
| NLI + MultiVerS ensemble (dev-chosen w = 0.6) | 0.658 | 0.552 | 0.782 | 0.430 | 0.444 |

Against the reproduced paper system, the averaged NLI model is +0.096 accuracy (right on 134
citations the paper system misses, wrong on 76 it gets; McNemar exact p = 8e-5) and +0.050
macro-F1 (paired bootstrap 95% CI [-0.009, +0.111]). Against the previous best ensemble it is
+0.038 accuracy (92 against 69, p = 0.08) and +0.037 macro-F1 (CI [-0.023, +0.094]). Each seed
alone beats both on both metrics.

Adding MultiVerS hurts. Dev puts weight 0.6 on MultiVerS because MultiVerS scores 0.674 macro-F1
on dev but 0.524 on test: dev (316 citations, 24 NEI) is too small to choose weights, as found
before. The NLI model alone is now the system to beat.

## What it does not fix

NOT ACCURATE F1 on test is 0.439, almost unchanged from the paper's 0.426, even though CV rose
from 0.454 to 0.513. The gain on test comes from ACCURATE and NEI. Detecting misrepresentation is
still the open problem, and it is what step 2 of the plan targets: synthetic contrastive
misrepresentations (overstated, causal, wrong population or number) generated from the ~1300
accurate training citations, to give this model far more than the ~330 real NOT ACCURATE examples
it has now.
