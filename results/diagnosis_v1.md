# Diagnosis: where the remaining error is, and what kind of problem it is

Six model changes (retrained BiomedBERT, nine labels, fine-tuned MultiVerS, zero-shot and
fine-tuned LLMs, ensembles) all land at test macro-F1 0.52 to 0.54. This note steps back from the
models to the task.

## 1. Evidence precision is not the bottleneck either

The paper's best classification setting (0.73 / 0.75) feeds only the gold evidence sentences. The
released MultiVerS, frozen, on dev:

| Evidence fed | Accuracy, all 316 | Macro-F1 | Accuracy, 255 with gold evidence | ACCURATE F1 | NOT ACCURATE F1 |
|---|---|---|---|---|---|
| Retrieved top-20, paper pipeline | 0.677 | 0.684 | 0.655 | 0.765 | 0.420 |
| Retrieved top-20, our reranker | 0.674 | 0.674 | 0.663 | 0.771 | 0.411 |
| Gold + 18 retrieved (oracle-20) | 0.655 | 0.656 | 0.639 | 0.757 | 0.350 |
| **Gold only** (~2 sentences; one retrieved sentence when there is no gold) | 0.560 | 0.464 | 0.588 | 0.759 | 0.296 |

Handed exactly the annotated evidence, the model still scores 0.30 F1 on NOT ACCURATE. Finding
the evidence (recall) and isolating it (precision) are both solved well enough; **judging it is
not**. (Gold-only is also shorter than anything the checkpoint was trained on, which likely
explains part of its drop.)

## 2. The error budget of the current best system

MultiVerS + 9-label BiomedBERT ensemble (`classifier_v3.md`), test, 217 errors:

| Group | n | Accuracy | Share of all errors | Main mistake |
|---|---|---|---|---|
| A. Has gold evidence, truly ACCURATE | 386 | 0.77 | 41% | 76 called NOT ACCURATE |
| B. Has gold evidence, misrepresents it (CONTRADICT, OVERSIMPLIFY, NOT_SUBSTANTIATE, MISQUOTE) | 100 | **0.24** | 35% | 66 called ACCURATE |
| C. No gold evidence, ETIQUETTE | 70 | 0.69 | 10% | 18 called ACCURATE |
| D. No gold evidence, IRRELEVANT | 49 | 0.43 | 13% | 19 called NOT ACCURATE |

**76% of the errors are one decision: given evidence that exists, does the citation represent it
faithfully?** That is a faithfulness, or entailment, judgement, the problem natural-language
inference (NLI) and scientific claim verification study. Every model here learns it from about
330 training citations that misrepresent their evidence.

## 3. Transfer from NLI is a plausible lever

`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, an NLI model trained on about a million
general-domain examples and **never trained on this data**, scoring dev citations that have gold
evidence (misrepresented 64, faithful 191), AUC for telling them apart:

| Model | AUC |
|---|---|
| NLI model, 1 − P(entailment), gold evidence as premise | **0.689** |
| NLI model, P(contradiction) | 0.626 |
| MultiVerS (fact-verification pretraining), 1 − P(ACCURATE), our top-20 | 0.683 |
| BiomedBERT E9 (no NLI pretraining), 1 − P(ACCURATE), our top-20 | 0.610 |

The comparison is not like for like (the NLI model is given the gold sentences), so it shows
direction, not size: with no task training at all, an NLI prior matches the task-trained
MultiVerS and clearly beats BiomedBERT, the only model here without such a prior.
