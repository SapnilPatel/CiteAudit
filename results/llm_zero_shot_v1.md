# Zero-shot open LLM: Qwen2.5-7B-Instruct, prompt v1, dev split

`scripts/llm_classify.py` via `notebooks/llm_classify_colab.ipynb` on a Colab T4 (4-bit NF4).
The model reads the citing sentence and our reranker's top-20 sentences and answers A/B/C; the
three answer-letter probabilities are the prediction. It is well-behaved as a classifier: on
average 1.000 of its next-token probability falls on the three letters. Dev only, 316 citations.

| Dev, our evidence | Accuracy | Macro-F1 | ACCURATE F1 | NOT ACCURATE F1 | NEI F1 |
|---|---|---|---|---|---|
| Released MultiVerS | 0.674 | 0.674 | — | — | — |
| MultiVerS + BiomedBERT ensemble (current best) | — | 0.698 | — | — | — |
| **Qwen2.5-7B zero-shot** | 0.636 | **0.394** | 0.782 | 0.325 | 0.074 |

**It says ACCURATE for 254 of 316 citations and IRRELEVANT for 3.** But the bias is not the
main problem; its probabilities also rank citations worse than MultiVerS's in every class:

| One-vs-rest AUC, dev | Qwen2.5-7B | MultiVerS |
|---|---|---|
| ACCURATE | 0.683 | 0.770 |
| NOT ACCURATE | 0.591 | 0.722 |
| NOT ENOUGH INFO | 0.834 | 0.979 |
| Misrepresented vs ACCURATE (OVERSIMPLIFY, CONTRADICT, NOT_SUBSTANTIATE, MISQUOTE; n = 61) | 0.607 | 0.679 |

- Even with per-class offsets fitted on dev itself, an optimistic upper bound, macro-F1 only
  reaches 0.487.
- Added to the MultiVerS + BiomedBERT ensemble, with all three weights fitted on dev, the best
  LLM weight is 0.0: it contributes nothing.
- It confuses IRRELEVANT with NOT ACCURATE (mean P(NOT ACCURATE) 0.65 on IRRELEVANT citations),
  probably because "cited for a point it does not make" (an ETIQUETTE-style NOT ACCURATE) and
  "has nothing to do with the statement" overlap in the prompt.

**Conclusion: zero-shot prompting a 7B model is not the way to the weak classes.** It is a weaker
discriminator than the task-trained models, not just a miscalibrated one, so prompt wording or
calibration alone is unlikely to close a 0.3 macro-F1 gap. Test was not run.
