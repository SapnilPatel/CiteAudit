# Fine-tuned LLM: Qwen2.5-3B-Instruct, QLoRA, 1 epoch

`notebooks/finetune_llm_colab.ipynb` on a Colab T4: `scripts/finetune_llm.py`, 4-bit weights,
LoRA r = 16 on every linear layer, lr 1e-4, 8 citations per step, sqrt class weights, 1 epoch
(268 steps) on our reranker's CV evidence (`ours_20_train_cv`), prompt v1. Log in
`classification/llm_ft/qwen3b/pred/config.json`.

| | Dev macro-F1 | Test accuracy | Test macro-F1 | ACCURATE F1 | NOT ACCURATE F1 | NEI F1 |
|---|---|---|---|---|---|---|
| Reproduced paper system | — | 0.584 | 0.524 | 0.692 | 0.426 | 0.455 |
| Qwen2.5-7B zero-shot (`llm_zero_shot_v1.md`) | 0.394 | — | — | — | — | — |
| **Qwen2.5-3B fine-tuned** | 0.384 | **0.663** | 0.391 | 0.798 | 0.335 | 0.039 |
| Current best ensemble (`classifier_v3.md`) | 0.726 | 0.642 | 0.538 | 0.764 | 0.426 | 0.424 |

- **Highest accuracy of any single model so far (0.663), but by predicting ACCURATE**: 506 of 606
  test citations, and NOT ENOUGH INFO once. Macro-F1 is far below the paper.
- **Fine-tuning did teach it something**: dev AUCs rose from the 7B zero-shot model's
  0.683 / 0.591 / 0.834 (ACCURATE / NOT ACCURATE / NEI) to 0.723 / 0.649 / 0.839, with a smaller
  model. They are still below BiomedBERT's (0.746 / 0.666 / 0.846) and MultiVerS's
  (0.770 / 0.722 / 0.979).
- **It adds nothing to the ensemble**: with all weights fitted on dev, its weight is 0.0.
- Training loss was still falling when the epoch ended (about 1.0 at step 50, 0.55 at step 250),
  so this run is undertrained; more epochs or the 7B model would likely do better, at 2 to 4
  times the Colab time.
