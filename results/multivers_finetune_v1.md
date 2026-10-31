# Fine-tuning MultiVerS on our reranker's evidence, first run

`notebooks/finetune_multivers_colab.ipynb` on a Colab T4: `scripts/finetune_multivers.py` continuing
the released checkpoint on top-20 evidence from our cross-validated reranker
(`ours_20_train_cv`, 2140 citations; one over 4096 tokens skipped), 2 epochs, lr 1e-5,
8 citations per step, fp16, seed 0. About 37 minutes an epoch. Epoch chosen on dev macro-F1, with
the unchanged checkpoint (epoch 0) as a candidate. Log in `classification/mv_ft/C_mvft_s0/config.json`.

| Epoch | Train loss | Dev macro-F1 | Dev accuracy |
|---|---|---|---|
| 0, released checkpoint | — | **0.677** | 0.677 |
| 1 | 0.652 | 0.638 | **0.706** |
| 2 | 0.427 | 0.667 | 0.703 |

**Fine-tuning did not raise dev macro-F1, so the released weights were kept** and the test
predictions equal the released checkpoint's on every one of the 606 citations: macro-F1 0.505 on
our evidence and 0.524 on the paper's. That they match exactly also confirms that the Colab fp16
path and the local fp32 path predict identically.

Same pattern as the BiomedBERT classifier (`classifier_v2.md`): training on our evidence raises
accuracy (dev 0.677 to 0.70) and lowers macro-F1, because the gain is in the majority ACCURATE
class while NOT ENOUGH INFO gets worse. The dev gap at epoch 2 is 0.010 on a split with 24 NEI
citations, so a different seed or schedule might land either side of the bar; this run gives no
reason to expect a large macro-F1 gain from plain fine-tuning.
