---
license: mit
language: en
library_name: transformers
pipeline_tag: text-classification
base_model: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
tags:
- citation-integrity
- biomedical
- natural-language-inference
- scientific-claim-verification
---

# CiteAudit citation classifier (DeBERTa-v3-large NLI)

This model judges whether a citation in a biomedical paper represents the cited paper accurately.
It is the classifier half of [CiteAudit](https://github.com/SapnilPatel/CiteAudit). The other half
is a reranker that finds the cited paper's most relevant sentences.

- **Input:** premise = evidence sentences from the cited paper, best first; hypothesis = the citing
  sentence with its citation markers removed.
- **Output:** `ACCURATE`, `NOT_ACCURATE` (the citation misrepresents the paper: contradicts it,
  is not supported by it, oversimplifies or misquotes it, or cites it for a point it only reports
  from others) or `NOT ENOUGH INFO` (the cited paper is irrelevant to the statement).

It is `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, fine-tuned on the 2141 training
citations of the Citation-Integrity dataset (Sarol et al., 2024). Its original NLI head is kept:
entailment, contradiction and neutral are relabelled ACCURATE, NOT_ACCURATE and NOT ENOUGH INFO.
The training settings were 3 epochs, lr 1e-5, effective batch 8 and sqrt class weights, with the
epoch chosen on dev.

## Results (Citation-Integrity test set, 1037 citations, evidence from the CiteAudit reranker)

| System | Accuracy | Macro-F1 |
|---|---|---|
| Sarol et al. 2024 (MultiVerS, BM25 + MonoT5 evidence), our reproduction | 0.584 | 0.524 |
| This model, mean of 3 training seeds | 0.668 ± 0.015 | 0.556 ± 0.015 |

The test score of this exact checkpoint is in `test_metrics.json`.

## Use

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

name = "REPO_ID"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.float32).eval()
evidence = " ".join(top_sentences)  # the cited paper's most relevant sentences, best first
enc = tok(evidence, citing_sentence, truncation="longest_first", max_length=512, return_tensors="pt")
probs = model(**enc).logits.softmax(-1)[0]
print({model.config.id2label[i]: round(p.item(), 3) for i, p in enumerate(probs)})
```

Or use the whole pipeline: `pip install git+https://github.com/SapnilPatel/CiteAudit` and then
`citeaudit check`.

## Limitations

The model was trained on 2141 citations from biomedical papers, so it may not transfer to other
fields. About a third of its decisions are wrong. Use it to flag citations for a human to check,
not as a verdict. It reads at most 512 tokens of evidence (about 12 sentences). It sees only the
citing sentence, not the paragraph around it.
