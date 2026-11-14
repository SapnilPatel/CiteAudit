---
license: mit
language: en
library_name: transformers
pipeline_tag: text-classification
base_model: microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
tags:
- citation-integrity
- biomedical
- cross-encoder
- reranking
---

# CiteAudit evidence reranker (PubMedBERT cross-encoder)

This model finds the sentences of a cited paper that bear on a citing sentence. It is the retrieval
half of [CiteAudit](https://github.com/SapnilPatel/CiteAudit). Its top 20 sentences go to the
[CiteAudit classifier](https://huggingface.co/CLASSIFIER_ID).

- **Input:** text_a = the citing sentence with its citation markers removed; text_b = one candidate
  sentence of the cited paper with up to 200 characters of each neighbour, at 192 tokens.
- **Output:** a single relevance logit. Higher means more relevant.
- **Use:** it reranks BM25's top 60 sentences (k1 0.9, b 0.4).
- **Training:** fine-tuned from BiomedBERT on the Citation-Integrity training split (Sarol et al.,
  2024) with a listwise loss over each gold sentence and BM25-sampled negatives.

## Results (Citation-Integrity test set, 20 articles, 606 citations with evidence)

| Retrieval | Recall@10 | Recall@20 | MRR |
|---|---|---|---|
| BM25 + MonoT5, Sarol et al. 2024 (reproduced) | 0.422 | 0.529 | 0.323 |
| BM25 + this reranker | 0.510 | 0.627 | 0.379 |

## Use

The [citeaudit package](https://github.com/SapnilPatel/CiteAudit) builds the windows exactly as in
training (`citeaudit.retrieval.window`). Note one detail of the training code: a previous sentence
of 100 to 200 characters is clipped to its last `200 - len` characters. Build windows another way
and scores shift.
