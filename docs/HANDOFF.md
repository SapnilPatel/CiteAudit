# CiteAudit: project handoff

*State of the project on 14 November 2026: the problem, the goal, everything tried and what it
gave, and where the next gains are most likely to be. Written so that a researcher or a model with
no prior context can continue the work. Every number here was measured in this repository; the
write-up each one comes from is linked.*

---

## 1. The problem

When a biomedical paper cites another paper, it claims that the cited paper supports a specific
statement. Often it does not: the statement contradicts the source, stretches it (drops a
qualifier, generalises from mice to humans, turns an association into a cause), gets a detail
wrong, says something the source never says, or cites a paper for a finding that paper only
repeats from others. Checking this by hand means reading the cited paper, so in practice almost
nobody does.

**Task.** Given a **citing sentence** and the **full text of the cited paper**, label the citation:

| Label | Meaning | Fine-grained labels it collapses from |
|---|---|---|
| ACCURATE | The cited paper supports the statement | ACCURATE, INDIRECT, INDIRECT_NOT_REVIEW |
| NOT_ACCURATE | The statement misrepresents the cited paper | CONTRADICT, NOT_SUBSTANTIATE, OVERSIMPLIFY, MISQUOTE, ETIQUETTE |
| NOT ENOUGH INFO | The cited paper is irrelevant to the statement | IRRELEVANT |

**Benchmark.** Citation-Integrity (Sarol et al., *Bioinformatics* 2024,
doi:10.1093/bioinformatics/btae420). Splits by cited article: **train 2,141 citations
(70 articles), dev 316, test 606 (20 articles)**. A cited article has about 239 sentences, of which
about 2.5 are gold evidence. Train fine labels: ACCURATE 1308, ETIQUETTE 304, NOT_SUBSTANTIATE 189,
IRRELEVANT 143, CONTRADICT 58, OVERSIMPLIFY 56, INDIRECT 35, INDIRECT_NOT_REVIEW 25, MISQUOTE 23.
Test gold: 386 ACCURATE, 170 NOT_ACCURATE, 50 NOT ENOUGH INFO.

**Metrics.** Macro-F1 over the three labels (the paper's headline) and accuracy. Significance:
McNemar exact test on accuracy, paired bootstrap CI on macro-F1.

**The paper's system.** BM25 keeps 60 sentences, MonoT5 (zero-shot) reranks to the top 20, and
MultiVerS (Longformer-large, fine-tuned) classifies. It scored 0.59 accuracy / 0.52 macro-F1 on
test. With *perfect* evidence and the gold citation context, the paper reports about 0.73 to 0.75
accuracy, which is the best ceiling estimate available.

## 2. The end goal

1. Reproduce the paper (done).
2. **Improve on it substantially**: the person leading the project wants a clear accuracy gain,
   not a marginal one.
3. Release it as **open source that any researcher can use**: an installable tool
   (`citeaudit check`) with the models on the Hugging Face Hub. That project lead has ruled out
   the tool being, or looking like, a wrapper around a commercial LLM ("anyone could just ask
   ChatGPT"). The contribution must be a small open model that does the job better than the
   published one.

## 3. Where things stand

| | Accuracy | Macro-F1 | NOT ACC F1 | NEI F1 |
|---|---|---|---|---|
| Paper, published | 0.59 | 0.52 | 0.43 | 0.42 |
| Paper, reproduced here | 0.584 | 0.524 | 0.426 | 0.455 |
| **CiteAudit now: NLI classifier, 3 seeds averaged** | **0.680** | **0.575** | 0.439 | 0.490 |
| CiteAudit now: one seed (mean ± sd over 3) | 0.668 ± 0.015 | 0.556 ± 0.015 | | |

(Test split, 606 citations.) Against the reproduced paper system: **+9.6 accuracy points,
McNemar p = 0.00008**. Macro-F1 gains +0.050, but the 95% CI (−0.009 to +0.111) still includes
zero.

**The plateau.** Six different attempts to go further all land at **0.67 to 0.69 accuracy and
0.56 to 0.58 macro-F1**, within seed-to-seed noise (± about 0.02 macro-F1). **NOT ACCURATE F1 has
not moved from the paper's 0.43**: every gain so far comes from recognising accurate and
irrelevant citations. Detecting misrepresentation is the unsolved core.

**Released or ready:**

- **Package.** `citeaudit` (in `citeaudit/`, installable from GitHub) runs BM25 → reranker → NLI
  classifier on a PDF, a text file or a PubMed Central id.
- **Reranker on the Hub.** `Sapnil/citeaudit-reranker` is published. It reproduces the research
  retrieval exactly: 60 of 60 top-20 lists identical, and test Recall@20 0.627.
- **Classifier pending.** `Sapnil/citeaudit-nli` is not published yet.
  `notebooks/release_model_colab.ipynb` trains 3 seeds, averages their weights (a model soup), and
  publishes the soup if it is at least as good on dev as the seeds' mean.
- **CI.** Offline tests pass, run by GitHub Actions (`tests/test_package.py`).

## 4. The pipeline as it stands

1. **Filter.** BM25 (k1 0.9, b 0.4) over all sentences of the cited paper keeps 60.
2. **Rerank.** A PubMedBERT cross-encoder (`scripts/train_reranker.py`) keeps 20. Query = the
   citing sentence with markers removed. Passage = the candidate sentence with up to 200
   characters of each neighbour, at 192 tokens.
3. **Classify.** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, fine-tuned
   (`scripts/train_nli.py`):
    - Premise = the 20 sentences in rank order. Hypothesis = the citing sentence with **all**
      citation markers removed.
    - The NLI head is kept: entailment / contradiction / neutral become ACCURATE / NOT ACCURATE /
      NOT ENOUGH INFO.
    - Settings: 3 epochs, lr 1e-5, effective batch 8, sqrt class weights, 512 tokens, epoch chosen
      on dev.

| Retrieval (test) | Recall@10 | Recall@20 | MRR |
|---|---|---|---|
| BM25 + MonoT5 (paper, reproduced) | 0.422 | 0.529 | 0.323 |
| BM25 + PubMedBERT reranker (ours, released retrain) | 0.510 | 0.627 | 0.379 |

## 5. Everything tried, and what it gave

**Protocol.** Choices are made by 3-fold cross-validation over train, split by article (2,141
out-of-fold predictions; `fold_articles`, seed 12345). Dev (316 citations, 24 NEI) proved too small
to choose with. Test is only for reporting, but it has now been reported many times, so treat new
small test gains with suspicion.

| # | Idea | Result | Verdict | Write-up |
|---|---|---|---|---|
| 1 | Better evidence for the paper's frozen MultiVerS (ours, or perfect gold) | Even perfect evidence does not help it (dev 0.655 vs 0.677) | MultiVerS is fitted to its own retrieval | `repro/README.md` |
| 2 | Fine-tune MultiVerS on our evidence | No gain | ✗ | `results/multivers_finetune_v1.md` |
| 3 | BiomedBERT classifier, 3 and 9 labels | Test 0.662 / 0.489 (3-label); 9-label 0.487 | Good accuracy, poor minority classes | `results/classifier_v3.md` |
| 4 | Ensemble MultiVerS + BiomedBERT | Test 0.642 / 0.538 | Best before NLI | `results/ensemble_v1.md` |
| 5 | Open LLMs: Qwen2.5-7B zero-shot, Qwen2.5-3B QLoRA fine-tuned | Macro-F1 0.39 (dev / test) | ✗ small LLMs are weak here | `results/llm_zero_shot_v1.md`, `llm_finetune_v1.md` |
| 6 | Citing context (window or paragraph) for BiomedBERT | CV 0.428 → 0.378 / 0.358 | ✗ hurts, as it was fed (concatenated) | `results/classifier_v4.md` |
| 7 | **NLI-pretrained DeBERTa-v3-large, NLI head kept** | **CV 0.428 → 0.570; test 0.680 / 0.575** | ✓ the one big win | `results/nli_classifier_v1.md` |
| 8 | NLI + MultiVerS ensemble | Test 0.552 | ✗ | same |
| 9 | ~1000 LLM-written misrepresentations of accurate citations (raw) | CV 0.542; misrepresentation labels +13 to 39 pts, ACCURATE −11 | ✗ the LLM's tidy style became a shortcut | `results/synthetic_v1.md` |
| 10 | Same, artefacts cleaned (`clean_misrep.py`) | CV 0.557 (full), 0.550 (half) | ✗ a trade between classes | `results/synthetic_v2.md` |
| 11 | Intermediate training on 15.6k human-labelled biomedical claims (HealthVer, SciFact) | CV 0.562 | ✗ | `results/nli_biomed_v1.md` |
| 12 | Same + 1024 tokens (reads all 20 sentences for 80% of citations) | CV 0.570 (tie), accuracy 0.689; test 0.673 / 0.580 | ≈ no reliable gain | same |
| 13 | CV-tuned decision biases per class | +0.013 accuracy on model 7, but lowered macro-F1 on model 12 | ✗ not reliable | `results/nli_classifier_v1.md` |
| 14 | Citation-marker features stacked on NLI probabilities (logistic regression) | +2 to 3 macro-F1 over the same stacker without markers; CV accuracy 0.707; still below NLI argmax on macro-F1 | Signal real, method wrong (§6.1) | this document |

### Diagnostics that matter

- **Error budget** of the best pre-NLI system (`results/diagnosis_v1.md`):
    - 41%: accurate citations with their evidence retrieved, called wrong.
    - 35%: misrepresenting citations with evidence retrieved (accuracy 0.24 on these).
    - 10%: ETIQUETTE.
    - 13%: IRRELEVANT.
    - So about **76% of errors are faithfulness judgements on evidence the model already had**.
      Retrieval is not the main bottleneck.
- **Zero-shot NLI** already ranked misrepresentation as well as the fine-tuned MultiVerS (AUC
  0.689 against 0.683; BiomedBERT 0.610). That is why the NLI model worked.
- **Truncation.** At 512 tokens, 97% of inputs are cut: the classifier reads about 12 of the 20
  sentences, and 14% of the retrieved gold evidence never reaches it. Fixing this (#12) did not help.
- **Synthetic data teaches artefacts.** 91% of LLM rewrites end in a full stop, against 59% of real
  citing sentences. Real citing sentences carry truncations and reference fragments.

## 6. What has not been tried, ranked by expected value

Every experiment in rows 7 to 13 changed the *model or its training data* while keeping the
*input* the same: an anonymous sentence with every citation marker removed, against the top 20
sentences. The leads below change what the model is told about the citation itself.

### 6.1 Show the model the citation's structure (highest value, cheapest)

`train_nli.py`'s `statement()` deletes every marker: `<|cit|>`, `<|multi_cit|>` (this reference is
one of several at that position) and `<|other_cit|>` (another paper cited elsewhere in the
sentence). Yet:

- **298 of the 304 ETIQUETTE citations in train (98%) carry `<|multi_cit|>`**, as do 71 of 73 in
  test. P(ETIQUETTE | multi) = 27% against 0.6% for a single citation.
- ETIQUETTE is NOT_ACCURATE, 14% of train and 10% of errors.
- A post-hoc logistic stacker shows the marker adds 2 to 3 macro-F1 over the identical stacker
  without it. CV accuracy reaches 0.707, the highest seen.

**Experiment:** keep readable markers in the hypothesis (`llm_classify.readable()` already maps
them to `[CITED]`, `[CITED, with others]`, `[other]`) and fine-tune, so the model learns the
interaction with the evidence. This is one CV run (about 75 minutes on a T4). It needs a
`--keep-markers` flag in `train_nli.py` and the same mapping in `citeaudit/text.py`.
`clean_citing()` currently strips user citations such as `[3, 5]`, which would then have to become
markers.

### 6.2 Citation scope: judge only the part of the sentence this reference supports

Long sentences cite several papers for different clauses. The model judges the whole sentence
against one paper, so a sentence that says more than this reference supports looks
NOT_SUBSTANTIATE. The annotations carry the annotators' **`citation_context`** span (in
`results/annotations/<split>.jsonl`; deliberately unused so far because a deployed tool cannot
have it). The paper's gain from context used exactly these gold spans.

**Experiment:** first an oracle run, with hypothesis = gold `citation_context` (CV, 75 minutes),
to measure the ceiling. If it is large, train a span extractor (marker position → scope) on train
annotations and use its output.

### 6.3 Decompose the label along the annotation's own logic

The fine labels mix three different questions:

1. **Relevance:** IRRELEVANT against the rest.
2. **Attribution:** is the claim the cited paper's own finding, or something it reports from
   others (INDIRECT / ETIQUETTE)? A strong cue is whether the matching evidence sentences in the
   cited paper carry their own citations, like "(12)" or "[3]".
3. **Faithfulness:** NLI.

One 3-way head conflates them. Separate heads or modules, each with its own features, chosen by
CV, may help ETIQUETTE and INDIRECT, which NLI cannot see.

### 6.4 Oracle-evidence diagnostic for the NLI model (cheap, informative)

The "perfect evidence does not help" finding is for MultiVerS only. It has never been run for the
NLI classifier: exports `data_train/target/oracle_20*` exist. If the NLI model jumps with gold
evidence, retrieval matters again and reranker work pays off. If not, focus entirely on 6.1 to 6.3.

### 6.5 Label-noise audit and ceiling estimate

The plateau may be the annotation agreement level. Read the model's most confident disagreements
with gold (`scripts/error_analysis.py` lists them), estimate how many are defensible, and check
whether the paper reports inter-annotator agreement (not yet checked here). If noise is high,
confident-learning-style relabelling or down-weighting of training labels is an option, and the
realistic target should be stated against it.

### 6.6 More compute

Everything so far ran on a free Colab T4 (16 GB). Larger cross-encoders or fine-tuned seq2seq
models (e.g. Flan-T5-XL/XXL, 3B to 11B) need an A100 (Colab Pro, USC CARC) and are untested.
Expect modest gains; the leads above are about information, not capacity.

### 6.7 On hold, by the project lead's choice: large LLMs

A frontier LLM as judge is excluded as the product (see §2). A compatible use is as a *teacher*:
distil its judgements or rationales into the small open model, so the released tool stays small
and open. Only revisit if 6.1 to 6.5 are exhausted.

## 7. Practical context for whoever continues

**Repository** (github.com/SapnilPatel/CiteAudit):

- `scripts/`: research pipeline.
    - `train_nli.py`: CV via `--folds/--fold`, seeds, `--predict`, `--save-model`, `--synthetic`,
      `--max-len`, `--grad-ckpt`.
    - `train_reranker.py`, `export_for_multivers.py` (builds top-k classifier inputs), `tune_bias.py`,
      `soup.py`, `error_analysis.py`, `pretrain_nli.py`, `build_pretrain_data.py`.
- `data_train/target/<name>`: classifier inputs, as `claims_<split>.jsonl` + `corpus.jsonl` (one
  synthetic document of the top-k sentences per citation).
    - `ours_20r_*` = our reranker, rank order. This is what the NLI model uses.
    - `ours_20r_train_cv` = train rankings from 3-fold CV rerankers, so no train citation is scored
      by a reranker trained on it.
- **Gold labels.** 3-way gold labels must come from `results/annotations/<split>.jsonl`; claims
  with no evidence are *not* all NEI. Fine labels: `data_train/fine_labels/<split>.json`; citing
  context: `data_train/context/`.
- `results/`: one write-up per experiment, plus run configs. Prediction `.jsonl` files are
  git-ignored.
- `notebooks/`: one Colab notebook per GPU experiment. Each clones the repo, writes resumably to
  `MyDrive/CiteAudit_runs/<run>`, and ends by downloading a zip that goes in the repo root.

**Gotchas already paid for:**

- **The reranker's window quirk.** The previous neighbour is clipped with `s[len(s)-200:]`. For a
  100 to 200 character sentence, that index is negative and keeps only the last `200−len`
  characters. The model was trained on this, so replicate it.
- **fp16 checkpoint.** The NLI checkpoint is published in fp16. Load it with `.float()` before
  fp16-autocast training, or GradScaler fails.
- **Truncation mode.** Use `truncation='longest_first'`: some claims exceed 512 tokens alone.
- **Memory on a T4.** Batch by token budget: 8 × 512 tokens is out of memory on a T4, while
  2 × 512 and 1 × 1024 (with gradient checkpointing) fit.
- **Phase gates.** Gates comparing CV against a baseline must use unrounded numbers; one slip let
  a tie through (`nli_biomed_v1.md`).
- **Noise level.** Seed sd is about 0.015 to 0.02 macro-F1. Differences under 0.03 on one seed
  are noise.

**Working conventions in this repository:** commit and push after every change (daily
contribution history matters to the owner); no AI co-author lines in commits or PRs; do not modify
the teammate's reproduction repository (git remote `teammate`); GPU work goes through Colab
notebooks, not the owner's laptop.

## 8. Recommended next steps

1. **6.1, markers in the hypothesis:** CV, one run.
2. **6.4, oracle evidence for NLI**, and **6.2, oracle scope:** together they say whether to
   invest in retrieval or in scope.
3. Depending on 2: build the scope extractor (6.2) or the attribution module (6.3).
4. Publish the classifier (release notebook, model soup) once the best configuration is fixed, and
   update `citeaudit/` if the input format changes (markers, scope).

**Realistic target:** with the paper's perfect-evidence-and-context ceiling at about 0.73 to 0.75
accuracy, an honest goal is accuracy above 0.71 and macro-F1 above 0.60, with NOT ACCURATE F1
clearly above 0.45.
