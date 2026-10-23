# Week 1 — Six-Person Split

Everyone works on **baseline reproduction only** this week.

## Person 1 — Official repo + environment
- Clone `ScienceNLP-Lab/Citation-Integrity`.
- Get its code environment running.
- Record Python/CUDA/package versions.
- Find the authors' released checkpoint instructions.
- Deliverable: `ENV_NOTES.md` or a short setup section in the group README.

## Person 2 — Dataset + split audit
- Inspect `Data/`.
- Verify train/dev/test organization (paper: 70/10/20 reference articles).
- Count citation instances and label frequencies.
- Verify the 8 original labels.
- Deliverable: dataset summary + any preprocessing needed for MultiVerS.

## Person 3 — Label mapping + evaluation
- Verify the paper's 8 -> 3 mapping.
- Make sure evaluation reports micro-F1, macro-F1, and per-class F1.
- Test `scripts/evaluate_classification.py` on a small file.
- Deliverable: one shared evaluator for the group.

## Person 4 — BM25 + MonoT5 retrieval baseline
- Run the authors' `bm25t5.ipynb`.
- Use the **citance** query first because that corresponds to the main top-20 classifier setup.
- Record Recall@1/5/10/20 and MRR.
- Target Recall@20: ~0.53 for citance.
- Deliverable: retrieved top-20 evidence files + retrieval metrics.

## Person 5 — MultiVerS top-20 baseline
- Use the authors' released/fine-tuned MultiVerS checkpoint if possible.
- Feed the top-20 retrieved evidence setup into classification.
- Record micro/macro/per-class F1.
- Targets: 0.59 micro, 0.52 macro.
- Deliverable: predictions + metrics + exact run command.

## Person 6 — Gold-evidence oracle + integration
- Prepare/run the gold-evidence version that bypasses retrieval mistakes.
- Target: 0.73 micro, 0.75 macro.
- Keep the master `results/ours.csv` updated.
- Merge commands from the other five people into one reproducibility checklist.
- Deliverable: paper-vs-ours table and list of blockers/discrepancies.

# Weekly meeting

Each person reports only:
1. command/code that now runs,
2. latest metric,
3. blocker,
4. next action.

The key Week-1 question is: **Can we reproduce the original pipeline before modifying it?**
