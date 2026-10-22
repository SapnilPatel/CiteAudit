#!/usr/bin/env python3
"""Turn a retrieval rankings file into the claims + corpus pair the MultiVerS stage consumes.

Each claim becomes **one synthetic document** holding the top-k sentences that run retrieved, so
the dataset has exactly one (claim, document) instance per claim. Two such exports — one from the
paper's reranker and one from ours — then differ in exactly one thing, which is what makes the
downstream F1 comparison attributable.

Why a synthetic corpus rather than rewriting `evidence` over the original one. The authors' reader
(`multivers/data_train.py`) builds one instance per (claim, cited_doc) pair and `metrics.py` scores
per instance. In this dataset a claim's `cited_doc_ids` is the whole reference article split into
paragraph chunks — median 79 of them, median 2 sentences each — so reusing the original corpus
yields 24797 instances for dev's 316 claims, almost all of them NOT ENOUGH INFO, and ~78
predictions per claim with no defined way to aggregate them. One document per claim makes
per-instance F1 equal per-citation F1, and makes `--topk` the knob it is meant to be.

Labels. The `claims-*.jsonl` files carry the **9 original fine-grained labels**, while the model's
`label_lookup` has 3 classes. ACCURATE stays ACCURATE, the eight other labels (CONTRADICT,
NOT_SUBSTANTIATE, OVERSIMPLIFY, MISQUOTE, ETIQUETTE, INDIRECT, INDIRECT_NOT_REVIEW, IRRELEVANT)
collapse to NOT_ACCURATE, and a claim with no gold evidence at all is NOT ENOUGH INFO — which is
also how the authors' reader derives NEI, from a cited doc that carries no evidence. Every claim
in this data has exactly one distinct label, so merging its evidence into one document cannot
create a label conflict.

The rankings file must cover **every** claim, including those with no gold evidence. Those are
exactly the NOT ENOUGH INFO class, and giving them an empty document while every other claim gets
k sentences would let the classifier read the label off the input length. Produce such a file with
`retrieval_harness.py --rank-all` (or a reranker, which ranks every citance anyway).

    python scripts/export_for_multivers.py \
        --rankings results/retrieval/dev_ce_pilot.jsonl --split dev --topk 20 \
        --out data_train/target/ours_20
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / 'upstream/Citation-Integrity/Data/multivers-format'

# The model's three classes, as `multivers/data.py` and `data_train.py` spell them.
ACCURATE, NOT_ACCURATE, NEI = 'ACCURATE', 'NOT_ACCURATE', 'NOT ENOUGH INFO'
# Synthetic doc ids must not collide with the real corpus, whose largest doc_id is 100070.
DOC_ID_BASE = 1_000_000


def read_jsonl(path):
    with open(path, encoding='utf8') as f:
        return [json.loads(x) for x in f if x.strip()]


# The paper's 8-to-3 collapse, as recorded in this repository's README:
#   ACCURATE, INDIRECT                                      -> ACCURATE      (MultiVerS Support)
#   CONTRADICT, NOT_SUBSTANTIATE, OVERSIMPLIFY, MISQUOTE,
#   ETIQUETTE                                               -> NOT_ACCURATE  (Refute)
#   IRRELEVANT                                              -> NEI
# INDIRECT_NOT_REVIEW is the ninth label the paper does not list (§6); it is treated as a variant
# of INDIRECT. It never occurs in dev (25 train / 9 test instances).
#
# This mapping CANNOT be read off `claims-*.jsonl`: that file stores a label only inside
# `evidence[doc][*]["label"]`, so the citances with no gold evidence sentences carry none at all —
# and they are not one class. In dev those 61 citances are 37 ETIQUETTE (which collapse to
# NOT_ACCURATE) and 24 IRRELEVANT (which collapse to NEI). Deriving NEI from "has no evidence"
# therefore mislabels 37 of 316 dev citances, so the fine-grained label is taken from the aligned
# annotations instead (`scripts/align_annotations.py`, §5).
LABEL_MAP = {
    'ACCURATE': ACCURATE,
    'INDIRECT': ACCURATE,
    'INDIRECT_NOT_REVIEW': ACCURATE,
    'CONTRADICT': NOT_ACCURATE,
    'NOT_SUBSTANTIATE': NOT_ACCURATE,
    'OVERSIMPLIFY': NOT_ACCURATE,
    'MISQUOTE': NOT_ACCURATE,
    'ETIQUETTE': NOT_ACCURATE,
    'IRRELEVANT': NEI,
}


def three_way(claim, original):
    """The 3-class label for one claim, from its ORIGINAL fine-grained annotation label.

    `original` maps claim id -> one of the nine labels, from the aligned annotations. The label
    inside the claim's own `evidence` is consistent with it but absent whenever the citance has no
    gold evidence sentences, which is exactly where the two disagree.
    """
    fine = original.get(claim['id'])
    if fine is None:
        raise SystemExit(f'claim {claim["id"]} has no aligned annotation; run '
                         f'scripts/align_annotations.py for this split first')
    if fine not in LABEL_MAP:
        raise SystemExit(f'claim {claim["id"]} has unmapped label {fine!r}; extend LABEL_MAP '
                         f'deliberately rather than guessing')
    return LABEL_MAP[fine]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rankings', type=Path, required=True,
                   help='{id, ranked:["doc_id:sent_idx", ...]} JSONL covering every claim')
    p.add_argument('--split', required=True, choices=['train', 'dev', 'test'])
    p.add_argument('--topk', type=int, default=20)
    p.add_argument('--order', choices=['doc', 'rank'], default='doc',
                   help="order of the retrieved sentences inside the synthetic document: 'doc' "
                        "keeps the article's own order so adjacent evidence stays adjacent and the "
                        "document reads as prose; 'rank' keeps retrieval order. The authors' "
                        "notebook emits rank order, but their conversion script is not published, "
                        "so this is a free choice and worth measuring both ways.")
    p.add_argument('--oracle', action='store_true',
                   help='put every gold evidence sentence in the document, then fill the rest of '
                        'the k slots from the rankings. This is the diagnostic for whether the '
                        'classifier uses its evidence at all: it keeps the document length and '
                        'the gold labels identical to a retrieved arm and raises evidence recall '
                        'to 1.0, so a frozen checkpoint that scores the same on both is not '
                        'reading the evidence. Filling to k matters — a document of gold '
                        'sentences alone would be much shorter than the arms it is compared with, '
                        'and length would confound the comparison.')
    p.add_argument('--random', action='store_true',
                   help='fill the document with k sentences drawn uniformly from the same '
                        'candidate pool instead of the retrieved ones. The lower anchor for the '
                        'oracle: it says how much of the output depends on the evidence at all, '
                        'and so separates "the classifier barely reads its evidence" from "the '
                        'classifier reads it but is saturated above some recall". Seeded by claim '
                        'id, so the arm is reproducible.')
    p.add_argument('--seed', type=int, default=0, help='seed for --random')
    p.add_argument('--annotations', type=Path,
                   help='aligned annotations JSONL from scripts/align_annotations.py; the\n'
                        'source of the fine-grained label. Defaults to\n'
                        'results/annotations/<split>.jsonl.')
    p.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.oracle and a.random:
        p.error('--oracle and --random choose the document contents; pass at most one')

    claims = read_jsonl(a.data_dir / f'claims-{a.split}.jsonl')
    corpus = {d['doc_id']: d for d in read_jsonl(a.data_dir / 'corpus.jsonl')}
    ranked = {r['id']: r['ranked'] for r in read_jsonl(a.rankings)}
    ann_path = a.annotations or ROOT / f'results/annotations/{a.split}.jsonl'
    if not ann_path.exists():
        raise SystemExit(f'{ann_path} not found; the fine-grained label lives in the\n'
                         f'annotations, not in claims-{a.split}.jsonl. Run\n'
                         f'scripts/align_annotations.py --split {a.split} first.')
    original = {r['id']: r['label'] for r in read_jsonl(ann_path)}

    missing = [c['id'] for c in claims if c['id'] not in ranked]
    if missing:
        raise SystemExit(
            f'{len(missing)} of {len(claims)} claims have no rankings (e.g. {missing[:5]}). '
            f'Every claim needs them, including the no-gold ones: re-run the harness with '
            f'--rank-all, or use a reranker rankings file.')

    def sentence(key):
        did, si = key.split(':')
        return corpus[int(did)]['abstract'][int(si)]

    a.out.mkdir(parents=True, exist_ok=True)
    labels = Counter()
    n_sents, lost_all, lost_some, kept_gold, total_gold = [], 0, 0, 0, 0

    with (a.out / f'claims_{a.split}.jsonl').open('w', encoding='utf8') as fc, \
         (a.out / 'corpus.jsonl').open('w', encoding='utf8') as fd:
        for c in claims:
            label = three_way(c, original)
            labels[label] += 1
            doc_id = DOC_ID_BASE + c['id']

            gold = {f'{d}:{s}' for d, items in c.get('evidence', {}).items()
                    for it in items for s in it['sentences']}

            if a.random:
                # Uniform from the whole candidate pool the ranker saw, so the arm differs from a
                # retrieved one in evidence quality and in nothing else. Seeding per claim keeps
                # the draw stable across k, so k=10 is a subset-like sample of the same pool
                # rather than an unrelated one.
                pool = ranked[c['id']]
                keys = random.Random(a.seed * 1_000_003 + c['id']).sample(
                    pool, min(a.topk, len(pool)))
            elif a.oracle:
                # Gold first, then the highest-ranked non-gold sentences, so the document still
                # holds k sentences and differs from a retrieved arm only in evidence recall.
                # A claim with more than k gold sentences keeps them all: truncating would make
                # the "oracle" miss evidence, which is the one thing it must not do.
                fill = [k for k in ranked[c['id']] if k not in gold]
                keys = sorted(gold) + fill[:max(a.topk - len(gold), 0)]
            else:
                keys = ranked[c['id']][:a.topk]
            if a.order == 'doc':
                keys = sorted(keys, key=lambda k: (int(k.split(':')[0]), int(k.split(':')[1])))
            rationale = [i for i, k in enumerate(keys) if k in gold]
            total_gold += len(gold)
            kept_gold += len(rationale)
            if gold and not rationale:
                lost_all += 1
            elif gold and len(rationale) < len(gold):
                lost_some += 1

            # `title` is "" for every document in this corpus, not None. Keep it that way: the
            # authors' `_tokenize` branches on `is not None`, so "" still prepends an eos and
            # `_get_abstract_sent_tokens` still skips two — matching their tokenisation exactly.
            fd.write(json.dumps({'doc_id': doc_id, 'title': '',
                                 'abstract': [sentence(k) for k in keys]}) + '\n')
            n_sents.append(len(keys))

            # NEI is expressed the way the reader derives it: a cited doc with no evidence entry.
            # For the other two classes the entry carries the 3-way label and the positions, inside
            # this document, of the gold sentences that survived retrieval — possibly none, which
            # is the honest representation of a retrieval miss.
            evidence = {} if label == NEI else {
                str(doc_id): [{'sentences': rationale, 'label': label}]}
            fc.write(json.dumps({'id': c['id'], 'claim': c['claim'],
                                 'evidence': evidence, 'cited_doc_ids': [doc_id]}) + '\n')

    meta = {'rankings': str(a.rankings), 'split': a.split, 'topk': a.topk, 'order': a.order,
            'claims': len(claims), 'labels': dict(labels),
            'mean_sentences_per_doc': sum(n_sents) / len(n_sents),
            'gold_sentences_total': total_gold, 'gold_sentences_retrieved': kept_gold,
            'claims_losing_all_evidence': lost_all, 'claims_losing_some_evidence': lost_some}
    (a.out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(f'wrote {a.out}/  claims={len(claims)}  labels={dict(labels)}')
    print(f'  mean sentences/doc {meta["mean_sentences_per_doc"]:.1f}   '
          f'gold retrieved {kept_gold}/{total_gold} ({kept_gold / total_gold:.3f})')
    print(f'  claims losing all evidence {lost_all}, losing some {lost_some}')


if __name__ == '__main__':
    main()
