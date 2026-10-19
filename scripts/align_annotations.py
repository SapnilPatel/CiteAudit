#!/usr/bin/env python3
"""Align `annotations.zip` records to claim ids in `Data/multivers-format/claims-*.jsonl`.

The annotations hold what the multivers-format files drop: the citing paragraph, the
citation-context spans the authors query with, the gold evidence spans, and `marker_span`
(which citation marker in the text this particular reference corresponds to). None of it is
usable until each annotation file is tied to a claim id.

Alignment is structural first, textual second:

  group  corpus doc_id = <reference number> * 1000 + <paragraph index>, so a claim's
         cited_doc_ids determine its reference article exactly, and the annotation directory
         is the one with that number. Verified: no claim spans two references, and every
         group holds equally many claims and annotation files.
  within claims and annotation files of one group are matched on
           K1  claim text == one of the citation_context texts
           K2  a gold evidence sentence == one of the evidence_segments
           K3  label agreement (constraint only, never the sole evidence for a pair)
         then maximum bipartite matching over those edges, strongest key first, with
         anything still ambiguous reported rather than guessed.

Output JSONL, one row per aligned claim: claim id, annotation path, the keys that matched,
the authors' query string (citation_context span sliced out of citing_paragraph, as
`bm25t5.ipynb` builds it), marker_span, and the label.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / 'upstream/Citation-Integrity/Data/multivers-format'
DEFAULT_ANN = ROOT / 'upstream/annotations_extracted/annotations'
SPLIT_DIR = {'train': 'Train', 'dev': 'Dev', 'test': 'Test'}
MARKERS = re.compile(r'<\|(?:multi_cit|cit)\|>')


def read_jsonl(path):
    with open(path, encoding='utf8') as f:
        return [json.loads(x) for x in f if x.strip()]


def norm(s):
    """Compare on letters and digits only: the two sources differ in markers and whitespace."""
    return re.sub(r'[^a-z0-9]', '', MARKERS.sub('', str(s)).lower())


def authors_query(rec):
    """The query `bm25t5.ipynb` builds: citing_paragraph sliced from the min start to the
    max end of the citation_context spans."""
    spans = rec.get('citation_context') or []
    if not spans:
        return ''
    start = min(s['start'] for s in spans)
    end = max(s['end'] for s in spans)
    return rec['citing_paragraph'][start:end].strip().replace('\n', ' ')


def load_annotations(ann_dir, split):
    """{reference number: {path: record}}"""
    out = defaultdict(dict)
    root = Path(ann_dir) / SPLIT_DIR[split] / 'citations'
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        num = int(d.name.split('_')[0])
        for f in sorted(d.glob('*.json')):
            out[num][f] = json.load(open(f, encoding='utf8'))
    return out


def claim_labels(claim):
    return {it['label'] for items in claim.get('evidence', {}).values() for it in items}


def gold_texts(claim, corpus):
    out = set()
    for doc, items in claim.get('evidence', {}).items():
        sents = corpus.get(int(doc), [])
        for it in items:
            for si in it['sentences']:
                if si < len(sents):
                    out.add(norm(sents[si]))
    return out


def match_edges(claims, anns, corpus):
    """Compatibility edges in strict tiers, strongest first.

    tier 1  K1: the citance is exactly one of the citation_context sentences. Near-unique —
            a citance appears once per reference it cites, and references are separate groups.
    tier 2  K2: a gold evidence sentence matches an evidence_segment. Weak on its own: several
            citances of one reference can rest on the same evidence sentence, so K2 must never
            outrank K1, or a K2 edge will be chosen over the correct K1 pair.
    tier 3  label-compatible only, resolved by elimination within the group.
    """
    edges = defaultdict(dict)  # claim id -> {path: (tier, keys)}
    for c in claims:
        ctext = norm(c['claim'])
        cgold = gold_texts(c, corpus)
        clabels = claim_labels(c)
        for path, rec in anns.items():
            if clabels and rec['label'] not in clabels:
                continue  # K3 as a hard constraint when the claim carries labels
            k1 = any(ctext == norm(s['text']) for s in rec.get('citation_context') or [])
            k2 = bool(cgold) and any(norm(e['text']) in cgold
                                     for e in rec.get('evidence_segments') or [])
            if k1:
                edges[c['id']][path] = (1, ['K1', 'K2'] if k2 else ['K1'])
            elif k2:
                edges[c['id']][path] = (2, ['K2'])
            else:
                edges[c['id']][path] = (3, ['K3'] if clabels else ['none'])
    return edges


def kuhn(left, adjacency):
    """Maximum bipartite matching; returns {right: left}."""
    match = {}

    def try_assign(u, seen):
        for v in adjacency.get(u, ()):
            if v in seen:
                continue
            seen.add(v)
            if v not in match or try_assign(match[v], seen):
                match[v] = u
                return True
        return False

    for u in left:
        try_assign(u, set())
    return match


def align_group(claims, anns, corpus):
    """Returns {claim id: (path, keys)}, the unresolved ids, and the ambiguous ones.

    A claim is flagged ambiguous when it was assigned at a tier where it had several
    compatible annotations and so could have been permuted with another claim.
    """
    edges = match_edges(claims, anns, corpus)
    ids = [c['id'] for c in claims]
    assigned, used, ambiguous = {}, set(), []
    for tier in (1, 2, 3):
        todo = [i for i in ids if i not in assigned]
        adjacency = {i: [p for p, (t, _) in edges[i].items() if t <= tier and p not in used]
                     for i in todo}
        for right, left in kuhn(todo, adjacency).items():
            assigned[left] = (right, edges[left][right][1])
            used.add(right)
            if len(adjacency[left]) > 1:
                ambiguous.append(left)
    return assigned, [i for i in ids if i not in assigned], ambiguous


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    p.add_argument('--ann-dir', type=Path, default=DEFAULT_ANN)
    p.add_argument('--split', default='test', choices=['train', 'dev', 'test'])
    p.add_argument('--out', type=Path)
    a = p.parse_args()

    corpus = {int(d['doc_id']): d['abstract'] for d in read_jsonl(a.data_dir / 'corpus.jsonl')}
    claims = read_jsonl(a.data_dir / f'claims-{a.split}.jsonl')
    anns = load_annotations(a.ann_dir, a.split)

    by_ref = defaultdict(list)
    for c in claims:
        by_ref[int(min(c['cited_doc_ids'])) // 1000].append(c)

    only_claims = set(by_ref) - set(anns)
    only_anns = set(anns) - set(by_ref)
    if only_claims or only_anns:
        raise SystemExit(f'reference groups disagree: claims-only {sorted(only_claims)}, '
                         f'annotations-only {sorted(only_anns)}')

    rows, unresolved, ambiguous, key_counts = [], [], [], defaultdict(int)
    for ref, group in sorted(by_ref.items()):
        if len(group) != len(anns[ref]):
            raise SystemExit(f'group {ref}: {len(group)} claims vs {len(anns[ref])} annotations')
        assigned, left, amb = align_group(group, anns[ref], corpus)
        unresolved += [(ref, i) for i in left]
        ambiguous += amb
        for c in group:
            if c['id'] not in assigned:
                continue
            path, keys = assigned[c['id']]
            rec = anns[ref][path]
            key_counts['+'.join(keys)] += 1
            rows.append({'id': c['id'], 'reference': ref,
                         'annotation': str(path.relative_to(ROOT)),
                         'match_keys': keys,
                         'ambiguous': c['id'] in amb,
                         'label': rec['label'],
                         'claim': c['claim'],
                         'authors_query': authors_query(rec),
                         'citing_paragraph': rec['citing_paragraph'],
                         'citation_context': rec.get('citation_context') or [],
                         'marker_span': rec.get('marker_span'),
                         'evidence_segments': rec.get('evidence_segments') or []})

    rows.sort(key=lambda r: r['id'])
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open('w', encoding='utf8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    uniq = len({r['annotation'] for r in rows})
    print(f'split={a.split}  claims={len(claims)}  aligned={len(rows)}  '
          f'distinct annotations used={uniq}  unresolved={len(unresolved)}')
    print('  match keys: ' + ', '.join(f'{k}={v}' for k, v in sorted(key_counts.items())))
    k1 = sum(v for k, v in key_counts.items() if 'K1' in k)
    k2 = sum(v for k, v in key_counts.items() if k == 'K2')
    print(f'  K1 (citance == citation_context, near-unique): {k1}/{len(claims)}'
          f'   K2 only: {k2}   label+elimination only: {len(rows) - k1 - k2}')
    print(f'  assigned where several annotations were compatible (permutable): {len(ambiguous)}')
    if unresolved:
        print(f'  unresolved claim ids: {[i for _, i in unresolved][:10]}')
    aq = sum(1 for r in rows if r['authors_query'])
    diff = sum(1 for r in rows if norm(r['authors_query']) != norm(r['claim']))
    print(f"  authors' query present for {aq}/{len(rows)}; differs from the citance for {diff}")


if __name__ == '__main__':
    main()
