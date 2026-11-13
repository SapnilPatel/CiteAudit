#!/usr/bin/env python3
"""Write the citing-paragraph context of every citation, for `train_classifier.py --context`.

Every model so far has read only the citing sentence. The release also carries each citation's
citing paragraph (about 170 words), which says why the source is cited: the information that
separates ETIQUETTE from IRRELEVANT, or an oversimplification from a fair summary. It is also
what a real user always has, since they hold the citing paper. The annotators' marked
`citation_context` span is gold information and is deliberately not used.

For each citation, located by its marker's offset in the paragraph (`<|cit|>` or
`<|multi_cit|>`), this writes:

  window      the citing sentence with the sentence before and after it;
  paragraph   the whole paragraph, or if longer than --max-chars the sentences closest to the
              citing one that fit, so the checked citation is never cut off.

Sentences are split with a simple punctuation rule, enough for choosing what to show the model.

    python scripts/build_context.py --split train --out data_train/context/train.json
"""
import argparse
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SENT_END = re.compile(r'(?<=[.!?\]\)])\s+(?=[A-Z0-9(\[<])')


def sentences(text):
    """(start, end) character spans of the paragraph's sentences."""
    spans, start = [], 0
    for m in SENT_END.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return spans


def marker_offset(rec):
    para = rec['citing_paragraph']
    m = rec['marker_span']
    m = ast.literal_eval(m) if isinstance(m, str) else m
    if para[m['start']:m['end']] in ('<|cit|>', '<|multi_cit|>'):
        return m['start'], True
    # Two citations in the release have offsets that miss the marker: find the claim instead.
    i = para.find(rec['claim'][:60])
    if i >= 0:
        return i, False
    i = min((x for x in (para.find('<|cit|>'), para.find('<|multi_cit|>')) if x >= 0), default=0)
    return i, False


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--split', required=True, choices=['train', 'dev', 'test'])
    p.add_argument('--annotations', type=Path)
    p.add_argument('--max-chars', type=int, default=1500)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()

    recs = [json.loads(x) for x in open(a.annotations or ROOT / f'results/annotations/{a.split}.jsonl')]
    out, fallback = {}, 0
    for r in recs:
        para = r['citing_paragraph']  # unnormalised: the marker offsets index into it
        off, exact = marker_offset(r)
        fallback += not exact
        spans = sentences(para)
        i = next((k for k, (s, e) in enumerate(spans) if s <= off < e + 1), len(spans) - 1)
        window = ' '.join(para[s:e] for s, e in spans[max(i - 1, 0):i + 2])
        lo = hi = i
        while True:  # grow outward from the citing sentence while it fits
            grew = False
            for nlo, nhi in ((lo - 1, hi), (lo, hi + 1)):
                if 0 <= nlo and nhi < len(spans) and spans[nhi][1] - spans[nlo][0] <= a.max_chars:
                    lo, hi, grew = nlo, nhi, True
            if not grew:
                break
        out[str(r['id'])] = {'article': r['reference'], 'window': window,
                             'paragraph': para[spans[lo][0]:spans[hi][1]]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(sorted(out.items(), key=lambda x: int(x[0]))), indent=0),
                     encoding='utf8')
    missing = sum(not any(m in v['window'] for m in ('<|cit|>', '<|multi_cit|>')) for v in out.values())
    print(f'{a.split}: {len(out)} citations, {fallback} located by text, '
          f'{missing} windows without the citation marker -> {a.out}')


if __name__ == '__main__':
    main()
