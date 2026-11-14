#!/usr/bin/env python3
"""Remove the surface artefacts and the non-edits from generate_misrep.py's synthetic citations.

Why this exists. Training on the raw synthetic citations lowered cross-validated macro-F1 from
0.570 to 0.542 (results/synthetic_v1.md) even though it raised accuracy on every real
misrepresentation label. The model also learned two things that are not misrepresentation:

  1. Style. The LLM tidies what it rewrites: 90% of rewrites end in a full stop against 59% of the
     real citing sentences, which carry truncations and leftover reference fragments (",7",
     "(Wang et al.,."). "Tidy sentence -> NOT ACCURATE" then penalises tidy accurate citations.
  2. Non-edits labelled as misrepresentation: a changed reference number ("-43" -> "-45"), a
     changed year, a completed word fragment, a punctuation or capitalisation fix.

So each rewrite is rebuilt from the original: the two are aligned token by token, and only edits
that change words are taken from the rewrite. Punctuation-only and case-only edits keep the
original's text, so the rewrite has exactly the original's surface form. A rewrite is dropped if
no word edit is left, if its only edits are to numbers that do not occur in the evidence (reference
numbers, years: nothing the evidence could contradict), or if an edit only completes or truncates
a word.

    python scripts/clean_misrep.py --raw data_train/synthetic/misrep_raw.jsonl \\
        --out data_train/synthetic/misrep_clean.jsonl
"""
import argparse
import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_misrep import sources  # noqa: E402
from llm_classify import read_jsonl  # noqa: E402

TOKEN_RE = re.compile(r'\w+|[^\w\s]')


def tokens(s):
    """(token, the token with the whitespace before it) pairs."""
    out, prev = [], 0
    for m in TOKEN_RE.finditer(s):
        out.append((m.group(), s[prev:m.end()]))
        prev = m.end()
    return out


def words(toks):
    return [t.lower() for t, _ in toks if re.match(r'\w', t)]


def rebuild(original, rewrite, evidence):
    """(rebuilt rewrite, reason) where exactly one is None."""
    a, b = tokens(original), tokens(rewrite)
    sm = difflib.SequenceMatcher(None, [t for t, _ in a], [t for t, _ in b], autojunk=False)
    ops = sm.get_opcodes()
    content = [o for o in ops if o[0] != 'equal' and words(a[o[1]:o[2]]) != words(b[o[3]:o[4]])]
    if len(content) == 1 and content[0][0] == 'delete' and content[0][2] == len(a):
        return None, 'truncated'  # the LLM stopped early: an accurate sentence, cut short
    pieces, edits = [], []
    for op, i1, i2, j1, j2 in ops:
        wa, wb = words(a[i1:i2]), words(b[j1:j2])
        if op == 'equal' or wa == wb:  # unchanged, or a punctuation / case-only change
            pieces += [ws for _, ws in a[i1:i2]]
        else:
            pieces += [ws for _, ws in b[j1:j2]]  # each token carries the whitespace before it
            edits.append((wa, wb))
    text = re.sub(r'\s+', ' ', ''.join(pieces)).strip()
    # The original's surface form at both ends: first-letter case and final full stop.
    if text and original:
        text = (text[0].upper() if original[0].isupper() else text[0].lower()) + text[1:]
        if original.rstrip().endswith('.') and not text.endswith('.'):
            text = text.rstrip(',;:') + '.'
        elif not original.rstrip().endswith('.') and text.endswith('.'):
            text = text[:-1].rstrip()
    if not edits:
        return None, 'no word edit'
    changed = [w for wa, wb in edits for w in wa + wb]
    if all(w.isdigit() for w in changed) and not any(re.search(rf'\b{w}\b', evidence) for w in changed):
        return None, 'only numbers absent from the evidence'
    for wa, wb in edits:
        if len(wa) == len(wb) == 1 and (wa[0].startswith(wb[0]) or wb[0].startswith(wa[0])):
            return None, 'word fragment'
    return text, None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, default=Path('data_train/target/ours_20r_train_cv'))
    p.add_argument('--fine-labels', type=Path, default=Path('data_train/fine_labels/train.json'))
    a = p.parse_args()

    evidence = {cid: ' '.join(ev) for cid, _, ev in sources(a.data_dir, a.fine_labels)}
    kept, dropped = [], Counter()
    for r in read_jsonl(a.raw):
        text, why = rebuild(r['original'], r['statement'], evidence[r['source_id']])
        if why:
            dropped[why] += 1
        else:
            kept.append({**r, 'statement': text, 'llm_statement': r['statement']})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(''.join(json.dumps(r) + '\n' for r in kept), encoding='utf8')
    end = lambda rows, k: sum(r[k].rstrip().endswith('.') for r in rows) / len(rows)
    print(f'kept {len(kept)}, dropped {dict(dropped)}; ends in a full stop: original '
          f'{end(kept, "original"):.2f}, raw rewrite {end(kept, "llm_statement"):.2f}, '
          f'cleaned {end(kept, "statement"):.2f}; types {dict(Counter(r["type"] for r in kept))}')


if __name__ == '__main__':
    main()
