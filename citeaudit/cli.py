"""citeaudit: check whether citations say what the cited paper says.

Single citation:
    citeaudit check --paper paper.pdf --citing "Statins reduce mortality in all adults [12]."
    citeaudit check --pmcid PMC3643595 --citing "loc285194 is a p53-regulated tumor suppressor."

Many citations, one JSON object per line, each with "citing" and one of "paper" (a path, relative
to the batch file), "pmcid", "text" or "sentences", plus an optional "id":
    citeaudit check --batch citations.jsonl --out results.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .pipeline import DEFAULT_CLASSIFIER, DEFAULT_RERANKER, CiteAuditor
from .sources import from_file, from_pmc
from .text import split_sentences

SYMBOL = {'ACCURATE': '✅', 'NOT_ACCURATE': '❌', 'NOT ENOUGH INFO': '➖'}


def paper_sentences(spec, base=Path('.'), cache={}):
    """Sentences of the cited paper described by a batch row or by CLI arguments."""
    if spec.get('sentences'):
        return spec['sentences']
    if spec.get('text'):
        return split_sentences(spec['text'])
    key = ('pmc', spec['pmcid']) if spec.get('pmcid') else ('file', str(base / spec['paper'])) \
        if spec.get('paper') else None
    if key is None:
        raise SystemExit('each citation needs the cited paper: "paper", "pmcid", "text" or "sentences"')
    if key not in cache:  # many citations usually share one paper
        cache[key] = from_pmc(key[1]) if key[0] == 'pmc' else from_file(key[1])
        if not cache[key]:
            raise SystemExit(f'no sentences found in {key[1]}')
    return cache[key]


def show(result, n_evidence):
    p = result.probabilities
    print(f'\n{SYMBOL[result.label]}  {result.label}  ({result.confidence:.0%}): {result.explanation}')
    print('   ' + '   '.join(f'{k} {v:.0%}' for k, v in p.items()))
    print(f'\n   Statement checked: {result.statement}')
    if n_evidence:
        print(f'\n   Most relevant sentences in the cited paper (of {len(result.evidence)} the '
              f'classifier read):')
        for e in result.evidence[:n_evidence]:
            print(f'   [{e.index}] {e.sentence}')
    print()


def main(argv=None):
    p = argparse.ArgumentParser(prog='citeaudit', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--version', action='version', version=f'citeaudit {__version__}')
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('check', help='check citations against the papers they cite')
    src = c.add_mutually_exclusive_group()
    src.add_argument('--paper', type=Path, help='the cited paper: .txt, .md or .pdf')
    src.add_argument('--pmcid', help='the cited paper by PubMed Central id (open access only)')
    c.add_argument('--citing', help='the citing sentence')
    c.add_argument('--batch', type=Path, help='JSONL file of citations (see above)')
    c.add_argument('--out', type=Path, help='with --batch: write results here (default stdout)')
    c.add_argument('--json', action='store_true', help='print the result as JSON')
    c.add_argument('--evidence', type=int, default=3, help='evidence sentences to show (default 3)')
    c.add_argument('--reranker', default=DEFAULT_RERANKER)
    c.add_argument('--classifier', default=DEFAULT_CLASSIFIER)
    c.add_argument('--device', default='auto', help='cpu, cuda or mps (default: best available)')
    a = p.parse_args(argv)

    if a.batch:
        rows = [json.loads(x) for x in a.batch.read_text(encoding='utf8').splitlines() if x.strip()]
        base = a.batch.parent
    elif a.citing and (a.paper or a.pmcid):
        rows = [{'citing': a.citing, 'paper': a.paper, 'pmcid': a.pmcid}]
        base = Path('.')
    else:
        c.error('give --citing with --paper or --pmcid, or --batch')

    papers = [paper_sentences(r, base) for r in rows]  # fail on a bad input before loading models
    print(f'loading models on {a.device} ...', file=sys.stderr, flush=True)
    auditor = CiteAuditor(a.reranker, a.classifier, a.device)
    out = a.out.open('w', encoding='utf8') if a.out else sys.stdout
    for n, (row, sents) in enumerate(zip(rows, papers), 1):
        result = auditor.check(row['citing'], sentences=sents)
        if a.batch or a.json:
            d = result.to_dict()
            d['evidence'] = d['evidence'][:max(a.evidence, 0)]
            out.write(json.dumps({'id': row.get('id', n), **d}, ensure_ascii=False) + '\n')
        else:
            show(result, a.evidence)
        if a.batch and a.out:
            print(f'  {n}/{len(rows)}', file=sys.stderr, flush=True)
    if a.out:
        out.close()
        print(f'wrote {a.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
