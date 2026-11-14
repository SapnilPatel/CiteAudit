"""Unit tests for the citeaudit package. No model downloads: the models are stubbed."""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from citeaudit import CiteAuditor, cli  # noqa: E402
from citeaudit import retrieval  # noqa: E402
from citeaudit.sources import sentences_from_jats  # noqa: E402
from citeaudit.text import clean_citing, split_sentences  # noqa: E402

PAPER = ('Statins lower LDL cholesterol. In this trial of 4,000 men aged 45-64 years, '
         'pravastatin reduced coronary events by 31% (p < 0.001). Fig. 2 shows the effect '
         'over 5.2 years, as reported by Smith et al. in earlier work.\n\n'
         'No effect was seen on all-cause mortality. Diet was not recorded.')


def test_split_sentences_keeps_abbreviations_and_decimals():
    s = split_sentences(PAPER)
    assert s == ['Statins lower LDL cholesterol.',
                 'In this trial of 4,000 men aged 45-64 years, pravastatin reduced coronary '
                 'events by 31% (p < 0.001).',
                 'Fig. 2 shows the effect over 5.2 years, as reported by Smith et al. in earlier work.',
                 'No effect was seen on all-cause mortality.', 'Diet was not recorded.']


def test_split_sentences_eg_and_single_letters():
    assert split_sentences('In mice, e.g. strain B6, i.e. males, drug A. Then vitamin D. Done (n = 4).') == \
        ['In mice, e.g. strain B6, i.e. males, drug A.', 'Then vitamin D.', 'Done (n = 4).']


@pytest.mark.parametrize('raw, clean', [
    ('Statins reduce mortality [12].', 'Statins reduce mortality.'),
    ('Statins reduce mortality [3, 5-7] in men.', 'Statins reduce mortality in men.'),
    ('Statins reduce mortality (Smith et al., 2020; Lee 2019a).', 'Statins reduce mortality.'),
    ('Statins reduce mortality [<|cit|>].', 'Statins reduce mortality.'),
    ('As <|multi_cit|> showed, it works (<|other_cit|>).', 'As showed, it works.'),
    ('In 2020 statins (a drug class) were common.', 'In 2020 statins (a drug class) were common.'),
])
def test_clean_citing(raw, clean):
    assert clean_citing(raw) == clean


def test_retrieval_matches_research_code():
    """The package's copy of BM25 and the reranker window must equal the research pipeline's."""
    import retrieval_harness
    import train_reranker
    sents = split_sentences(PAPER) + ['x' * 450, 'Short.'] + ['y' * n for n in (99, 100, 150, 199, 200, 201)]
    q = 'pravastatin reduced coronary events in men'
    assert retrieval.bm25_scores(q, sents) == retrieval_harness.bm25_scores(q, sents, 0.9, 0.4)
    for i in range(len(sents)):
        assert retrieval.window(sents, i) == train_reranker.window(sents, i)


class StubReranker:
    def __call__(self, query, passages):  # word overlap with the query
        q = set(retrieval.tok(query))
        return [float(len(q & set(retrieval.tok(p)))) for p in passages]


class StubClassifier:
    def __init__(self):
        self.calls = []

    def __call__(self, evidence, statement):
        self.calls.append((evidence, statement))
        return {'ACCURATE': 0.2, 'NOT_ACCURATE': 0.7, 'NOT ENOUGH INFO': 0.1}


def test_pipeline_with_stub_models():
    clf = StubClassifier()
    auditor = CiteAuditor(StubReranker(), clf, device='cpu')
    r = auditor.check('Pravastatin reduced all-cause mortality [4].', paper_text=PAPER)
    assert r.label == 'NOT_ACCURATE' and r.confidence == 0.7
    assert r.statement == 'Pravastatin reduced all-cause mortality.'
    assert len(r.evidence) == 5
    assert [e.score for e in r.evidence] == sorted((e.score for e in r.evidence), reverse=True)
    assert 'No effect was seen on all-cause mortality.' in [e.sentence for e in r.evidence[:2]]
    evidence, statement = clf.calls[0]
    assert evidence[0] == r.evidence[0].sentence and statement == r.statement
    json.dumps(r.to_dict())


def test_jats_skips_tables_and_references():
    root = ET.fromstring(
        '<article><front><article-meta><title-group><article-title>A trial</article-title>'
        '</title-group><abstract><p>We tested drug A. It worked.</p></abstract></article-meta>'
        '</front><body><sec><title>Results</title><p>Drug A cut pain by half.</p>'
        '<table-wrap><table><tr><td>Table cell.</td></tr></table></table-wrap></sec></body>'
        '<back><ref-list><ref>Smith J. A paper. 2020.</ref></ref-list></back></article>')
    assert sentences_from_jats(root) == ['A trial', 'We tested drug A.', 'It worked.', 'Results',
                                         'Drug A cut pain by half.']


def test_cli_batch(tmp_path, monkeypatch, capsys):
    (tmp_path / 'paper.txt').write_text(PAPER)
    (tmp_path / 'batch.jsonl').write_text(
        json.dumps({'id': 'a', 'citing': 'Pravastatin cut coronary events [1].', 'paper': 'paper.txt'}) + '\n' +
        json.dumps({'citing': 'Statins lower LDL [2].', 'text': PAPER}) + '\n')
    monkeypatch.setattr(cli, 'CiteAuditor', lambda *a: CiteAuditor(StubReranker(), StubClassifier()))
    cli.main(['check', '--batch', str(tmp_path / 'batch.jsonl'), '--out', str(tmp_path / 'out.jsonl'),
              '--evidence', '2'])
    rows = [json.loads(x) for x in (tmp_path / 'out.jsonl').read_text().splitlines()]
    assert [r['id'] for r in rows] == ['a', 2]
    assert all(r['label'] == 'NOT_ACCURATE' and len(r['evidence']) == 2 for r in rows)
