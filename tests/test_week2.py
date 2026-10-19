import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_pipeline(tmp_path):
    corpus=tmp_path/'corpus.jsonl';claims=tmp_path/'claims.jsonl';out=tmp_path/'retrieval.jsonl'
    corpus.write_text(json.dumps({'doc_id':1,'title':'Paper','abstract':['Completely unrelated sentence.','Drug A significantly reduced inflammation.','No further findings.']})+'\n')
    claims.write_text(json.dumps({'id':12,'claim':'Drug A reduced inflammation','evidence':{'1':[{'sentences':[1],'label':'SUPPORT'}]}})+'\n')
    subprocess.run([sys.executable,str(ROOT/'scripts/week2_pipeline.py'),'--claims',str(claims),'--corpus',str(corpus),'--out',str(out)],check=True,capture_output=True)
    row=json.loads(out.read_text());assert row['retrieved'][0]==1
    assert json.loads(out.with_suffix('.metrics.json').read_text())['Recall@1']==1

def test_compare(tmp_path):
    one=tmp_path/'one.csv';two=tmp_path/'two.csv';res=tmp_path/'result.json'
    one.write_text('id,gold,prediction\n1,ACCURATE,IRRELEVANT\n2,NOT_ACCURATE,NOT_ACCURATE\n')
    two.write_text('id,gold,prediction\n1,ACCURATE,ACCURATE\n2,NOT_ACCURATE,NOT_ACCURATE\n')
    subprocess.run([sys.executable,str(ROOT/'scripts/week2_compare.py'),'--retrieved',str(one),'--oracle',str(two),'--out',str(res)],check=True,capture_output=True)
    values=json.loads(res.read_text());assert values['oracle_minus_retrieved_micro_f1']==0.5

def test_retrieval_harness(tmp_path):
    # one article split over two chunks; gold sentence sits in the second chunk
    (tmp_path/'corpus.jsonl').write_text(
        json.dumps({'doc_id':10,'title':'','abstract':['Unrelated opening.','Methods were standard.']})+'\n'+
        json.dumps({'doc_id':11,'title':'','abstract':['Drug A significantly reduced inflammation.']})+'\n')
    (tmp_path/'claims-test.jsonl').write_text(
        json.dumps({'id':0,'claim':'Drug A reduced inflammation <|cit|>.','evidence':{'11':[{'sentences':[0],'label':'ACCURATE'}]},'cited_doc_ids':[10,11]})+'\n'+
        json.dumps({'id':1,'claim':'No evidence here.','evidence':{},'cited_doc_ids':[10,11]})+'\n')
    rk=tmp_path/'r.jsonl';met=tmp_path/'m.json'
    subprocess.run([sys.executable,str(ROOT/'scripts/retrieval_harness.py'),'--data-dir',str(tmp_path),'--bootstrap','10',
                    '--rankings-out',str(rk),'--metrics-out',str(met)],check=True,capture_output=True)
    assert json.loads(rk.read_text())['ranked'][0]=='11:0'
    m=json.loads(met.read_text())
    assert m['queries_scored']==1 and m['skipped_no_gold']==1 and m['metrics']['Recall@1']==1
    # a reversed external ranking puts gold last (rank 3)
    rk.write_text(json.dumps({'id':0,'ranked':['10:0','10:1','11:0']})+'\n')
    subprocess.run([sys.executable,str(ROOT/'scripts/retrieval_harness.py'),'--data-dir',str(tmp_path),'--ranker','file',
                    '--rankings-in',str(rk),'--bootstrap','0','--metrics-out',str(met)],check=True,capture_output=True)
    assert abs(json.loads(met.read_text())['metrics']['MRR']-1/3)<1e-9

def tiny_bert(path):
    """A randomly initialised 2-layer BERT saved locally, so the test needs no download."""
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + [
        w for w in ('drug a reduced inflammation in mice significantly filler sentence '
                    'about weather more cit').split()]
    (path / 'vocab.txt').parent.mkdir(parents=True, exist_ok=True)
    (path / 'vocab.txt').write_text('\n'.join(vocab) + '\n')
    tok = BertTokenizerFast(vocab_file=str(path / 'vocab.txt'))
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=64, num_labels=1)
    BertForSequenceClassification(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path


def test_reranker_train_and_predict(tmp_path):
    """One training epoch and a prediction pass on a synthetic corpus, CPU only."""
    data = tmp_path / 'data'
    data.mkdir()
    corpus = [{'doc_id': 1000 + i, 'title': '', 'abstract': [f'Filler sentence {i} about weather.',
               'Drug A significantly reduced inflammation in mice.', f'More filler {i}.']}
              for i in range(3)]
    (data / 'corpus.jsonl').write_text('\n'.join(json.dumps(d) for d in corpus) + '\n')
    claims = [{'id': i, 'claim': 'Drug A reduced inflammation <|cit|>.',
               'evidence': {'1001': [{'sentences': [1], 'label': 'ACCURATE'}]},
               'cited_doc_ids': [1000, 1001, 1002]} for i in range(4)]
    for split in ('train', 'dev'):
        (data / f'claims-{split}.jsonl').write_text('\n'.join(json.dumps(c) for c in claims) + '\n')
    out = tmp_path / 'ckpt'
    rk = tmp_path / 'ranked.jsonl'
    model = tiny_bert(tmp_path / 'tiny')
    r = subprocess.run([sys.executable, str(ROOT / 'scripts/train_reranker.py'),
                        '--data-dir', str(data), '--model', str(model),
                        '--epochs', '1', '--negatives', '2', '--neg-from', '0', '--neg-to', '9',
                        '--out', str(out), '--cache-dir', str(tmp_path / 'cache'),
                        '--predict', 'dev', '--rankings-out', str(rk), '--depth', '9',
                        '--device', 'cpu', '--max-len', '64'],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    rows = [json.loads(l) for l in rk.read_text().splitlines()]
    assert len(rows) == 4
    assert all(len(set(x['ranked'])) == len(x['ranked']) for x in rows)   # no duplicate keys
    assert all(set(x['gold']) <= set(x['ranked']) for x in rows)          # gold stays rankable
    # the rankings file must feed straight back into the evaluator
    met = tmp_path / 'm.json'
    subprocess.run([sys.executable, str(ROOT / 'scripts/retrieval_harness.py'),
                    '--data-dir', str(data), '--split', 'dev', '--ranker', 'file',
                    '--rankings-in', str(rk), '--bootstrap', '0', '--metrics-out', str(met)],
                   check=True, capture_output=True)
    assert 'Recall@20_micro' in json.loads(met.read_text())['metrics']
