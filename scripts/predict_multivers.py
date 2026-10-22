#!/usr/bin/env python3
"""Run the paper's released MultiVerS checkpoint to predict 3-way citation labels.

Why this exists rather than the authors' `multivers/predict.py`. That script cannot run as
shipped: it opens `../train_configs.json`, which is not in the repository, and it does not build
its dataloader from `--input_file`/`--corpus_file` at all — those are only used at the end to
recover claim ids, while the data comes from `data_train.ConcatDataModule.test_dataloader()` with
the directory hardcoded. Its dependencies (`pytorch-lightning==1.2.1`, `torch==1.7.1`,
`bertviz`, `focal_loss`) are also a decade apart from anything that runs on this machine.

So this reimplements the *label* path only, which is all the 3-way F1 needs, and reimplements it
to match the original exactly:

  input      `claim </s> title </s> sent1 </s> sent2 ...`, tokenized by
             `allenai/longformer-large-4096` plus the 13 additional tokens `data.py` adds, in
             that order — their ids depend on insertion order and the checkpoint's embedding
             matrix is 50278 = 50265 + 13 rows.
  global     attention on every `<s>`/`</s>` and on every claim token (everything before the
  attention first `</s>`), per `_get_global_attention_mask`.
  head       `pooler_output` -> Linear(1024,1024) -> GELU -> Linear(1024,3) -> argmax, per
             `model.py`'s `FeedForward(num_layers=2, activations=[GELU, Identity])`. Dropout is
             a no-op in eval mode. `label_threshold` is None in the checkpoint's hparams, so the
             prediction is a plain argmax.
  labels     `{0: NOT_ACCURATE, 1: NOT ENOUGH INFO, 2: ACCURATE}`.
  skipping   instances tokenizing to more than 4096 tokens are dropped, as the authors do with
             `print("input id > 4096!"); continue`. They are reported, not silently ignored.

The rationale head is deliberately not reimplemented: `rationale_weight` is 0.0 in the released
checkpoint's hparams and rationales do not enter the label prediction.

    python scripts/predict_multivers.py --data-dir data_train/target/ours_20 --split dev \
        --out results/classification/ours_20.jsonl
"""
import argparse
import importlib.abc
import importlib.machinery
import json
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn as nn

# The order `multivers/data.py` adds them in; token ids follow insertion order.
ADDITIONAL_TOKENS = ["<|sec|>", "</|sec|>", "<|sec-title|>", "</|sec-title|>", "<|abs|>",
                     "</|abs|>", "<|title|>", "</|title|>", "<|sent|>", "<|par|>", "<|cit|>",
                     "<|multi_cit|>", "<|other_cit|>"]
ENCODER = 'allenai/longformer-large-4096'
LABELS = {0: 'NOT_ACCURATE', 1: 'NOT ENOUGH INFO', 2: 'ACCURATE'}
MAX_TOKENS = 4096


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Manufacture empty *packages* for modules the pickled checkpoint references.

    The checkpoint was written by pytorch-lightning 1.2.1 and its `hyper_parameters` pickle
    references modules that are not installed and not needed to read weights. Each stub has to be
    a package (`__path__ = []`), or `import pytorch_lightning.callbacks` fails with
    "'pytorch_lightning.callbacks' is not a package" rather than with a plain ImportError.
    """
    ROOTS = ('pytorch_lightning', 'focal_loss', 'bertviz',
             'allennlp_nn_util', 'allennlp_feedforward')

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in self.ROOTS:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__path__ = []
        return m

    def exec_module(self, module):
        module.__getattr__ = lambda name: type(name, (), {})


class LabelHead(nn.Module):
    """`label_classifier` from model.py, named so the checkpoint's keys load unchanged."""

    def __init__(self, hidden=1024, num_labels=3):
        super().__init__()
        self._linear_layers = nn.ModuleList(
            [nn.Linear(hidden, hidden), nn.Linear(hidden, num_labels)])

    def forward(self, x):
        return self._linear_layers[1](nn.functional.gelu(self._linear_layers[0](x)))


def get_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ENCODER)
    tok.add_tokens(ADDITIONAL_TOKENS)
    return tok


def build_model(ckpt_path, device, output_attentions=False):
    from transformers import LongformerConfig, LongformerModel

    sys.meta_path.insert(0, _StubFinder())
    # mmap keeps the 5.25 GB file (weights plus optimizer state) off the heap; only the tensors
    # actually touched are read.
    ck = torch.load(ckpt_path, map_location='cpu', mmap=True, weights_only=False)
    sd = ck['state_dict']

    cfg = LongformerConfig.from_pretrained(ENCODER)
    cfg.output_attentions = output_attentions
    encoder = LongformerModel(cfg)  # random init; every weight is replaced below
    encoder.resize_token_embeddings(len(ADDITIONAL_TOKENS) + cfg.vocab_size)
    head = LabelHead(cfg.hidden_size, 3)

    enc_sd = {k[len('encoder.'):]: v for k, v in sd.items() if k.startswith('encoder.')}
    head_sd = {k[len('label_classifier.'):]: v
               for k, v in sd.items() if k.startswith('label_classifier.')}
    # Fail loudly on any mismatch: a silently missing tensor leaves random weights in place and
    # still produces plausible-looking predictions.
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    missing = [k for k in missing if k != 'embeddings.position_ids']
    unexpected = [k for k in unexpected if k != 'embeddings.position_ids']
    if missing or unexpected:
        raise SystemExit(f'encoder weights do not match: missing={missing[:5]} '
                         f'unexpected={unexpected[:5]}')
    head.load_state_dict(head_sd, strict=True)

    encoder.to(device).eval()
    head.to(device).eval()
    meta = {'epoch': ck.get('epoch'), 'global_step': ck.get('global_step'),
            'pl_version': ck.get('pytorch-lightning_version')}
    return encoder, head, meta


def encode(tok, claim, sentences, title):
    """`_tokenize` plus `_get_global_attention_mask` from multivers/data.py.

    `title` is "" rather than None for every document in this corpus; the original branches on
    `is not None`, so an empty title still contributes its own `</s>`. Reproduced as-is.
    """
    cited = tok.eos_token.join(sentences)
    if title is not None:
        cited = title + tok.eos_token + cited
    enc = tok(claim + tok.eos_token + cited)
    ids = torch.tensor(enc['input_ids'])
    is_special = (ids == tok.bos_token_id) | (ids == tok.eos_token_id)
    first_eos = torch.where(ids == tok.eos_token_id)[0][0]
    is_claim = torch.arange(len(ids)) < first_eos
    return {'input_ids': ids,
            'attention_mask': torch.tensor(enc['attention_mask']),
            'global_attention_mask': (is_special | is_claim).to(torch.int64)}


def read_jsonl(path):
    with open(path, encoding='utf8') as f:
        return [json.loads(x) for x in f if x.strip()]


def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', type=Path, required=True,
                   help='an export directory: claims_<split>.jsonl + corpus.jsonl')
    p.add_argument('--split', default='dev', choices=['train', 'dev', 'test'])
    p.add_argument('--checkpoint', type=Path, default=Path('checkpoints/bestModel.ckpt'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--limit', type=int, help='only the first N claims, for a smoke test')
    p.add_argument('--output-attentions', action='store_true',
                   help="pass output_attentions=True as the authors' forward does. It cannot "
                        'affect pooler_output, so it is off by default to save memory; use it to '
                        'verify that.')
    p.add_argument('--resume', action='store_true',
                   help='keep predictions already in --out and only run the remaining claims')
    a = p.parse_args()

    claims = read_jsonl(a.data_dir / f'claims_{a.split}.jsonl')
    corpus = {d['doc_id']: d for d in read_jsonl(a.data_dir / 'corpus.jsonl')}
    if a.limit:
        claims = claims[:a.limit]

    done = {}
    if a.resume and a.out.exists():
        done = {r['id']: r for r in read_jsonl(a.out)}
        print(f'resuming: {len(done)} of {len(claims)} claims already predicted', flush=True)

    device = pick_device(a.device)
    tok = get_tokenizer()
    encoder, head, meta = build_model(a.checkpoint, device, a.output_attentions)
    print(f'MultiVerS checkpoint {a.checkpoint} (epoch {meta["epoch"]}, '
          f'step {meta["global_step"]}, PL {meta["pl_version"]}) on {device}', flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    out = a.out.open('a' if done else 'w', encoding='utf8')
    skipped, t0 = [], time.time()
    for n, c in enumerate(claims, 1):
        if c['id'] in done:
            continue
        # The export gives each claim exactly one document; keep the loop general anyway so a
        # multi-document export would still predict per (claim, document) pair.
        for doc_id in c['cited_doc_ids']:
            doc = corpus[doc_id]
            gold = (next(iter(c['evidence'].values()))[0]['label']
                    if c.get('evidence') else 'NOT ENOUGH INFO')
            enc = encode(tok, c['claim'], doc['abstract'], doc.get('title'))
            if len(enc['input_ids']) > MAX_TOKENS:
                skipped.append((c['id'], doc_id, len(enc['input_ids'])))
                continue
            batch = {k: v.unsqueeze(0).to(device) for k, v in enc.items()}
            with torch.no_grad():
                pooled = encoder(**batch).pooler_output
                logits = head(pooled)
                probs = torch.softmax(logits.float(), dim=1)[0].tolist()
            out.write(json.dumps({'id': c['id'], 'doc_id': doc_id, 'gold': gold,
                                  'prediction': LABELS[int(logits.argmax(dim=1)[0])],
                                  'probs': {LABELS[i]: round(probs[i], 5) for i in range(3)},
                                  'n_tokens': len(enc['input_ids'])}) + '\n')
            out.flush()  # so an interrupted run stays resumable
        if n % 25 == 0:
            rate = (n - len(done)) / max(time.time() - t0, 1e-9)
            print(f'  {n}/{len(claims)}  {rate:.2f} claims/s  '
                  f'eta {(len(claims) - n) / max(rate, 1e-9) / 60:.0f} min', flush=True)
    out.close()

    if skipped:
        print(f'skipped {len(skipped)} instances over {MAX_TOKENS} tokens '
              f'(as the authors do): {skipped[:5]}')
    print(f'wrote {a.out}  ({len(read_jsonl(a.out))} predictions)')


if __name__ == '__main__':
    main()
