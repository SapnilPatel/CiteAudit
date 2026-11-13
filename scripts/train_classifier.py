#!/usr/bin/env python3
"""Train a 3-way citation classifier on retrieved evidence, as a lighter stand-in for MultiVerS.

Why this exists. The released MultiVerS checkpoint is frozen to the evidence distribution it was
trained on: feeding it better retrieval, or even perfect evidence, does not raise its F1 (findings
§10, and on test: 0.505 vs 0.524 macro-F1, p = 0.631). Beating the paper therefore means training
a classifier on the evidence it will actually see. Retraining MultiVerS (Longformer-large, 4096
tokens) needs a CUDA GPU; a BERT-base model trains on an Apple M4 in minutes per epoch, needs no
5 GB checkpoint at prediction time, and is what an end user would want to download anyway.

The input does not fit in 512 tokens. A citing sentence plus 20 evidence sentences is a median of
926 BiomedBERT tokens on dev, and every citation exceeds 512. So the evidence is packed, in the
export's order, into windows that each fit alongside the citing sentence:

  window     `[CLS] citing sentence [SEP] sent_i ... sent_j [SEP]`, at most --max-len tokens; a
             sentence never straddles two windows, and one longer than a window is truncated.
  pooling    the [CLS] vectors of a citation's windows are mean- and max-pooled, concatenated,
             and passed through a linear head. Max lets one decisive window carry the label;
             mean keeps the cases where evidence is spread thin.
  loss       cross-entropy, optionally class-weighted: NOT ENOUGH INFO is 8% of train and the
             target metric is macro-F1, so --class-weight sqrt is the default.

Input is an export directory from `export_for_multivers.py` (claims_<split>.jsonl + corpus.jsonl,
one synthetic document per citation), and predictions are written in `predict_multivers.py`'s
format, so `score_multivers.py` scores both models the same way and pairs them with McNemar.
The checkpoint kept is the epoch with the best dev macro-F1; test is never looked at to choose it.

    python scripts/train_classifier.py --train data_train/target/paper_20_train \\
        --dev data_train/target/paper_20 --out checkpoints/classifier \\
        --predict test=data_train/target/paper_20_test --predict test=data_train/target/ours_20_test
"""
import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

LABELS = ['NOT_ACCURATE', 'NOT ENOUGH INFO', 'ACCURATE']
NEI = 'NOT ENOUGH INFO'
# The nine annotation labels and the paper's collapse to three (INDIRECT_NOT_REVIEW, which the
# paper does not list, as a variant of INDIRECT). With --labels 9 the model is trained on these
# and its probabilities are summed per group, so it is still scored as a 3-way classifier.
FINE = ['ACCURATE', 'INDIRECT', 'INDIRECT_NOT_REVIEW', 'CONTRADICT', 'NOT_SUBSTANTIATE',
        'OVERSIMPLIFY', 'MISQUOTE', 'ETIQUETTE', 'IRRELEVANT']
COLLAPSE = {'ACCURATE': 'ACCURATE', 'INDIRECT': 'ACCURATE', 'INDIRECT_NOT_REVIEW': 'ACCURATE',
            'CONTRADICT': 'NOT_ACCURATE', 'NOT_SUBSTANTIATE': 'NOT_ACCURATE',
            'OVERSIMPLIFY': 'NOT_ACCURATE', 'MISQUOTE': 'NOT_ACCURATE',
            'ETIQUETTE': 'NOT_ACCURATE', 'IRRELEVANT': NEI, 'NOT_ACCURATE': 'NOT_ACCURATE', NEI: NEI}
DEFAULT_MODEL = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'


def read_jsonl(path):
    with open(path, encoding='utf8') as f:
        return [json.loads(x) for x in f if x.strip()]


def gold_of(claim):
    """The 3-way gold label as the export encodes it: NEI is an empty evidence dict."""
    ev = claim.get('evidence')
    return next(iter(ev.values()))[0]['label'] if ev else NEI


def load_export(d, split, fine=None, context=None):
    """(id, doc_id, claim text, evidence sentences, 3-way gold, training target) per citation.

    The target is the 3-way gold, or with `fine` (claim id -> annotation label) the fine label,
    which must collapse to the 3-way gold: a mismatch means the two files disagree on the data.
    With `context` (claim id -> text), that text replaces the citing sentence as the model's
    first segment; it contains the citing sentence, with its citation marker, and its neighbours.
    """
    claims = read_jsonl(Path(d) / f'claims_{split}.jsonl')
    corpus = {x['doc_id']: x['abstract'] for x in read_jsonl(Path(d) / 'corpus.jsonl')}
    rows = []
    for c in claims:
        if len(c['cited_doc_ids']) != 1:
            raise SystemExit(f'{d}: claim {c["id"]} has {len(c["cited_doc_ids"])} documents; '
                             f'expected the one-document-per-claim export')
        doc = c['cited_doc_ids'][0]
        target = fine[str(c['id'])] if fine else gold_of(c)
        if COLLAPSE[target] != gold_of(c):
            raise SystemExit(f'{d}: claim {c["id"]} is {gold_of(c)} in the export but {target} '
                             f'in the fine labels')
        text = context[str(c['id'])] if context else c['claim']
        rows.append((c['id'], doc, text, corpus[doc], gold_of(c), target))
    return rows


def fold_articles(articles, folds, seed=12345):
    """Partition reference articles into `folds` groups, as train_reranker.py does.

    Folds are cut by article, never by citation: citations of one article share its text, so a
    split within an article would let the held-out fold see what it is tested on.
    """
    arts = sorted(set(articles))
    random.Random(seed).shuffle(arts)
    return [set(arts[i::folds]) for i in range(folds)]


def windows(tok, claim, sents, max_len, max_claim):
    """Token-id windows `[CLS] claim [SEP] sentences [SEP]`, packed greedily in export order."""
    c = tok(claim, add_special_tokens=False)['input_ids'][:max_claim]
    room = max_len - len(c) - 3
    out, cur = [], []
    for s in sents:
        ids = tok(s, add_special_tokens=False)['input_ids'][:room]
        if cur and len(cur) + len(ids) > room:
            out.append(cur)
            cur = []
        cur = cur + ids
    if cur or not out:
        out.append(cur)
    return [[tok.cls_token_id] + c + [tok.sep_token_id] + w + [tok.sep_token_id] for w in out]


def encode_rows(tok, rows, a):
    return [(r, windows(tok, r[2], r[3], a.max_len, a.max_claim)) for r in rows]


def build(a, torch):
    from transformers import AutoModel

    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(a.model)
            h = self.encoder.config.hidden_size
            self.dropout = torch.nn.Dropout(0.1)
            self.head = torch.nn.Linear(2 * h, len(a.targets))

        def forward(self, ids, mask, owner, n):
            # Pool in fp32 even under autocast: the -1e4 fill and the division are not fp16-safe.
            cls = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0].float()
            # Pool windows back to their citation: owner[i] is the citation of window i.
            idx = owner.unsqueeze(1).expand(-1, cls.size(1))
            mean = torch.zeros(n, cls.size(1), device=cls.device, dtype=cls.dtype)
            mean = mean.index_add(0, owner, cls) / torch.bincount(owner, minlength=n).unsqueeze(1)
            mx = torch.full((n, cls.size(1)), -1e4, device=cls.device, dtype=cls.dtype)
            mx = mx.scatter_reduce(0, idx, cls, reduce='amax')
            return self.head(self.dropout(torch.cat([mean, mx], dim=1)))

    return Classifier()


def collate(batch, pad_id, torch, device):
    flat, owner = [], []
    for i, (_, ws) in enumerate(batch):
        flat += ws
        owner += [i] * len(ws)
    width = max(len(w) for w in flat)
    ids = torch.tensor([w + [pad_id] * (width - len(w)) for w in flat], device=device)
    mask = torch.tensor([[1] * len(w) + [0] * (width - len(w)) for w in flat], device=device)
    return ids, mask, torch.tensor(owner, device=device), len(batch)


def predict(model, data, a, torch, device, pad_id):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(data), a.eval_batch):
            batch = data[i:i + a.eval_batch]
            with amp(torch, device, a):
                logits = model(*collate(batch, pad_id, torch, device))
            probs = torch.softmax(logits.float(), dim=1)
            for (r, ws), p in zip(batch, probs.tolist()):
                p3 = {c: sum(x for t, x in zip(a.targets, p) if COLLAPSE[t] == c) for c in LABELS}
                row = {'id': r[0], 'doc_id': r[1], 'gold': r[4],
                       'prediction': max(LABELS, key=p3.get),
                       'probs': {c: round(x, 5) for c, x in p3.items()}, 'n_windows': len(ws)}
                if a.targets != LABELS:
                    row['fine_probs'] = {t: round(x, 5) for t, x in zip(a.targets, p)}
                out.append(row)
    model.train()
    return out


def amp(torch, device, a):
    """fp16 autocast on CUDA when --amp is set; a no-op context everywhere else."""
    import contextlib
    if a.amp and device.type == 'cuda':
        return torch.autocast('cuda', dtype=torch.float16)
    return contextlib.nullcontext()


def macro_f1(rows):
    f1s = []
    for c in LABELS:
        tp = sum(r['gold'] == c and r['prediction'] == c for r in rows)
        fp = sum(r['gold'] != c and r['prediction'] == c for r in rows)
        fn = sum(r['gold'] == c and r['prediction'] != c for r in rows)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.)
    return sum(f1s) / len(f1s), sum(r['gold'] == r['prediction'] for r in rows) / len(rows)


def pick_device(torch, name):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf8')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train', type=Path, required=True, help='export directory with claims_train')
    p.add_argument('--dev', type=Path, required=True, help='export directory with claims_dev')
    p.add_argument('--out', type=Path, required=True, help='checkpoint directory')
    p.add_argument('--predict', action='append', default=[], metavar='SPLIT=DIR',
                   help='after training, predict this export with the best checkpoint; repeatable')
    p.add_argument('--pred-dir', type=Path, default=Path('results/classification/classifier'))
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--batch', type=int, default=4, help='citations per optimizer step')
    p.add_argument('--eval-batch', type=int, default=8)
    p.add_argument('--max-len', type=int, default=512)
    p.add_argument('--max-claim', type=int, default=128)
    p.add_argument('--class-weight', default='sqrt', choices=['none', 'sqrt', 'inverse'])
    p.add_argument('--labels', type=int, default=3, choices=[3, 9],
                   help='train on the 3 collapsed labels, or on the 9 annotation labels')
    p.add_argument('--fine-labels', type=Path, default=Path('data_train/fine_labels'),
                   help='directory of <split>.json files mapping claim id to annotation label')
    p.add_argument('--context', default='none', choices=['none', 'window', 'paragraph'],
                   help='what the model reads in place of the citing sentence: the sentence alone, '
                        'with its neighbours, or with its paragraph (build_context.py)')
    p.add_argument('--context-dir', type=Path, default=Path('data_train/context'))
    p.add_argument('--folds', type=int, default=0,
                   help='k-fold cross-validation over the training articles (0 = off): train on '
                        'the other folds, choose the epoch on --dev, and write predictions for '
                        'the held-out fold to <pred-dir>/oof.jsonl')
    p.add_argument('--fold', type=int, default=0, help='which fold to hold out, 0-based')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='auto')
    p.add_argument('--amp', action='store_true',
                   help='fp16 mixed precision on CUDA, about 2x faster on a T4; ignored elsewhere')
    p.add_argument('--limit', type=int, help='only the first N citations of each split, for a smoke test')
    a = p.parse_args()
    a.targets = FINE if a.labels == 9 else LABELS

    def fine(split):
        if a.labels == 3:
            return None
        return json.loads((a.fine_labels / f'{split}.json').read_text(encoding='utf8'))

    def meta(split):
        return json.loads((a.context_dir / f'{split}.json').read_text(encoding='utf8'))

    def context(split):
        return None if a.context == 'none' else {k: v[a.context] for k, v in meta(split).items()}

    import torch
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = pick_device(torch, a.device)
    tok = AutoTokenizer.from_pretrained(a.model)

    train = encode_rows(tok, load_export(a.train, 'train', fine('train'), context('train'))[:a.limit], a)
    dev = encode_rows(tok, load_export(a.dev, 'dev', fine('dev'), context('dev'))[:a.limit], a)
    heldout = []
    if a.folds:
        article = {int(k): v['article'] for k, v in meta('train').items()}
        held = fold_articles(article.values(), a.folds)[a.fold]
        heldout = [x for x in train if article[x[0][0]] in held]
        train = [x for x in train if article[x[0][0]] not in held]
        print(f'fold {a.fold}/{a.folds}: training on {len(train)} citations, holding out '
              f'{len(heldout)} from {len(held)} articles', flush=True)
    counts = Counter(r[5] for r, _ in train)
    n_win = [len(w) for _, w in train]
    print(f'train {len(train)} citations {dict(counts)}, windows per citation mean '
          f'{sum(n_win) / len(n_win):.2f} max {max(n_win)}; dev {len(dev)}; {device}', flush=True)

    if a.class_weight == 'none':
        w = [1.] * 3
    else:
        power = 0.5 if a.class_weight == 'sqrt' else 1.
        w = [(len(train) / (len(a.targets) * max(counts[c], 1))) ** power for c in a.targets]
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))
    print('class weights ' + ', '.join(f'{c} {x:.2f}' for c, x in zip(a.targets, w)), flush=True)

    model = build(a, torch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(train) / a.batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    y = {c: i for i, c in enumerate(a.targets)}
    # A disabled scaler passes every call straight through, so one code path serves fp32 too.
    scaler = torch.amp.GradScaler('cuda', enabled=a.amp and device.type == 'cuda')

    best, log = -1., []
    for ep in range(1, a.epochs + 1):
        random.shuffle(train)
        t0, total = time.time(), 0.
        for i in range(0, len(train), a.batch):
            batch = train[i:i + a.batch]
            gold = torch.tensor([y[r[5]] for r, _ in batch], device=device)
            with amp(torch, device, a):
                logits = model(*collate(batch, tok.pad_token_id, torch, device))
            loss = loss_fn(logits.float(), gold)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad()
            total += loss.item() * len(batch)
        preds = predict(model, dev, a, torch, device, tok.pad_token_id)
        f1, acc = macro_f1(preds)
        log.append({'epoch': ep, 'train_loss': total / len(train), 'dev_macro_f1': f1,
                    'dev_accuracy': acc, 'minutes': (time.time() - t0) / 60})
        print(f'epoch {ep}: loss {total / len(train):.4f}  dev macro-F1 {f1:.3f}  '
              f'accuracy {acc:.3f}  ({(time.time() - t0) / 60:.1f} min)', flush=True)
        if f1 > best:
            best = f1
            a.out.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), a.out / 'model.pt')
            tok.save_pretrained(a.out)
            write(a.pred_dir / 'dev.jsonl', preds)

    (a.out / 'config.json').write_text(json.dumps(
        {'model': a.model, 'labels': a.targets, 'max_len': a.max_len, 'max_claim': a.max_claim,
         'best_dev_macro_f1': best, 'args': {k: str(v) for k, v in vars(a).items()},
         'log': log}, indent=2))

    if a.predict or heldout:
        model.load_state_dict(torch.load(a.out / 'model.pt', map_location=device))
    if heldout:
        preds = predict(model, heldout, a, torch, device, tok.pad_token_id)
        write(a.pred_dir / 'oof.jsonl', preds)
        f1, acc = macro_f1(preds)
        print(f'held-out fold {a.fold}: macro-F1 {f1:.3f}  accuracy {acc:.3f}', flush=True)
    if a.predict:
        for spec in a.predict:
            split, d = spec.split('=', 1)
            rows = encode_rows(tok, load_export(d, split, fine(split), context(split))[:a.limit], a)
            preds = predict(model, rows, a, torch, device, tok.pad_token_id)
            dest = a.pred_dir / f'{Path(d).name}.jsonl'
            write(dest, preds)
            f1, acc = macro_f1(preds)
            print(f'{d} ({split}): macro-F1 {f1:.3f}  accuracy {acc:.3f}  -> {dest}')


if __name__ == '__main__':
    main()
