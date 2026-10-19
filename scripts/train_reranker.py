#!/usr/bin/env python3
"""Fine-tune a biomedical cross-encoder to rerank evidence sentences.

Replaces the MonoT5 stage of the paper's pipeline. BM25 stays as the first stage.

Design, and why each piece is the way it is:

candidates  The pool and the prev/next context window are imported from
            `retrieval_harness`, not reimplemented, so training sees exactly the
            candidates evaluation will rank. Windows cross chunk boundaries, which
            matters because half the corpus chunks hold a single sentence.

shapes      Every batch is padded to exactly `--max-len`. Dynamic padding makes MPS compile
            and cache a separate graph per input shape; with one shape per batch length the
            cache grew until the process was swapping 6 GB on a 16 GB machine.

input       `citance [SEP] prev center next`, and the label is a property of the *center*
            sentence alone. That label is what fixes the blur which costs zero-shot MonoT5
            its top-rank precision (R@1 0.100 → 0.063, MRR 0.323 → 0.253 once context is
            on): a neighbour's window contains the gold sentence's text and scores nearly
            as high, because MonoT5 has no notion of which sentence the window is about.

            Delimiting the centre with added `[C]`/`[/C]` tokens (`--center-markers`) was
            the obvious way to make that explicit and it turns out to be unnecessary: on
            dev, paired over articles, it moves R@20 by +0.003 [-0.010,+0.023], R@1 by
            -0.007 and MRR by -0.008. The training signal carries it; the markers do not.
            Default is off, so the checkpoint stays a plain fine-tune with no vocabulary
            change.

negatives   Sampled from BM25 ranks `--neg-from`..`--neg-to` (default 5-100) of the same
            claim, with every gold sentence of that claim removed. The top ranks are
            skipped on purpose: annotation gives a *minimal* evidence set (2.5 sentences out
            of 240), so the highest-ranked non-gold sentences are the likeliest false
            negatives. Candidates all come from the cited article already, so they are hard
            negatives by construction — no extra mining needed.

loss        Listwise: one gold against N negatives, softmax cross-entropy within the group.
            Ranking is the actual task, and a group-normalised loss is far less sensitive to
            a mislabelled negative than per-pair BCE would be.

predict     Writes `{id, ranked}` JSONL, which `retrieval_harness.py --ranker file` scores,
            so model numbers come out of the same metric code as every baseline — including
            the paper's pooled definitions. Reranks BM25's top `--depth` (default 60, the
            authors' depth and the development configuration) and appends the rest in BM25
            order so metrics beyond that depth stay defined.

Baselines to beat (zero-shot MonoT5, same depth, same context, paper's metrics):
    dev   R@20 0.585   MRR 0.300   AllGold@20 0.447
    test  R@20 0.575   MRR 0.253   AllGold@20 0.460   (paper's own pipeline: 0.529 / 0.323)
"""
import argparse
import contextlib
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval_harness import (DEFAULT_DATA, bm25_scores, load_split, rank_by,  # noqa: E402
                               with_context)

CENTER_OPEN, CENTER_CLOSE = '[C]', '[/C]'
DEFAULT_MODEL = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'


def device_of(name):
    import torch
    if name != 'auto':
        return name
    return ('cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available() else 'cpu')


def bm25_order(queries, cache_path=None):
    """BM25 ranking per query, cached because it is pure python and recomputed often."""
    if cache_path and Path(cache_path).exists():
        cached = json.loads(Path(cache_path).read_text())
        if set(cached) == {str(q['id']) for q in queries}:
            return {int(k): v for k, v in cached.items()}
    out = {}
    for n, q in enumerate(queries, 1):
        out[q['id']] = rank_by(bm25_scores(q['query'], q['texts'], 0.9, 0.4))
        if n % 500 == 0:
            print(f'  BM25 {n}/{len(queries)}', flush=True)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps({str(k): v for k, v in out.items()}))
    return out


def window(texts, i, neighbour_chars=200, mark=True, context=True):
    """The model's text_b: neighbours in order, centre delimited.

    Neighbours are clipped here, keeping the side adjacent to the centre, rather than left
    to the tokenizer. `truncation` cuts the tail of the longer sequence, which on a long
    window removes the closing marker and can remove the centre sentence itself — the one
    the label refers to. Sentence lengths are heavy-tailed (p50 136 tokens, p99 485, max
    2810 for query + window), so this is not a rare case.
    """
    centre = f'{CENTER_OPEN} {texts[i]} {CENTER_CLOSE}' if mark else texts[i]
    if not context:
        return centre
    # `s[-n:]` with n == 0 returns the whole string, so clip by an explicit end index
    prev = texts[i - 1][len(texts[i - 1]) - neighbour_chars:] if i > 0 and neighbour_chars else ''
    nxt = texts[i + 1][:neighbour_chars] if i + 1 < len(texts) and neighbour_chars else ''
    return ' '.join(x for x in (prev, centre, nxt) if x)


def build_groups(queries, order, n_neg, lo, hi, rng):
    """One group per gold sentence: (query, gold index, [negative indices])."""
    groups, skipped = [], 0
    for q in queries:
        gold_idx = {i for i, k in enumerate(q['keys']) if k in q['gold']}
        if not gold_idx:
            skipped += 1
            continue
        pool = [i for i in order[q['id']][lo:hi] if i not in gold_idx]
        if len(pool) < n_neg:  # tiny articles: fall back to anything non-gold
            pool = [i for i in order[q['id']] if i not in gold_idx]
        if len(pool) < n_neg:
            skipped += 1
            continue
        for g in sorted(gold_idx):
            groups.append((q, g, rng.sample(pool, n_neg)))
    return groups, skipped


def encode(tokzr, query, texts, idxs, max_len, device, mark=True, context=True, nchars=200):
    """One query against several candidate windows (prediction path)."""
    enc = tokzr([query] * len(idxs),
                [window(texts, i, nchars, mark, context) for i in idxs],
                padding='max_length', truncation='longest_first', max_length=max_len,
                return_tensors='pt')
    return {k: v.to(device) for k, v in enc.items()}


def load_model(model_name, device, checkpoint=None, add_markers=True):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    src = checkpoint or model_name
    tokzr = AutoTokenizer.from_pretrained(src)
    if add_markers and CENTER_OPEN not in tokzr.get_vocab():
        tokzr.add_special_tokens({'additional_special_tokens': [CENTER_OPEN, CENTER_CLOSE]})
    model = AutoModelForSequenceClassification.from_pretrained(src, num_labels=1)
    model.resize_token_embeddings(len(tokzr))
    return tokzr, model.to(device), torch


def train(a, device):
    tokzr, model, torch = load_model(a.model, device, a.init_from, a.center_markers)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)

    queries = [q for q in load_split(a.data_dir, a.train_split) if q['gold']]
    print(f'{a.train_split}: {len(queries)} citances with evidence', flush=True)
    order = bm25_order(queries, a.cache_dir / f'bm25_{a.train_split}.json')
    groups, skipped = build_groups(queries, order, a.negatives, a.neg_from, a.neg_to, rng)
    print(f'{len(groups)} training groups (1 gold + {a.negatives} negatives), '
          f'{skipped} citances skipped', flush=True)

    steps_per_epoch = math.ceil(len(groups) / a.groups_per_batch)
    total = steps_per_epoch * a.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    warmup = int(a.warmup * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(warmup, 1) if s < warmup else
        max(0., (total - s) / max(total - warmup, 1)))
    listwise = a.loss == 'listwise'
    loss_fn = torch.nn.CrossEntropyLoss() if listwise else torch.nn.BCEWithLogitsLoss()
    if not listwise:  # 1 for the gold at position 0 of each group, 0 for its negatives
        bce_target = torch.zeros(a.groups_per_batch, a.negatives + 1, device=device)
        bce_target[:, 0] = 1.

    model.train()
    amp = (lambda: torch.autocast(device.split(':')[0], dtype=torch.bfloat16)) if a.amp \
        else contextlib.nullcontext
    step, t0 = 0, time.time()
    for epoch in range(a.epochs):
        rng.shuffle(groups)
        running = []
        for b in range(0, len(groups), a.groups_per_batch):
            batch = groups[b:b + a.groups_per_batch]
            # one forward over every pair in the batch, reshaped to (groups, 1 + negatives):
            # all groups are the same width, and a single forward is much faster than one per group
            pairs = [(q['query'], q['texts'], i) for q, gold, negs in batch for i in [gold] + negs]
            enc = tokzr([q for q, _, _ in pairs],
                        [window(t, i, a.neighbour_chars, a.center_markers, a.context)
                         for _, t, i in pairs],
                        padding='max_length', truncation='longest_first',
                        max_length=a.max_len, return_tensors='pt').to(device)
            with amp():
                scores = model(**enc).logits.view(len(batch), a.negatives + 1).float()
                loss = (loss_fn(scores, torch.zeros(len(batch), dtype=torch.long, device=device))
                        if listwise else loss_fn(scores, bce_target[:len(batch)]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            running.append(loss.item())
            step += 1
            if step % a.log_every == 0:
                rate = step / (time.time() - t0)
                print(f'  epoch {epoch + 1} step {step}/{total} '
                      f'loss {sum(running[-a.log_every:]) / a.log_every:.4f} '
                      f'lr {sched.get_last_lr()[0]:.2e} '
                      f'{rate:.2f} steps/s eta {(total - step) / rate / 60:.0f} min', flush=True)
        print(f'epoch {epoch + 1}: mean loss {sum(running) / len(running):.4f}', flush=True)
        if a.out:
            a.out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(a.out)
            tokzr.save_pretrained(a.out)
            print(f'saved to {a.out}', flush=True)
    return tokzr, model


def predict(a, device, tokzr=None, model=None):
    import torch
    if model is None:
        tokzr, model, torch = load_model(a.model, device, a.init_from or a.out,
                                         a.center_markers)
    model.eval()
    amp = (lambda: torch.autocast(device.split(':')[0], dtype=torch.bfloat16)) if a.amp \
        else contextlib.nullcontext
    queries = load_split(a.data_dir, a.predict)
    order = bm25_order(queries, a.cache_dir / f'bm25_{a.predict}.json')
    a.rankings_out.parent.mkdir(parents=True, exist_ok=True)
    with a.rankings_out.open('w', encoding='utf8') as f:
        for n, q in enumerate(queries, 1):
            head = order[q['id']][:a.depth]
            scores = []
            for lo in range(0, len(head), a.batch_size):
                chunk = head[lo:lo + a.batch_size]
                enc = encode(tokzr, q['query'], q['texts'], chunk, a.max_len, device,
                             a.center_markers, a.context, a.neighbour_chars)
                with torch.inference_mode(), amp():
                    scores.extend(model(**enc).logits.squeeze(-1).float().tolist())
            ranked = [head[i] for i in rank_by(scores)] + order[q['id']][a.depth:]
            f.write(json.dumps({'id': q['id'], 'article': q['article'],
                                'gold': sorted(q['gold']),
                                'ranked': [q['keys'][i] for i in ranked]}) + '\n')
            if n % 50 == 0:
                print(f'  predict {n}/{len(queries)}', flush=True)
    print(f'wrote {a.rankings_out}\nscore it with:\n'
          f'  python scripts/retrieval_harness.py --split {a.predict} --ranker file '
          f'--rankings-in {a.rankings_out}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--init-from', type=Path, help='resume from / predict with this checkpoint')
    p.add_argument('--out', type=Path, default=Path('checkpoints/reranker'))
    p.add_argument('--cache-dir', type=Path, default=Path('results/cache'))
    p.add_argument('--train-split', default='train', choices=['train', 'dev', 'test'])
    p.add_argument('--no-train', action='store_true', help='predict only')
    p.add_argument('--predict', choices=['train', 'dev', 'test'],
                   help='split to rerank after training')
    p.add_argument('--rankings-out', type=Path)
    p.add_argument('--depth', type=int, default=60, help="BM25 candidates reranked (authors' depth)")
    p.add_argument('--negatives', type=int, default=7, help='negatives per group')
    p.add_argument('--neg-from', type=int, default=5, help='first BM25 rank sampled as negative')
    p.add_argument('--neg-to', type=int, default=100)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--groups-per-batch', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=32, help='pairs per forward at predict time')
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--warmup', type=float, default=0.1)
    p.add_argument('--max-len', type=int, default=192,
                   help='citance + a 3-sentence window is ~120 tokens; 192 leaves headroom')
    p.add_argument('--loss', default='listwise', choices=['listwise', 'bce'],
                   help='listwise: softmax cross-entropy inside the group. bce: independent '
                        'per-pair sigmoid, the ablation for whether group competition matters')
    p.add_argument('--no-context', dest='context', action='store_false',
                   help='ablation: score the candidate sentence alone, no neighbours')
    p.add_argument('--neighbour-chars', type=int, default=200,
                   help='characters kept from each neighbour, on the side facing the centre')
    p.add_argument('--center-markers', action='store_true',
                   help='delimit the centre sentence with added [C]/[/C] tokens. Measured '
                        'equivalent to leaving them out (see the module docstring), so off '
                        'by default: in-domain training supplies the distinction by itself')
    p.add_argument('--amp', action='store_true',
                   help='bfloat16 autocast for training and inference. Worth it on CUDA '
                        '(tensor cores); measured no faster than fp32 on M2 MPS')
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='auto')
    a = p.parse_args()
    if a.predict and not a.rankings_out:
        p.error('--predict needs --rankings-out')
    if a.no_train and not a.predict:
        p.error('--no-train leaves nothing to do; pass --predict')

    device = device_of(a.device)
    print(f'device={device}  model={a.init_from or a.model}', flush=True)
    tokzr = model = None
    if not a.no_train:
        tokzr, model = train(a, device)
    if a.predict:
        predict(a, device, tokzr, model)


if __name__ == '__main__':
    main()
