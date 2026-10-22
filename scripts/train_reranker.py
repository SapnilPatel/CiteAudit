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


def window(texts, i, neighbour_chars=200, mark=False, context=True, neighbours=1):
    """The model's text_b: `neighbours` sentences each side in order, centre delimited.

    `mark` defaults to False to match `--center-markers`, which is off in the recipe. The
    two defaults used to disagree, so a direct call to `window()` silently produced marked
    text the trained model had never seen — a diagnostic script built that way reproduced
    the pipeline's score to within 0.005 while sharing almost none of its top-20 sets.

    Neighbours are clipped here, keeping the side adjacent to the centre, rather than left
    to the tokenizer. `truncation` cuts the tail of the longer sequence, which on a long
    window removes the closing marker and can remove the centre sentence itself — the one
    the label refers to. Sentence lengths are heavy-tailed (p50 136 tokens, p99 485, max
    2810 for query + window), so this is not a rare case.

    `neighbour_chars` caps each neighbour individually, so widening `neighbours` and
    lengthening `neighbour_chars` are separate knobs: more context either way still needs
    `--max-len` raised to match, or the tokenizer silently takes back what was added.
    """
    centre = f'{CENTER_OPEN} {texts[i]} {CENTER_CLOSE}' if mark else texts[i]
    if not context or not neighbour_chars or not neighbours:
        return centre
    before, after = [], []
    for d in range(1, neighbours + 1):
        if i - d >= 0:
            s = texts[i - d]
            # `s[-n:]` with n == 0 returns the whole string, so clip by an explicit end index
            before.append(s[len(s) - neighbour_chars:])
        if i + d < len(texts):
            after.append(texts[i + d][:neighbour_chars])
    # nearest neighbour sits next to the centre on both sides
    return ' '.join(x for x in (*reversed(before), centre, *after) if x)


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


def is_monot5(name):
    return 'monot5' in str(name).lower() or '/t5' in str(name).lower()


def load_model(model_name, device, checkpoint=None, add_markers=True):
    import torch
    from transformers import (AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
                              AutoTokenizer)
    src = checkpoint or model_name
    tokzr = AutoTokenizer.from_pretrained(src)
    if add_markers and CENTER_OPEN not in tokzr.get_vocab():
        tokzr.add_special_tokens({'additional_special_tokens': [CENTER_OPEN, CENTER_CLOSE]})
    if is_monot5(model_name):
        model = AutoModelForSeq2SeqLM.from_pretrained(src)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(src, num_labels=1)
    model.resize_token_embeddings(len(tokzr))
    return tokzr, model.to(device), torch


class Scorer:
    """One scoring function for both architectures, used by training AND prediction.

    A reranker comparison is only valid if the score being optimised is the score being
    ranked by, so there is deliberately no separate inference path here:

      bert     a single logit from a sequence-pair classification head
      monot5   `log_softmax` over the {true, false} logits at the first decoder step of
               `Query: ... Document: ... Relevant:`, exactly as retrieval_harness.py scores
               the zero-shot model. Fine-tuning therefore moves the same quantity the
               published pipeline ranks by, so the two are comparable.
    """

    def __init__(self, tokzr, model, torch, model_name, max_len, device):
        self.tokzr, self.model, self.torch = tokzr, model, torch
        self.max_len, self.device = max_len, device
        self.monot5 = is_monot5(model_name)
        if self.monot5:
            self.true_id = tokzr.encode('true', add_special_tokens=False)[0]
            self.false_id = tokzr.encode('false', add_special_tokens=False)[0]
            self.start = model.config.decoder_start_token_id

    def __call__(self, queries, windows):
        """Scores for a flat list of (query, window) pairs, differentiable."""
        torch = self.torch
        if self.monot5:
            enc = self.tokzr([f'Query: {q} Document: {w} Relevant:'
                              for q, w in zip(queries, windows)],
                             padding='max_length', truncation=True, max_length=self.max_len,
                             return_tensors='pt').to(self.device)
            dec = torch.full((len(queries), 1), self.start, dtype=torch.long,
                             device=self.device)
            logits = self.model(**enc, decoder_input_ids=dec).logits[:, -1,
                                                                    [self.true_id, self.false_id]]
            return torch.log_softmax(logits.float(), dim=-1)[:, 0]
        enc = self.tokzr(queries, windows, padding='max_length', truncation='longest_first',
                         max_length=self.max_len, return_tensors='pt').to(self.device)
        return self.model(**enc).logits.squeeze(-1).float()


def fold_articles(queries, folds, seed=12345):
    """Partition the split's reference articles into `folds` groups, deterministically.

    Folds must be cut by article, never by citance: every citance of one reference ranks
    sentences of that same reference, so a citance in train and another in eval would share
    the text being ranked and the eval score would be optimistic.
    """
    arts = sorted({q['article'] for q in queries})
    rng = random.Random(seed)
    rng.shuffle(arts)
    return [set(arts[i::folds]) for i in range(folds)]


def train(a, device):
    tokzr, model, torch = load_model(a.model, device, a.init_from, a.center_markers)
    scorer = Scorer(tokzr, model, torch, a.model, a.max_len, device)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)

    queries = [q for q in load_split(a.data_dir, a.train_split) if q['gold']]
    if a.folds:
        held = fold_articles(queries, a.folds)[a.fold]
        queries = [q for q in queries if q['article'] not in held]
        print(f'fold {a.fold}/{a.folds}: training on '
              f'{len({q["article"] for q in queries})} articles, '
              f'holding out {len(held)}', flush=True)
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

    # Mid-epoch checkpointing, so a machine that loses power costs minutes rather than the whole
    # run. The epoch's shuffle is reproduced rather than stored: the RNG state is saved as it was
    # BEFORE the shuffle, so restoring it and shuffling again yields the identical order, and a
    # resumed run follows exactly the trajectory the uninterrupted one would have.
    state_path = Path(a.out) / 'train_state.pt' if a.out else None
    start_epoch, start_b, step = 0, 0, 0
    if state_path and state_path.exists() and not a.no_resume:
        st = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(st['model'])
        opt.load_state_dict(st['opt'])
        sched.load_state_dict(st['sched'])
        rng.setstate(st['rng'])
        torch.set_rng_state(st['torch_rng'])
        # dropout on MPS/CUDA draws from that device's own generator, not the CPU one, so
        # restoring only torch.set_rng_state leaves the trajectory off
        if st.get('mps_rng') is not None and torch.backends.mps.is_available():
            torch.mps.set_rng_state(st['mps_rng'])
        if st.get('cuda_rng') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st['cuda_rng'])
        start_epoch, start_b, step = st['epoch'], st['b'], st['step']
        print(f'resuming from {state_path}: epoch {start_epoch + 1}, '
              f'group {start_b}, step {step}/{total}', flush=True)

    def save_state(epoch, b, epoch_rng):
        if not state_path:
            return
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_suffix('.tmp')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict(), 'rng': epoch_rng,
                    'torch_rng': torch.get_rng_state(),
                    'mps_rng': torch.mps.get_rng_state()
                    if torch.backends.mps.is_available() else None,
                    'cuda_rng': torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None,
                    'epoch': epoch, 'b': b, 'step': step}, tmp)
        tmp.replace(state_path)  # atomic: a crash mid-write cannot corrupt the checkpoint

    t0 = time.time()
    for epoch in range(start_epoch, a.epochs):
        epoch_rng = rng.getstate()  # the state the shuffle below consumes
        rng.shuffle(groups)
        running = []
        for b in range(0, len(groups), a.groups_per_batch):
            if epoch == start_epoch and b < start_b:
                continue  # already trained on in the interrupted run
            batch = groups[b:b + a.groups_per_batch]
            # one forward over every pair in the batch, reshaped to (groups, 1 + negatives):
            # all groups are the same width, and a single forward is much faster than one per group
            pairs = [(q['query'], q['texts'], i) for q, gold, negs in batch for i in [gold] + negs]
            with amp():
                scores = scorer([q for q, _, _ in pairs],
                                [window(t, i, a.neighbour_chars, a.center_markers, a.context,
                                        a.neighbours) for _, t, i in pairs]
                                ).view(len(batch), a.negatives + 1)
                loss = (loss_fn(scores, torch.zeros(len(batch), dtype=torch.long, device=device))
                        if listwise else loss_fn(scores, bce_target[:len(batch)]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            running.append(loss.item())
            step += 1
            if a.save_every and step % a.save_every == 0:
                save_state(epoch, b + a.groups_per_batch, epoch_rng)
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
    if state_path and state_path.exists():
        state_path.unlink()  # or the next run with this --out would resume a finished one
    return tokzr, model


def predict(a, device, tokzr=None, model=None):
    import torch
    if model is None:
        tokzr, model, torch = load_model(a.model, device, a.init_from or a.out,
                                         a.center_markers)
    model.eval()
    scorer = Scorer(tokzr, model, torch, a.model, a.max_len, device)
    amp = (lambda: torch.autocast(device.split(':')[0], dtype=torch.bfloat16)) if a.amp \
        else contextlib.nullcontext
    queries = load_split(a.data_dir, a.predict)
    order = bm25_order(queries, a.cache_dir / f'bm25_{a.predict}.json')
    if a.folds:
        # evaluate only the articles this fold held out, so no training article is scored
        held = fold_articles([q for q in queries if q['gold']], a.folds)[a.fold]
        queries = [q for q in queries if q['article'] in held]
        print(f'fold {a.fold}/{a.folds}: scoring {len(queries)} citances '
              f'on {len(held)} held-out articles', flush=True)
    a.rankings_out.parent.mkdir(parents=True, exist_ok=True)
    with a.rankings_out.open('w', encoding='utf8') as f:
        for n, q in enumerate(queries, 1):
            head = order[q['id']][:a.depth]
            scores = []
            for lo in range(0, len(head), a.batch_size):
                chunk = head[lo:lo + a.batch_size]
                with torch.inference_mode(), amp():
                    scores.extend(scorer(
                        [q['query']] * len(chunk),
                        [window(q['texts'], i, a.neighbour_chars, a.center_markers,
                                a.context, a.neighbours) for i in chunk]).tolist())
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
    p.add_argument('--folds', type=int, default=0,
                   help='k-fold cross-validation over the split\'s articles (0 = off). '
                        'Both --train-split and --predict must name the same split.')
    p.add_argument('--fold', type=int, default=0, help='which fold to hold out, 0-based')
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
    p.add_argument('--neighbours', type=int, default=1,
                   help='neighbour sentences each side of the centre')
    p.add_argument('--neighbour-chars', type=int, default=200,
                   help='characters kept from each neighbour, on the side facing the centre')
    p.add_argument('--center-markers', action='store_true',
                   help='delimit the centre sentence with added [C]/[/C] tokens. Measured '
                        'equivalent to leaving them out (see the module docstring), so off '
                        'by default: in-domain training supplies the distinction by itself')
    p.add_argument('--amp', action='store_true',
                   help='bfloat16 autocast for training and inference. Worth it on CUDA '
                        '(tensor cores); measured no faster than fp32 on M2 MPS')
    p.add_argument('--save-every', type=int, default=200,
                   help='checkpoint the full training state every N steps (0 = off), so an '
                        'interrupted run resumes from there instead of from scratch')
    p.add_argument('--no-resume', action='store_true',
                   help='ignore an existing train_state.pt and start over')
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='auto')
    a = p.parse_args()
    if a.predict and not a.rankings_out:
        p.error('--predict needs --rankings-out')
    if a.no_train and not a.predict:
        p.error('--no-train leaves nothing to do; pass --predict')
    if a.folds:
        if not 0 <= a.fold < a.folds:
            p.error(f'--fold must be in [0,{a.folds})')
        if a.predict and a.predict != a.train_split:
            # otherwise the held-out articles are computed from one split and the model
            # trained on another, and the eval set is not held out at all
            p.error(f'--folds needs --predict to name --train-split ({a.train_split}), '
                    f'got {a.predict}')

    device = device_of(a.device)
    print(f'device={device}  model={a.init_from or a.model}', flush=True)
    tokzr = model = None
    if not a.no_train:
        tokzr, model = train(a, device)
    if a.predict:
        predict(a, device, tokzr, model)


if __name__ == '__main__':
    main()
