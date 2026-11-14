"""Evidence retrieval: BM25 over the cited paper's sentences, then a cross-encoder reranker.

These are the research pipeline's exact settings (scripts/retrieval_harness.py and
scripts/train_reranker.py): BM25 with k1 0.9 and b 0.4 keeps 60 candidates, and the reranker scores
each one with a neighbour on each side (up to 200 characters each) at 192 tokens.
tests/test_package.py checks that this copy stays identical to the research code.
"""
import math
import re
from collections import Counter

BM25_K1, BM25_B = 0.9, 0.4
DEPTH = 60
NEIGHBOUR_CHARS = 200
RERANKER_MAX_LEN = 192


def tok(s):
    return re.findall(r'[a-z0-9]+', s.lower())


def bm25_scores(query, texts, k1=BM25_K1, b=BM25_B):
    docs = [tok(t) for t in texts]
    n = len(docs)
    avg = max(sum(map(len, docs)) / max(n, 1), 1)
    df = Counter(w for d in docs for w in set(d))
    qs = set(tok(query))
    idf = {w: math.log(1 + (n - df[w] + .5) / (df[w] + .5)) for w in qs if w in df}
    out = []
    for d in docs:
        f = Counter(d)
        out.append(sum(idf[w] * f[w] * (k1 + 1) / (f[w] + k1 * (1 - b + b * len(d) / avg))
                       for w in idf if f[w]))
    return out


def rank_by(scores):
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def window(texts, i, neighbour_chars=NEIGHBOUR_CHARS):
    """Sentence i with the tail of the previous sentence and the head of the next one.

    The previous sentence is clipped with `s[len(s) - neighbour_chars:]` exactly as in training.
    For a sentence of 100 to 200 characters that index is negative and keeps only the last
    `neighbour_chars - len(s)` characters, not the whole sentence. The reranker was trained on
    those windows, so it is kept here.
    """
    before = texts[i - 1][len(texts[i - 1]) - neighbour_chars:] if i > 0 else ''
    after = texts[i + 1][:neighbour_chars] if i + 1 < len(texts) else ''
    return ' '.join(x for x in (before, texts[i], after) if x)


class Reranker:
    def __init__(self, name, device, batch_size=32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.device, self.batch_size = torch, device, batch_size
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name, num_labels=1, torch_dtype=torch.float32).to(device).eval()

    def __call__(self, query, passages):
        scores = []
        for lo in range(0, len(passages), self.batch_size):
            chunk = passages[lo:lo + self.batch_size]
            enc = self.tok([query] * len(chunk), chunk, padding=True, truncation='longest_first',
                           max_length=RERANKER_MAX_LEN, return_tensors='pt').to(self.device)
            with self.torch.inference_mode():
                scores.extend(self.model(**enc).logits.squeeze(-1).float().tolist())
        return scores


def retrieve(query, sentences, reranker, k=20, depth=DEPTH):
    """Indices of the k best evidence sentences, best first, with their reranker scores."""
    if not sentences:
        return []
    head = rank_by(bm25_scores(query, sentences))[:depth]
    scores = reranker(query, [window(sentences, i) for i in head])
    order = rank_by(scores)
    return [(head[j], scores[j]) for j in order[:k]]
