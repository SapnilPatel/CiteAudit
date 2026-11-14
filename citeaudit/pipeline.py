"""The CiteAudit pipeline: retrieve evidence from the cited paper, then classify the citation."""
import os
from dataclasses import asdict, dataclass, field

from .classify import Classifier
from .retrieval import Reranker, retrieve
from .text import clean_citing, split_sentences

DEFAULT_RERANKER = os.environ.get('CITEAUDIT_RERANKER', 'Sapnil/citeaudit-reranker')
DEFAULT_CLASSIFIER = os.environ.get('CITEAUDIT_CLASSIFIER', 'Sapnil/citeaudit-nli')
TOP_K = 20  # evidence sentences given to the classifier, as in training

EXPLANATIONS = {
    'ACCURATE': 'the cited paper supports the statement',
    'NOT_ACCURATE': 'the statement misrepresents the cited paper (contradicts it, is not '
                    'supported by it, overstates or misquotes it)',
    'NOT ENOUGH INFO': 'the cited paper does not address the statement',
}


@dataclass
class Evidence:
    sentence: str
    index: int      # position of the sentence in the cited paper
    score: float    # reranker score: higher is more relevant


@dataclass
class Result:
    label: str
    probabilities: dict
    citing_sentence: str
    statement: str  # the citing sentence as the classifier read it
    evidence: list = field(default_factory=list)

    @property
    def confidence(self):
        return self.probabilities[self.label]

    @property
    def explanation(self):
        return EXPLANATIONS[self.label]

    def to_dict(self):
        return asdict(self)


def pick_device(device='auto'):
    if device != 'auto':
        return device
    import torch
    return ('cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available() else 'cpu')


class CiteAuditor:
    """Load the two models once, then call `check` for each citation.

    Models are downloaded from the Hugging Face Hub on first use. Pass a local directory or another
    Hub id to use different ones (or set CITEAUDIT_RERANKER / CITEAUDIT_CLASSIFIER).
    """

    def __init__(self, reranker=DEFAULT_RERANKER, classifier=DEFAULT_CLASSIFIER, device='auto'):
        self.device = pick_device(device)
        self.reranker = Reranker(reranker, self.device) if isinstance(reranker, str) else reranker
        self.classifier = Classifier(classifier, self.device) if isinstance(classifier, str) \
            else classifier

    def check(self, citing_sentence, paper_text=None, sentences=None):
        """Classify one citation against the cited paper (its text, or its sentences)."""
        if sentences is None:
            if paper_text is None:
                raise ValueError('pass the cited paper as paper_text or sentences')
            sentences = split_sentences(paper_text)
        statement = clean_citing(citing_sentence)
        if not statement:
            raise ValueError('the citing sentence is empty once citations are removed')
        top = retrieve(statement, sentences, self.reranker, k=TOP_K)
        evidence = [Evidence(sentences[i], i, round(s, 4)) for i, s in top]
        probs = self.classifier([e.sentence for e in evidence], statement)
        label = max(probs, key=probs.get)
        return Result(label, {k: round(v, 4) for k, v in probs.items()}, citing_sentence,
                      statement, evidence)
