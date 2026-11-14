"""Text handling: sentence splitting of the cited paper and cleaning of the citing sentence."""
import re

# Abbreviations that end in a full stop without ending a sentence. Lowercased, without the dot.
ABBREVIATIONS = {
    'al', 'e.g', 'i.e', 'eg', 'ie', 'etc', 'fig', 'figs', 'vs', 'ca', 'approx', 'resp', 'ref',
    'refs', 'eq', 'eqs', 'no', 'nos', 'vol', 'tab', 'suppl', 'dr', 'prof', 'mr', 'mrs', 'ms', 'st',
    'sp', 'spp', 'cf', 'viz', 'inc', 'ltd', 'co', 'corp', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul',
    'aug', 'sep', 'sept', 'oct', 'nov', 'dec', 'min', 'max', 'mol', 'wt', 'conc', 'temp', 'u.s',
}
_BOUNDARY = re.compile(r'([.!?])(["\')\]]*)\s+(?=["\'(\[]?[A-Z0-9])')


def split_sentences(text):
    """Split running text into sentences, keeping abbreviations, initials and decimals intact.

    Paragraph breaks (blank lines) always end a sentence. This approximates the sentence split
    of the training data, which came pre-split, closely enough for retrieval.
    """
    out = []
    for para in re.split(r'\n\s*\n', text):
        para = re.sub(r'\s+', ' ', para).strip()
        if not para:
            continue
        start = 0
        for m in _BOUNDARY.finditer(para):
            if m.group(1) == '.':
                word = re.search(r'(\S+)$', para[start:m.start()])
                word = word.group(1).lower().lstrip('([') if word else ''
                # "et al.", "Fig.", "No. 5", "e.g.", "U.S.". A lone capital ("drug A.", "vitamin
                # D.") ends a sentence: in a paper's body that is far commoner than an initial.
                if word in ABBREVIATIONS or re.fullmatch(r'(?:[a-z]\.)+[a-z]', word):
                    continue
            out.append(para[start:m.end(2)].strip())
            start = m.end()
        tail = para[start:].strip()
        if tail:
            out.append(tail)
    return [s for s in out if re.search(r'[A-Za-z]{2}', s)]


# Training data marks citations as <|cit|>, <|multi_cit|>, <|other_cit|>, with any brackets.
_MARKER_RE = re.compile(r'[\[(]?\s*<\|(multi_cit|other_cit|cit)\|>\s*[\])]?')
# Numeric citations: [12], [3, 4], [5-8], [5–8], [3,5-7]
_NUMERIC_RE = re.compile(r'\s*\[\s*\d+(?:\s*[-–—,;]\s*\d+)*\s*\]')
# Author-year citations: (Smith et al., 2020), (Smith and Lee 2019; Wang 2018a), (Smith, 2020)
_AUTHOR_YEAR_RE = re.compile(
    r'\s*\((?:[^()]*?\b(?:et al\.?|and|&)?[^()]*?,?\s*(?:19|20)\d{2}[a-z]?\s*[;,]?\s*)+\)')


def clean_citing(sentence):
    """The citing sentence as the classifier was trained to read it: citations removed."""
    s = _MARKER_RE.sub(' ', sentence)
    s = _NUMERIC_RE.sub('', s)
    s = _AUTHOR_YEAR_RE.sub('', s)
    return re.sub(r'\s+([.,;:])', r'\1', re.sub(r'\s+', ' ', s)).strip()
