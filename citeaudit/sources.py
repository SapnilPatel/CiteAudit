"""Loading the cited paper: a text file, a PDF, or an open-access paper from PubMed Central."""
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from .text import split_sentences

EUROPE_PMC = 'https://www.ebi.ac.uk/europepmc/webservices/rest/{}/fullTextXML'


def from_file(path):
    """Sentences of a .txt/.md file, or of a .pdf (needs `pip install citeaudit[pdf]`)."""
    path = Path(path)
    if path.suffix.lower() == '.pdf':
        try:
            from pypdf import PdfReader
        except ImportError:
            raise SystemExit('reading PDFs needs pypdf: pip install "citeaudit[pdf]"')
        pages = [p.extract_text() or '' for p in PdfReader(str(path)).pages]
        # PDF lines break mid-sentence: join hyphenated and wrapped lines, keep blank lines.
        text = '\n\n'.join(re.sub(r'-\n(?=[a-z])', '', p) for p in pages)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        return _drop_references(split_sentences(text))
    return split_sentences(path.read_text(encoding='utf8', errors='replace'))


def _drop_references(sentences):
    """Cut a PDF's reference list, which would otherwise flood retrieval with citation strings."""
    for i in range(len(sentences) - 1, len(sentences) // 2, -1):
        if re.fullmatch(r'(references|bibliography|literature cited)\b.*', sentences[i], re.I):
            return sentences[:i]
    return sentences


def from_pmc(pmcid, timeout=30):
    """Title, abstract and body of an open-access PubMed Central paper, via Europe PMC."""
    pmcid = pmcid.strip().upper()
    if not pmcid.startswith('PMC'):
        pmcid = 'PMC' + pmcid
    req = urllib.request.Request(EUROPE_PMC.format(pmcid), headers={'User-Agent': 'citeaudit'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            root = ET.fromstring(r.read())
    except Exception as e:
        raise SystemExit(f'could not fetch {pmcid} from Europe PMC ({e}). Only open-access '
                         f'papers have full text there; otherwise pass the paper as a file.')
    return sentences_from_jats(root)


def sentences_from_jats(root):
    """Sentences from a JATS XML article (the format of PMC and Europe PMC)."""
    paras = []
    title = root.find('.//article-meta//article-title')
    if title is not None:
        paras.append(''.join(title.itertext()))
    for part in [*root.findall('.//article-meta/abstract'), *root.findall('.//body')]:
        _collect(part, paras)
    return split_sentences('\n\n'.join(p.strip() for p in paras if p.strip()))


def _collect(el, paras):
    """Paragraphs and section titles in document order, skipping tables and reference lists."""
    if el.tag in ('table-wrap', 'ref-list'):
        return
    if el.tag in ('p', 'title'):
        paras.append(''.join(el.itertext()))
        return
    for child in el:
        _collect(child, paras)
