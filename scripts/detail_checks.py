#!/usr/bin/env python3
"""Rule-based checks of the details a citation states against the evidence it is judged on.

Why this exists. The label audit (results/label_audit_v1.md) found that the NLI classifier's clear
errors are narrow and checkable: a number in the citation that the evidence does not contain (or
contains as its complement: "35.6% female" against "64.4% were male"), a gene, variant or drug name
the evidence does not mention (BRCA1/2 where only BRCA1 holds), and a citation that cites a paper
for the opposite of what the sentence says ("Unlike other reports [X], ..."). A transformer reads
these softly; here they are counted exactly, and `stack_checks.py` learns how much each count
should move the classifier's probabilities.

Every feature uses only the citing sentence and the evidence sentences the pipeline retrieved, so
it is available to the deployed tool too. No model, no LLM.
"""
import math
import re

MARKER = r'<\|(?:multi_cit|other_cit|cit)\|>'
# A marker plus the reference-number fragments that often trail it: "[<|cit|>,7", "<|multi_cit|>–5".
MARKER_TAIL_RE = re.compile(r'[\[(]?\s*' + MARKER + r'\s*[\])]?(?:\s*[,–-]\s*\d+[\])]?)*')
NUM_RE = re.compile(r'(?<![A-Za-z]-)(?<![A-Za-z0-9.])(\d+(?:[.,]\d+)*)(?![A-Za-z0-9])')
# Gene, protein, variant or drug-code tokens: letters and digits together (BRCA1, N501Y, IL-6,
# ACE2, MFN2), or 3+ capitals (HIV, ASD, NLRP).
ENT_RE = re.compile(r'\b(?:(?=[\w-]*[A-Z])(?=[\w-]*\d)[A-Za-z][\w-]*|[A-Z]{3,}[a-z]?)\b')
SLASH_RE = re.compile(r'\b([A-Za-z]+)(\d+)/(\d+)\b')  # BRCA1/2 -> BRCA1 BRCA2
WORD_RE = re.compile(r'[a-z][a-z0-9-]+')

CONTRAST_RE = re.compile(r'\b(unlike|in contrast (?:to|with)|contrary to|as opposed to|'
                         r'differ(?:s|ed)? from|inconsistent with|in disagreement with)\b', re.I)
NEG_RE = re.compile(r'\b(not|no|none|neither|nor|without|fail(?:s|ed)?|lack(?:s|ed|ing)?|'
                    r'absence|absent|unable|cannot)\b', re.I)
HEDGE_RE = re.compile(r'\b(may|might|could|suggest(?:s|ed|ing)?|possibl[ey]|potential(?:ly)?|'
                      r'likely|appear(?:s|ed)?|indicate(?:s|d)?|preliminary|hypothes\w+|'
                      r'propose[sd]?|partial(?:ly)?|in part|whether|remains? (?:to be|unclear))\b', re.I)
BOOST_RE = re.compile(r'\b(demonstrat\w+|prov(?:e|ed|en|es)|establish\w*|clearly|confirm\w*|'
                      r'definitive\w*|always|all|essential|critical|key|major|largely|'
                      r'significant(?:ly)?|cause[sd]?|responsible)\b', re.I)
STOP = set('the and for with that this from are was were has have had been its their which also '
           'than such these those into between among both over under after before during within '
           'our we us in of to on by as at an or be is it not can may other others studies study '
           'shown show showed reported report previous previously recent recently'.split())


def strip_markers(claim):
    return re.sub(r'\s+', ' ', MARKER_TAIL_RE.sub(' ', claim)).strip()


def _num(s):
    s = s.replace(',', '') if re.fullmatch(r'\d{1,3}(,\d{3})+', s) else s.replace(',', '.')
    try:
        x = float(s)
    except ValueError:
        return None
    return x


def numbers(text):
    out = []
    for m in NUM_RE.finditer(text):
        x = _num(m.group(1))
        if x is not None:
            out.append(x)
    return out


def entities(text):
    text = SLASH_RE.sub(lambda m: f'{m.group(1)}{m.group(2)} {m.group(1)}{m.group(3)}', text)
    return {e.lower().replace('-', '') for e in ENT_RE.findall(text)}


def content_words(text):
    return {w for w in WORD_RE.findall(text.lower()) if w not in STOP and len(w) > 2}


def features(claim, sentences):
    """Feature dict for one citation. `claim` may contain the dataset's <|cit|> markers."""
    c = strip_markers(claim)
    premise = ' '.join(sentences)
    pnums = numbers(premise)
    # A claim number is "found" if the evidence has it, or its complement to 100 for percentages
    # ("35.6% female" / "64.4% male"), within rounding.
    def found(x):
        return any(abs(x - y) < 0.051 or (x <= 100 and abs(100 - x - y) < 0.051) for y in pnums)
    cnums = [x for x in numbers(c) if not (1900 <= x <= 2030 and float(x).is_integer())]  # years
    n_missing = sum(not found(x) for x in cnums)

    cents = entities(c)
    pflat = premise.lower().replace('-', '')
    e_missing = sum(e not in pflat for e in cents)

    cw = content_words(c)
    sw = [content_words(s) for s in sentences] or [set()]
    recall = len(cw & set().union(*sw)) / max(1, len(cw))
    per_sent = sorted((len(cw & s) / max(1, len(cw)) for s in sw), reverse=True)
    best = sentences[max(range(len(sw)), key=lambda i: len(cw & sw[i]))] if sentences else ''

    return {
        'has_num': float(bool(cnums)),
        'num_missing': float(n_missing),
        'num_missing_frac': n_missing / len(cnums) if cnums else 0.,
        'has_ent': float(bool(cents)),
        'ent_missing': float(e_missing),
        'ent_missing_frac': e_missing / len(cents) if cents else 0.,
        'contrast': float(bool(CONTRAST_RE.search(c))),
        'claim_neg': float(len(NEG_RE.findall(c))),
        'best_neg': float(len(NEG_RE.findall(best))),
        'claim_hedge': float(len(HEDGE_RE.findall(c))),
        'best_hedge': float(len(HEDGE_RE.findall(best))),
        'claim_boost': float(len(BOOST_RE.findall(c))),
        'overlap_recall': recall,
        'overlap_best': per_sent[0] if per_sent else 0.,
        'overlap_top3': sum(per_sent[:3]) / 3,
        'claim_len': math.log1p(len(cw)),
        'multi_cit': float('<|multi_cit|>' in claim),
        'other_cit': float('<|other_cit|>' in claim),
    }
