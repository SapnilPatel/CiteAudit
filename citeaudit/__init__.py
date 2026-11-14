"""CiteAudit: check whether a citation in a biomedical paper says what the cited paper says.

    from citeaudit import CiteAuditor
    auditor = CiteAuditor()
    result = auditor.check("Statins reduce mortality in all adults [12].", paper_text=open("paper.txt").read())
    print(result.label, result.probabilities)
    for ev in result.evidence[:3]:
        print(ev.score, ev.sentence)
"""
from .pipeline import CiteAuditor, Evidence, Result

__all__ = ['CiteAuditor', 'Evidence', 'Result']
__version__ = '0.1.0'
