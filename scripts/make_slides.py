#!/usr/bin/env python3
"""Build the two-slide progress deck as a local .pptx.

Mirrors the published artifact (claude.ai/artifact/V7XThXktANxs3BSqKPGKAi) so the two stay
in step. Lay out in CSS pixels on a 1920x1080 canvas exactly as the artifact's slides do,
then scale once into a standard 13.333x7.5in widescreen slide: 1px = 6350 EMU. Font sizes
convert at half their pixel value (60px title -> 30pt), which keeps every element the same
fraction of slide height it occupies in the browser.

PowerPoint does no layout, so every box here carries an explicit height. The heights were
computed from the text: characters per line at ~0.52x the font size for Helvetica Neue,
then lines x size x line-height plus padding. Change a string and its box may need a new
height — the comment on each block says how many lines it assumes.

    python scripts/make_slides.py [--out slides/week2_progress.pptx]
"""
import argparse
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Pt

PX = 6350                      # EMU per CSS pixel at this slide size
CANVAS_W, CANVAS_H = 1920, 1080

INK = RGBColor(0x14, 0x21, 0x3D)        # near-black navy, body text
MUTED = RGBColor(0x4A, 0x55, 0x68)      # secondary text
GREY = RGBColor(0x6A, 0x71, 0x79)       # captions, neutral accents
ORANGE = RGBColor(0xB4, 0x53, 0x0A)     # our arm
BLUE = RGBColor(0x2D, 0x5F, 0x8B)       # the paper's arm
PAGE = RGBColor(0xFB, 0xFB, 0xF8)       # off-white ground
CARD = RGBColor(0xF1, 0xF2, 0xEE)       # card fill
STAGE = RGBColor(0xEC, 0xEE, 0xE8)      # unchanged-stage fill, table header
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BORDER = RGBColor(0xD8, 0xDB, 0xD5)
ARROW = RGBColor(0xA8, 0xAF, 0xA8)
NAVY = RGBColor(0x14, 0x21, 0x3D)       # dark panel ground
ON_NAVY = RGBColor(0xDC, 0xE4, 0xE8)    # body text on navy
ON_NAVY_HI = RGBColor(0xFB, 0xFB, 0xF8)
ON_NAVY_LO = RGBColor(0x8F, 0xA8, 0xB8)
AMBER = RGBColor(0xE8, 0xA9, 0x6A)      # the oracle row, the open question

SERIF = 'Georgia'               # stands in for Source Serif 4
SANS = 'Helvetica Neue'         # stands in for IBM Plex Sans

MARGIN = 128                    # left and right page margin


def px(v):
    return Emu(int(round(v * PX)))


def pt(v):
    """CSS px -> points. Half, so proportions survive the 1920px -> 13.333in scaling."""
    return Pt(v / 2)


def rect(slide, x, y, w, h, fill=None, line=None, line_px=1, radius=None):
    if radius:
        shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, px(x), px(y), px(w), px(h))
        # adjustment is the corner radius as a fraction of the shorter side
        shp.adjustments[0] = radius / min(w, h)
    else:
        shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, px(x), px(y), px(w), px(h))
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = px(line_px)
    shp.shadow.inherit = False
    shp.text_frame.text = ''
    return shp


def text(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP):
    """`paras`: list of (runs, opts). A run is (string, {size, bold, italic, color, font})."""
    box = slide.shapes.add_textbox(px(x), px(y), px(w), px(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, (runs, opt) in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = opt.get('align', PP_ALIGN.LEFT)
        p.line_spacing = opt.get('leading', 1.3)
        if opt.get('before'):
            p.space_before = pt(opt['before'])
        for s, st in runs:
            r = p.add_run()
            r.text = s
            f = r.font
            f.name = st.get('font', SANS)
            f.size = pt(st.get('size', 24))
            f.bold = st.get('bold', False)
            f.italic = st.get('italic', False)
            f.color.rgb = st.get('color', INK)
            if st.get('spacing'):     # letter-spacing, in px, for the eyebrow rows
                r.font._rPr.set('spc', str(int(st['spacing'] * 50)))
    return box


def one(s, **st):
    """Shorthand for a single-run paragraph."""
    return ([(s, st)], {'leading': st.pop('leading', 1.3)})


def header(slide, eyebrow, title, subtitle=None):
    """Eyebrow + h1 (+ optional one-line subtitle). Returns the y to continue from."""
    y = 96
    text(slide, MARGIN, y, CANVAS_W - 2 * MARGIN, 30,
         [one(eyebrow, size=24, bold=True, color=ORANGE, spacing=3, leading=1.2)])
    y += 38
    text(slide, MARGIN, y, CANVAS_W - 2 * MARGIN, 66,
         [one(title, size=60, bold=True, color=INK, font=SERIF, leading=1.08)])
    y += 74
    if subtitle:
        text(slide, MARGIN, y, 1500, 36, [one(subtitle, size=26, color=MUTED, leading=1.35)])
        y += 36
    return y


def table(slide, x, y, w, rows, widths, header_fill=STAGE, size=24, row_h=44,
          grid=BORDER, cell_pad=14):
    """`rows`: list of lists of (string, {align, color, bold}). Row 0 is the header."""
    shape = slide.shapes.add_table(len(rows), len(rows[0]), px(x), px(y), px(w),
                                   px(row_h * len(rows)))
    tbl = shape.table
    tbl.first_row = False       # no banding or bold from the built-in style
    tbl.horz_banding = False
    for i, frac in enumerate(widths):
        tbl.columns[i].width = px(w * frac)
    for r, row in enumerate(rows):
        tbl.rows[r].height = px(row_h)
        for c, (s, st) in enumerate(row):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_fill if r == 0 else (
                st.get('fill') or PAGE)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = px(cell_pad)
            cell.margin_top = cell.margin_bottom = px(8)
            p = cell.text_frame.paragraphs[0]
            p.alignment = st.get('align', PP_ALIGN.LEFT)
            r_ = p.add_run()
            r_.text = s
            r_.font.name = SANS
            r_.font.size = pt(size)
            r_.font.bold = st.get('bold', r == 0)
            r_.font.color.rgb = st.get('color', INK)
    if grid is not None:
        # python-pptx exposes no border API; the enclosing stroke reads as the table's frame
        rect(slide, x, y, w, row_h * len(rows), fill=None, line=grid)
    return shape


def notes(slide, body):
    slide.notes_slide.notes_text_frame.text = body


# ---------------------------------------------------------------- slide 1

# The three stages, as the general pipeline: a column each, with the gutter that carries the
# PAPER / OURS labels on their left. Widths resolve to 458.67px per stage box.
GUTTER = 104
ARROW_W = 44
COL_GAP = 16
STAGE_W = (CANVAS_W - 2 * MARGIN - GUTTER - 2 * ARROW_W - 6 * COL_GAP) / 3


def stage_x(i):
    return MARGIN + GUTTER + COL_GAP + i * (STAGE_W + COL_GAP + ARROW_W + COL_GAP)


def method_row(slide, y, h, label, accent, cells):
    """One arm's concrete models, aligned under the generic stage boxes above.

    `cells` is three (runs, ...) lists, one per stage; runs that carry no colour read as
    unchanged-from-the-paper grey, so only the stage we actually replaced is coloured.
    """
    text(slide, MARGIN, y, GUTTER, h, [one(label, size=24, bold=True, color=accent,
                                           spacing=2)], anchor=MSO_ANCHOR.MIDDLE)
    for i, runs in enumerate(cells):
        text(slide, stage_x(i) + 20, y, STAGE_W - 40, h,
             [(runs, {'leading': 1.3})], anchor=MSO_ANCHOR.TOP)


def slide_problem(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, CANVAS_W, CANVAS_H, fill=PAGE)

    y = header(s, 'CITEAUDIT · SAROL ET AL., BIOINFORMATICS 40(7) 2024 · '
                  'REPRODUCTION AND EXTENSION',
               'Does this citation say what the source actually said?')
    y += 24

    # ---- the problem, given room: definition on the left, the scale of it on the right
    panel_h = 290
    rect(s, MARGIN, y, CANVAS_W - 2 * MARGIN, panel_h, fill=CARD)
    rect(s, MARGIN, y, 5, panel_h, fill=ORANGE)
    inner = CANVAS_W - 2 * MARGIN - 5 - 64
    left_w = inner * 0.60
    right_x = MARGIN + 5 + 32 + left_w + 40
    right_w = inner - left_w - 40

    # body wraps to 5 lines at 957px / 28px
    text(s, MARGIN + 37, y + 24, left_w, panel_h - 48, [
        one('THE PROBLEM', size=24, bold=True, color=ORANGE, spacing=1, leading=1.2),
        ([('A ', {'size': 28}),
          ('citance', {'size': 28, 'italic': True}),
          (' is a sentence in one paper that cites another. Citation integrity asks whether '
           'it represents the cited work accurately \u2014 and the verdict has to rest on '
           'specific sentences inside the cited article, not on the article as a whole. The '
           'label is one of three: ', {'size': 28}),
          ('ACCURATE', {'size': 28, 'bold': True}),
          (', ', {'size': 28}),
          ('NOT ACCURATE', {'size': 28, 'bold': True}),
          (', or ', {'size': 28}),
          ('NOT ENOUGH INFO', {'size': 28, 'bold': True}),
          ('.', {'size': 28})], {'leading': 1.4, 'before': 10}),
    ])

    text(s, right_x, y + 24, right_w, 30,
         [one('WHY IT IS HARD', size=24, bold=True, color=GREY, spacing=1, leading=1.2)])
    sy = y + 24 + 40
    for num, lab in [('239', 'candidate sentences per citance'),
                     ('2.5', 'of them marked as the evidence'),
                     ('20', 'the classifier ever gets to see')]:
        text(s, right_x, sy, 110, 52, [one(num, size=42, bold=True, font=SERIF, color=INK,
                                           align=PP_ALIGN.RIGHT)], anchor=MSO_ANCHOR.MIDDLE)
        text(s, right_x + 126, sy, right_w - 126, 52, [one(lab, size=24, color=MUTED)],
             anchor=MSO_ANCHOR.MIDDLE)
        sy += 52
    y += panel_h + 28

    # ---- the general pipeline, with a band marking the one stage we replace
    band_x = stage_x(1) - 14
    rect(s, band_x, y - 14, STAGE_W + 28, 339, fill=RGBColor(0xF7, 0xEF, 0xE6))

    for i, (step, name, detail) in enumerate([
            ('STAGE 1', 'Retrieve', '~239 sentences \u2192 60'),
            ('STAGE 2', 'Rerank', '60 \u2192 the top 20'),
            ('STAGE 3', 'Classify', '20 \u2192 one of 3 labels')]):
        x = stage_x(i)
        rect(s, x, y, STAGE_W, 149, fill=STAGE, line=BORDER)
        text(s, x + 20, y + 18, STAGE_W - 40, 113, [
            one(step, size=24, bold=True, color=GREY, spacing=1, leading=1.2),
            ([(name, {'size': 36, 'bold': True, 'font': SERIF})], {'leading': 1.15,
                                                                   'before': 4}),
            ([(detail, {'size': 24, 'color': MUTED})], {'leading': 1.3, 'before': 6}),
        ])
        if i < 2:
            text(s, x + STAGE_W + COL_GAP, y, ARROW_W, 149,
                 [one('\u2192', size=32, color=ARROW, align=PP_ALIGN.CENTER)],
                 anchor=MSO_ANCHOR.MIDDLE)
    y += 149 + 12

    # ---- the concrete models, smaller, under the stage they sit in
    method_row(s, y, 70, 'PAPER', BLUE, [
        [('BM25', {'size': 24, 'bold': True, 'color': GREY}),
         ('  k1 0.9 \u00b7 b 0.4 \u00b7 top-60', {'size': 24, 'color': GREY})],
        [('MonoT5-base', {'size': 24, 'bold': True, 'color': BLUE}),
         (', zero-shot  222.9M \u00b7 each sentence scored alone',
          {'size': 24, 'color': MUTED})],
        [('MultiVerS', {'size': 24, 'bold': True, 'color': GREY}),
         (' (Longformer-large), fine-tuned', {'size': 24, 'color': GREY})],
    ])
    y += 70 + 10
    method_row(s, y, 70, 'OURS', ORANGE, [
        [('BM25', {'size': 24, 'bold': True, 'color': GREY}),
         ('  identical settings', {'size': 24, 'color': GREY})],
        [('PubMedBERT-base', {'size': 24, 'bold': True, 'color': ORANGE}),
         (', fine-tuned  109.5M \u00b7 a sentence with \u00b11 neighbour',
          {'size': 24, 'color': MUTED})],
        [('MultiVerS', {'size': 24, 'bold': True, 'color': GREY}),
         ('  identical frozen checkpoint', {'size': 24, 'color': GREY})],
    ])

    text(s, MARGIN, 951, CANVAS_W - 2 * MARGIN, 65, [
        one('Counts are means over the 487 test citances. Stages 1 and 3 are the paper\u2019s, '
            'unchanged \u2014 same candidate pool, same BM25 settings and depth, same frozen '
            'checkpoint \u2014 so the whole measured difference is attributable to stage 2.',
            size=24, color=GREY, leading=1.35)])

    notes(s, NOTES_PROBLEM)
    return s


# ---------------------------------------------------------------- slide 2

def kv_row(slide, x, y, w, h, label, value, label_c=INK, value_c=INK, bold=False,
           split=0.60):
    """A label on the left, a right-aligned figure on the right, both vertically centred."""
    text(slide, x, y, w * split, h, [one(label, size=24, color=label_c)],
         anchor=MSO_ANCHOR.MIDDLE)
    text(slide, x + w * split, y, w * (1 - split), h,
         [one(value, size=24, color=value_c, bold=bold, align=PP_ALIGN.RIGHT)],
         anchor=MSO_ANCHOR.MIDDLE)


def slide_takeaway(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, CANVAS_W, CANVAS_H, fill=PAGE)

    y = header(s, 'MAIN TAKEAWAY SO FAR \u00b7 AND WHAT REMAINS',
               'Retrieval improved. The classifier is the ceiling.')
    y += 24

    inner = CANVAS_W - 2 * MARGIN - 34          # two columns, flex 1.12 / 1
    left_w = inner / 2.12 * 1.12
    right_w = inner / 2.12
    right_x = MARGIN + left_w + 34
    row_h = 347

    # ---- left: the headline retrieval result
    text(s, MARGIN, y, left_w, 40,
         [one('Retrieval \u2014 held-out test, measured once', size=32, bold=True,
              font=SERIF, leading=1.2)])
    R = lambda s_, **k: (s_, dict(align=PP_ALIGN.RIGHT, **k))   # noqa: E731
    L = lambda s_, **k: (s_, dict(align=PP_ALIGN.LEFT, **k))    # noqa: E731
    table(s, MARGIN, y + 52, left_w, [
        [L('Metric'), R('Paper'), R('Ours'), R('\u0394 (95% CI)')],
        [L('Recall@20'), R('0.529', color=BLUE), R('0.638', color=ORANGE, bold=True),
         R('+0.109 \u00b10.024', color=ORANGE)],
        [L('Recall@10'), R('0.422', color=BLUE), R('0.523', color=ORANGE, bold=True),
         R('+0.101 \u00b10.040', color=ORANGE)],
        [L('AllGold@20'), R('0.372', color=BLUE), R('0.493', color=ORANGE, bold=True),
         R('+0.097 \u00b10.036', color=ORANGE)],
        [L('MRR'), R('0.323', color=BLUE), R('0.365', color=ORANGE, bold=True),
         R('+0.043 \u00b10.036', color=ORANGE)],
    ], widths=[0.34, 0.19, 0.19, 0.28])
    text(s, MARGIN, y + 52 + 220 + 10, left_w, 65, [
        one('Wins on 17 of 20 articles; every interval excludes zero. Half the parameters, '
            '2.2\u00d7 faster, 1.36\u00d7 the FLOPs as configured.',
            size=24, color=GREY, leading=1.35)])

    # ---- right: the experiment that sent us to stage 2 rather than stage 1
    rect(s, right_x, y, right_w, row_h, fill=CARD)
    rect(s, right_x, y, right_w, 5, fill=BLUE)
    bx, bw = right_x + 28, right_w - 56
    text(s, bx, y + 26, bw, 30,
         [one('EXPERIMENT 1 \u00b7 IS THE FILTER THE PROBLEM?', size=24, bold=True,
              color=BLUE, spacing=1, leading=1.2)])
    text(s, bx, y + 64, bw, 32,
         [one('Drop BM25, rerank the whole pool', size=26, bold=True, leading=1.2)])
    ry = y + 110
    kv_row(s, bx, ry, bw, 36, 'BM25 top-60 \u2192 MonoT5 (the paper)', '0.529',
           value_c=BLUE, bold=True)
    kv_row(s, bx, ry + 36, bw, 36, 'No BM25 \u00b7 MonoT5 over all 239', '0.541',
           value_c=INK, bold=True)
    text(s, bx, ry + 84, bw, 130, [
        ([('\u0394 +0.012 \u00b10.014', {'size': 24, 'bold': True}),
          (' \u2014 the interval includes zero and it wins on only 10 of 20 articles, at '
           '4.0\u00d7 the candidates. The depth-60 cut does discard 29% of the gold '
           'sentences, yet MonoT5 given all of them recovers none of it. The limit is the '
           'reranker, not the filter.', {'size': 24, 'color': MUTED})],
         {'leading': 1.35}),
    ])
    y += row_h + 20

    # ---- the downstream finding, compressed to the one line the title rests on
    rect(s, MARGIN, y, CANVAS_W - 2 * MARGIN, 63, fill=NAVY, radius=6)
    text(s, MARGIN + 28, y, CANVAS_W - 2 * MARGIN - 56, 63, [
        ([('Downstream: ', {'size': 24, 'bold': True, 'color': AMBER}),
          ('under the frozen classifier both arms tie (McNemar p = 1.000), and perfect '
           'oracle evidence is the worst arm of all.', {'size': 24, 'color': ON_NAVY})],
         {'leading': 1.3}),
    ], anchor=MSO_ANCHOR.MIDDLE)
    y += 63 + 20

    text(s, MARGIN, y, CANVAS_W - 2 * MARGIN, 40,
         [one('What remains before the final report', size=32, bold=True, font=SERIF,
              leading=1.2)])
    y += 50

    # four cards: 1-line headings, 2-line bodies at 361px / 24px
    w = (CANVAS_W - 2 * MARGIN - 3 * 20) / 4
    h = 175
    todo = [
        (ORANGE, 'DECISIVE', 'Retrain MultiVerS per arm',
         'Both arms share one frozen checkpoint today.'),
        (GREY, 'HEADROOM', 'Finish the length sweep',
         'A quarter of inputs truncate at 192 tokens.'),
        (GREY, 'COST', 'Report FLOPs, not seconds',
         'Dynamic padding cuts 1.36\u00d7 to 1.06\u00d7 free.'),
        (GREY, 'HYGIENE', 'Rerun both arms together',
         'Cross-date numbers drifted 0.012 on dev.'),
    ]
    for i, (accent, label, head, body) in enumerate(todo):
        x = MARGIN + i * (w + 20)
        rect(s, x, y, w, h, fill=CARD)
        rect(s, x, y, w, 5, fill=accent)
        text(s, x + 20, y + 21, w - 40, h - 39, [
            one(label, size=24, bold=True, color=accent, spacing=1, leading=1.2),
            ([(head, {'size': 26, 'bold': True})], {'leading': 1.2, 'before': 6}),
            ([(body, {'size': 24, 'color': MUTED})], {'leading': 1.3, 'before': 6}),
        ])

    text(s, MARGIN, 951, CANVAS_W - 2 * MARGIN, 65, [
        one('Pooled (micro) metrics, the paper\u2019s definitions; intervals from a paired '
            'bootstrap resampling articles. The classification arms share one frozen '
            'checkpoint, so only the arm-to-arm comparison carries weight \u2014 absolute '
            'F1 is not comparable to the paper\u2019s.', size=24, color=GREY, leading=1.35)])

    notes(s, NOTES_TAKEAWAY)
    return s


NOTES_PROBLEM = """\
Start with the problem block, it carries the slide. A citance is a sentence in one paper \
that cites another. Citation integrity asks whether that sentence represents the cited \
work accurately. The part that makes it an NLP problem rather than a judgement call is \
that the verdict has to rest on specific sentences inside the cited article — you cannot \
answer it from the abstract or the title.

The three numbers on the right are the scale of it. Averaged over the 487 test citances, a \
citance has 239 candidate sentences drawn from the article it cites, the annotators marked \
about 2.5 of them as the evidence, and the classifier only ever sees the twenty that \
retrieval hands it. So retrieval is a hard filter on everything downstream, which is why \
we went after it first.

The pipeline is three stages and it is the same three stages in both arms: retrieve, \
rerank, classify. Stage 1 narrows 239 candidate sentences to 60 with BM25. Stage 2 reorders \
those 60 and keeps the top 20. Stage 3 reads those 20 and emits one of the three labels.

Prior work is Sarol et al. 2024 in Bioinformatics — the blue row. BM25, then MonoT5-base \
applied zero-shot, then MultiVerS, a Longformer-large model fine-tuned for the three-way \
label. We reproduced their published retrieval numbers before touching anything: 0.529 \
Recall@20 against their 0.53, 0.323 MRR against their 0.32.

Our method is the orange row, and the point of the shaded column is that only stage 2 \
differs. Same citances, same candidate pool, same BM25 settings and depth 60, same metric \
code, same frozen MultiVerS checkpoint. The reranker becomes a PubMedBERT cross-encoder, \
fine-tuned on this task's own training articles, scoring each candidate sentence together \
with one neighbour on each side rather than alone.

Two differentiations worth naming out loud. Supervision: their reranker is a general-domain \
model transferred zero-shot, ours is trained in domain. And the unit being scored: a lone \
sentence versus a sentence in context — biomedical evidence sentences frequently need the \
sentence before them for the referent. Both are also cheaper, not more expensive: 109.5M \
parameters against 222.9M."""


NOTES_TAKEAWAY = """\
Take the right-hand box first, because it is chronologically first and it explains why we \
worked on the reranker at all.

The obvious suspect was the cheap filter. BM25 keeps only 60 of a citance's 239 candidate \
sentences before anything smart looks at them, and that cut really does throw away 29% of \
the annotated evidence — BM25's own recall at depth 60 is 0.712, so no reranker working \
behind it can exceed that. So we removed the filter and let MonoT5 score all 239.

It bought nothing. Recall@20 went 0.529 to 0.541, a paired delta of +0.012 whose interval \
includes zero, winning on only 10 of 20 articles — a coin flip — and AllGold@20 and MRR \
both came out flat or slightly negative, all at four times the candidates scored. So the \
evidence the filter discards is evidence MonoT5 would not have ranked highly anyway. The \
limit was the reranker, not the filter. That is what sent us to stage 2.

Now the left table, which is the main result. Held-out test Recall@20 goes 0.529 to 0.638, \
a paired gain of +0.109 plus or minus 0.024, winning on 17 of the 20 test articles. All \
four metrics move and no interval touches zero. The test split was measured exactly once, \
after the recipe was frozen on dev.

Read the pattern across the metrics, not just the headline. Recall@20 and AllGold@20 both \
move about ten points; MRR moves four and wins on only 13 of 20 articles. We capture more \
evidence and more often capture all of it, but we are only slightly better at putting a \
correct sentence first. That matches the change we made: the context window helps recognise \
relevance and slightly hurts top-1 precision. For this pipeline that trade is fine, because \
the classifier receives the top-20 as a set and does not care about the order.

Cost: 109.5M parameters against 222.9M and 2.2x the throughput, but as configured we use \
1.36x the FLOPs. Both numbers are real. We run faster despite more arithmetic because there \
is no serial decoder step.

The navy line is the finding that caps the slide. Under the released MultiVerS checkpoint \
the better evidence changes no label — both arms tie at McNemar p = 1.000 — and an oracle \
with perfect evidence is the worst arm of all. Random evidence does cost 19 to 25 points at \
p < 0.001, so the classifier is reading its input; it simply cannot exploit anything above \
the retrieval distribution it was trained on.

Which makes the first remaining item the decisive one. The paper's own gold-evidence row is \
+0.23 macro-F1 over its top-20 row, but their gold-evidence model was trained on gold \
evidence. Retraining the classifier on each arm's output is the experiment that decides \
whether our retrieval gain is worth anything downstream. It needs the CUDA machine: \
Longformer-large, likely with gradient checkpointing at 8 GB.

The other three are smaller. A quarter of reranker inputs truncate at 192 tokens, so the \
length sweep has headroom — and our 0.638 is now within 0.074 of BM25's depth-60 ceiling of \
0.712, so the filter depth is becoming a live constraint for us even though it was not one \
for MonoT5. Dynamic padding should cut the FLOPs overhead from 1.36x to 1.06x at no measured \
speed cost. And both arms must be rerun inside one session for the final table: numbers \
taken on different dates drifted by 0.012 on dev, which we traced to the environment rather \
than to our code."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('slides/week2_progress.pptx'))
    a = ap.parse_args()

    prs = Presentation()
    prs.slide_width = px(CANVAS_W)
    prs.slide_height = px(CANVAS_H)
    slide_problem(prs)
    slide_takeaway(prs)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(a.out)
    print(f'wrote {a.out}  ({a.out.stat().st_size / 1024:.0f} KB, {len(prs.slides.__iter__.__self__._sldIdLst)} slides)')


if __name__ == '__main__':
    main()
