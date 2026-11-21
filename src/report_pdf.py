"""
Build report/Milestone2_Report.pdf.

  Page 1   one-page summary of observations
  Page 2   early samples at the chosen config, plus the schedule comparison grid
  Page 3   CFG sweep grid
  Page 4+  training log: run settings, per-config summary, every generation

Run src/make_report.py first (it writes the figures, samples and
report/training_log.csv this script reads).

Usage (from the repo root):
  python src/report_pdf.py
"""
import io
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
FIG = REPORT / "figures"
OUT = REPORT / "Milestone2_Report.pdf"
REPO_URL = "https://github.com/tanoj22/text2img-coco"
TEAM = "Sai Tanoj Salehundam, Tanatswa Manzini, Raymond Wang, Rajiv Praveen"

ACCENT = colors.HexColor("#1f3a5f")
INK = colors.HexColor("#1b1b1b")
INK_2 = colors.HexColor("#55534e")
RULE = colors.HexColor("#d9d7d0")
SHADE = colors.HexColor("#f3f2ee")
HILITE = colors.HexColor("#e6eef8")

PAGE_W, PAGE_H = letter
MARGIN = 0.55 * inch
TEXT_W = PAGE_W - 2 * MARGIN - 12  # the frame keeps 6 pt padding on each side

LABELS = {"ddim": "DDIM", "euler": "Euler", "dpmpp_karras": "DPM++ 2M Karras"}


def styles():
    def st(name, **kw):
        opts = dict(fontName="Helvetica", textColor=INK, alignment=TA_LEFT)
        opts.update(kw)
        return ParagraphStyle(name, **opts)

    return {
        "title": st("title", fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=ACCENT),
        "sub": st("sub", fontSize=8.6, leading=11, textColor=INK_2),
        "h": st("h", fontName="Helvetica-Bold", fontSize=10, leading=12, textColor=ACCENT,
                spaceBefore=5, spaceAfter=2),
        "body": st("body", fontSize=8.4, leading=10.6),
        "bullet": st("bullet", fontSize=8.4, leading=10.4, leftIndent=9, bulletIndent=0, spaceAfter=1.2),
        "cap": st("cap", fontSize=7.2, leading=8.8, textColor=INK_2),
        "cell": st("cell", fontSize=7.6, leading=9),
    }


S = styles()


def header_rule(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(ACCENT)
    canvas.setLineWidth(2)
    canvas.line(MARGIN, PAGE_H - 0.38 * inch, PAGE_W - MARGIN, PAGE_H - 0.38 * inch)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(INK_2)
    canvas.drawString(MARGIN, 0.33 * inch, "Text-to-Image Generation on MS COCO  |  Milestone 2")
    canvas.drawRightString(PAGE_W - MARGIN, 0.33 * inch, f"Page {doc.page}")
    canvas.restoreState()


def _jpeg(im):
    """Photo grids go in as JPEG, which keeps the PDF small enough to upload."""
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=88, optimize=True)
    buf.seek(0)
    return buf


def fig(path, width, photo=True):
    with PILImage.open(path) as im:
        w, h = im.size
        src = _jpeg(im) if photo else str(path)
    return Image(src, width=width, height=width * h / w)


def cfg_grid_fig(max_w, max_h, crop_top=85):
    """The CFG grid without its title band (the page caption says the same), as large as fits."""
    with PILImage.open(FIG / "cfg_grid.png") as im:
        im = im.crop((0, crop_top, im.width, im.height))
        w, h = im.size
        buf = _jpeg(im)
    scale = min(max_w / w, max_h / h)
    return Image(buf, width=w * scale, height=h * scale)


def table(rows, widths, header_rows=1, highlight=(), align_right_from=1, font=7.6, zebra=True, pad=1.6):
    t = Table(rows, colWidths=widths, repeatRows=header_rows)
    cmds = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", font),
        ("FONT", (0, 0), (-1, header_rows - 1), "Helvetica-Bold", font),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), ACCENT),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.8, ACCENT),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE),
        ("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), pad),
        ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    if zebra:
        for r in range(header_rows, len(rows)):
            if (r - header_rows) % 2 == 1:
                cmds.append(("BACKGROUND", (0, r), (-1, r), SHADE))
    for r in highlight:
        cmds += [("BACKGROUND", (0, r), (-1, r), HILITE), ("FONT", (0, r), (-1, r), "Helvetica-Bold", font)]
    t.setStyle(TableStyle(cmds))
    return t


def bullets(items):
    return [Paragraph(text, S["bullet"], bulletText="•") for text in items]


# ---------------------------------------------------------------- data


def load():
    cfg = pd.read_csv(FIG / "cfg_summary.csv")
    sched = pd.read_csv(FIG / "schedule_summary.csv")
    log = pd.read_csv(REPORT / "training_log.csv")
    real = pd.read_csv(ROOT / "outputs" / "real_clip_scores.csv")
    return cfg, sched, log, real


def cfg_table(cfg, chosen=7.5):
    rows = [["CFG", "CLIP mean", "Std", "vs real", "s / image"]]
    hl = []
    for i, r in enumerate(cfg.itertuples(), start=1):
        rows.append([f"{r.cfg:g}", f"{r.clip_mean:.2f}", f"{r.clip_std:.2f}", f"{r.vs_real:+.2f}",
                     f"{r.time_mean:.2f}"])
        if r.cfg == chosen:
            hl.append(i)
    w = TEXT_W * 0.40
    return table(rows, [w * 0.16, w * 0.24, w * 0.18, w * 0.21, w * 0.21], highlight=hl)


def sched_table(sched, chosen):
    rows = [["Sampler", "Steps", "Spacing", "CLIP mean", "Std", "s / image"]]
    hl = []
    for i, r in enumerate(sched.itertuples(), start=1):
        rows.append([LABELS[r.scheduler], str(r.steps), r.spacing, f"{r.clip_mean:.2f}", f"{r.clip_std:.2f}",
                     f"{r.time_mean:.1f}"])
        if (r.scheduler, r.steps, r.spacing) == chosen:
            hl.append(i)
    w = TEXT_W * 0.57
    return table(rows, [w * 0.27, w * 0.11, w * 0.16, w * 0.17, w * 0.12, w * 0.17], highlight=hl,
                 align_right_from=3)


# ---------------------------------------------------------------- pages


def page_summary(cfg, sched, log, real, text):
    story = [
        Paragraph("Text-to-Image Generation on MS COCO: Milestone 2", S["title"]),
        Spacer(1, 2),
        Paragraph(f"<b>Team:</b> {TEAM} &nbsp;|&nbsp; <b>GitHub:</b> "
                  f'<link href="{REPO_URL}" color="#1f5fa8">github.com/tanoj22/text2img-coco</link>', S["sub"]),
        Spacer(1, 3),
        Paragraph("Setup", S["h"]),
        Paragraph(text["setup"], S["body"]),
        Spacer(1, 4),
    ]

    chosen = text["chosen"]
    left = [Paragraph("<b>Classifier-free guidance</b> (DDIM, 30 steps, CUDA)", S["cap"]), Spacer(1, 2),
            cfg_table(cfg), Spacer(1, 7),
            Paragraph("<b>Schedule: CLIP vs steps</b> (leading spacing, ±1 standard error)", S["cap"]),
            fig(FIG / "schedule_plot.png", TEXT_W * 0.385, photo=False)]
    right = [Paragraph("<b>Noise schedule</b> (CFG 7.5, Apple M4 GPU)", S["cap"]), Spacer(1, 2),
             sched_table(sched, chosen)]
    pair = Table([[left, right]], colWidths=[TEXT_W * 0.41, TEXT_W * 0.59])
    pair.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                              ("RIGHTPADDING", (0, 0), (0, 0), 10), ("RIGHTPADDING", (1, 0), (1, 0), 0)]))
    story += [pair, Spacer(1, 2),
              Paragraph(text["table_note"], S["cap"]),
              Paragraph("Observations", S["h"])]
    story += bullets(text["observations"])
    story += [Paragraph("Chosen configuration and next steps", S["h"]), Paragraph(text["next"], S["body"]),
              PageBreak()]
    return story


def page_samples(text):
    return [
        Paragraph("Early samples", S["title"]),
        Paragraph(text["samples_caption"], S["sub"]),
        Spacer(1, 6),
        fig(FIG / "samples_grid.png", TEXT_W),
        Spacer(1, 8),
        Paragraph("Same prompt across samplers and step counts", S["h"]),
        Paragraph(text["grid_caption"], S["sub"]),
        Spacer(1, 4),
        fig(FIG / "schedule_grid.png", TEXT_W),
        PageBreak(),
        Paragraph("Guidance scale sweep", S["title"]),
        Paragraph(text["cfg_grid_caption"], S["sub"]),
        Spacer(1, 4),
        cfg_grid_fig(TEXT_W, PAGE_H - 2.1 * inch),
        PageBreak(),
    ]


def page_log(cfg, sched, log, text):
    story = [Paragraph("Training log", S["title"]), Paragraph(text["log_intro"], S["sub"]), Spacer(1, 6)]

    env = [[Paragraph(f"<b>{k}</b>", S["cell"]), Paragraph(v, S["cell"])] for k, v in text["env"]]
    et = Table(env, colWidths=[TEXT_W * 0.2, TEXT_W * 0.8])
    et.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    story += [Paragraph("Run settings", S["h"]), et, Spacer(1, 6)]

    # per-config summary across both sweeps
    rows = [["Run", "Sweep", "Device", "Sampler", "Steps", "Spacing", "CFG", "Images", "CLIP mean", "Std",
             "s / image", "s / step"]]
    groups = log.groupby(["sweep", "device", "scheduler", "steps", "spacing", "cfg"], sort=False)
    for i, ((sweep, device, s, n, sp, c), g) in enumerate(groups, start=1):
        rows.append([str(i), sweep, device.upper() if device == "mps" else "CUDA", LABELS[s], str(n), sp,
                     f"{c:g}", str(len(g)), f"{g.clip_score.mean():.2f}", f"{g.clip_score.std():.2f}",
                     f"{g.time_sec.mean():.2f}", f"{(g.time_sec / g.steps).mean():.3f}"])
    w = TEXT_W
    story += [Paragraph("Per-configuration summary", S["h"]),
              table(rows, [w * x for x in (0.05, 0.09, 0.07, 0.15, 0.06, 0.09, 0.06, 0.07, 0.1, 0.07, 0.1, 0.09)],
                    align_right_from=4),
              Spacer(1, 8)]

    # every generation
    rows = [["#", "Sweep", "Prompt", "Sampler", "Steps", "Spacing", "CFG", "Seed", "Time (s)", "CLIP"]]
    for i, r in enumerate(log.itertuples(), start=1):
        rows.append([str(i), r.sweep, f"{r.prompt_id}", LABELS[r.scheduler], str(r.steps), r.spacing,
                     f"{r.cfg:g}", str(r.seed), f"{r.time_sec:.2f}", f"{r.clip_score:.2f}"])
    story += [Paragraph(f"All {len(log)} generations", S["h"]),
              Paragraph(text["log_rows_note"], S["cap"]), Spacer(1, 3),
              table(rows, [w * x for x in (0.06, 0.1, 0.08, 0.2, 0.08, 0.11, 0.07, 0.08, 0.11, 0.11)],
                    align_right_from=4, font=6.8, pad=1.05)]
    return story


def build(text):
    cfg, sched, log, real = load()
    doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=0.5 * inch, bottomMargin=0.5 * inch,
                            title="Text-to-Image Generation on MS COCO: Milestone 2",
                            author=TEAM, subject="Milestone 2 report", creator="", producer="")
    story = page_summary(cfg, sched, log, real, text) + page_samples(text) + page_log(cfg, sched, log, text)
    doc.build(story, onFirstPage=header_rule, onLaterPages=header_rule)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    from report_text import TEXT
    build(TEXT)
