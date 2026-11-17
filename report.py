"""
report.py - experiment summaries, CSV / PNG / PDF export bundles.

A ReportBundle describes *what* can be exported from a page (parameters, metrics, tables and lazily-built
Matplotlib figures).  Figures are only rendered when the user asks for a PNG or PDF, so ordinary slider
interaction never pays for export work.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import os
import zipfile
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

import physics_engine as pe
import visualizations as viz


@dataclass
class ReportBundle:
    key: str
    title: str
    params: Dict[str, object] = field(default_factory=dict)
    metrics: Dict[str, object] = field(default_factory=dict)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    figure_builders: Dict[str, Callable] = field(default_factory=dict)
    notes: str = ""

    def signature(self) -> str:
        raw = repr(sorted(self.params.items(), key=lambda kv: kv[0])) + repr(sorted(self.metrics.items(), key=lambda kv: kv[0]))
        raw += "".join(f"{n}:{t.shape}:{float(np.nansum(t.select_dtypes('number').to_numpy())):.8g}" for n, t in self.tables.items())
        return hashlib.md5(raw.encode()).hexdigest()


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:.6g}"
    return str(v)


# --------------------------------------------------------------------------
def result_to_dataframe(res: pe.SimulationResult) -> pd.DataFrame:
    """Time series of every dashboard metric (one row per stored snapshot)."""
    return pd.DataFrame(dict(t=res.times, total_probability=res.norm, reflection_R=res.R, transmission_T=res.T,
                             inside_B=res.B, R_plus_T_plus_B=res.R + res.T + res.B, energy=res.energy,
                             mean_x=res.mean_x, delta_x=res.delta_x, mean_p=res.mean_p, delta_p=res.delta_p,
                             delta_x_delta_p=res.delta_x * res.delta_p))


def config_params(cfg: pe.SimConfig) -> Dict[str, object]:
    p = {"x_min": cfg.x_min, "x_max": cfg.x_max, "dx": cfg.dx, "dt": cfg.dt, "t_max": cfg.t_max,
         "mass m": cfg.mass, "x0": cfg.x0, "sigma": cfg.sigma, "k0": cfg.k0, "E0 = k0^2/2m": cfg.energy_nominal,
         "potential": cfg.potential_kind}
    p.update({f"potential: {k}": v for k, v in cfg.potential_params})
    p["units"] = "dimensionless (hbar = 1)"
    return p


def result_metrics(res: pe.SimulationResult, i: Optional[int] = None) -> Dict[str, object]:
    i = res.n_snap - 1 if i is None else i
    m = res.metrics_at(i)
    return {"time t": m["t"], "total probability": m["total"], "reflection R": m["R"], "transmission T": m["T"],
            "inside region B": m["B"], "R+T+B": m["R"] + m["T"] + m["B"], "<E>": m["energy"],
            "Delta x": m["delta_x"], "Delta p": m["delta_p"], "Delta x * Delta p": m["delta_x"] * m["delta_p"]}


def standard_bundle(key: str, title: str, res: pe.SimulationResult, i: Optional[int] = None,
                    notes: str = "") -> ReportBundle:
    i = res.n_snap - 1 if i is None else int(i)
    return ReportBundle(
        key=key, title=title, params=config_params(res.cfg), metrics=result_metrics(res, i),
        tables={"time_series": result_to_dataframe(res)},
        figure_builders={
            f"snapshot_t{res.times[i]:.2f}": lambda: viz.mpl_panels(res, i),
            "regions": lambda: viz.mpl_regions(res, i),
            "spacetime": lambda: viz.mpl_spacetime(res),
            "probability_vs_time": lambda: viz.mpl_lines(
                res.times, {"R (left)": res.R, "T (right)": res.T, "B (inside)": res.B, "R+T+B": res.R + res.T + res.B},
                "t", "probability", "Probability in each region"),
        },
        notes=notes)


# --------------------------------------------------------------------------
def bundle_csv(b: ReportBundle) -> bytes:
    if len(b.tables) == 1:
        return next(iter(b.tables.values())).to_csv(index=False).encode()
    out = io.StringIO()
    for name, t in b.tables.items():
        out.write(f"# {name}\n")
        t.to_csv(out, index=False)
        out.write("\n")
    return out.getvalue().encode()


def bundle_summary(b: ReportBundle) -> str:
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# {b.title}", f"*QWaveLab experiment summary – generated {now}*", "", "## Parameters"]
    lines += [f"- **{k}**: {_fmt(v)}" for k, v in b.params.items()]
    lines += ["", "## Numerical results"]
    lines += [f"- **{k}**: {_fmt(v)}" for k, v in b.metrics.items()]
    for n, t in b.tables.items():
        lines += ["", f"## Table: {n} ({len(t)} rows)", "", t.head(12).to_markdown(index=False) if _has_tabulate() else t.head(12).to_string(index=False)]
    if b.notes:
        lines += ["", "## Interpretation", "", b.notes]
    lines += ["", "---", "All values were computed by the Crank–Nicolson solver in dimensionless units (ħ = 1)."]
    return "\n".join(lines)


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except Exception:
        return False


def bundle_pngs(b: ReportBundle) -> tuple:
    """Return (bytes, filename, mime): a PNG for one figure, a ZIP for several."""
    figs = {n: viz.fig_to_png(f()) for n, f in b.figure_builders.items()}
    if len(figs) == 1:
        n, data = next(iter(figs.items()))
        return data, f"{b.key}_{n}.png", "image/png"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n, data in figs.items():
            z.writestr(f"{b.key}_{n}.png", data)
    return buf.getvalue(), f"{b.key}_figures.zip", "application/zip"


# --------------------------------------------------------------------------
def _register_unicode_font() -> str:
    """Use DejaVu Sans (shipped with Matplotlib) so Greek letters render in the PDF."""
    try:
        import matplotlib
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        base = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
        pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(base, "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(base, "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold")
        return "DejaVu"
    except Exception:
        return "Helvetica"


def bundle_pdf(b: ReportBundle) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak
    from xml.sax.saxutils import escape

    font = _register_unicode_font()
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Title"], fontName=font, fontSize=19, textColor=colors.HexColor("#0e7490"))
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName=font, fontSize=12.5, textColor=colors.HexColor("#4c1d95"),
                        spaceBefore=10)
    BODY = ParagraphStyle("B", parent=ss["BodyText"], fontName=font, fontSize=9.2, leading=13)
    SMALL = ParagraphStyle("S", parent=BODY, fontSize=7.8, textColor=colors.grey)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                            bottomMargin=1.6 * cm, title=b.title, author="QWaveLab")
    story = [Paragraph(escape(b.title), H1),
             Paragraph(f"QWaveLab – Interactive Quantum Wave Packet Simulation Laboratory · generated "
                       f"{_dt.datetime.now():%Y-%m-%d %H:%M}", SMALL), Spacer(1, 8)]

    def kv_table(d, title):
        story.append(Paragraph(title, H2))
        data = [[str(k), _fmt(v)] for k, v in d.items()]
        t = Table(data, colWidths=[7 * cm, 9.5 * cm])
        t.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), font), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                               ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#f1f5f9"), colors.white]),
                               ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story.append(t)

    kv_table(b.params, "Selected parameters")
    kv_table(b.metrics, "Numerical results")
    if b.notes:
        story.append(Paragraph("Interpretation", H2))
        for para in b.notes.replace("**", "").split("\n\n"):
            story.append(Paragraph(escape(para), BODY))
            story.append(Spacer(1, 4))
    for n, f in b.figure_builders.items():
        story.append(Paragraph(f"Figure: {escape(n.replace('_', ' '))}", H2))
        png = viz.fig_to_png(f(), dpi=130)
        story.append(Image(io.BytesIO(png), width=16 * cm, height=16 * cm, kind="proportional"))
    for n, t in b.tables.items():
        story.append(PageBreak())
        story.append(Paragraph(f"Table: {escape(n)} (first 25 of {len(t)} rows)", H2))
        sub = t.head(25)
        cols = list(sub.columns)[:8]
        data = [[str(c) for c in cols]] + [[_fmt(v) for v in row] for row in sub[cols].to_numpy()]
        tb = Table(data, repeatRows=1)
        tb.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), font), ("FONTSIZE", (0, 0), (-1, -1), 6.8),
                                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0f2fe")),
                                ("GRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#cbd5e1"))]))
        story.append(tb)
    story += [Spacer(1, 10), Paragraph("All numbers were produced by the Crank–Nicolson finite-difference solver "
                                       "(dimensionless units, ħ = 1). Dirichlet walls; see the application's "
                                       "Numerical Validation page for accuracy and limitations.", SMALL)]
    doc.build(story)
    return buf.getvalue()
