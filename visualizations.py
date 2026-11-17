"""
visualizations.py - Plotly figures for the interactive UI and Matplotlib figures for PNG / PDF export.

Every figure is drawn from arrays produced by the numerical engine; nothing here invents data.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import physics_engine as pe
import potentials as pot

# --------------------------------------------------------------------------
# Palette (dark scientific theme)
# --------------------------------------------------------------------------
CYAN = "#22d3ee"
BLUE = "#4f8cff"
VIOLET = "#a78bfa"
PINK = "#f472b6"
AMBER = "#fbbf24"
GREEN = "#34d399"
GRID = "rgba(148,163,184,0.14)"
TEXT = "#cbd5e1"
SERIES = [CYAN, VIOLET, AMBER, GREEN, PINK, BLUE]
HEAT = [[0.0, "#050816"], [0.22, "#2e1a78"], [0.5, "#3b6cf5"], [0.78, "#22d3ee"], [1.0, "#f3fdff"]]


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def _style(fig: go.Figure, title: Optional[str] = None, height: int = 420, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,13,34,0.55)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=TEXT, size=13),
        margin=dict(l=58, r=22, t=56 if title else 24, b=60),
        height=height,
        hovermode="x unified",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#0b1230", font_size=12),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, xanchor="left", font=dict(size=15)))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(148,163,184,0.3)")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(148,163,184,0.3)")
    return fig


def _stride(n: int, target: int = 1600) -> int:
    return max(1, n // target)


def _ranges(res: pe.SimulationResult) -> Dict[str, float]:
    """Fixed axis limits for the whole run so that animated frames do not rescale."""
    cache = res.__dict__.get("_rng")
    if cache is None:
        d = res.density
        cache = dict(dmax=float(d.max()), pmax=float(np.abs(res.psi).max()),
                     vmax=float(np.abs(res.V).max()))
        res.__dict__["_rng"] = cache
    return cache


def _shade_barriers(fig, segments, row=None, col=None):
    for (l, r, h) in segments:
        color = AMBER if h > 0 else GREEN
        kw = dict(row=row, col=col) if row else {}
        fig.add_vrect(x0=l, x1=r, fillcolor=_rgba(color, 0.10), line_width=0, layer="below", **kw)


def segments_of(res: pe.SimulationResult):
    return pot.segments_for(res.cfg.potential_kind, res.cfg.pot_dict)


# ==========================================================================
# Wave-function panels
# ==========================================================================
def fig_snapshot_panels(res: pe.SimulationResult, i: int, xlim: Optional[Tuple[float, float]] = None,
                        height: int = 620) -> go.Figure:
    """2x2 panel: |psi|^2 (+ scaled V), Re psi, Im psi, V(x)."""
    s = _stride(len(res.x))
    x = res.x[::s]
    psi = res.psi[i][::s]
    dens = pe.probability_density(psi)
    V = res.V[::s]
    r = _ranges(res)
    vs = 0.85 * r["dmax"] / max(r["vmax"], 1e-12)

    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.16, horizontal_spacing=0.09,
                        subplot_titles=("Probability density |ψ(x,t)|²", "Real part Re ψ(x,t)",
                                        "Imaginary part Im ψ(x,t)", "Potential energy V(x)"))
    fig.add_trace(go.Scatter(x=x, y=dens, name="|ψ|²", line=dict(color=CYAN, width=2), fill="tozeroy",
                             fillcolor=_rgba(CYAN, 0.18), hovertemplate="%{y:.5f}"), row=1, col=1)
    if r["vmax"] > 0:
        fig.add_trace(go.Scatter(x=x, y=V * vs, name="V(x) (scaled)", line=dict(color=AMBER, width=1.5, dash="dash"),
                                 customdata=V, hovertemplate="V=%{customdata:.3f}"), row=1, col=1)
    env = np.sqrt(dens)
    fig.add_trace(go.Scatter(x=x, y=env, line=dict(color=_rgba(VIOLET, 0.0)), showlegend=False, hoverinfo="skip"),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=-env, line=dict(color=_rgba(VIOLET, 0.35), width=1, dash="dot"),
                             name="±|ψ| envelope", fill="tonexty", fillcolor=_rgba(VIOLET, 0.07),
                             hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=psi.real, name="Re ψ", line=dict(color=BLUE, width=1.6),
                             hovertemplate="%{y:.5f}"), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=psi.imag, name="Im ψ", line=dict(color=PINK, width=1.6),
                             hovertemplate="%{y:.5f}"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=V, name="V(x)", line=dict(color=AMBER, width=2), fill="tozeroy",
                             fillcolor=_rgba(AMBER, 0.15), hovertemplate="%{y:.3f}"), row=2, col=2)
    E = float(res.energy[i])
    fig.add_hline(y=E, line=dict(color=GREEN, dash="dot", width=1), row=2, col=2,
                  annotation_text=f"⟨E⟩ = {E:.2f}", annotation_font_color=GREEN, annotation_position="top left")
    pm = 1.12 * r["pmax"]
    fig.update_yaxes(range=[-0.02 * r["dmax"], 1.18 * r["dmax"]], row=1, col=1)
    fig.update_yaxes(range=[-pm, pm], row=1, col=2)
    fig.update_yaxes(range=[-pm, pm], row=2, col=1)
    vlo = min(0.0, float(res.V.min())); vhi = max(float(res.V.max()), E)
    pad = 0.15 * max(vhi - vlo, 1e-6)
    fig.update_yaxes(range=[vlo - pad, vhi + pad], row=2, col=2)
    for rr, cc in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        fig.update_xaxes(title_text="x", range=list(xlim) if xlim else None, row=rr, col=cc)
    _style(fig, None, height, legend=False)
    fig.update_layout(hovermode="x")
    for a in fig.layout.annotations:
        a.font = dict(size=13, color=TEXT)
    return fig


def fig_density_regions(res: pe.SimulationResult, i: int, xlim: Optional[Tuple[float, float]] = None,
                        height: int = 420, title: Optional[str] = None) -> go.Figure:
    """|psi|^2 split into left / inside / right of the interaction region, with the scaled potential."""
    s = _stride(len(res.x))
    x = res.x[::s]
    dens = res.density[i][::s]
    left, inside, right = pot.region_masks(x, res.region)
    r = _ranges(res)
    fig = go.Figure()
    free = res.region[1] <= res.region[0]
    labels = ("x < 0 (left)", "", "x > 0 (right)") if free else (
        "Left of region (incoming / reflected)", "Inside region", "Right of region (transmitted)")
    for mask, name, col in ((left, labels[0], VIOLET), (inside, labels[1], AMBER), (right, labels[2], CYAN)):
        if not mask.any():
            continue
        yv = np.where(mask, dens, np.nan)
        fig.add_trace(go.Scatter(x=x, y=yv, name=name, line=dict(color=col, width=2), fill="tozeroy",
                                 fillcolor=_rgba(col, 0.22), connectgaps=False, hovertemplate="%{y:.5f}"))
    if r["vmax"] > 0:
        vs = 0.85 * r["dmax"] / r["vmax"]
        fig.add_trace(go.Scatter(x=x, y=res.V[::s] * vs, name="V(x) (scaled)", customdata=res.V[::s],
                                 line=dict(color=AMBER, width=1.5, dash="dash"),
                                 hovertemplate="V=%{customdata:.3f}"))
    lo = min(0.0, float(res.V.min()) * 0.85 * r["dmax"] / max(r["vmax"], 1e-12)) * 1.1
    fig.update_yaxes(range=[lo, 1.18 * r["dmax"]], title_text="|ψ(x,t)|²")
    fig.update_xaxes(title_text="Position x", range=list(xlim) if xlim else None)
    _style(fig, title, height)
    return fig


def fig_spacetime(res: pe.SimulationResult, xlim: Optional[Tuple[float, float]] = None, height: int = 460,
                  title: str = "Space–time probability density |ψ(x,t)|²") -> go.Figure:
    """Heat-map of |psi|^2 in the (x, t) plane; barriers show as vertical bands."""
    sx = _stride(len(res.x), 700)
    st = _stride(res.n_snap, 260)
    x = res.x[::sx]
    z = res.density[::st, ::sx]
    fig = go.Figure(go.Heatmap(x=x, y=res.times[::st], z=z, colorscale=HEAT,
                               colorbar=dict(title="|ψ|²", thickness=12),
                               hovertemplate="x=%{x:.2f}<br>t=%{y:.2f}<br>|ψ|²=%{z:.4f}<extra></extra>"))
    a, b = res.region
    if res.cfg.potential_kind != "free":
        for xe in {a, b}:
            fig.add_vline(x=xe, line=dict(color=AMBER, dash="dash", width=1.2))
    fig.update_xaxes(title_text="Position x", range=list(xlim) if xlim else None)
    fig.update_yaxes(title_text="Time t")
    _style(fig, title, height, legend=False)
    fig.update_layout(hovermode="closest")
    return fig


def fig_potential(x: np.ndarray, V: np.ndarray, title: str = "Potential energy V(x)", energy: Optional[float] = None,
                  xlim: Optional[Tuple[float, float]] = None, height: int = 300) -> go.Figure:
    s = _stride(len(x))
    fig = go.Figure(go.Scatter(x=x[::s], y=V[::s], name="V(x)", line=dict(color=AMBER, width=2.4),
                               fill="tozeroy", fillcolor=_rgba(AMBER, 0.15), hovertemplate="V=%{y:.3f}"))
    if energy is not None:
        fig.add_hline(y=energy, line=dict(color=GREEN, dash="dot"), annotation_text=f"E₀ = {energy:.2f}",
                      annotation_font_color=GREEN, annotation_position="top left")
    pad = 0.2 * max(np.ptp(V), abs(energy or 0), 0.5)
    top = max(float(V.max()), energy or 0) + pad
    fig.update_yaxes(title_text="V(x)", range=[min(0.0, float(V.min())) - pad, top])
    fig.update_xaxes(title_text="Position x", range=list(xlim) if xlim else None)
    _style(fig, title, height, legend=False)
    return fig


# ==========================================================================
# Probabilities and generic curves
# ==========================================================================
def fig_RTB_time(res: pe.SimulationResult, i: Optional[int] = None, height: int = 340) -> go.Figure:
    free = res.region[1] <= res.region[0]
    fig = go.Figure()
    names = ("P(x<0)", "P(x>0)", "") if free else ("R (left of region)", "T (right of region)", "B (inside region)")
    fig.add_trace(go.Scatter(x=res.times, y=res.R, name=names[0], line=dict(color=VIOLET, width=2)))
    fig.add_trace(go.Scatter(x=res.times, y=res.T, name=names[1], line=dict(color=CYAN, width=2)))
    if not free:
        fig.add_trace(go.Scatter(x=res.times, y=res.B, name=names[2], line=dict(color=AMBER, width=2)))
    fig.add_trace(go.Scatter(x=res.times, y=res.R + res.T + res.B, name="R+T+B",
                             line=dict(color=GREEN, width=1.4, dash="dot")))
    if i is not None:
        fig.add_vline(x=float(res.times[i]), line=dict(color="rgba(255,255,255,0.45)", dash="dot", width=1))
    fig.update_yaxes(title_text="Probability", range=[-0.02, 1.05])
    fig.update_xaxes(title_text="Time t")
    return _style(fig, "Probability in each region vs time", height)


def fig_RTB_bars(R: float, T: float, B: float, free: bool = False, height: int = 300) -> go.Figure:
    labels = ["P(x<0)", "P(x>0)"] if free else ["Reflection R", "Transmission T", "Inside barrier B"]
    vals = [R, T] if free else [R, T, B]
    cols = [VIOLET, CYAN] if free else [VIOLET, CYAN, AMBER]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=cols, text=[f"{v * 100:.2f}%" for v in vals],
                           textposition="outside", hovertemplate="%{x}: %{y:.5f}<extra></extra>"))
    fig.update_yaxes(title_text="Probability", range=[0, 1.15])
    fig = _style(fig, "Final probabilities", height, legend=False)
    fig.update_layout(hovermode="closest")
    return fig


def fig_lines(x, ys: Dict[str, Sequence[float]], xlabel: str, ylabel: str, title: Optional[str] = None,
              log_y: bool = False, markers: bool = False, height: int = 380, dashes: Optional[Dict[str, str]] = None,
              yrange: Optional[Tuple[float, float]] = None, colors: Optional[Dict[str, str]] = None) -> go.Figure:
    fig = go.Figure()
    for j, (name, y) in enumerate(ys.items()):
        col = (colors or {}).get(name, SERIES[j % len(SERIES)])
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines+markers" if markers else "lines",
                                 line=dict(color=col, width=2, dash=(dashes or {}).get(name, "solid")),
                                 marker=dict(size=7), hovertemplate="%{y:.5g}"))
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel, type="log" if log_y else "linear",
                     range=list(yrange) if yrange else None)
    return _style(fig, title, height)


def fig_TE(E, T, peaks=None, packet_points=None, V_ref: Optional[float] = None, title="Transmission probability vs energy",
           height: int = 430, log_y: bool = False) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=E, y=T, name="T(E) transfer matrix", line=dict(color=CYAN, width=2.2),
                             fill="tozeroy", fillcolor=_rgba(CYAN, 0.08), hovertemplate="T=%{y:.5f}"))
    if peaks is not None and len(peaks):
        fig.add_trace(go.Scatter(x=peaks["E_res"], y=peaks["T_peak"], mode="markers+text", name="Resonance peaks",
                                 marker=dict(color=PINK, size=10, symbol="diamond"),
                                 text=[f"E{n + 1}" for n in range(len(peaks))], textposition="top center",
                                 hovertemplate="E_res=%{x:.4f}<br>T=%{y:.4f}<extra></extra>"))
    if packet_points is not None and len(packet_points):
        fig.add_trace(go.Scatter(x=packet_points["E0"], y=packet_points["T"], mode="markers", name="Wave-packet simulation",
                                 marker=dict(color=AMBER, size=9, symbol="x"),
                                 hovertemplate="E₀=%{x:.4f}<br>T=%{y:.4f}<extra></extra>"))
    if V_ref is not None:
        fig.add_vline(x=V_ref, line=dict(color=AMBER, dash="dash", width=1),
                      annotation_text="barrier top", annotation_font_color=AMBER)
    fig.update_xaxes(title_text="Energy E = ħ²k²/2m")
    fig.update_yaxes(title_text="Transmission T", type="log" if log_y else "linear",
                     range=None if log_y else [-0.02, 1.08])
    return _style(fig, title, height)


def fig_momentum(md: Dict[str, np.ndarray], title="Momentum-space probability |φ(p)|²", height: int = 360,
                 ref: Optional[Dict[str, np.ndarray]] = None, mass: float = 1.0) -> go.Figure:
    p, d = md["p"], md["density"]
    cum = np.cumsum(d) / d.sum()
    lo = p[max(0, int(np.searchsorted(cum, 1e-5)) - 5)]
    hi = p[min(len(p) - 1, int(np.searchsorted(cum, 1 - 1e-5)) + 5)]
    pad = 0.15 * (hi - lo)
    fig = go.Figure()
    neg, pos = p < 0, p >= 0
    fig.add_trace(go.Scatter(x=p[neg], y=d[neg], name=f"p < 0  (weight {md['weight_neg']:.4f})", line=dict(color=VIOLET, width=2),
                             fill="tozeroy", fillcolor=_rgba(VIOLET, 0.25), hovertemplate="%{y:.5f}"))
    fig.add_trace(go.Scatter(x=p[pos], y=d[pos], name=f"p > 0  (weight {md['weight_pos']:.4f})", line=dict(color=CYAN, width=2),
                             fill="tozeroy", fillcolor=_rgba(CYAN, 0.25), hovertemplate="%{y:.5f}"))
    if ref is not None:
        fig.add_trace(go.Scatter(x=ref["p"], y=ref["density"], name="Initial packet (t = 0)",
                                 line=dict(color=AMBER, width=1.4, dash="dash"), hovertemplate="%{y:.5f}"))
    fig.update_xaxes(title_text="Momentum p = ħk", range=[lo - pad, hi + pad])
    fig.update_yaxes(title_text="|φ(p)|²")
    return _style(fig, title, height)


def fig_interference(x, dens, d_right, d_left, inter, xlim=None, height: int = 380) -> go.Figure:
    s = _stride(len(x))
    x = x[::s]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=dens[::s], name="|ψ|² (total)", line=dict(color=CYAN, width=2.2)))
    fig.add_trace(go.Scatter(x=x, y=d_right[::s], name="|ψ₊|² (right-moving)", line=dict(color=BLUE, width=1.3, dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=d_left[::s], name="|ψ₋|² (left-moving)", line=dict(color=VIOLET, width=1.3, dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=inter[::s], name="Interference 2Re(ψ₊*ψ₋)", line=dict(color=PINK, width=1.6),
                             fill="tozeroy", fillcolor=_rgba(PINK, 0.15)))
    fig.update_xaxes(title_text="Position x", range=list(xlim) if xlim else None)
    fig.update_yaxes(title_text="Probability density")
    return _style(fig, "Position space: |ψ|² = |ψ₊|² + |ψ₋|² + interference", height)


def fig_classical_quantum(res: pe.SimulationResult, i: int, pos_classical: np.ndarray, xlim=None, height: int = 420) -> go.Figure:
    """Quantum |psi|^2 against the density of an ensemble of classical particles at the same time."""
    s = _stride(len(res.x))
    x = res.x[::s]
    dx = res.cfg.dx
    edges = np.append(res.x - dx / 2, res.x[-1] + dx / 2)
    bw = max(1, int(round(0.25 / dx)))                       # classical histogram bins ~0.25 wide
    edges = edges[::bw]
    h, e = np.histogram(pos_classical, bins=edges, density=False)
    cl = h / (len(pos_classical) * np.diff(e))
    xc = 0.5 * (e[1:] + e[:-1])
    r = _ranges(res)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=res.density[i][::s], name="Quantum |ψ|²", line=dict(color=CYAN, width=2.2),
                             fill="tozeroy", fillcolor=_rgba(CYAN, 0.16)))
    fig.add_trace(go.Scatter(x=xc, y=cl, name="Classical ensemble density", line=dict(color=PINK, width=1.6, shape="hv")))
    if r["vmax"] > 0:
        vs = 0.85 * r["dmax"] / r["vmax"]
        fig.add_trace(go.Scatter(x=x, y=res.V[::s] * vs, name="V(x) (scaled)", customdata=res.V[::s],
                                 line=dict(color=AMBER, width=1.4, dash="dash"), hovertemplate="V=%{customdata:.3f}"))
    fig.update_yaxes(title_text="Probability density", range=[0, 1.25 * max(r["dmax"], float(cl.max()) * 0.9)])
    fig.update_xaxes(title_text="Position x", range=list(xlim) if xlim else None)
    return _style(fig, None, height)


# ==========================================================================
# Matplotlib (PNG / PDF export; light, print-friendly)
# ==========================================================================
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 110})
    return plt


C_CY, C_VI, C_AM, C_BL, C_PK, C_GR = "#0891b2", "#7c3aed", "#d97706", "#2563eb", "#db2777", "#059669"


def mpl_panels(res: pe.SimulationResult, i: int):
    plt = _mpl()
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    x, psi = res.x, res.psi[i]
    d = pe.probability_density(psi)
    ax[0, 0].plot(x, d, color=C_CY, lw=2, label="|ψ|²")
    if res.V.max() > 0:
        ax[0, 0].plot(x, res.V / res.V.max() * d.max() * 0.85 if d.max() else res.V, "--", color=C_AM, label="V (scaled)")
    ax[0, 0].set_title("Probability density"); ax[0, 0].legend()
    ax[0, 1].plot(x, psi.real, color=C_BL); ax[0, 1].set_title("Re ψ")
    ax[1, 0].plot(x, psi.imag, color=C_PK); ax[1, 0].set_title("Im ψ")
    ax[1, 1].plot(x, res.V, color=C_AM); ax[1, 1].set_title("Potential V(x)")
    for a in ax.flat:
        a.set_xlabel("x")
    fig.suptitle(f"Wave packet at t = {res.times[i]:.2f}   (total probability {res.norm[i]:.6f})")
    fig.tight_layout()
    return fig


def mpl_spacetime(res: pe.SimulationResult):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(res.density, aspect="auto", origin="lower", cmap="magma",
                   extent=[res.x[0], res.x[-1], res.times[0], res.times[-1]])
    for xe in set(res.region):
        if res.cfg.potential_kind != "free":
            ax.axvline(xe, color="cyan", ls="--", lw=1)
    ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_title("Space–time probability density")
    ax.grid(False)
    fig.colorbar(im, label="|ψ|²")
    fig.tight_layout()
    return fig


def mpl_lines(x, ys: Dict[str, Sequence[float]], xlabel: str, ylabel: str, title: str, markers: bool = False,
              log_y: bool = False, dashes: Optional[Dict[str, str]] = None, vlines: Optional[Dict[str, float]] = None):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cols = [C_CY, C_VI, C_AM, C_GR, C_PK, C_BL]
    for j, (n, y) in enumerate(ys.items()):
        ax.plot(x, y, ("-o" if markers else "-") if (dashes or {}).get(n) is None else (dashes or {})[n],
                color=cols[j % len(cols)], lw=1.8, ms=4, label=n)
    for n, v in (vlines or {}).items():
        ax.axvline(v, color="gray", ls=":", lw=1, label=n)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def mpl_TE(E, T, peaks=None, title="Transmission vs energy"):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(E, T, color=C_CY, lw=1.8, label="T(E)")
    if peaks is not None and len(peaks):
        ax.plot(peaks["E_res"], peaks["T_peak"], "D", color=C_PK, label="resonances")
    ax.set_xlabel("E"); ax.set_ylabel("T"); ax.set_title(title); ax.legend()
    fig.tight_layout()
    return fig


def mpl_regions(res: pe.SimulationResult, i: int, title: str = ""):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    left, inside, right = pot.region_masks(res.x, res.region)
    d = res.density[i]
    for m, c, n in ((left, C_VI, "left"), (inside, C_AM, "inside"), (right, C_CY, "right")):
        if m.any():
            ax.fill_between(res.x, np.where(m, d, 0), color=c, alpha=0.35, label=n)
    if res.V.max() > 0:
        ax.plot(res.x, res.V / res.V.max() * res.density.max() * 0.85, "--", color=C_AM, lw=1.2, label="V (scaled)")
    ax.set_xlabel("x"); ax.set_ylabel("|ψ|²"); ax.set_title(title or f"Probability density at t = {res.times[i]:.2f}")
    ax.legend()
    fig.tight_layout()
    return fig


def mpl_momentum(md, title="Momentum distribution"):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    p, d = md["p"], md["density"]
    ax.fill_between(p[p < 0], d[p < 0], color=C_VI, alpha=0.4, label="p<0")
    ax.fill_between(p[p >= 0], d[p >= 0], color=C_CY, alpha=0.4, label="p>0")
    cum = np.cumsum(d) / d.sum()
    ax.set_xlim(p[max(0, np.searchsorted(cum, 1e-5) - 5)], p[min(len(p) - 1, np.searchsorted(cum, 1 - 1e-5) + 5)])
    ax.set_xlabel("p"); ax.set_ylabel("|φ(p)|²"); ax.set_title(title); ax.legend()
    fig.tight_layout()
    return fig


def fig_to_png(fig, dpi: int = 160) -> bytes:
    import io
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def fig_multi_xy(series: Dict[str, Tuple[Sequence[float], Sequence[float]]], xlabel: str, ylabel: str,
                 title: Optional[str] = None, height: int = 360, log_y: bool = False) -> go.Figure:
    """Curves that do not share an x array: {name: (x, y)}."""
    fig = go.Figure()
    for j, (name, (x, y)) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=x, y=y, name=name, line=dict(color=SERIES[j % len(SERIES)], width=2),
                                 hovertemplate="%{y:.5g}"))
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel, type="log" if log_y else "linear")
    fig = _style(fig, title, height)
    fig.update_layout(hovermode="closest")
    return fig
