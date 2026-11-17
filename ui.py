"""
ui.py - Streamlit helpers: theme/CSS, metric cards, explanation panels, animation player, export panel.
(No physics here: the numerical engine lives in physics_engine.py / analysis.py.)
"""
from __future__ import annotations

import html
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import streamlit as st

import report as rpt

PLOT_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2},
               "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');
:root{--cyan:#22d3ee;--blue:#4f8cff;--violet:#a78bfa;--amber:#fbbf24;--ink:#e6edf7;--mute:#93a4c0;
--panel:rgba(14,21,48,.62);--edge:rgba(120,150,255,.16);}
html,body,[class*="css"]{font-family:'Inter',system-ui,sans-serif;}
.stApp{background:
 radial-gradient(1100px 560px at 8% -8%,rgba(79,140,255,.20),transparent 60%),
 radial-gradient(900px 520px at 100% 2%,rgba(167,139,250,.16),transparent 60%),
 radial-gradient(700px 500px at 50% 120%,rgba(34,211,238,.08),transparent 60%),#04071a;}
[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:2.2rem;max-width:1320px;}
h1,h2,h3,h4{font-family:'Space Grotesk','Inter',sans-serif;letter-spacing:-.01em;color:var(--ink);}
[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(7,12,32,.97),rgba(4,7,22,.97));
 border-right:1px solid var(--edge);}
[data-testid="stSidebar"] .stRadio label{padding:.38rem .6rem;border-radius:10px;transition:background .2s,transform .2s;}
[data-testid="stSidebar"] .stRadio label:hover{background:rgba(79,140,255,.12);transform:translateX(2px);}
.qw-brand{font-family:'Space Grotesk';font-weight:700;font-size:1.35rem;
 background:linear-gradient(90deg,var(--cyan),var(--violet));-webkit-background-clip:text;background-clip:text;color:transparent;}
.qw-brand-sub{color:var(--mute);font-size:.74rem;letter-spacing:.14em;text-transform:uppercase;margin-bottom:.8rem;}
.qw-hero{padding:2.1rem 2rem;border-radius:22px;border:1px solid var(--edge);position:relative;overflow:hidden;
 background:linear-gradient(135deg,rgba(20,32,76,.72),rgba(18,12,52,.62));backdrop-filter:blur(10px);margin-bottom:1.2rem;}
.qw-hero:before{content:"";position:absolute;inset:-40% -10% auto auto;width:520px;height:520px;
 background:radial-gradient(circle,rgba(34,211,238,.20),transparent 62%);pointer-events:none;}
.qw-kicker{color:var(--cyan);font-size:.78rem;letter-spacing:.18em;text-transform:uppercase;font-weight:600;}
.qw-title{font-family:'Space Grotesk';font-size:clamp(1.8rem,4.2vw,3rem);font-weight:700;line-height:1.08;margin:.3rem 0 .6rem;
 background:linear-gradient(95deg,#fff 10%,#9fd8ff 45%,#c4b5fd 90%);-webkit-background-clip:text;background-clip:text;color:transparent;}
.qw-lead{color:#b6c4de;font-size:1.04rem;max-width:62ch;line-height:1.6;}
.qw-section{margin:.4rem 0 .8rem;}
.qw-section h3{margin:0;font-size:1.25rem;}
.qw-section p{margin:.15rem 0 0;color:var(--mute);font-size:.93rem;}
.qw-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(142px,1fr));gap:.65rem;margin:.4rem 0 1rem;}
.qw-card{background:var(--panel);border:1px solid var(--edge);border-radius:14px;padding:.7rem .85rem;backdrop-filter:blur(8px);
 transition:transform .2s,border-color .2s;}
.qw-card:hover{transform:translateY(-2px);border-color:rgba(34,211,238,.45);}
.qw-label{color:var(--mute);font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;}
.qw-value{font-family:'JetBrains Mono',monospace;font-weight:600;font-size:1.28rem;color:#f3f8ff;margin-top:.12rem;}
.qw-sub{color:var(--mute);font-size:.72rem;margin-top:.05rem;}
.qw-ok{color:#34d399!important;}.qw-warn{color:#fbbf24!important;}.qw-cy{color:var(--cyan)!important;}.qw-vi{color:var(--violet)!important;}
.qw-feature{background:var(--panel);border:1px solid var(--edge);border-radius:16px;padding:1rem 1.1rem;height:100%;
 transition:transform .25s,border-color .25s,box-shadow .25s;}
.qw-feature:hover{transform:translateY(-3px);border-color:rgba(167,139,250,.5);box-shadow:0 8px 30px rgba(79,140,255,.12);}
.qw-feature h4{margin:.1rem 0 .3rem;font-size:1.02rem;}
.qw-feature p{color:#aebbd4;font-size:.9rem;margin:0;line-height:1.5;}
.qw-tag{display:inline-block;font-size:.68rem;padding:.12rem .5rem;border-radius:99px;border:1px solid var(--edge);color:var(--cyan);
 margin-right:.3rem;letter-spacing:.05em;}
div[data-testid="stVerticalBlockBorderWrapper"]{background:var(--panel);border-radius:16px;backdrop-filter:blur(8px);}
.stButton>button,.stDownloadButton>button{border-radius:11px;border:1px solid var(--edge);
 background:linear-gradient(135deg,rgba(34,211,238,.14),rgba(167,139,250,.14));color:var(--ink);transition:all .2s;}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--cyan);transform:translateY(-1px);
 box-shadow:0 4px 18px rgba(34,211,238,.18);color:#fff;}
.stTabs [data-baseweb="tab"]{border-radius:10px 10px 0 0;}
.stTabs [aria-selected="true"]{color:var(--cyan)!important;}
details{border-radius:14px!important;border:1px solid var(--edge)!important;background:rgba(10,16,40,.5)!important;}
.qw-eq{background:rgba(6,10,30,.6);border:1px solid var(--edge);border-radius:14px;padding:.4rem .8rem;margin:.5rem 0;}
.qw-note{border-left:3px solid var(--violet);padding:.5rem .9rem;background:rgba(167,139,250,.07);border-radius:0 10px 10px 0;
 color:#c7d2e6;font-size:.92rem;margin:.5rem 0;}
@media (max-width:720px){.block-container{padding:1.2rem .8rem;}.qw-hero{padding:1.3rem 1.1rem;}.qw-value{font-size:1.05rem;}}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Small wrappers that survive Streamlit API changes
# --------------------------------------------------------------------------
def st_plot(fig, key: str) -> None:
    try:
        st.plotly_chart(fig, width="stretch", key=key, config=PLOT_CONFIG, theme=None)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key, config=PLOT_CONFIG, theme=None)


def st_df(df, **kw) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def hero(kicker: str, title: str, lead: str) -> None:
    st.markdown(f'<div class="qw-hero"><div class="qw-kicker">{html.escape(kicker)}</div>'
                f'<div class="qw-title">{html.escape(title)}</div><div class="qw-lead">{lead}</div></div>',
                unsafe_allow_html=True)


def section(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="qw-section"><h3>{html.escape(title)}</h3>'
                + (f"<p>{sub}</p>" if sub else "") + "</div>", unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="qw-note">{text}</div>', unsafe_allow_html=True)


def cards(items: Sequence[Tuple], key: Optional[str] = None) -> None:
    """items: (label, value, sub=None, css_class=None)"""
    out = ['<div class="qw-grid">']
    for it in items:
        label, value = it[0], it[1]
        sub = it[2] if len(it) > 2 and it[2] else ""
        cls = it[3] if len(it) > 3 and it[3] else ""
        out.append(f'<div class="qw-card"><div class="qw-label">{html.escape(str(label))}</div>'
                   f'<div class="qw-value {cls}">{html.escape(str(value))}</div>'
                   + (f'<div class="qw-sub">{html.escape(str(sub))}</div>' if sub else "") + "</div>")
    out.append("</div>")
    st.markdown("".join(out), unsafe_allow_html=True)


def science_panel(phenomenon: str, equation: Optional[str], observe: str, why: str, meaning: str,
                  title: str = "🔬 Scientific explanation", expanded: bool = False) -> None:
    with st.expander(title, expanded=expanded):
        st.markdown(f"**What phenomenon is demonstrated?**  \n{phenomenon}")
        if equation:
            st.markdown("**Governing equation**")
            st.latex(equation)
        st.markdown(f"**What should you observe?**  \n{observe}")
        st.markdown(f"**Why does it happen?**  \n{why}")
        st.markdown(f"**What does it mean physically?**  \n{meaning}")


def show_messages(errors: List[str], warnings: List[str]) -> bool:
    """Display validation output.  Returns True if the run must be blocked."""
    for e in errors:
        st.error(e, icon="⛔")
    for w in warnings:
        st.warning(w, icon="⚠️")
    return bool(errors)


def sliders_from_specs(specs, prefix: str, defaults: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    out = {}
    for sp in specs:
        d = (defaults or {}).get(sp.name, sp.default)
        out[sp.name] = st.slider(sp.label, float(sp.lo), float(sp.hi), float(d), float(sp.step),
                                 key=f"{prefix}_{sp.name}", help=sp.help or None)
    return out


# --------------------------------------------------------------------------
# Animation player (Play / Pause / Reset / step / time slider / speed)
# --------------------------------------------------------------------------
def player(key: str, res, draw: Callable, default_speed: float = 2.0) -> int:
    """Interactive time-evolution player.

    The body runs inside a Streamlit *fragment* that re-runs itself on a timer while playing, so only the
    plots update (not the whole page).  Pause/Play changes the timer, which needs one full rerun.
    `draw(res, idx)` renders everything that depends on the current snapshot.
    """
    ss = st.session_state
    n = res.n_snap
    if ss.get(f"{key}_sig") != res.signature:
        ss[f"{key}_sig"] = res.signature
        ss[f"{key}_t"] = 0
        ss[f"{key}_play"] = False
    ss[f"{key}_t"] = int(min(max(int(ss.get(f"{key}_t", 0)), 0), n - 1))

    speed = st.select_slider("Animation speed", options=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0], value=default_speed,
                             key=f"{key}_speed", format_func=lambda v: f"{v:g}×")
    step = max(1, int(speed / 2))
    interval = max(0.08, 0.2 * step / speed)
    run_every = interval if ss.get(f"{key}_play", False) else None

    @st.fragment(run_every=run_every)
    def _body():
        playing = bool(ss.get(f"{key}_play", False))
        if playing:                                   # advance BEFORE the slider is created
            nxt = ss[f"{key}_t"] + step
            if nxt >= n - 1:
                ss[f"{key}_t"] = n - 1
                ss[f"{key}_play"] = False
                st.rerun()                            # full rerun stops the timer
            else:
                ss[f"{key}_t"] = nxt
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("⏮ Reset", key=f"{key}_b_reset"):
            ss[f"{key}_t"] = 0
            if playing:
                ss[f"{key}_play"] = False
                st.rerun()
        if c2.button("◀ Step back", key=f"{key}_b_back"):
            ss[f"{key}_t"] = max(0, ss[f"{key}_t"] - 1)
            if playing:
                ss[f"{key}_play"] = False
                st.rerun()
        if c3.button("⏸ Pause" if playing else "▶ Play", key=f"{key}_b_play"):
            if not playing and ss[f"{key}_t"] >= n - 1:
                ss[f"{key}_t"] = 0                    # replay from the start
            ss[f"{key}_play"] = not playing
            st.rerun()
        if c4.button("Step forward ▶", key=f"{key}_b_fwd"):
            ss[f"{key}_t"] = min(n - 1, ss[f"{key}_t"] + 1)
            if playing:
                ss[f"{key}_play"] = False
                st.rerun()
        st.slider("Time slider (snapshot index)", 0, n - 1, key=f"{key}_t",
                  help=f"{n} stored snapshots between t = 0 and t = {res.times[-1]:.2f}")
        draw(res, int(ss[f"{key}_t"]))

    _body()
    return int(ss[f"{key}_t"])


# --------------------------------------------------------------------------
# Export panel
# --------------------------------------------------------------------------
def export_panel(b: rpt.ReportBundle, extra_csv: Optional[Dict[str, bytes]] = None) -> None:
    ss = st.session_state
    with st.expander("📦 Export: CSV · PNG · summary · PDF report"):
        c1, c2, c3, c4 = st.columns(4)
        c1.download_button("⬇ CSV data", bundle_csv_cached(b), file_name=f"{b.key}.csv", mime="text/csv", key=f"{b.key}_dl_csv")
        c2.download_button("📝 Summary (.md)", rpt.bundle_summary(b).encode(), file_name=f"{b.key}_summary.md",
                           mime="text/markdown", key=f"{b.key}_dl_md")
        sig = b.signature()
        if c3.button("🖼 Prepare PNG graphs", key=f"{b.key}_prep_png"):
            with st.spinner("Rendering figures…"):
                ss[f"{b.key}_png"] = (sig, *rpt.bundle_pngs(b))
        stored = ss.get(f"{b.key}_png")
        if stored and stored[0] == sig:
            c3.download_button("⬇ Download PNG", stored[1], file_name=stored[2], mime=stored[3], key=f"{b.key}_dl_png")
        if c4.button("📄 Prepare PDF report", key=f"{b.key}_prep_pdf"):
            with st.spinner("Building PDF report…"):
                ss[f"{b.key}_pdf"] = (sig, rpt.bundle_pdf(b))
        stored = ss.get(f"{b.key}_pdf")
        if stored and stored[0] == sig:
            c4.download_button("⬇ Download PDF", stored[1], file_name=f"{b.key}_report.pdf", mime="application/pdf",
                               key=f"{b.key}_dl_pdf")
        st.caption("Every interactive chart also has a camera icon (top-right on hover) that saves it as a PNG.")


def bundle_csv_cached(b: rpt.ReportBundle) -> bytes:
    return rpt.bundle_csv(b)
