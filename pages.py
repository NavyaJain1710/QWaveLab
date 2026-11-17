"""
pages.py - the nine pages of QWaveLab.  Each page_* function renders one page.
All numbers shown come from physics_engine / analysis / experiments (no hard-coded results).
"""
from __future__ import annotations

import dataclasses
import math
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

import analysis as an
import experiments as ex
import physics_engine as pe
import potentials as pot
import report as rpt
import ui
import visualizations as viz
from physics_engine import HBAR

PAGE_NAMES = [
    "🏠 Home – Quantum Laboratory",
    "🌊 Wave Packet Simulator",
    "🧪 Potential Laboratory",
    "🚇 Quantum Tunnelling Explorer",
    "🎯 Resonant Tunnelling Lab",
    "〰️ Interference & Momentum",
    "⚖️ Classical vs Quantum",
    "📊 Experiment Analysis",
    "✅ Numerical Validation",
]

DX_OPTIONS = [0.2, 0.1, 0.05, 0.04, 0.025]
DT_OPTIONS = [0.05, 0.02, 0.01, 0.005]


# ==========================================================================
# Shared helpers
# ==========================================================================
@st.cache_resource(show_spinner=False, max_entries=12)
def _cached_sim(key: tuple) -> pe.SimulationResult:
    """Cache simulations by configuration: moving a *display* control never re-solves the PDE."""
    return pe.simulate(pe.SimConfig.from_key(key))


def run_sim(cfg: Optional[pe.SimConfig]) -> Optional[pe.SimulationResult]:
    if cfg is None:
        return None
    errs, warns = pe.validate_config(cfg)
    if ui.show_messages(errs, warns):
        return None
    with st.spinner("Solving the time-dependent Schrödinger equation (Crank–Nicolson)…"):
        return _cached_sim(cfg.key())


def auto_cfg(kind, params, k0, sigma, mass, dx, dt, extra_time: float = 0.0) -> Optional[pe.SimConfig]:
    try:
        return ex.auto_scattering_config(kind, params, k0, sigma, mass, dx, dt, extra_time=extra_time)
    except ValueError as e:
        st.error(str(e), icon="⛔")
        return None


def numerics_expander(prefix: str, dx=0.1, dt=0.02):
    with st.expander("Numerical resolution"):
        d = st.select_slider("Grid spacing Δx", DX_OPTIONS, value=dx, key=f"{prefix}_dx",
                             help="Smaller Δx resolves shorter wavelengths but costs more memory/time.")
        t = st.select_slider("Time step Δt", DT_OPTIONS, value=dt, key=f"{prefix}_dt",
                             help="Crank–Nicolson is unconditionally stable and unitary; Δt controls accuracy.")
    return d, t


def pct(v: float) -> str:
    return f"{100 * v:.2f}%"


def region_labels(res):
    free = res.region[1] <= res.region[0]
    return ("P(x<0)", "P(x>0)", "—") if free else ("Reflection R", "Transmission T", "Inside region B")


def metrics_cards(res, i: int, extra=None):
    m = res.metrics_at(i)
    lr, lt, lb = region_labels(res)
    items = [("Time t", f"{m['t']:.2f}"),
             ("Total probability", f"{m['total']:.8f}", "∫|ψ|²dx", "qw-ok" if abs(m['total'] - 1) < 1e-6 else "qw-warn"),
             (lr, pct(m["R"]), None, "qw-vi"), (lt, pct(m["T"]), None, "qw-cy")]
    if res.region[1] > res.region[0]:
        items.append((lb, pct(m["B"]), None, "qw-warn"))
    items += [("Expected energy ⟨E⟩", f"{m['energy']:.4f}"), ("Position spread Δx", f"{m['delta_x']:.3f}"),
              ("Momentum spread Δp", f"{m['delta_p']:.3f}")]
    ui.cards(items + (extra or []))


def draw_regions(key: str, xlim=None, title=None):
    def _draw(res, i):
        metrics_cards(res, i)
        ui.st_plot(viz.fig_density_regions(res, i, xlim=xlim, title=title), f"{key}_dens")
    return _draw


def prob_tabs(res, key: str, i: Optional[int] = None):
    free = res.region[1] <= res.region[0]
    c1, c2 = st.columns([1.5, 1])
    with c1:
        ui.st_plot(viz.fig_RTB_time(res, i), f"{key}_rtb_t")
    with c2:
        R, T, B = res.final_RTB()
        ui.st_plot(viz.fig_RTB_bars(R, T, B, free), f"{key}_rtb_bar")


def active_xlim(res, floor: float = 1e-3, pad: float = 3.0):
    """x-range where |psi|^2 ever exceeds `floor` x its maximum, so plots zoom on the action."""
    d = np.abs(res.psi).max(axis=0) ** 2
    idx = np.flatnonzero(d > floor * d.max())
    lo, hi = float(res.x[idx[0]]) - pad, float(res.x[idx[-1]]) + pad
    return (max(lo, float(res.x[0])), min(hi, float(res.x[-1])))


def _goto(name: str):
    st.session_state["nav"] = name


# ==========================================================================
# PAGE 1 - Home
# ==========================================================================
def page_home():
    ui.hero("Interactive Quantum Wave Packet Simulation Laboratory", "QWaveLab",
            "Watch a quantum particle move, spread, reflect, interfere and <b>tunnel</b> through barriers. "
            "Every curve and every percentage on this site comes from a live numerical solution of the "
            "time-dependent Schrödinger equation (Crank–Nicolson finite differences) – not from canned data.")
    st.button("🚀 Enter the simulation laboratory", on_click=_goto, args=(PAGE_NAMES[1],), key="home_cta")

    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        ui.section("What is a wave packet?")
        st.markdown(
            "A quantum particle is described by a complex wave function ψ(x,t). A **wave packet** is a superposition "
            "of plane waves, localised in space (width σ) and with a narrow spread of momenta around ħk₀. "
            "Its centre moves like a classical particle with velocity ħk₀/m, but unlike a classical particle it "
            "**spreads**, can be partly **reflected** by a barrier it classically could cross, and can "
            "**tunnel** through a barrier it classically could not.")
        st.markdown("**Time-dependent Schrödinger equation**")
        st.latex(r"i\hbar\,\frac{\partial \psi}{\partial t}=\left[-\frac{\hbar^{2}}{2m}\frac{\partial^{2}}{\partial x^{2}}+V(x)\right]\psi")
        st.markdown("**Initial Gaussian packet**")
        st.latex(r"\psi(x,0)=\left(2\pi\sigma^2\right)^{-1/4}\,e^{-(x-x_0)^2/4\sigma^2}\,e^{ik_0x}")
        st.markdown("**Crank–Nicolson step (unitary, unconditionally stable)**")
        st.latex(r"\left(I+\frac{i\Delta t}{2\hbar}H\right)\psi^{n+1}=\left(I-\frac{i\Delta t}{2\hbar}H\right)\psi^{n}")
    with c2:
        ui.section("Project objectives")
        st.markdown(
            "- Solve the 1-D time-dependent Schrödinger equation numerically.\n"
            "- Study **propagation and spreading** of a Gaussian packet.\n"
            "- Measure **reflection R**, **transmission T** and the probability inside the barrier **B**.\n"
            "- Explore **tunnelling**, **interference** and **resonant tunnelling**.\n"
            "- Verify **probability / energy conservation** and the **uncertainty principle**.\n"
            "- Quantify numerical accuracy by **convergence tests**.")
        st.markdown("".join(f'<span class="qw-tag">{t}</span>' for t in
                            ["propagation", "spreading", "reflection", "transmission", "tunnelling", "interference",
                             "resonance", "uncertainty", "conservation"]), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        res = _cached_sim(pe.SimConfig.make("rect_barrier", dict(V0=2.5, width=1.5)).key())
        ui.st_plot(viz.fig_spacetime(res, xlim=(-30, 30), height=300, title="Live preview: |ψ(x,t)|² (default experiment)"),
                   "home_preview")

    ui.section("Experiments in the laboratory", "Each page is a self-contained numerical experiment with its own explanation panel.")
    feats = [("🌊 Wave Packet Simulator", "Full control of x₀, σ, k₀, m, V₀, barrier width, duration and grid. Play, pause, step and scrub through time."),
             ("🧪 Potential Laboratory", "Free particle, rectangular, Gaussian and double barriers, finite well. R and T for each."),
             ("🚇 Tunnelling Explorer", "Incoming, reflected and transmitted packets with live R, T, B and the R+T+B check."),
             ("🎯 Resonant Tunnelling", "Double barrier: T(E) from the transfer matrix, resonances found numerically, lifetimes."),
             ("〰️ Interference & Momentum", "FFT momentum spectrum, ±p components, interference term, Δx·Δp ≥ ħ/2."),
             ("⚖️ Classical vs Quantum", "A classical ensemble and a quantum packet meet the same barrier at the same energy."),
             ("📊 Experiment Analysis", "Sweeps over V₀, barrier width and k₀; tables, plots, interpretation, CSV export."),
             ("✅ Numerical Validation", "Norm & energy conservation, exact free-packet comparison, convergence orders.")]
    for row in range(0, len(feats), 4):
        cols = st.columns(4)
        for col, (t, d) in zip(cols, feats[row:row + 4]):
            col.markdown(f'<div class="qw-feature"><h4>{t}</h4><p>{d}</p></div>', unsafe_allow_html=True)
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

    with st.expander("📐 Units, scientific assumptions and limitations"):
        st.markdown(
            "**Units.** Dimensionless by default, ħ = 1 and m = 1. Lengths are in units L₀, energies in ħ²/(m₀L₀²), "
            "times in m₀L₀²/ħ. To interpret physically, choose L₀ and m₀: e.g. for an electron with L₀ = 1 nm, "
            "the energy unit is ħ²/(m_eL₀²) ≈ 0.0381 eV and the time unit ≈ 17.3 fs.\n\n"
            "**Assumptions & limitations.**\n"
            "- One spatial dimension, single particle, time-independent potential, no spin or interactions.\n"
            "- The box has **hard (Dirichlet) walls**: ψ = 0 at both ends. Results are only physical until a packet "
            "reaches a wall; the app warns when that can happen.\n"
            "- Finite differences have O(Δx²) lattice dispersion; use ≥ 10 grid points per de Broglie wavelength.\n"
            "- Crank–Nicolson is exactly unitary, so norm and ⟨H⟩ are conserved to round-off; the error shows up as "
            "phase/shape error that scales as O(Δt²).\n"
            "- R, T and B are probabilities integrated over fixed regions (left of / inside / right of the potential). "
            "They are meaningful once the collision is over; the app flags when it is not.\n"
            "- A wave packet has a spread of energies, so its T differs from the single-energy (plane-wave) T(E).")


# ==========================================================================
# PAGE 2 - Wave Packet Simulator
# ==========================================================================
def page_simulator():
    ui.section("Wave Packet Simulator", "Gaussian packet meeting a rectangular barrier – every control re-solves (or re-draws) the simulation.")
    left, right = st.columns([1, 2.6], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown("##### 🌊 Wave packet")
            x0 = st.slider("Initial position x₀", -45.0, 20.0, -12.0, 0.5, key="sim_x0")
            sigma = st.slider("Width σ", 0.3, 6.0, 1.2, 0.1, key="sim_sigma")
            k0 = st.slider("Wave number k₀", 0.0, 6.0, 2.0, 0.1, key="sim_k0")
            mass = st.slider("Particle mass m", 0.2, 5.0, 1.0, 0.1, key="sim_m")
            st.markdown("##### 🧱 Barrier")
            V0 = st.slider("Height V₀", 0.0, 10.0, 2.5, 0.1, key="sim_V0")
            bw = st.slider("Width a", 0.1, 6.0, 1.5, 0.1, key="sim_bw")
            bc = st.slider("Centre", -10.0, 10.0, 0.0, 0.5, key="sim_bc")
            st.markdown("##### ⏱ Simulation")
            t_max = st.slider("Duration t_max", 1.0, 80.0, 14.0, 1.0, key="sim_tmax")
            L = st.slider("Domain half-length L", 20.0, 150.0, 50.0, 5.0, key="sim_L")
            dx, dt = numerics_expander("sim")
            st.markdown("##### 👁 Display")
            view = st.slider("Plotted x-range", -float(L), float(L), (-min(L, 35.0), min(L, 35.0)), 1.0, key="sim_view")
        E0 = HBAR ** 2 * k0 ** 2 / (2 * mass)
        st.caption(f"Carrier energy E₀ = ħ²k₀²/2m = **{E0:.3f}**  ·  barrier V₀ = **{V0:.2f}**  ·  "
                   + ("tunnelling regime (E₀ < V₀)" if E0 < V0 else "over-barrier regime (E₀ ≥ V₀)"))
    cfg = pe.SimConfig.make("rect_barrier", dict(V0=V0, width=bw, center=bc), x_min=-L, x_max=L, dx=dx, dt=dt,
                            t_max=t_max, mass=mass, x0=x0, sigma=sigma, k0=k0)
    with right:
        res = run_sim(cfg)
        if res is None:
            return
        ui.player("sim", res, lambda r, i: _sim_draw(r, i, tuple(view)))
    tabs = st.tabs(["Space–time map", "Probabilities", "Conservation", "Data & export"])
    with tabs[0]:
        ui.st_plot(viz.fig_spacetime(res, xlim=tuple(view)), "sim_heat")
    with tabs[1]:
        prob_tabs(res, "sim")
        st.caption("R = probability left of the barrier, T = right of it, B = inside it. At t = 0 all probability is on the left, "
                   "so R only equals the *reflection coefficient* after the collision is over.")
        if not res.scattering_complete():
            st.info("The collision is not finished at the final time (B is still non-negligible, or the packet hasn't "
                    "reached the barrier). Increase the duration to read off final R and T.")
    with tabs[2]:
        _conservation_plots(res, "sim")
    with tabs[3]:
        df = rpt.result_to_dataframe(res)
        ui.st_df(df.round(8))
    ui.science_panel(
        "Propagation, spreading, reflection, transmission and tunnelling of a Gaussian wave packet at a rectangular barrier.",
        r"i\hbar\,\partial_t\psi=-\tfrac{\hbar^2}{2m}\partial_x^2\psi+V(x)\psi",
        "The packet moves right at v = ħk₀/m while its width grows. At the barrier |ψ|² splits into a reflected and a "
        "transmitted packet; fringes appear on the left where incident and reflected waves overlap. "
        "Re ψ and Im ψ oscillate inside the |ψ| envelope with wavelength 2π/k₀.",
        "A wave must be continuous with continuous slope at the barrier edges. Matching the oscillating solution outside "
        "to an exponentially decaying (E < V₀) or oscillating-with-different-wavelength (E > V₀) solution inside forces "
        "part of the wave to be reflected and part transmitted. Spreading follows from the momentum spread ħ/2σ: faster "
        "components outrun slower ones.",
        "The particle has no definite trajectory; |ψ|² gives probabilities. R + T + B = 1 at all times because the "
        "Crank–Nicolson evolution is unitary. Raising V₀ or the width lowers T; raising k₀ raises it.")
    ui.export_panel(rpt.standard_bundle("simulator", "Wave Packet Simulator – experiment report", res,
                                        None))


def _sim_draw(res, i, view):
    metrics_cards(res, i)
    ui.st_plot(viz.fig_snapshot_panels(res, i, xlim=view), "sim_panels")


def _conservation_plots(res, key: str):
    c1, c2 = st.columns(2)
    dev = np.abs(res.step_norm - res.norm[0])
    with c1:
        ui.st_plot(viz.fig_lines(res.step_times, {"|∫|ψ|²dx − 1|": np.maximum(dev, 1e-17)}, "Time t",
                                 "Norm deviation", "Probability conservation (every time step)", log_y=True), f"{key}_cons_p")
    with c2:
        ui.st_plot(viz.fig_lines(res.times, {"⟨H⟩": res.energy, "⟨K⟩": res.kin_energy, "⟨V⟩": res.pot_energy},
                                 "Time t", "Energy", "Energy expectation values"), f"{key}_cons_e")
    e = an.conservation_errors(res)
    ui.cards([("Max |norm − 1|", f"{e['max_norm_deviation']:.1e}"),
              ("Max relative ⟨H⟩ drift", f"{e['max_energy_rel_drift']:.1e}"),
              ("Final norm", f"{e['final_norm']:.10f}"), ("Initial ⟨H⟩", f"{e['initial_energy']:.5f}")])
    st.caption("Total energy is conserved while kinetic and potential energy are exchanged during the collision.")


# ==========================================================================
# PAGE 3 - Potential Laboratory
# ==========================================================================
KIND_TEXT = {
    "free": ("Free propagation and dispersive spreading of a Gaussian packet (no force).",
             "The packet drifts at v = ħk₀/m, its width follows σ(t) = σ√(1+(ħt/2mσ²)²); P(x<0) falls to 0 and P(x>0) to 1 as it crosses the origin.",
             "Different momentum components travel at different speeds (ω = ħk²/2m is not linear in k), so the packet disperses.",
             "Even without forces a localised quantum particle becomes less localised with time."),
    "rect_barrier": ("Quantum reflection and transmission at a rectangular barrier, including tunnelling when E < V₀.",
                     "A reflected and a transmitted packet; T rises with energy and falls with V₀ and width.",
                     "Boundary conditions (ψ and ψ′ continuous) at both edges; inside the barrier the wave decays as e^(−κx) when E < V₀.",
                     "Classical particles are reflected for E < V₀ and transmitted for E > V₀; waves do neither perfectly."),
    "gauss_barrier": ("Scattering from a smooth (Gaussian) barrier.",
                      "Compared with a sharp barrier of similar height, the over-barrier reflection is much weaker.",
                      "A smooth potential reflects only if its variation is rapid on the scale of the de Broglie wavelength.",
                      "Sharp edges are an efficient source of quantum reflection; smooth potentials are almost reflectionless at high energy (adiabatic regime)."),
    "double_barrier": ("Interference between two barriers: the Fabry–Pérot analogue for matter waves.",
                       "Parts of the packet bounce between the barriers and leak out in a train of delayed sub-packets; T depends strongly on k₀.",
                       "Waves reflected from the two barriers interfere; when the round-trip phase is a multiple of 2π they add constructively inside the well.",
                       "Resonant tunnelling – the principle behind resonant-tunnelling diodes (see the Resonant Tunnelling page)."),
    "finite_well": ("Quantum reflection from an attractive well – impossible classically.",
                    "A fraction of the packet is reflected even though the potential is attractive; T oscillates with energy (Ramsauer–Townsend-like).",
                    "The abrupt change of local wavelength at the well edges reflects part of the wave; interference of the two edge reflections gives energy-dependent T.",
                    "Reflection comes from changes of the wave's properties, not from repulsive forces.")}


def page_potential_lab():
    ui.section("Potential Laboratory", "Choose a potential, tune it, and measure reflection and transmission from the numerical solution.")
    kind = st.selectbox("Potential configuration", list(pot.KINDS), format_func=pot.KINDS.get, key="pl_kind")
    left, right = st.columns([1, 2.6], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown("##### Potential parameters")
            pp = ui.sliders_from_specs(pot.PARAM_SPECS[kind], f"pl_{kind}")
            if kind == "free":
                st.caption("No parameters: V(x) = 0.")
            st.markdown("##### Wave packet")
            sigma = st.slider("Width σ", 1.0, 8.0, 3.0, 0.1, key="pl_sigma")
            k0 = st.slider("Wave number k₀", 0.3, 5.0, 2.0, 0.05, key="pl_k0")
            mass = st.slider("Mass m", 0.2, 5.0, 1.0, 0.1, key="pl_m")
            dx, dt = numerics_expander("pl")
    cfg = auto_cfg(kind, pp, k0, sigma, mass, dx, dt)
    with right:
        if cfg is None:
            return
        xg = pe.make_grid(cfg.x_min, cfg.x_max, cfg.dx)
        P = pot.build_potential(kind, xg, pp)
        a, b = P.region
        span = (a - 4 * sigma - 2, b + 6 * sigma) if kind != "free" else (-30, 30)
        ui.st_plot(viz.fig_potential(xg, P.V, f"Potential: {pot.KINDS[kind]}", energy=cfg.energy_nominal,
                                     xlim=(min(span[0], cfg.x0 - 3 * sigma), max(span[1], 10))), "pl_pot")
        res = run_sim(cfg)
        if res is None:
            return
        ui.player("pl", res, draw_regions("pl", xlim=active_xlim(res)))
    R, T, B = res.final_RTB()
    free = kind == "free"
    ui.section("Final probabilities", "Computed at the end of the run." + ("" if res.scattering_complete() else " ⚠ collision not fully complete."))
    ui.cards([("Reflection R" if not free else "P(x<0)", pct(R), None, "qw-vi"),
              ("Transmission T" if not free else "P(x>0)", pct(T), None, "qw-cy")]
             + ([] if free else [("Inside region B", pct(B), None, "qw-warn")])
             + [("R + T + B", f"{R + T + B:.10f}", None, "qw-ok")])
    prob_tabs(res, "pl")
    if not free:
        E = np.linspace(0.02, max(2.0 * cfg.energy_nominal, 1.3 * float(np.abs(res.V).max()), 1.0), 700)
        Tth = an.transmission_vs_energy(res.x, res.V, E, mass)
        pk = pd.DataFrame(dict(E0=[cfg.energy_nominal], T=[T]))
        ui.st_plot(viz.fig_TE(E, Tth, packet_points=pk, title="Plane-wave T(E) (transfer matrix) and this packet's T"), "pl_TE")
        st.caption("The orange × is the wave packet's T from the time-dependent simulation. It differs from T(E₀) when T(E) "
                   "varies over the packet's energy spread ΔE ≈ ħ²k₀Δk/m with Δk = 1/2σ.")
    t = KIND_TEXT[kind]
    ui.science_panel(t[0], r"i\hbar\,\partial_t\psi=\left[-\tfrac{\hbar^2}{2m}\partial_x^2+V(x)\right]\psi", t[1], t[2], t[3])
    ui.export_panel(rpt.standard_bundle(f"potential_{kind}", f"Potential Laboratory – {pot.KINDS[kind]}", res))


# ==========================================================================
# PAGE 4 - Quantum Tunnelling Explorer
# ==========================================================================
def page_tunnelling():
    ui.section("Quantum Tunnelling Explorer", "A packet with E₀ below the barrier top: watch the incoming, reflected and transmitted parts.")
    left, right = st.columns([1, 2.6], gap="medium")
    with left:
        with st.container(border=True):
            V0 = st.slider("Barrier height V₀", 0.2, 10.0, 3.0, 0.1, key="tn_V0")
            a = st.slider("Barrier width a", 0.1, 5.0, 1.0, 0.1, key="tn_a")
            k0 = st.slider("Wave number k₀", 0.5, 5.0, 2.0, 0.05, key="tn_k0")
            sigma = st.slider("Packet width σ", 1.0, 8.0, 3.0, 0.1, key="tn_sigma")
            mass = st.slider("Mass m", 0.2, 5.0, 1.0, 0.1, key="tn_m")
            dx, dt = numerics_expander("tn")
    E0 = HBAR ** 2 * k0 ** 2 / (2 * mass)
    pp = dict(V0=V0, width=a, center=0.0)
    cfg = auto_cfg("rect_barrier", pp, k0, sigma, mass, dx, dt)
    with right:
        if E0 < V0:
            kappa = math.sqrt(2 * mass * (V0 - E0)) / HBAR
            st.success(f"Tunnelling regime: E₀ = {E0:.3f} < V₀ = {V0:.2f}. Classically T = 0. "
                       f"Inside the barrier the wave decays as e^(−κx), κ = {kappa:.3f} (penetration depth 1/κ = {1 / kappa:.3f}).", icon="🚇")
        else:
            st.info(f"Over-barrier regime: E₀ = {E0:.3f} ≥ V₀ = {V0:.2f}. Classically T = 1, but quantum reflection still occurs.", icon="🌊")
        res = run_sim(cfg)
        if res is None:
            return
        ui.player("tn", res, _tunnel_draw(active_xlim(res)))
    R, T, B = res.final_RTB()
    ui.section("Final probabilities (dynamic)", "Integrated from |ψ|² of the last snapshot.")
    ui.cards([("R – reflection", pct(R), None, "qw-vi"), ("T – transmission", pct(T), None, "qw-cy"),
              ("B – inside barrier", pct(B), None, "qw-warn"),
              ("R + T + B", f"{R + T + B:.10f}", f"|R+T+B−1| = {abs(R + T + B - 1):.1e}", "qw-ok")])
    # four independent estimates of T
    Tpw = float(an.transmission_vs_energy(res.x, res.V, np.array([E0]), mass)[0])
    Tcf = float(an.rect_barrier_transmission_analytic(np.array([E0]), V0, a, mass)[0])
    rows = [("Wave-packet simulation (Crank–Nicolson)", T), ("Plane wave, transfer matrix on the same grid", Tpw),
            ("Plane wave, closed-form formula", Tcf)]
    if E0 < V0:
        kappa = math.sqrt(2 * mass * (V0 - E0)) / HBAR
        rows.append(("WKB estimate exp(−2κa)", math.exp(-2 * kappa * a)))
    ui.st_df(pd.DataFrame(rows, columns=["Method", "Transmission T"]).assign(**{"Transmission T": lambda d: d["Transmission T"].map("{:.6f}".format)}))
    st.caption("A wave packet contains a *range* of energies, and T grows steeply with E, so the packet's T exceeds T(E₀) "
               "when σ is small. Increase σ to approach the plane-wave value.")
    prob_tabs(res, "tn")
    ui.science_panel(
        "Quantum tunnelling: a particle crossing a barrier higher than its energy.",
        r"\psi''=\kappa^2\psi\ \ (E<V_0),\qquad \kappa=\frac{\sqrt{2m(V_0-E)}}{\hbar},\qquad T\approx e^{-2\kappa a}",
        "A weak transmitted packet appears on the right of the barrier while the reflected packet carries most of the "
        "probability back. B rises briefly as the packet is inside the barrier, then drops to ≈ 0.",
        "Inside the barrier the allowed wave function is real-exponential rather than oscillatory, but it does not vanish "
        "instantly, so a non-zero amplitude emerges at the far side. The amplitude falls by e^(−κa), hence the probability by e^(−2κa).",
        "Tunnelling is a direct consequence of wave behaviour. It drives alpha decay, scanning tunnelling microscopy and "
        "tunnel diodes; it is exponentially sensitive to barrier width and height.")
    ui.export_panel(rpt.standard_bundle("tunnelling", "Quantum Tunnelling Explorer – report", res,
                                        notes=f"E0 = {E0:.4f}, V0 = {V0:.3f}. Simulated T = {T:.5f}, R = {R:.5f}, B = {B:.2e}."))


def _tunnel_draw(xlim):
    def _draw(res, i):
        m = res.metrics_at(i)
        ui.cards([("Time t", f"{m['t']:.2f}"), ("R (left)", pct(m["R"]), None, "qw-vi"),
                  ("T (right)", pct(m["T"]), None, "qw-cy"), ("B (inside)", pct(m["B"]), None, "qw-warn"),
                  ("R+T+B", f"{m['R'] + m['T'] + m['B']:.8f}", None, "qw-ok")])
        ui.st_plot(viz.fig_density_regions(res, i, xlim=xlim), "tn_dens")
    return _draw


# ==========================================================================
# PAGE 5 - Resonant Tunnelling Laboratory
# ==========================================================================
@st.cache_data(show_spinner=False, max_entries=16)
def _resonance_scan(pp: tuple, e_lo: float, e_hi: float, n: int, well_width: float, well_floor: float):
    """T(E) by transfer matrix on a fine grid + numerically located resonances (cached)."""
    p = dict(pp)
    L = p["w1"] + p["gap"] + p["w2"]
    x = pe.make_grid(-(L / 2 + 6), L / 2 + 6, 0.02)
    V = pot.build_potential("double_barrier", x, p).V
    E = np.linspace(e_lo, e_hi, n)
    T = an.transmission_vs_energy(x, V, E)
    peaks = an.find_resonances(x, V, E, T, well_width=well_width, well_floor=well_floor)
    return E, T, peaks


@st.cache_data(show_spinner=False, max_entries=8)
def _packet_scan(pp: tuple, ks: tuple, sigma: float, dx: float, dt: float):
    return ex.packet_transmission_scan("double_barrier", dict(pp), list(ks), sigma, 1.0, dx, dt)


def page_resonant():
    ui.section("Resonant Tunnelling Laboratory",
               "Double barrier with a central well: transmission peaks computed numerically, then tested with wave packets.")
    left, right = st.columns([1, 2.4], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown("##### Structure")
            V1 = st.slider("First barrier height V₁", 0.5, 10.0, 3.0, 0.1, key="rs_V1")
            w1 = st.slider("First barrier width", 0.2, 3.0, 0.8, 0.1, key="rs_w1")
            V2 = st.slider("Second barrier height V₂", 0.5, 10.0, 3.0, 0.1, key="rs_V2")
            w2 = st.slider("Second barrier width", 0.2, 3.0, 0.8, 0.1, key="rs_w2")
            gap = st.slider("Distance between barriers", 1.0, 12.0, 4.0, 0.1, key="rs_gap")
            Vw = st.slider("Central-region potential", -2.0, 1.0, 0.0, 0.1, key="rs_Vw")
            st.markdown("##### Energy scan")
            e_hi = st.slider("Maximum energy", 1.0, 12.0, 4.0, 0.5, key="rs_ehi")
            npts = st.select_slider("Energy samples", [500, 1000, 2000, 4000], value=2000, key="rs_n")
    pp = dict(V1=V1, w1=w1, V2=V2, w2=w2, gap=gap, V_well=Vw, center=0.0)
    E, T, peaks = _resonance_scan(tuple(sorted(pp.items())), 0.01, float(e_hi), int(npts), float(gap), float(Vw))
    with right:
        L = w1 + gap + w2
        xg = pe.make_grid(-(L / 2 + 6), L / 2 + 6, 0.02)
        ui.st_plot(viz.fig_potential(xg, pot.build_potential("double_barrier", xg, pp).V, "Double-barrier potential", height=260), "rs_pot")
        c1, c2 = st.columns([4, 1])
        logy = c2.checkbox("log T axis", value=False, key="rs_log")
        pk_all = None
        saved = st.session_state.get("rs_scan")
        sig = (tuple(sorted(pp.items())), float(e_hi))
        if saved and saved[0] == sig:
            pk_all = saved[1]
        ui.st_plot(viz.fig_TE(E, np.maximum(T, 1e-12) if logy else T, peaks, packet_points=pk_all,
                              V_ref=min(V1, V2), title="Transmission probability vs energy (transfer matrix)", log_y=logy), "rs_TE")
    ui.cards([("Resonances found", str(len(peaks))),
              ("Lowest resonance E₁", f"{peaks['E_res'].iloc[0]:.4f}" if len(peaks) else "—"),
              ("T at E₁", f"{peaks['T_peak'].iloc[0]:.4f}" if len(peaks) else "—"),
              ("Max T on scan", f"{T.max():.4f}"), ("Min T on scan", f"{T.min():.2e}")])
    if len(peaks):
        show = peaks.rename(columns={"E_res": "E_res", "T_peak": "T at peak", "FWHM": "width Γ (FWHM)", "lifetime": "lifetime ħ/Γ",
                                     "k_res": "k_res", "n_est": "n", "E_infinite_well": "E_n (infinite well, same width)"})
        ui.st_df(show.round(5))
        st.caption("Peaks are located with `scipy.signal.find_peaks`, refined on a fine local energy grid, and widths come from "
                   "the half-maximum points. The infinite-well column is a calculated reference: real resonances lie *below* it "
                   "because the wave leaks into the finite barriers (effective well is wider).")
    else:
        st.info("No resonance peaks above the prominence threshold in this energy window. Try taller/wider barriers or a longer well.")

    ui.section("Excite a resonance with a wave packet",
               "Choose k₀ from the computed peaks (or off-resonance) and watch the probability get trapped between the barriers.")
    opts = {}
    for j in range(min(len(peaks), 6)):
        opts[f"Resonance #{j + 1}  (E = {peaks['E_res'].iloc[j]:.4f})"] = ("res", float(peaks["E_res"].iloc[j]))
    for j in range(min(len(peaks) - 1, 3)):
        m = (E > peaks["E_res"].iloc[j]) & (E < peaks["E_res"].iloc[j + 1])
        if m.any():
            Ev = float(E[m][np.argmin(T[m])])
            opts[f"Off-resonance valley between #{j + 1} and #{j + 2}  (E = {Ev:.4f}, T = {T[m].min():.3f})"] = ("val", Ev)
    opts["Custom k₀"] = ("custom", None)
    c1, c2 = st.columns([1.6, 1])
    with c1:
        choice = st.selectbox("Packet energy", list(opts), key="rs_choice")
    kind_, Ech = opts[choice]
    k0 = math.sqrt(2 * Ech) if Ech else st.slider("Wave number k₀", 0.4, 4.0, 1.0, 0.01, key="rs_k0c")
    with c2:
        sigma = st.slider("Packet width σ", 4.0, 20.0, 10.0, 0.5, key="rs_sigma",
                          help="Large σ → narrow energy spread ΔE ≈ ħ²k₀/(2mσ), needed to resolve narrow resonances.")
    c3, c4 = st.columns(2)
    extra = c3.slider("Extra time to watch the leakage", 0.0, 400.0, 60.0, 10.0, key="rs_extra")
    with c4:
        dx, dt = numerics_expander("rs", 0.1, 0.05)
    st.caption(f"Packet: k₀ = {k0:.4f}, E₀ = {k0 ** 2 / 2:.4f}, energy spread ΔE ≈ {k0 * (1 / (2 * sigma)):.4f}"
               + (f" – resonance width Γ = {peaks['FWHM'].iloc[int(choice.split('#')[1].split()[0]) - 1]:.4f}"
                  if kind_ == "res" and not np.isnan(peaks['FWHM'].iloc[int(choice.split('#')[1].split()[0]) - 1]) else ""))
    cfg = auto_cfg("double_barrier", pp, k0, sigma, 1.0, dx, dt, extra_time=extra)
    res = run_sim(cfg)
    if res is None:
        return
    ui.player("rs", res, draw_regions("rs", xlim=active_xlim(res), title="Probability density; shaded = left / inside structure / right"))
    R, T_, B = res.final_RTB()
    Tth = float(an.transmission_vs_energy(res.x, res.V, np.array([k0 ** 2 / 2]))[0])
    ui.cards([("T (simulation)", pct(T_), None, "qw-cy"), ("T(E₀) plane-wave", pct(Tth)), ("R (simulation)", pct(R), None, "qw-vi"),
              ("Peak trapped probability", pct(float(res.B.max())), "max over time of B", "qw-warn"),
              ("Final B", pct(B))])
    with st.expander("Compare resonant vs off-resonant packets (two simulations)"):
        keys = [k for k, v in opts.items() if v[0] == "res"][:1] + [k for k, v in opts.items() if v[0] == "val"][:1]
        if len(keys) == 2:
            series, rows = {}, []
            for lab, kk in zip(("on resonance", "off resonance"), keys):
                k_ = math.sqrt(2 * opts[kk][1])
                c_ = auto_cfg("double_barrier", pp, k_, sigma, 1.0, dx, dt, extra_time=extra)
                r_ = run_sim(c_)
                if r_ is None:
                    continue
                series[f"B(t), {lab}"] = (r_.times, r_.B)
                rows.append(dict(packet=lab, k0=k_, E0=k_ ** 2 / 2, T_sim=r_.T[-1], R_sim=r_.R[-1], max_B=r_.B.max(),
                                 T_planewave=float(an.transmission_vs_energy(r_.x, r_.V, np.array([k_ ** 2 / 2]))[0])))
            if rows:
                ui.st_plot(viz.fig_multi_xy(series, "Time t", "Probability inside the structure"), "rs_cmp")
                ui.st_df(pd.DataFrame(rows).round(5))
        else:
            st.write("Need at least one resonance and one valley in the scan window.")
    with st.expander("Check T(E) with wave-packet simulations"):
        st.caption("Runs one wave-packet simulation per computed resonance/valley energy (σ as chosen above) and overlays the result "
                   "on the T(E) curve. Packets are broader in energy than narrow resonances, so their T is an *average* of T(E) over ΔE.")
        if st.button("Run wave-packet scan", key="rs_scan_btn"):
            Es = [v[1] for v in opts.values() if v[0] in ("res", "val")][:8]
            ks = tuple(round(math.sqrt(2 * e), 5) for e in Es)
            prog = st.progress(0.0, text="Running wave-packet simulations…")
            try:
                dfp = _packet_scan(tuple(sorted(pp.items())), ks, float(sigma), float(dx), float(dt))
                st.session_state["rs_scan"] = (sig, dfp)
                prog.empty()
                st.rerun()
            except ValueError as e:
                prog.empty()
                st.error(str(e))
        if saved and saved[0] == sig:
            ui.st_df(saved[1][["k0", "E0", "T", "R", "T_plane_wave"]].round(5))
    ui.science_panel(
        "Resonant tunnelling: a double barrier is almost fully transparent at discrete energies even though each barrier alone is nearly opaque.",
        r"T(E)=\frac{1}{1+\left[\ldots\right]\sin^{2}\!\delta},\qquad 2k_wL+\phi_1+\phi_2=2\pi n\ \Rightarrow\ T\to 1",
        "T(E) has sharp peaks (T ≈ 1) at the computed resonance energies; with a wave packet tuned to a peak, probability piles up "
        "between the barriers (B rises) and then leaks out in both directions over the lifetime ħ/Γ. Off resonance the packet mostly bounces off.",
        "Waves reflected back and forth between the barriers interfere. When the round-trip phase (2k_w L plus the phase shifts on reflection) "
        "equals 2πn, the successive reflections add **constructively** inside the well, building a large standing-wave amplitude; the "
        "leakage through the two barriers then cancels the reflection on the incident side. Thicker/taller barriers → narrower peaks, longer lifetime.",
        "The well supports quasi-bound states; resonant tunnelling is conduction *through* those states. It is the working principle of "
        "resonant-tunnelling diodes and an analogue of a Fabry–Pérot cavity for matter waves.")
    b = rpt.standard_bundle("resonant", "Resonant Tunnelling Laboratory – report", res,
                            notes=f"{len(peaks)} resonances found numerically from the transfer-matrix T(E).")
    b.tables["resonances"] = peaks
    b.tables["T_of_E"] = pd.DataFrame(dict(E=E, T=T))
    b.figure_builders["T_of_E"] = lambda: viz.mpl_TE(E, T, peaks)
    ui.export_panel(b)


# ==========================================================================
# PAGE 6 - Interference and momentum analysis
# ==========================================================================
def page_momentum():
    ui.section("Quantum Interference & Momentum Analysis",
               "Position space ↔ momentum space (FFT), the ± momentum components, interference fringes and Δx·Δp ≥ ħ/2.")
    left, right = st.columns([1, 2.6], gap="medium")
    with left:
        with st.container(border=True):
            sigma = st.slider("Initial width σ", 0.5, 6.0, 1.5, 0.1, key="mo_sigma")
            k0 = st.slider("Wave number k₀", 0.5, 4.0, 2.0, 0.1, key="mo_k0")
            kind = st.radio("Potential", ["Free particle", "Rectangular barrier"], key="mo_pot", horizontal=True)
            pp = {}
            if kind == "Rectangular barrier":
                pp = dict(V0=st.slider("Barrier height V₀", 0.5, 8.0, 3.0, 0.1, key="mo_V0"),
                          width=st.slider("Barrier width", 0.2, 4.0, 1.0, 0.1, key="mo_w"), center=0.0)
            t_max = st.slider("Duration", 5.0, 60.0, 26.0, 1.0, key="mo_t")
            dx, dt = numerics_expander("mo", 0.1, 0.02)
    x0 = -(15 + 3 * sigma)
    Lh = math.ceil((abs(x0) + k0 * t_max + 10 * sigma + 10) / 10) * 10
    cfg = pe.SimConfig.make("rect_barrier" if pp else "free", pp, x_min=-Lh, x_max=Lh, dx=dx, dt=dt, t_max=t_max,
                            mass=1.0, x0=x0, sigma=sigma, k0=k0)
    with right:
        res = run_sim(cfg)
        if res is None:
            return
        xl = active_xlim(res)
        ui.player("mo", res, lambda r, i: _mom_draw(r, i, xl))
    ui.section("Uncertainty principle")
    prod = res.delta_x * res.delta_p
    ui.st_plot(viz.fig_lines(res.times, {"Δx·Δp": prod, "ħ/2": np.full_like(prod, HBAR / 2)}, "Time t", "Δx·Δp",
                             "Δx·Δp versus time (must stay ≥ ħ/2)", dashes={"ħ/2": "dash"}), "mo_unc")
    ui.cards([("Initial Δx·Δp", f"{prod[0]:.6f}", f"ħ/2 = {HBAR / 2}"), ("Minimum over time", f"{prod.min():.6f}", None,
              "qw-ok" if prod.min() >= HBAR / 2 - 1e-6 else "qw-warn"),
              ("Bound satisfied?", "YES" if prod.min() >= HBAR / 2 - 1e-6 else "NO", None, "qw-ok" if prod.min() >= HBAR / 2 - 1e-6 else "qw-warn")])
    sig_vals = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    scan = an.uncertainty_scan(sig_vals, k0=k0, dx=0.05)
    scan["Δp theory ħ/(2σ)"] = HBAR / (2 * scan["sigma"])
    c1, c2 = st.columns([1.2, 1])
    with c1:
        ui.st_plot(viz.fig_lines(scan["sigma"], {"Δp (FFT of grid ψ)": scan["delta_p"], "ħ/(2σ)": scan["Δp theory ħ/(2σ)"]},
                                 "Initial width σ", "Momentum spread Δp", "Narrower packet → broader momentum distribution",
                                 markers=True, dashes={"ħ/(2σ)": "dash"}), "mo_scan")
    with c2:
        ui.st_df(scan.rename(columns={"delta_x": "Δx", "delta_p": "Δp", "product": "Δx·Δp", "bound": "ħ/2", "ratio": "ratio to ħ/2"})
                 .drop(columns=["Δp theory ħ/(2σ)", "satisfied"]).round(6))
    ui.science_panel(
        "Position–momentum duality and quantum interference.",
        r"\phi(p)=\frac{1}{\sqrt{2\pi\hbar}}\int\psi(x)e^{-ipx/\hbar}dx,\qquad \Delta x\,\Delta p\ \ge\ \frac{\hbar}{2}",
        "The momentum distribution is a Gaussian centred on ħk₀. A narrower packet (small σ) has a wider distribution. "
        "After the packet hits the barrier a negative-momentum (p<0) lobe appears, and on the left of the barrier |ψ|² shows "
        "fringes of spacing π/k₀: the pink interference term 2Re(ψ₊*ψ₋) oscillates between positive and negative.",
        "The Fourier transform connects the two descriptions: localising a wave in space requires adding many wavelengths. "
        "Reflection reverses part of the momentum (p → −p). The incident (p>0) and reflected (p<0) waves are coherent, so "
        "their amplitudes add before squaring, producing the cross term.",
        "Δx·Δp = ħ/2 for a free Gaussian (minimum uncertainty) and grows later or when the packet splits in two. "
        "Interference redistributes probability in space; its integral over x is ≈ 0, so total probability is unchanged.")
    ui.export_panel(rpt.standard_bundle("momentum", "Interference and Momentum Analysis – report", res))


def _mom_draw(res, i, xl):
    psi = res.psi[i]
    md = an.momentum_distribution(psi, res.dx)
    ref = an.momentum_distribution(res.psi[0], res.dx)
    dec = an.momentum_decomposition(psi, res.dx)
    prod = res.delta_x[i] * res.delta_p[i]
    ui.cards([("Time t", f"{res.times[i]:.2f}"), ("Δx", f"{res.delta_x[i]:.4f}"), ("Δp", f"{res.delta_p[i]:.4f}"),
              ("Δx·Δp", f"{prod:.5f}", "≥ ħ/2 = 0.5", "qw-ok" if prod >= HBAR / 2 - 1e-6 else "qw-warn"),
              ("Weight p>0", f"{md['weight_pos']:.4f}", None, "qw-cy"), ("Weight p<0", f"{md['weight_neg']:.4f}", None, "qw-vi"),
              ("∫ interference dx", f"{dec['integral']:.1e}", "≈ 0: no net probability")])
    c1, c2 = st.columns(2)
    with c1:
        ui.st_plot(viz.fig_interference(res.x, res.density[i], dec["density_R"], dec["density_L"], dec["interference"], xlim=xl), "mo_pos")
    with c2:
        ui.st_plot(viz.fig_momentum(md, ref=ref), "mo_mom")


# ==========================================================================
# PAGE 7 - Classical vs Quantum
# ==========================================================================
def page_classical():
    ui.section("Classical vs Quantum", "The same barrier, the same energy distribution: Newton's particles versus a Schrödinger wave packet.")
    left, right = st.columns([1, 2.6], gap="medium")
    with left:
        with st.container(border=True):
            E = st.slider("Particle energy E", 0.3, 8.0, 2.0, 0.1, key="cq_E")
            V0 = st.slider("Barrier height V₀", 0.3, 8.0, 3.0, 0.1, key="cq_V0")
            a = st.slider("Barrier width a", 0.2, 4.0, 1.0, 0.1, key="cq_a")
            sigma = st.slider("Packet width σ", 1.5, 8.0, 3.0, 0.1, key="cq_sigma")
            mass = st.slider("Mass m", 0.3, 4.0, 1.0, 0.1, key="cq_m")
            dx, dt = numerics_expander("cq")
    k0 = math.sqrt(2 * mass * E) / HBAR
    pp = dict(V0=V0, width=a, center=0.0)
    cfg = auto_cfg("rect_barrier", pp, k0, sigma, mass, dx, dt)
    with right:
        if E < V0:
            st.success(f"E = {E:.2f} < V₀ = {V0:.2f}: a classical particle with this energy can **never** cross the barrier (T_classical = 0).", icon="⚖️")
        else:
            st.info(f"E = {E:.2f} ≥ V₀ = {V0:.2f}: a classical particle always crosses (T_classical = 1) – quantum waves still reflect partially.", icon="⚖️")
        res = run_sim(cfg)
        if res is None:
            return
        xs, vs = an.sample_classical_ensemble(sigma, k0, cfg.x0, mass)
        lo, hi = res.region
        xl = active_xlim(res)

        def _draw(r, i):
            pos = an.classical_positions(xs, vs, float(r.times[i]), lo, hi, V0, mass)
            cR, cB, cT = float((pos < lo).mean()), float(((pos >= lo) & (pos <= hi)).mean()), float((pos > hi).mean())
            ui.cards([("Time t", f"{r.times[i]:.2f}"), ("Quantum R", pct(r.R[i]), None, "qw-vi"), ("Quantum T", pct(r.T[i]), None, "qw-cy"),
                      ("Classical R", pct(cR), None, "qw-vi"), ("Classical T", pct(cT), None, "qw-cy"),
                      ("Quantum B", pct(r.B[i])), ("Classical B", pct(cB))])
            ui.st_plot(viz.fig_classical_quantum(r, i, pos, xlim=xl), "cq_dens")
        ui.player("cq", res, _draw)
    pos_end = an.classical_positions(xs, vs, float(res.times[-1]), lo, hi, V0, mass)
    Tcl_ens = float((pos_end > hi).mean())
    Tcl_thr = an.classical_transmission_fraction(sigma, k0, mass, V0)
    Tpw = float(an.transmission_vs_energy(res.x, res.V, np.array([E]), mass)[0])
    ui.section("Final comparison")
    ui.st_df(pd.DataFrame([
        ("Classical, single particle at energy E", 1.0 if E > V0 else 0.0),
        ("Classical ensemble (same x and p distribution as the packet)", Tcl_ens),
        ("Classical threshold fraction  P(p²/2m > V₀)", Tcl_thr),
        ("Quantum plane wave T(E) (transfer matrix)", Tpw),
        ("Quantum wave packet (Crank–Nicolson)", float(res.T[-1]))], columns=["Model", "Transmission probability"])
        .assign(**{"Transmission probability": lambda d: d["Transmission probability"].map("{:.5f}".format)}))
    # trajectory of the packet centre vs a classical particle
    t = res.times
    xcl = np.array([an.classical_positions(np.array([cfg.x0]), np.array([k0 / mass]), float(tt), lo, hi, V0, mass)[0] for tt in t])
    c1, c2 = st.columns(2)
    with c1:
        ui.st_plot(viz.fig_lines(t, {"Quantum ⟨x⟩ (whole packet)": res.mean_x, "Classical particle x(t)": xcl}, "Time t", "Position",
                                 "Mean position vs classical trajectory", dashes={"Classical particle x(t)": "dash"}), "cq_traj")
    with c2:
        Eg = np.linspace(0.02, max(2.0 * V0, 2.0 * E), 600)
        Tq = an.transmission_vs_energy(res.x, res.V, Eg, mass)
        fig = viz.fig_lines(Eg, {"Quantum T(E)": Tq, "Classical T(E)": (Eg > V0).astype(float)}, "Energy E", "Transmission",
                            "T versus energy", dashes={"Classical T(E)": "dash"})
        ui.st_plot(fig, "cq_TE")
    ui.science_panel(
        "Classical barrier reflection versus quantum tunnelling and quantum reflection.",
        r"\text{classical: } \tfrac12 m v^2+V=E \qquad\text{quantum: } T(E)\approx e^{-2\kappa a},\ \kappa=\tfrac{\sqrt{2m(V_0-E)}}{\hbar}",
        "For E < V₀ the pink classical ensemble bounces back completely, while the cyan quantum density develops a transmitted packet. "
        "For E > V₀ the classical ensemble passes (slowing down inside), but part of the quantum packet is still reflected. "
        "The classical step in T(E) is replaced by a smooth, oscillating quantum curve.",
        "Classically the kinetic energy cannot be negative, so the region E < V is forbidden. A wave has no such restriction: "
        "its amplitude merely decays exponentially in the forbidden region, and if the barrier is thin the amplitude that "
        "survives to the other side radiates a transmitted wave. Reflection above the barrier arises from the abrupt change of wavelength.",
        "Particles are not 'borrowing energy': the energy distribution of the packet is intact; its probability amplitude is simply "
        "non-zero where classical motion is forbidden. Tunnelling probability depends exponentially on a, V₀ and √m.")
    ui.export_panel(rpt.standard_bundle("classical_vs_quantum", "Classical vs Quantum – report", res,
                                        notes=f"E = {E:.3f}, V0 = {V0:.3f}. Quantum T = {res.T[-1]:.5f}; classical ensemble T = {Tcl_ens:.5f}."))


# ==========================================================================
# PAGE 8 - Experiment Analysis
# ==========================================================================
@st.cache_data(show_spinner=False, max_entries=24)
def _cached_experiment(exp_key: str, values: tuple, base_items: tuple) -> pd.DataFrame:
    return ex.run_experiment(exp_key, values, dict(base_items))


EXP_TEXT = {
    "A": ("Dependence of reflection and transmission on barrier height.",
          r"T(E)=\left[1+\frac{V_0^2\sinh^2\kappa a}{4E(V_0-E)}\right]^{-1}\ (E<V_0)",
          "T falls smoothly as V₀ rises; there is no sharp step at V₀ = E₀, as a classical particle would show.",
          "Higher barriers increase the mismatch between the wave inside and outside (more reflection) and increase κ (faster decay).",
          "Energy alone does not decide the outcome: the barrier shape and the wave nature of the particle matter."),
    "B": ("Dependence of tunnelling on barrier width.",
          r"T\approx \frac{16E(V_0-E)}{V_0^2}\,e^{-2\kappa a}",
          "For E < V₀, ln T decreases roughly linearly with width (slope ≈ −2κ); for E > V₀ T oscillates with width.",
          "The amplitude decays as e^(−κx) across the barrier, so the probability decays as e^(−2κa).",
          "Tunnelling is exponentially sensitive to width – the basis of scanning-tunnelling microscopy's atomic resolution."),
    "C": ("Dependence of transmission on initial momentum (energy).",
          r"E=\frac{\hbar^2k_0^2}{2m},\qquad T=T(E)",
          "T increases with k₀, passes through its steepest rise near E₀ ≈ V₀ and approaches 1 for E ≫ V₀ (with oscillations for wide barriers).",
          "Higher energy means smaller κ (slower decay) and a smaller wavelength mismatch at the barrier edges.",
          "Fast particles are transmitted more easily, but never perfectly except at special resonance energies.")}


def experiment_panel(exp_key: str, pfx: str):
    spec = ex.EXPERIMENTS[exp_key]
    p = spec["param"]
    st.markdown(f"#### {spec['title']}")
    bounds = {"V0": (0.2, 8.0), "width": (0.1, 5.0), "k0": (0.9, 5.0)}[p]
    c1, c2, c3 = st.columns(3)
    with c1:
        rng = st.slider(f"Range of {spec['label']}", bounds[0], bounds[1], spec["default_range"], 0.1, key=f"{pfx}_rng")
        npts = st.slider("Number of simulations", 4, 25, spec["default_n"], 1, key=f"{pfx}_n")
    base = dict(ex.BASE_DEFAULTS)
    with c2:
        if p != "V0":
            base["V0"] = st.slider("Fixed barrier height V₀", 0.2, 8.0, 2.5, 0.1, key=f"{pfx}_V0")
        if p != "width":
            base["width"] = st.slider("Fixed barrier width a", 0.1, 5.0, 1.5, 0.1, key=f"{pfx}_w")
        if p != "k0":
            base["k0"] = st.slider("Fixed wave number k₀", 0.9, 5.0, 2.0, 0.1, key=f"{pfx}_k0")
    with c3:
        base["sigma"] = st.slider("Packet width σ", 1.5, 8.0, 3.0, 0.1, key=f"{pfx}_sig",
                                  help="Wider packets have a narrower energy spread, so T approaches the plane-wave value.")
    values = tuple(float(v) for v in np.round(np.linspace(rng[0], rng[1], npts), 6))
    try:
        with st.spinner(f"Running {npts} wave-packet simulations…"):
            df = _cached_experiment(exp_key, values, tuple(sorted(base.items())))
    except ValueError as e:
        st.error(str(e), icon="⛔")
        return None
    fixed = {k: v for k, v in base.items() if k != p}
    show = df.rename(columns={p: spec["label"], "E0": "E₀", "T_plane_wave": "T plane-wave", "complete": "collision complete",
                              "total": "R+T+B", "runtime_s": "run time (s)"})
    ui.st_df(show.round(6))
    cc1, cc2 = st.columns(2)
    with cc1:
        ui.st_plot(viz.fig_lines(df[p], {"Reflection R (simulation)": df["R"], "1 − T plane wave": 1 - df["T_plane_wave"]},
                                 spec["label"], "Reflection probability R", "Reflection probability", markers=True,
                                 dashes={"1 − T plane wave": "dash"}), f"{pfx}_R")
    with cc2:
        logt = st.checkbox("log scale", value=False, key=f"{pfx}_log")
        ui.st_plot(viz.fig_lines(df[p], {"Transmission T (simulation)": np.maximum(df["T"], 1e-12) if logt else df["T"],
                                         "T plane wave (transfer matrix)": np.maximum(df["T_plane_wave"], 1e-12) if logt else df["T_plane_wave"]},
                                 spec["label"], "Transmission probability T", "Transmission probability", markers=True, log_y=logt,
                                 dashes={"T plane wave (transfer matrix)": "dash"}), f"{pfx}_T")
    text = ex.interpret_experiment(exp_key, df, fixed)
    st.markdown("##### 🧭 Interpretation of this run")
    ui.note(text.replace("\n\n", "<br><br>").replace("**", ""))
    t = EXP_TEXT[exp_key]
    ui.science_panel(*t[:1], t[1], *t[2:], title="🔬 Scientific explanation")
    st.download_button("⬇ Download comparison table (CSV)", df.to_csv(index=False).encode(), file_name=f"experiment_{exp_key}.csv",
                       mime="text/csv", key=f"{pfx}_csv")
    b = rpt.ReportBundle(
        key=f"exp_{pfx}", title=spec["title"], params={**fixed, "scanned": p, "range": f"{values[0]}..{values[-1]}", "n": len(values),
                                                         "dx": base["dx"], "dt": base["dt"], "units": "dimensionless"},
        metrics={"max |R+T+B-1|": float(np.max(np.abs(df['total'] - 1))), "T min": float(df['T'].min()), "T max": float(df['T'].max())},
        tables={f"experiment_{exp_key}": df},
        figure_builders={"reflection": lambda: viz.mpl_lines(df[p], {"R": df["R"]}, spec["label"], "R", "Reflection probability", markers=True),
                         "transmission": lambda: viz.mpl_lines(df[p], {"T (simulation)": df["T"], "T plane wave": df["T_plane_wave"]},
                                                               spec["label"], "T", "Transmission probability", markers=True,
                                                               dashes={"T plane wave": "--"})},
        notes=text)
    ui.export_panel(b)
    return df


def page_analysis():
    ui.section("Experiment Analysis", "Parameter sweeps: each row is a full wave-packet simulation. Tables, plots, interpretation and export.")
    tabs = st.tabs(["Experiment A · barrier height", "Experiment B · barrier width", "Experiment C · momentum", "Compare two experiments"])
    for tab, k in zip(tabs[:3], "ABC"):
        with tab:
            experiment_panel(k, f"ex{k}")
    with tabs[3]:
        st.caption("Pick any two experiments and view them side by side (each side keeps its own settings).")
        cl, cr = st.columns(2, gap="large")
        with cl:
            a = st.selectbox("Left experiment", list("ABC"), index=0, key="cmp_a", format_func=lambda k: ex.EXPERIMENTS[k]["title"])
            experiment_panel(a, "cmpL")
        with cr:
            b = st.selectbox("Right experiment", list("ABC"), index=1, key="cmp_b", format_func=lambda k: ex.EXPERIMENTS[k]["title"])
            experiment_panel(b, "cmpR")


# ==========================================================================
# PAGE 9 - Numerical Validation
# ==========================================================================
@st.cache_data(show_spinner=False, max_entries=8)
def _convergence(kind: str, t_end: float) -> pd.DataFrame:
    return an.convergence_study(kind, t_end=t_end)


@st.cache_data(show_spinner=False, max_entries=8)
def _transmission_convergence(V0: float, width: float, k0: float, sigma: float) -> pd.DataFrame:
    rows = []
    for dxv in (0.2, 0.1, 0.05, 0.025):
        r = ex.run_scattering("rect_barrier", dict(V0=V0, width=width, center=0.0), k0, sigma, 1.0, dxv, 0.01)
        rows.append(dict(dx=dxv, T=r["T"], R=r["R"], T_plane_wave=r["T_plane_wave"]))
    df = pd.DataFrame(rows)
    df["|T − T(finest)|"] = (df["T"] - df["T"].iloc[-1]).abs()
    return df


def page_validation():
    ui.section("Numerical Validation", "How far can the numbers be trusted? Conservation laws, an exact solution, and convergence orders.")
    with st.container(border=True):
        c = st.columns(6)
        sigma = c[0].slider("σ", 0.6, 4.0, 1.2, 0.1, key="va_sigma")
        k0 = c[1].slider("k₀", 0.5, 4.0, 2.0, 0.1, key="va_k0")
        V0 = c[2].slider("V₀", 0.5, 6.0, 2.5, 0.1, key="va_V0")
        bw = c[3].slider("Barrier width", 0.2, 3.0, 1.5, 0.1, key="va_bw")
        dx = c[4].select_slider("Δx", DX_OPTIONS, value=0.1, key="va_dx")
        dt = c[5].select_slider("Δt", [0.05, 0.02, 0.01, 0.005], value=0.01, key="va_dt")
    t_max = 14.0
    free_cfg = pe.SimConfig.make("free", {}, dx=dx, dt=dt, t_max=t_max, x0=-12.0, sigma=sigma, k0=k0)
    bar_cfg = pe.SimConfig.make("rect_barrier", dict(V0=V0, width=bw), dx=dx, dt=dt, t_max=t_max, x0=-12.0, sigma=sigma, k0=k0)
    rf, rb = run_sim(free_cfg), run_sim(bar_cfg)
    if rf is None or rb is None:
        return

    # ---- 1. probability conservation -------------------------------------
    ui.section("1 · Total probability conservation")
    ef, eb = an.conservation_errors(rf), an.conservation_errors(rb)
    ui.cards([("Free: max |norm − 1|", f"{ef['max_norm_deviation']:.1e}"), ("Barrier: max |norm − 1|", f"{eb['max_norm_deviation']:.1e}"),
              ("Steps taken", f"{len(rf.step_norm)}")])
    ui.st_plot(viz.fig_lines(rf.step_times, {"free particle": np.maximum(np.abs(rf.step_norm - rf.norm[0]), 1e-17),
                                             "with barrier": np.maximum(np.abs(rb.step_norm - rb.norm[0]), 1e-17)},
                             "Time t", "|∫|ψ|²dx − 1|", "Probability conservation error per time step", log_y=True), "va_norm")
    st.caption("Crank–Nicolson is the Cayley transform of a Hermitian matrix, hence exactly unitary: the deviation is floating-point round-off "
               "(≈10⁻¹³) and grows only like √steps.")

    # ---- 2. energy conservation -----------------------------------------
    ui.section("2 · Energy conservation")
    rel = lambda r: np.maximum(np.abs(r.energy - r.energy[0]) / abs(r.energy[0]), 1e-17)
    ui.st_plot(viz.fig_lines(rf.times, {"free particle": rel(rf), "with barrier": rel(rb)}, "Time t", "|⟨H⟩(t) − ⟨H⟩(0)| / ⟨H⟩(0)",
                             "Relative energy drift", log_y=True), "va_en")
    e_cont = HBAR ** 2 * k0 ** 2 / 2 + HBAR ** 2 / (8 * sigma ** 2)
    ui.cards([("⟨H⟩(0) numerical (free)", f"{rf.energy[0]:.6f}"), ("Continuum  k₀²/2 + 1/(8σ²)", f"{e_cont:.6f}"),
              ("Difference (lattice, O(Δx²))", f"{rf.energy[0] - e_cont:+.2e}")])
    c1, c2 = st.columns(2)
    with c1:
        ui.st_plot(viz.fig_lines(rb.times, {"⟨H⟩": rb.energy, "⟨K⟩": rb.kin_energy, "⟨V⟩": rb.pot_energy}, "Time t", "Energy",
                                 "Barrier run: K and V trade, H stays constant"), "va_en2")
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("The total energy ⟨H⟩ is exactly conserved by the *discrete* evolution (H commutes with its own Cayley transform). The small "
                    "offset from the continuum value is the finite-difference lattice correction, which shrinks as Δx².")

    # ---- 3. spreading vs theory ----------------------------------------
    ui.section("3 · Free-particle spreading: numerical vs theory")
    th = an.free_spreading_theory(sigma, rf.times)
    err = np.abs(rf.delta_x - th)
    ui.st_plot(viz.fig_lines(rf.times, {"Numerical Δx(t)": rf.delta_x, "Theory σ√(1+(ħt/2mσ²)²)": th}, "Time t", "Δx(t)",
                             "Wave-packet width", dashes={"Theory σ√(1+(ħt/2mσ²)²)": "dash"}), "va_spread")
    exact_err = np.array([np.sqrt((np.abs(rf.psi[i] - an.exact_free_packet(rf.x, free_cfg.x0, sigma, k0, rf.times[i])) ** 2).sum() * rf.dx)
                          for i in range(rf.n_snap)])
    c1, c2 = st.columns(2)
    with c1:
        ui.st_plot(viz.fig_lines(rf.times, {"|Δx_num − Δx_theory|": np.maximum(err, 1e-17)}, "Time t", "Absolute error",
                                 "Width error", log_y=True), "va_spread_err")
    with c2:
        ui.st_plot(viz.fig_lines(rf.times, {"‖ψ_num − ψ_exact‖₂": np.maximum(exact_err, 1e-17)}, "Time t", "L² error",
                                 "Error against the exact free-packet wave function", log_y=True), "va_l2")
    ui.cards([("Max width error", f"{err.max():.2e}"), ("Max relative width error", f"{(err / th).max():.2e}"),
              ("Final L² error vs exact ψ", f"{exact_err[-1]:.2e}")])

    # ---- 4. convergence --------------------------------------------------
    ui.section("4 · Grid and time-step convergence", "Error of the numerical wave function against the exact free-packet solution at t = 4 (σ = 1.2, k₀ = 2).")
    with st.spinner("Running convergence studies…"):
        cdx, cdt = _convergence("dx", 4.0), _convergence("dt", 4.0)
        ctr = _transmission_convergence(float(V0), float(bw), float(k0), 3.0)
    c1, c2 = st.columns(2)
    for col, df_, name, lab in ((c1, cdx, "dx", "Grid spacing Δx (Δt = 0.005)"), (c2, cdt, "dt", "Time step Δt (Δx = 0.02)")):
        with col:
            h = df_[name].to_numpy()
            ref = df_["L2"].iloc[0] * (h / h[0]) ** 2
            fig = viz.fig_lines(h, {"L² error": df_["L2"], "slope 2 (reference)": ref}, lab, "L² error", None, log_y=True,
                                markers=True, dashes={"slope 2 (reference)": "dash"})
            fig.update_xaxes(type="log")
            ui.st_plot(fig, f"va_conv_{name}")
            ui.st_df(df_[[name, "L2", "Linf", "observed_order"]].rename(columns={"L2": "L² error", "Linf": "max error", "observed_order": "observed order"}))
    st.caption("Observed order ≈ 2 confirms the second-order accuracy of the centred difference (in Δx) and of the Crank–Nicolson step (in Δt). "
               "In the Δt study the error eventually flattens at the spatial-error floor.")
    st.markdown("**Transmission coefficient convergence** (wave packet σ = 3 on the barrier above)")
    ui.st_df(ctr.round(7))

    # ---- 5. error analysis ----------------------------------------------
    ui.section("5 · Numerical error summary")
    summary = pd.DataFrame([
        ("Norm drift, free run", ef["max_norm_deviation"], "round-off (~1e-13)"),
        ("Norm drift, barrier run", eb["max_norm_deviation"], "round-off (~1e-13)"),
        ("Relative ⟨H⟩ drift, free run", ef["max_energy_rel_drift"], "round-off"),
        ("Relative ⟨H⟩ drift, barrier run", eb["max_energy_rel_drift"], "round-off"),
        ("Max relative width error (free)", float((err / th).max()), "O(Δx², Δt²)"),
        ("Final L² error vs exact ψ (free)", float(exact_err[-1]), "O(Δx², Δt²)"),
        ("Observed order in Δx", float(cdx["observed_order"].iloc[-1]), "≈ 2"),
        ("Observed order in Δt", float(cdt["observed_order"].iloc[1:].mean()), "≈ 2 (until spatial floor)"),
        ("Max |T(Δx) − T(Δx=0.025)|", float(ctr["|T − T(finest)|"].iloc[:-1].max()), "→ 0 with Δx")],
        columns=["Quantity", "Value", "Expected behaviour"])
    ui.st_df(summary.assign(Value=lambda d: d["Value"].map("{:.3e}".format)))
    with st.expander("⚠️ Limitations of the numerical model"):
        st.markdown(
            "- Hard walls at the domain edges reflect any probability that reaches them; the app warns when this can occur.\n"
            "- Lattice dispersion makes short wavelengths (k·Δx ≳ 0.6) propagate too slowly. Keep ≥ 10 points per wavelength.\n"
            "- Sharp potential steps are represented on the grid by cell-averaged heights, giving O(Δx) uncertainty in effective width "
            "unless edges fall on cell boundaries (the discretisation preserves ∫V dx).\n"
            "- A packet's T is an energy-average of T(E); it equals the plane-wave value only in the narrow-ΔE limit (large σ).")
    b = rpt.ReportBundle(
        key="validation", title="Numerical Validation – report",
        params=dict(sigma=sigma, k0=k0, V0=V0, barrier_width=bw, dx=dx, dt=dt, t_max=t_max),
        metrics={q: float(v) for q, v in zip(summary["Quantity"], summary["Value"].astype(float))},
        tables={"time_series_free": pd.DataFrame(dict(t=rf.times, width_num=rf.delta_x, width_theory=th, L2_error=exact_err,
                                                      energy=rf.energy)),
                "convergence_dx": cdx, "convergence_dt": cdt, "transmission_convergence": ctr},
        figure_builders={
            "norm_conservation": lambda: viz.mpl_lines(rf.step_times, {"free": np.maximum(np.abs(rf.step_norm - 1), 1e-17),
                                                                      "barrier": np.maximum(np.abs(rb.step_norm - 1), 1e-17)},
                                                       "t", "|norm-1|", "Probability conservation", log_y=True),
            "spreading": lambda: viz.mpl_lines(rf.times, {"numerical": rf.delta_x, "theory": th}, "t", "Δx", "Free spreading",
                                               dashes={"theory": "--"}),
            "convergence_dx": lambda: viz.mpl_lines(cdx["dx"], {"L2 error": cdx["L2"]}, "dx", "error", "Grid convergence", markers=True,
                                                    log_y=True)})
    ui.export_panel(b)
