"""
experiments.py - automated scattering experiments (parameter sweeps) built on the CN engine.

Every row of every table is produced by an actual wave-packet simulation; the plane-wave
transfer-matrix transmission is added as an independent theoretical reference column.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

import analysis as an
import physics_engine as pe
import potentials as pot
from physics_engine import HBAR, SimConfig


# --------------------------------------------------------------------------
# Automatic domain / duration for a scattering run
# --------------------------------------------------------------------------
def auto_scattering_config(kind: str, pot_params: Dict[str, float], k0: float, sigma: float,
                           mass: float = 1.0, dx: float = 0.1, dt: float = 0.02,
                           start_gap: float = 6.0, max_snapshots: int = 300, extra_time: float = 0.0) -> SimConfig:
    """Choose x0, domain and duration so that the whole collision fits inside the box.

    The packet starts `start_gap` widths (+2) to the left of the interaction region, and the run
    lasts until the transmitted packet's centre is ~8 widths beyond the region.
    """
    a, b = pot.region_for(kind, pot_params)
    x0 = a - start_gap * sigma - 2.0
    v = HBAR * k0 / mass
    if k0 <= 0 or k0 * sigma / mass * HBAR < 2.5:
        raise ValueError(
            f"Packet too slow / too narrow for an automatic scattering run: need ħk₀σ/m ≳ 2.5 "
            f"(here {HBAR * k0 * sigma / mass:.2f}). The packet would spread faster than it travels – "
            "increase k₀ or σ.")
    # time for the packet edge (4.5 widths, including free spreading) to clear the region
    t = (b - x0 + 4.5 * sigma) / v
    for _ in range(40):
        s_t = float(an.free_spreading_theory(sigma, t, mass))
        t_new = (b - x0 + 4.5 * s_t) / v
        if abs(t_new - t) < 1e-3:
            break
        t = t_new
    s_t = float(an.free_spreading_theory(sigma, t, mass))
    t = t_new + 1.0
    # left wall: initial packet or reflected packet, whichever extends further left
    x_min = min(x0 - 6 * sigma - 5, a - (b - a) - 10.5 * s_t - 5)
    x_max = b + 10.5 * s_t + 5
    if extra_time > 0:                       # keep watching (e.g. resonant leakage); widen the box accordingly
        t += extra_time
        x_min -= v * extra_time
        x_max += v * extra_time
    x_min = math.floor(x_min / dx) * dx
    x_max = math.ceil(x_max / dx) * dx
    return SimConfig.make(kind, pot_params, x_min=x_min, x_max=x_max, dx=dx, dt=dt, t_max=round(t, 2),
                          mass=mass, x0=x0, sigma=sigma, k0=k0, max_snapshots=max_snapshots)


# --------------------------------------------------------------------------
# One scattering run -> R, T, B
# --------------------------------------------------------------------------
def run_scattering(kind, pot_params, k0, sigma, mass=1.0, dx=0.1, dt=0.02) -> Dict[str, float]:
    cfg = auto_scattering_config(kind, pot_params, k0, sigma, mass, dx, dt, max_snapshots=20)
    errs, _ = pe.validate_config(cfg)
    if errs:
        raise ValueError(" ".join(errs))
    res = pe.simulate(cfg)
    R, T, B = res.final_RTB()
    E0 = HBAR ** 2 * k0 ** 2 / (2 * mass)
    E_ref = np.array([E0])
    T_plane = float(an.transmission_vs_energy(res.x, res.V, E_ref, mass)[0])
    return dict(E0=E0, R=R, T=T, B=B, total=R + T + B, T_plane_wave=T_plane,
                complete=res.scattering_complete(), runtime_s=res.runtime)


# --------------------------------------------------------------------------
# Experiments A, B, C
# --------------------------------------------------------------------------
EXPERIMENTS: Dict[str, Dict] = {
    "A": dict(title="Experiment A – Barrier height variation", param="V0", label="Barrier height V₀",
              default_range=(0.5, 6.0), default_n=12),
    "B": dict(title="Experiment B – Barrier width variation", param="width", label="Barrier width a",
              default_range=(0.2, 3.0), default_n=12),
    "C": dict(title="Experiment C – Initial momentum variation", param="k0", label="Initial wave number k₀",
              default_range=(1.2, 4.0), default_n=12),
}

BASE_DEFAULTS = dict(V0=2.5, width=1.5, k0=2.0, sigma=3.0, mass=1.0, dx=0.1, dt=0.02)


def run_experiment(exp_key: str, values: Iterable[float], base: Optional[Dict[str, float]] = None,
                   progress: Optional[Callable[[float], None]] = None) -> pd.DataFrame:
    """Sweep one parameter of a rectangular-barrier scattering problem."""
    spec = EXPERIMENTS[exp_key]
    b = dict(BASE_DEFAULTS)
    b.update(base or {})
    values = list(values)
    rows = []
    for i, val in enumerate(values):
        p = dict(b)
        p[spec["param"]] = float(val)
        out = run_scattering("rect_barrier", dict(V0=p["V0"], width=p["width"], center=0.0),
                             p["k0"], p["sigma"], p["mass"], p["dx"], p["dt"])
        rows.append({spec["param"]: float(val), **out})
        if progress:
            progress((i + 1) / len(values))
    df = pd.DataFrame(rows)
    for k, v in b.items():
        if k != spec["param"]:
            df.attrs[k] = v
    df.attrs["experiment"] = exp_key
    df.attrs["param"] = spec["param"]
    df.attrs["fixed"] = {k: v for k, v in b.items() if k != spec["param"]}
    return df


# --------------------------------------------------------------------------
# Interpretation: sentences generated from the computed numbers
# --------------------------------------------------------------------------
def interpret_experiment(exp_key: str, df: pd.DataFrame, fixed: Optional[Dict[str, float]] = None) -> str:
    """Plain-language reading of the sweep, with every number taken from *df*."""
    fixed = fixed if fixed is not None else df.attrs.get("fixed", {})
    m = fixed.get("mass", 1.0)
    p = EXPERIMENTS[exp_key]["param"]
    x = df[p].to_numpy(); T = df["T"].to_numpy(); R = df["R"].to_numpy()
    lines: List[str] = []
    dec = bool(np.all(np.diff(T) <= 1e-3))
    inc = bool(np.all(np.diff(T) >= -1e-3))
    trend = "decreases monotonically" if dec else ("increases monotonically" if inc else "varies non-monotonically")
    lines.append(f"Over the scanned range, the transmission probability **T {trend}** "
                 f"from {T[0]:.3f} at {p} = {x[0]:.2f} to {T[-1]:.3f} at {p} = {x[-1]:.2f}; "
                 f"reflection moves from {R[0]:.3f} to {R[-1]:.3f}.")
    if exp_key == "A":
        E0 = df["E0"].iloc[0]
        lines.append(f"The carrier energy is E₀ = ħ²k₀²/2m = {E0:.3f}. "
                     "A classical particle would be transmitted only for V₀ < E₀, giving a sharp step in T; "
                     "the wave packet instead shows a smooth fall-off because the barrier is reflective "
                     "even for V₀ < E₀ and partly transparent for V₀ > E₀ (tunnelling).")
        below = df[df["V0"] > E0]
        if len(below):
            lines.append(f"For V₀ > E₀ (tunnelling regime) T is still nonzero: e.g. T = {below['T'].iloc[0]:.3f} "
                         f"at V₀ = {below['V0'].iloc[0]:.2f}.")
    elif exp_key == "B":
        V0 = fixed.get("V0", 2.5); E0 = df["E0"].iloc[0]
        if V0 > E0 and (T > 1e-8).sum() >= 3:
            kappa = math.sqrt(2 * m * (V0 - E0)) / HBAR
            sel = T > 1e-8
            slope = np.polyfit(x[sel][-min(6, sel.sum()):], np.log(T[sel][-min(6, sel.sum()):]), 1)[0]
            lines.append(f"Since V₀ = {V0:.2f} > E₀ = {E0:.3f}, the wave decays inside the barrier with "
                         f"κ = √(2m(V₀−E₀))/ħ = {kappa:.3f}. For wide barriers T ∝ exp(−2κa), so ln T should have "
                         f"slope ≈ −2κ = {-2 * kappa:.3f}. The simulated slope over the widest barriers is "
                         f"{slope:.3f} (differences come from the finite momentum spread of the packet, which "
                         "favours its high-energy tail).")
        else:
            lines.append("Because E₀ ≥ V₀ here, T oscillates with width: reflections from the two barrier edges "
                         "interfere, and T = 1 occurs when the barrier holds an integer number of half-wavelengths.")
    elif exp_key == "C":
        V0 = fixed.get("V0", 2.5)
        kc = math.sqrt(2 * m * V0) / HBAR
        lines.append(f"Transmission rises with momentum. The classical threshold is k = √(2mV₀)/ħ = {kc:.3f} "
                     f"(E = V₀ = {V0:.2f}); quantum transmission is already nonzero below it and still below 1 "
                     "just above it.")
    err = float(np.max(np.abs(df["total"] - 1)))
    lines.append(f"Probability bookkeeping: max |R+T+B−1| = {err:.1e}. "
                 f"{int((~df['complete']).sum())} of {len(df)} runs had not completed the collision.")
    diff = float(np.max(np.abs(df["T"] - df["T_plane_wave"])))
    lines.append(f"The last column compares with the plane-wave (single-energy) transmission from the "
                 f"transfer matrix; the largest gap is {diff:.3f}, caused by the packet's energy spread.")
    return "\n\n".join(lines)


# --------------------------------------------------------------------------
# Wave-packet check of T(E) near resonances
# --------------------------------------------------------------------------
def packet_transmission_scan(kind, pot_params, k_values, sigma=6.0, mass=1.0, dx=0.1, dt=0.05,
                             progress: Optional[Callable[[float], None]] = None) -> pd.DataFrame:
    rows = []
    for i, k in enumerate(k_values):
        out = run_scattering(kind, pot_params, float(k), sigma, mass, dx, dt)
        rows.append(dict(k0=float(k), **out))
        if progress:
            progress((i + 1) / len(k_values))
    return pd.DataFrame(rows)
