"""
analysis.py - physics analysis built on top of the Crank-Nicolson engine.

* stationary scattering (transfer matrix) -> T(E) curves and resonance finding
* momentum-space decomposition and interference term
* classical particle / ensemble comparison
* numerical validation helpers (free spreading, convergence, error norms)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import physics_engine as pe
import potentials as pot
from physics_engine import HBAR


# ==========================================================================
# 1. Stationary scattering: transfer matrix on the (discretised) potential
# ==========================================================================
def potential_layers(x: np.ndarray, V: np.ndarray, tol: float = 1e-12) -> List[Tuple[float, float]]:
    """Piecewise-constant layers (thickness, V) describing the grid potential.

    Grid point x_i represents the cell of width dx around it; consecutive cells with
    equal V are merged.  Only the region where V != 0 is returned (V = 0 outside).
    """
    dx = x[1] - x[0]
    nz = np.flatnonzero(np.abs(V) > tol)
    if nz.size == 0:
        return []
    seg = V[nz[0]: nz[-1] + 1]
    layers: List[Tuple[float, float]] = []
    start = 0
    for i in range(1, len(seg) + 1):
        if i == len(seg) or abs(seg[i] - seg[start]) > tol:
            layers.append(((i - start) * dx, float(seg[start])))
            start = i
    return layers


def transmission_transfer_matrix(layers: List[Tuple[float, float]], energies, mass: float = 1.0) -> np.ndarray:
    """Transmission probability T(E) for a plane wave through piecewise-constant layers.

    In each layer psi'' = -q^2 psi with q^2 = 2m(E - V)/hbar^2, so (psi, psi') is propagated by
        M = [[cos(qd),  sin(qd)/q], [-q sin(qd), cos(qd)]]
    (analytic in q^2, so it is valid for tunnelling q^2 < 0 where cos->cosh, sin(qd)/q->sinh/kappa).
    Matching incident + reflected waves on the left and a transmitted wave on the right gives T.
    """
    E = np.atleast_1d(np.asarray(energies, dtype=float))
    out = np.empty_like(E)
    pos = E > 0
    out[~pos] = 0.0
    Ep = E[pos]
    if Ep.size == 0:
        return out
    k = np.sqrt(2 * mass * Ep) / HBAR
    M11 = np.ones_like(Ep); M12 = np.zeros_like(Ep)
    M21 = np.zeros_like(Ep); M22 = np.ones_like(Ep)
    with np.errstate(all="ignore"):
        for d, Vj in layers:
            q2 = 2 * mass * (Ep - Vj) / HBAR ** 2
            q = np.sqrt(q2.astype(complex))
            c = np.cos(q * d).real
            small = np.abs(q) < 1e-9
            s = np.where(small, d, (np.sin(q * np.where(small, 1.0, q) * 0 + q * d) /
                                   np.where(small, 1.0, q)).real)
            l11, l12, l21, l22 = c, s, -q2 * s, c
            M11, M12, M21, M22 = (l11 * M11 + l12 * M21, l11 * M12 + l12 * M22,
                                  l21 * M11 + l22 * M21, l21 * M12 + l22 * M22)
        ik = 1j * k
        denom = (M22 - ik * M12) + M11 - M21 / ik
        s_ = 2.0 / denom
        u = (M22 - ik * M12) * s_
        w = (ik * M11 - M21) * s_
        tau = M11 * u + M12 * w
        T = np.abs(tau) ** 2
    T = np.where(np.isfinite(T), T, 0.0)
    out[pos] = np.clip(T, 0.0, 1.0)
    return out


def transmission_vs_energy(x, V, energies, mass=1.0) -> np.ndarray:
    return transmission_transfer_matrix(potential_layers(x, V), energies, mass)


def rect_barrier_transmission_analytic(E, V0, a, mass=1.0):
    """Textbook closed form for a rectangular barrier (used to cross-check the transfer matrix)."""
    E = np.asarray(E, dtype=float)
    out = np.empty_like(E)
    lo = E < V0
    with np.errstate(all="ignore"):
        kappa = np.sqrt(2 * mass * (V0 - E[lo])) / HBAR
        out[lo] = 1.0 / (1.0 + V0 ** 2 * np.sinh(kappa * a) ** 2 / (4 * E[lo] * (V0 - E[lo])))
        q = np.sqrt(2 * mass * (E[~lo] - V0)) / HBAR
        den = 4 * E[~lo] * (E[~lo] - V0)
        out[~lo] = np.where(den > 0, 1.0 / (1.0 + V0 ** 2 * np.sin(q * a) ** 2 / np.where(den > 0, den, 1)), 1.0)
    return out


def find_resonances(x, V, E_grid, T_grid, mass=1.0, well_width: Optional[float] = None,
                    well_floor: float = 0.0, prominence: float = 0.02) -> pd.DataFrame:
    """Locate resonant-transmission peaks numerically.

    Peaks of T(E) are found with scipy.signal.find_peaks and then refined by re-evaluating the
    transfer-matrix transmission on a fine local grid.  FWHM gives the resonance width Gamma
    and lifetime tau = hbar/Gamma.  If *well_width* is given, the energy of the n-th level of an
    infinite well of that width is listed as a comparison (calculated, not tabulated).
    """
    layers = potential_layers(x, V)
    E_grid = np.asarray(E_grid)
    dE = E_grid[1] - E_grid[0]
    idx, _ = find_peaks(T_grid, prominence=prominence)
    rows = []
    for i in idx:
        lo, hi = E_grid[max(i - 3, 0)], E_grid[min(i + 3, len(E_grid) - 1)]
        Ef = np.linspace(lo, hi, 400)
        Tf = transmission_transfer_matrix(layers, Ef, mass)
        j = int(np.argmax(Tf))
        E_res, T_pk = float(Ef[j]), float(Tf[j])
        # FWHM: widen the search window until both half-maximum crossings are bracketed
        fwhm = float("nan")
        half = T_pk / 2
        span = 12 * dE
        for _ in range(7):
            Ew = np.linspace(max(E_res - span, 1e-6), E_res + span, 4000)
            Tw = transmission_transfer_matrix(layers, Ew, mass)
            above = Tw >= half
            jj = int(np.argmin(np.abs(Ew - E_res)))
            l = jj
            while l > 0 and above[l]:
                l -= 1
            r = jj
            while r < len(Ew) - 1 and above[r]:
                r += 1
            if l > 0 and r < len(Ew) - 1:
                fwhm = float(Ew[r] - Ew[l])
                break
            span *= 4
        rows.append(dict(E_res=E_res, T_peak=T_pk, FWHM=fwhm,
                         lifetime=(HBAR / fwhm) if fwhm and np.isfinite(fwhm) and fwhm > 0 else float("nan"),
                         k_res=math.sqrt(2 * mass * E_res) / HBAR))
    df = pd.DataFrame(rows, columns=["E_res", "T_peak", "FWHM", "lifetime", "k_res"])
    if well_width and len(df):
        # n-th resonance <-> n-th quasi-bound level (peaks are ordered by energy)
        df["n_est"] = np.arange(1, len(df) + 1)
        df["E_infinite_well"] = well_floor + (df["n_est"] * math.pi * HBAR / well_width) ** 2 / (2 * mass)
    return df


# ==========================================================================
# 2. Momentum space and interference
# ==========================================================================
def momentum_distribution(psi: np.ndarray, dx: float) -> Dict[str, np.ndarray]:
    """|phi(p)|^2 with positive/negative momentum weights, mean and spread."""
    p, phi = pe.momentum_space(psi, dx)
    dens = pe.probability_density(phi)
    dp = p[1] - p[0]
    mean, sd = pe.moments(dens, p, dp)
    return dict(p=p, density=dens, mean=float(mean), delta_p=float(sd),
                weight_pos=float(dens[p > 0].sum() * dp), weight_neg=float(dens[p < 0].sum() * dp))


def momentum_decomposition(psi: np.ndarray, dx: float) -> Dict[str, np.ndarray]:
    """Split psi = psi_R + psi_L into right-moving (p>0) and left-moving (p<0) parts.

        |psi|^2 = |psi_R|^2 + |psi_L|^2 + 2 Re(psi_R^* psi_L)

    The last term is the interference contribution.  Because psi_R and psi_L are orthogonal,
    it integrates to ~0: interference redistributes probability in space, it does not create any.
    """
    N = len(psi)
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    F = np.fft.fft(psi)
    psi_R = np.fft.ifft(np.where(k > 0, F, 0))
    psi_L = np.fft.ifft(np.where(k < 0, F, 0))
    rest = psi - psi_R - psi_L               # k == 0 component (negligible)
    psi_R = psi_R + rest
    inter = 2 * np.real(np.conj(psi_R) * psi_L)
    return dict(psi_R=psi_R, psi_L=psi_L, density_R=pe.probability_density(psi_R),
                density_L=pe.probability_density(psi_L), interference=inter,
                integral=float(inter.sum() * dx))


def uncertainty_product(dx_unc, dp_unc):
    return np.asarray(dx_unc) * np.asarray(dp_unc)


def uncertainty_scan(sigmas, k0=2.0, mass=1.0, dx=0.05, L=None) -> pd.DataFrame:
    """Δx, Δp and Δx·Δp of the initial packet for several widths (computed from the grid by FFT)."""
    rows = []
    for s in sigmas:
        half = max(10 * s, 20.0)
        x = pe.make_grid(-half, half, dx)
        psi = pe.gaussian_wave_packet(x, 0.0, s, k0)
        psi /= math.sqrt(pe.total_probability(psi, dx))
        _, sx = pe.moments(pe.probability_density(psi), x, dx)
        md = momentum_distribution(psi, dx)
        prod = float(sx * md["delta_p"])
        rows.append(dict(sigma=s, delta_x=float(sx), delta_p=md["delta_p"], product=prod,
                         bound=HBAR / 2, ratio=prod / (HBAR / 2), satisfied=prod >= HBAR / 2 - 1e-6))
    return pd.DataFrame(rows)


# ==========================================================================
# 3. Classical comparison
# ==========================================================================
def sample_classical_ensemble(sigma, k0, x0, mass=1.0, n=40000, seed=7):
    """Classical ensemble with the same position and momentum distributions as the quantum packet.

    For a minimum-uncertainty Gaussian the Wigner function is a product of Gaussians, so positions
    ~ N(x0, sigma) and momenta ~ N(hbar k0, hbar/(2 sigma)) are independent.
    """
    rng = np.random.default_rng(seed)
    xs = rng.normal(x0, sigma, n)
    ps = rng.normal(HBAR * k0, HBAR / (2 * sigma), n)
    return xs, ps / mass          # positions, velocities


def classical_positions(xs, vs, t, a, b, V0, mass=1.0):
    """Exact Newtonian motion through a rectangular barrier [a, b] of height V0 at time t.

    Particles with E < V0 are reflected elastically at x = a; those with E > V0 slow down inside
    the barrier (speed sqrt(2(E-V0)/m)) and continue.  Energy conservation fixes the motion
    piecewise, so no time-stepping error is involved.
    """
    xs = np.asarray(xs); vs = np.asarray(vs)
    E = 0.5 * mass * vs ** 2
    pos = xs + vs * t                                     # free motion (left movers / before barrier)
    with np.errstate(all="ignore"):
        t1 = (a - xs) / vs
        hit = (vs > 0) & (xs < a) & (t >= t1)
        pass_ = hit & (E > V0)
        refl = hit & (E <= V0)
        vb = np.sqrt(2 * np.maximum(E - V0, 0) / mass)
        tb = (b - a) / np.where(vb > 0, vb, np.inf)
        inside = pass_ & (t < t1 + tb)
        after = pass_ & ~inside
        pos = np.where(inside, a + vb * (t - t1), pos)
        pos = np.where(after, b + vs * (t - t1 - tb), pos)
        pos = np.where(refl, a - vs * (t - t1), pos)
    return pos


def classical_transmission_fraction(sigma, k0, mass, V0) -> float:
    """Fraction of the packet's momentum distribution with kinetic energy above V0 (moving right)."""
    L = max(60.0, 40 * sigma)
    x = pe.make_grid(-L, L, 0.05)
    psi = pe.gaussian_wave_packet(x, 0.0, sigma, k0)
    psi /= math.sqrt(pe.total_probability(psi, 0.05))
    md = momentum_distribution(psi, 0.05)
    p, dens = md["p"], md["density"]
    dp = p[1] - p[0]
    ok = (p > 0) & (p ** 2 / (2 * mass) > V0)
    return float(dens[ok].sum() * dp)


# ==========================================================================
# 4. Numerical validation
# ==========================================================================
def free_spreading_theory(sigma, t, mass=1.0):
    """sigma(t) = sigma sqrt(1 + (hbar t / 2 m sigma^2)^2) for a free Gaussian packet."""
    return sigma * np.sqrt(1 + (HBAR * np.asarray(t) / (2 * mass * sigma ** 2)) ** 2)


def exact_free_packet(x, x0, sigma, k0, t, mass=1.0):
    """Exact free-particle solution (Galilean boost of the spreading Gaussian)."""
    tau = HBAR * t / (2 * mass * sigma ** 2)
    A = (2 * np.pi * sigma ** 2) ** -0.25
    xc = x - x0 - HBAR * k0 * t / mass
    return (A / np.sqrt(1 + 1j * tau)) * np.exp(-xc ** 2 / (4 * sigma ** 2 * (1 + 1j * tau))
                                               + 1j * k0 * x - 1j * HBAR * k0 ** 2 * t / (2 * mass))


def error_vs_exact(dx, dt, t_end=4.0, sigma=1.2, k0=2.0, x0=-6.0, mass=1.0, half=30.0):
    """Max-norm and L2 error of the CN solution against the exact free packet at t_end."""
    x = pe.make_grid(-half, half, dx)
    psi0 = pe.gaussian_wave_packet(x, x0, sigma, k0)
    snaps, times, norms = pe.simulate_wave_packet(x, psi0, np.zeros_like(x), dx, dt, t_end,
                                                  store_every=10 ** 9, mass=mass)
    ex = exact_free_packet(x, x0, sigma, k0, times[-1], mass)
    diff = snaps[-1] - ex
    return dict(L2=float(math.sqrt((np.abs(diff) ** 2).sum() * dx)), Linf=float(np.abs(diff).max()),
                norm_drift=float(np.abs(norms - norms[0]).max()), t_end=float(times[-1]))


def convergence_study(kind="dx", values=None, **kw) -> pd.DataFrame:
    """Grid (dx) or time-step (dt) convergence versus the exact free-packet solution."""
    rows = []
    if kind == "dx":
        values = values or [0.4, 0.2, 0.1, 0.05]
        for v in values:
            r = error_vs_exact(dx=v, dt=kw.get("dt", 0.005), **{k: w for k, w in kw.items() if k != "dt"})
            rows.append(dict(dx=v, dt=kw.get("dt", 0.005), **r))
    else:
        values = values or [0.2, 0.1, 0.05, 0.025]
        for v in values:
            r = error_vs_exact(dx=kw.get("dx", 0.02), dt=v, **{k: w for k, w in kw.items() if k != "dx"})
            rows.append(dict(dx=kw.get("dx", 0.02), dt=v, **r))
    df = pd.DataFrame(rows)
    col = "dx" if kind == "dx" else "dt"
    err = df["L2"].to_numpy()
    h = df[col].to_numpy()
    order = [np.nan] + [math.log(err[i - 1] / err[i]) / math.log(h[i - 1] / h[i]) if err[i] > 0 and err[i - 1] > 0
                        else np.nan for i in range(1, len(df))]
    df["observed_order"] = order
    return df


def conservation_errors(res: pe.SimulationResult) -> Dict[str, float]:
    return dict(
        max_norm_deviation=float(np.max(np.abs(res.step_norm - res.norm[0]))),
        max_energy_rel_drift=float(np.max(np.abs(res.energy - res.energy[0])) / max(abs(res.energy[0]), 1e-300)),
        final_norm=float(res.norm[-1]),
        initial_energy=float(res.energy[0]),
    )
