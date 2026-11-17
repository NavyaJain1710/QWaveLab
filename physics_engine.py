"""
physics_engine.py - numerical solver for the 1-D time-dependent Schroedinger equation.

    i hbar dpsi/dt = H psi,      H = -hbar^2/(2m) d^2/dx^2 + V(x)

Method (reused from the Colab notebook)
---------------------------------------
* Finite-difference Hamiltonian on a uniform grid, Dirichlet (psi = 0) walls.
* Crank-Nicolson time stepping:   (I + i dt H / 2hbar) psi^{n+1} = (I - i dt H / 2hbar) psi^n
  This is the Cayley transform of a Hermitian H, so it is *unitary*:
  total probability and <H> are conserved to round-off, for any dt.
* The sparse (tridiagonal) left-hand matrix is LU-factorised ONCE and re-used
  for every time step.

This module has no dependency on Streamlit / Plotly so it can be tested and
re-used on its own.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import factorized

import potentials as pot

HBAR = 1.0           # dimensionless units (hbar = m = 1 by default)
MAX_WORK = 6.0e7     # grid points x time steps allowed before we refuse to run


# ==========================================================================
# Configuration
# ==========================================================================
@dataclass(frozen=True)
class SimConfig:
    """Everything that defines one simulation (hashable via .key())."""
    x_min: float = -50.0
    x_max: float = 50.0
    dx: float = 0.1
    dt: float = 0.01
    t_max: float = 14.0
    mass: float = 1.0
    x0: float = -12.0
    sigma: float = 1.2
    k0: float = 2.0
    potential_kind: str = "rect_barrier"
    potential_params: tuple = (("V0", 2.5), ("width", 1.5), ("center", 0.0))
    max_snapshots: int = 300

    # -- helpers -----------------------------------------------------------
    def key(self) -> tuple:
        return dataclasses.astuple(self)

    @staticmethod
    def from_key(key: tuple) -> "SimConfig":
        return SimConfig(*key)

    @staticmethod
    def make(potential_kind="rect_barrier", potential_params=None, **kw) -> "SimConfig":
        params = pot.full_params(potential_kind, potential_params)
        return SimConfig(potential_kind=potential_kind,
                         potential_params=tuple(sorted(params.items())), **kw)

    @property
    def pot_dict(self) -> Dict[str, float]:
        return dict(self.potential_params)

    @property
    def n_points(self) -> int:
        return int(round((self.x_max - self.x_min) / self.dx)) + 1

    @property
    def n_steps(self) -> int:
        return max(1, int(round(self.t_max / self.dt)))

    @property
    def energy_nominal(self) -> float:
        """Kinetic energy of the carrier wave, hbar^2 k0^2 / 2m."""
        return HBAR ** 2 * self.k0 ** 2 / (2 * self.mass)


# ==========================================================================
# Notebook functions (kept, generalised for hbar / mass arguments)
# ==========================================================================
def make_grid(x_min: float, x_max: float, dx: float) -> np.ndarray:
    n = int(round((x_max - x_min) / dx)) + 1
    return x_min + dx * np.arange(n)


def gaussian_wave_packet(x, x0, sigma, k0):
    """Normalised Gaussian packet.  |psi|^2 is Gaussian with standard deviation sigma:
        psi = (2 pi sigma^2)^(-1/4) exp(-(x-x0)^2 / 4 sigma^2) exp(i k0 x)
    """
    A = (1.0 / (2.0 * np.pi * sigma ** 2)) ** 0.25
    return A * np.exp(-((x - x0) ** 2) / (4.0 * sigma ** 2)) * np.exp(1j * k0 * x)


def build_hamiltonian(V: np.ndarray, dx: float, mass: float = 1.0, hbar: float = HBAR):
    """Sparse tridiagonal H: diagonal hbar^2/(m dx^2)+V, off-diagonals -hbar^2/(2 m dx^2)."""
    N = len(V)
    main = hbar ** 2 / (mass * dx ** 2) + V
    off = np.full(N - 1, -hbar ** 2 / (2 * mass * dx ** 2))
    return diags([off, main, off], [-1, 0, 1], format="csc")


def simulate_wave_packet(x, psi_initial, V, dx, dt, t_max, store_every=10,
                         mass: float = 1.0, hbar: float = HBAR):
    """Crank-Nicolson evolution (the notebook solver).

    Returns (snapshots[nsnap, N], snapshot_times, norm_history[steps]).
    The final step is always stored.
    """
    N = len(x)
    steps = max(1, int(round(t_max / dt)))
    H = build_hamiltonian(V, dx, mass, hbar)
    I = eye(N, format="csc")
    A = I + (1j * dt / (2 * hbar)) * H          # implicit side
    B = I - (1j * dt / (2 * hbar)) * H          # explicit side
    solve_A = factorized(A)                      # factorise once, solve many times

    psi = np.array(psi_initial, dtype=complex)
    snaps = [psi.copy()]
    times = [0.0]
    norms = np.empty(steps)
    for n in range(steps):
        psi = solve_A(B @ psi)
        norms[n] = (psi.real ** 2 + psi.imag ** 2).sum() * dx
        if (n + 1) % store_every == 0 or n == steps - 1:
            snaps.append(psi.copy())
            times.append((n + 1) * dt)
    return np.array(snaps), np.array(times), norms


# ==========================================================================
# Observables
# ==========================================================================
def probability_density(psi: np.ndarray) -> np.ndarray:
    return psi.real ** 2 + psi.imag ** 2


def total_probability(psi: np.ndarray, dx: float) -> np.ndarray:
    """Norm  int |psi|^2 dx  for one wave function or a stack (last axis = x)."""
    return probability_density(psi).sum(axis=-1) * dx


def momentum_space(psi: np.ndarray, dx: float, hbar: float = HBAR):
    """Momentum-space wave function via FFT.

    Returns (p, phi) with p = hbar k (ascending order) and phi normalised so that
    sum |phi|^2 dp = sum |psi|^2 dx (Parseval).  Works on stacks (last axis = x).
    """
    N = psi.shape[-1]
    k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    phi = np.fft.fftshift(np.fft.fft(psi, axis=-1), axes=-1) * dx / np.sqrt(2 * np.pi)
    # phi(k) normalised in k; convert to p = hbar k:  phi_p = phi_k / sqrt(hbar)
    return hbar * k, phi / np.sqrt(hbar)


def moments(density: np.ndarray, grid: np.ndarray, step: float):
    """Mean and standard deviation of a distribution given on a uniform grid."""
    norm = density.sum(axis=-1) * step
    mean = (density * grid).sum(axis=-1) * step / norm
    var = (density * grid ** 2).sum(axis=-1) * step / norm - mean ** 2
    return mean, np.sqrt(np.maximum(var, 0.0))


# ==========================================================================
# Result container
# ==========================================================================
@dataclass
class SimulationResult:
    cfg: SimConfig
    x: np.ndarray
    V: np.ndarray
    region: Tuple[float, float]
    times: np.ndarray
    psi: np.ndarray                 # (n_snap, N) complex
    step_times: np.ndarray          # time of every CN step
    step_norm: np.ndarray           # norm after every CN step
    norm: np.ndarray                # per snapshot
    energy: np.ndarray              # <H>
    pot_energy: np.ndarray          # <V>
    mean_x: np.ndarray
    delta_x: np.ndarray
    mean_p: np.ndarray
    delta_p: np.ndarray
    R: np.ndarray
    T: np.ndarray
    B: np.ndarray
    runtime: float = 0.0
    signature: str = ""

    @property
    def n_snap(self) -> int:
        return len(self.times)

    @property
    def dx(self) -> float:
        return self.cfg.dx

    @property
    def density(self) -> np.ndarray:
        return probability_density(self.psi)

    @property
    def kin_energy(self) -> np.ndarray:
        return self.energy - self.pot_energy

    def metrics_at(self, i: int) -> Dict[str, float]:
        return dict(t=self.times[i], total=self.norm[i], R=self.R[i], T=self.T[i], B=self.B[i],
                    energy=self.energy[i], delta_x=self.delta_x[i], delta_p=self.delta_p[i])

    def final_RTB(self) -> Tuple[float, float, float]:
        return float(self.R[-1]), float(self.T[-1]), float(self.B[-1])

    def scattering_complete(self, tol: float = 0.01) -> bool:
        """True once the packet entered the interaction region and has (almost) left it again."""
        if self.region[1] <= self.region[0]:
            return True
        entered = np.max(self.B) > 1e-4 or self.T[-1] > 1e-4
        return bool(entered and self.B[-1] < tol)


# ==========================================================================
# Validation of inputs
# ==========================================================================
def validate_config(cfg: SimConfig) -> Tuple[List[str], List[str]]:
    """Return (errors, warnings).  Errors block the run; warnings are shown beside results."""
    errs: List[str] = []
    warns: List[str] = []
    finite = all(np.isfinite(v) for v in (cfg.x_min, cfg.x_max, cfg.dx, cfg.dt, cfg.t_max,
                                          cfg.mass, cfg.x0, cfg.sigma, cfg.k0))
    if not finite:
        return ["All numerical parameters must be finite numbers."], warns
    if cfg.mass <= 0:
        errs.append("Particle mass m must be positive.")
    if cfg.sigma <= 0:
        errs.append("Wave-packet width σ must be positive.")
    if cfg.dx <= 0 or cfg.dt <= 0:
        errs.append("Grid spacing Δx and time step Δt must be positive.")
    if cfg.t_max <= 0:
        errs.append("Simulation duration must be positive.")
    if cfg.x_max <= cfg.x_min:
        errs.append("Domain is empty: x_max must be larger than x_min.")
    errs += pot.validate_potential_params(cfg.potential_kind, cfg.pot_dict)
    if errs:
        return errs, warns

    N, steps = cfg.n_points, cfg.n_steps
    if N < 50:
        errs.append(f"Only {N} grid points – increase the domain or decrease Δx.")
    if N * steps > MAX_WORK:
        errs.append(f"Simulation too large ({N} points × {steps} steps). "
                    "Increase Δx / Δt, shorten the duration or shrink the domain.")
    if not (cfg.x_min < cfg.x0 < cfg.x_max):
        errs.append("Initial packet position x₀ lies outside the computational domain.")
    a, b = pot.region_for(cfg.potential_kind, cfg.pot_dict)
    if cfg.potential_kind != "free" and (a < cfg.x_min or b > cfg.x_max):
        errs.append("The potential extends beyond the computational domain.")
    if errs:
        return errs, warns

    v = HBAR * cfg.k0 / cfg.mass
    sig_end = cfg.sigma * math.sqrt(1 + (HBAR * cfg.t_max / (2 * cfg.mass * cfg.sigma ** 2)) ** 2)
    if cfg.x0 - 5 * cfg.sigma < cfg.x_min or cfg.x0 + 5 * cfg.sigma > cfg.x_max:
        warns.append("The initial packet is closer than 5σ to a domain wall and is clipped.")
    x_front = cfg.x0 + v * cfg.t_max
    if cfg.potential_kind == "free":
        if x_front + 4 * sig_end > cfg.x_max or x_front - 4 * sig_end < cfg.x_min:
            warns.append("The packet reaches the domain wall before the end of the run; "
                         "walls are perfectly reflecting (Dirichlet) so late-time data are unphysical.")
    else:
        # reflected packet returns toward x_min, transmitted one travels toward x_max
        x_left = cfg.x0 - 0 * v   # reflected packet cannot go further left than x0 (energy conserved)
        if x_front - (b - a) + 4 * sig_end > cfg.x_max and cfg.k0 > 0:
            warns.append("The transmitted packet may hit the right domain wall before the run ends. "
                         "Enlarge the domain or shorten the duration.")
        if x_left - 4 * sig_end < cfg.x_min:
            warns.append("The reflected packet may hit the left domain wall (x_min is too close to x₀).")
    ppw = 2 * math.pi / (abs(cfg.k0) * cfg.dx) if cfg.k0 != 0 else np.inf
    if ppw < 10:
        warns.append(f"Only {ppw:.1f} grid points per de Broglie wavelength – the lattice dispersion "
                     "distorts the packet. Use Δx ≤ λ/10.")
    k_hi = abs(cfg.k0) + 3 / (2 * cfg.sigma)
    if k_hi * cfg.dx > 1.0:
        warns.append("Momentum components of the packet approach the grid (Nyquist) limit; "
                     "decrease Δx or increase σ.")
    e_char = cfg.energy_nominal + max([abs(v_) for _, v_ in cfg.potential_params
                                       if isinstance(v_, (int, float))] + [0])
    if cfg.dt * e_char > 0.5:
        warns.append("Δt·E is large: Crank–Nicolson stays stable and unitary, but phase errors grow. "
                     "Try a smaller Δt.")
    if cfg.potential_kind != "free":
        widths = [vv for kk, vv in cfg.potential_params if kk in ("width", "w1", "w2")]
        if widths and min(widths) < 3 * cfg.dx:
            warns.append("A barrier is thinner than 3 grid cells – it is poorly resolved. Decrease Δx.")
    if cfg.k0 <= 0:
        warns.append("k₀ ≤ 0: the packet moves away from (or stays near) the barrier, so no scattering occurs.")
    return errs, warns


# ==========================================================================
# Main entry point
# ==========================================================================
def simulate(cfg: SimConfig) -> SimulationResult:
    """Run a complete simulation described by *cfg* and compute all observables."""
    import time as _time
    t_start = _time.perf_counter()
    errs, _ = validate_config(cfg)
    if errs:
        raise ValueError(" ".join(errs))

    x = make_grid(cfg.x_min, cfg.x_max, cfg.dx)
    P = pot.build_potential(cfg.potential_kind, x, cfg.pot_dict)
    psi0 = gaussian_wave_packet(x, cfg.x0, cfg.sigma, cfg.k0)
    # re-normalise on the grid so that sum|psi|^2 dx = 1 exactly
    psi0 = psi0 / math.sqrt(total_probability(psi0, cfg.dx))

    steps = cfg.n_steps
    store_every = max(1, math.ceil(steps / max(2, cfg.max_snapshots - 1)))
    snaps, times, step_norm = simulate_wave_packet(
        x, psi0, P.V, cfg.dx, cfg.dt, steps * cfg.dt, store_every, cfg.mass, HBAR)

    res = _observables(cfg, x, P, snaps, times, step_norm)
    res.runtime = _time.perf_counter() - t_start
    res.signature = str(abs(hash(cfg.key())))
    return res


def _observables(cfg, x, P, snaps, times, step_norm) -> SimulationResult:
    dx = cfg.dx
    dens = probability_density(snaps)
    norm = dens.sum(axis=1) * dx

    # <H> = Re <psi|H|psi> / <psi|psi>  with the *same* discrete H used for the evolution
    H = build_hamiltonian(P.V, dx, cfg.mass)
    Hpsi = (H @ snaps.T).T
    energy = np.real(np.sum(np.conj(snaps) * Hpsi, axis=1)) * dx / norm
    pot_energy = (dens * P.V).sum(axis=1) * dx / norm

    mean_x, delta_x = moments(dens, x, dx)

    p, phi = momentum_space(snaps, dx)
    dp = p[1] - p[0]
    dens_p = probability_density(phi)
    mean_p, delta_p = moments(dens_p, p, dp)

    left, inside, right = pot.region_masks(x, P.region)
    R = dens[:, left].sum(axis=1) * dx
    T = dens[:, right].sum(axis=1) * dx
    B = dens[:, inside].sum(axis=1) * dx

    step_times = cfg.dt * np.arange(1, len(step_norm) + 1)
    return SimulationResult(cfg, x, P.V, P.region, times, snaps, step_times, step_norm, norm,
                            energy, pot_energy, mean_x, delta_x, mean_p, delta_p, R, T, B)
