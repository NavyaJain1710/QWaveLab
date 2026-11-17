"""
potentials.py - one-dimensional potential energy profiles V(x) for QWaveLab.

All potentials are returned in dimensionless units (hbar = m = 1 by default).

Discretisation note
-------------------
A "box" of height h between x = left and x = right is placed on the grid by
*cell-overlap weighting*: grid point x_i represents the cell
[x_i - dx/2, x_i + dx/2] and receives V_i = h * (overlap of cell with box) / dx.
When the box edges fall on cell boundaries this is identical to the notebook's
"V = V0 for left <= x <= right", but when they do not, the integral of V(x)
(and hence the effective barrier strength) is preserved to machine precision
instead of being rounded to the nearest grid point.  This removes an O(dx)
systematic error in the barrier width.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Parameter specification (used by the UI to build sliders generically)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    lo: float
    hi: float
    default: float
    step: float
    help: str = ""


@dataclass
class Potential:
    """A discretised potential together with its analytic description."""
    kind: str
    label: str
    V: np.ndarray
    region: Tuple[float, float]            # interaction region [a, b] used for R/T/B
    params: Dict[str, float]
    segments: List[Tuple[float, float, float]] = field(default_factory=list)  # (left,right,height)


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
def box(x: np.ndarray, left: float, right: float, height: float) -> np.ndarray:
    """Rectangular profile of given height on [left, right], cell-overlap weighted."""
    dx = x[1] - x[0]
    lo = np.maximum(x - dx / 2, left)
    hi = np.minimum(x + dx / 2, right)
    overlap = np.clip(hi - lo, 0.0, dx)
    return height * overlap / dx


def free_particle(x, **_):
    return np.zeros_like(x)


def rectangular_barrier(x, V0=2.5, width=1.5, center=0.0):
    """V = V0 for |x - center| <= width/2, else 0 (the notebook's create_potential)."""
    return box(x, center - width / 2, center + width / 2, V0)


def gaussian_barrier(x, V0=2.5, sigma_v=0.8, center=0.0):
    """V(x) = V0 exp(-(x-c)^2 / (2 sigma_v^2))."""
    return V0 * np.exp(-((x - center) ** 2) / (2 * sigma_v ** 2))


def double_barrier_edges(w1, w2, gap, center=0.0):
    """Return ((l1, r1), (l2, r2)) edges of the two barriers of a double-barrier structure."""
    total = w1 + gap + w2
    l1 = center - total / 2
    r1 = l1 + w1
    l2 = r1 + gap
    r2 = l2 + w2
    return (l1, r1), (l2, r2)


def double_barrier(x, V1=3.0, w1=0.8, V2=3.0, w2=0.8, gap=4.0, V_well=0.0, center=0.0):
    """Two rectangular barriers separated by a central region at potential V_well."""
    (l1, r1), (l2, r2) = double_barrier_edges(w1, w2, gap, center)
    V = box(x, l1, r1, V1) + box(x, l2, r2, V2)
    if V_well != 0.0:
        V = V + box(x, r1, l2, V_well)
    return V


def finite_well(x, depth=3.0, width=4.0, center=0.0):
    """V = -depth for |x - center| <= width/2 (attractive well)."""
    return box(x, center - width / 2, center + width / 2, -depth)


# --------------------------------------------------------------------------
# Registry used by the UI
# --------------------------------------------------------------------------
KINDS: Dict[str, str] = {
    "free": "Free particle (V = 0)",
    "rect_barrier": "Single rectangular barrier",
    "gauss_barrier": "Gaussian potential barrier",
    "double_barrier": "Double rectangular barrier",
    "finite_well": "Finite potential well",
}

PARAM_SPECS: Dict[str, List[ParamSpec]] = {
    "free": [],
    "rect_barrier": [
        ParamSpec("V0", "Barrier height V₀", 0.1, 10.0, 2.5, 0.1),
        ParamSpec("width", "Barrier width a", 0.1, 8.0, 1.5, 0.1),
        ParamSpec("center", "Barrier centre", -10.0, 10.0, 0.0, 0.5),
    ],
    "gauss_barrier": [
        ParamSpec("V0", "Peak height V₀", 0.1, 10.0, 2.5, 0.1),
        ParamSpec("sigma_v", "Barrier width σ_V", 0.2, 4.0, 0.8, 0.1),
        ParamSpec("center", "Barrier centre", -10.0, 10.0, 0.0, 0.5),
    ],
    "double_barrier": [
        ParamSpec("V1", "First barrier height V₁", 0.1, 10.0, 3.0, 0.1),
        ParamSpec("w1", "First barrier width", 0.1, 4.0, 0.8, 0.1),
        ParamSpec("V2", "Second barrier height V₂", 0.1, 10.0, 3.0, 0.1),
        ParamSpec("w2", "Second barrier width", 0.1, 4.0, 0.8, 0.1),
        ParamSpec("gap", "Distance between barriers (well width)", 0.5, 15.0, 4.0, 0.1),
        ParamSpec("V_well", "Central-region potential", -3.0, 2.0, 0.0, 0.1),
        ParamSpec("center", "Structure centre", -10.0, 10.0, 0.0, 0.5),
    ],
    "finite_well": [
        ParamSpec("depth", "Well depth V_w", 0.1, 10.0, 3.0, 0.1),
        ParamSpec("width", "Well width", 0.2, 12.0, 4.0, 0.1),
        ParamSpec("center", "Well centre", -10.0, 10.0, 0.0, 0.5),
    ],
}

_BUILDERS: Dict[str, Callable] = {
    "free": free_particle,
    "rect_barrier": rectangular_barrier,
    "gauss_barrier": gaussian_barrier,
    "double_barrier": double_barrier,
    "finite_well": finite_well,
}


def default_params(kind: str) -> Dict[str, float]:
    return {p.name: p.default for p in PARAM_SPECS[kind]}


def full_params(kind: str, params: Optional[Dict[str, float]]) -> Dict[str, float]:
    out = default_params(kind)
    out.update(dict(params or {}))
    return out


def region_for(kind: str, params: Optional[Dict[str, float]] = None) -> Tuple[float, float]:
    """Interaction region [a, b]: where the potential is non-negligible.

    R, T and B are the probabilities of finding the particle at x<a, x>b and
    a<=x<=b respectively.  For a free particle a=b=0 (left/right of the origin).
    """
    p = full_params(kind, params)
    if kind == "free":
        return (0.0, 0.0)
    if kind == "rect_barrier":
        return (p["center"] - p["width"] / 2, p["center"] + p["width"] / 2)
    if kind == "gauss_barrier":
        return (p["center"] - 4 * p["sigma_v"], p["center"] + 4 * p["sigma_v"])
    if kind == "double_barrier":
        (l1, _), (_, r2) = double_barrier_edges(p["w1"], p["w2"], p["gap"], p["center"])
        return (l1, r2)
    if kind == "finite_well":
        return (p["center"] - p["width"] / 2, p["center"] + p["width"] / 2)
    raise ValueError(f"Unknown potential kind: {kind}")


def segments_for(kind: str, params: Optional[Dict[str, float]] = None):
    """Analytic (left, right, height) description, used for shading plots."""
    p = full_params(kind, params)
    if kind == "rect_barrier":
        a, b = region_for(kind, p)
        return [(a, b, p["V0"])]
    if kind == "double_barrier":
        (l1, r1), (l2, r2) = double_barrier_edges(p["w1"], p["w2"], p["gap"], p["center"])
        segs = [(l1, r1, p["V1"]), (l2, r2, p["V2"])]
        if p["V_well"] != 0:
            segs.append((r1, l2, p["V_well"]))
        return segs
    if kind == "finite_well":
        a, b = region_for(kind, p)
        return [(a, b, -p["depth"])]
    return []


def build_potential(kind: str, x: np.ndarray, params: Optional[Dict[str, float]] = None) -> Potential:
    if kind not in _BUILDERS:
        raise ValueError(f"Unknown potential kind: {kind}")
    p = full_params(kind, params)
    V = np.asarray(_BUILDERS[kind](x, **p), dtype=float)
    return Potential(kind, KINDS[kind], V, region_for(kind, p), p, segments_for(kind, p))


def region_masks(x: np.ndarray, region: Tuple[float, float]):
    """Boolean masks (left, inside, right) that partition the grid exactly."""
    a, b = region
    if b <= a:                       # free particle: split at the origin, nothing "inside"
        left = x < a
        return left, np.zeros_like(left), ~left
    left = x < a
    inside = (x >= a) & (x <= b)
    right = x > b
    return left, inside, right


def validate_potential_params(kind: str, params: Dict[str, float]) -> List[str]:
    """Return a list of human-readable errors for invalid potential parameters."""
    errs = []
    p = full_params(kind, params)
    for spec in PARAM_SPECS[kind]:
        v = p[spec.name]
        if not np.isfinite(v):
            errs.append(f"{spec.label} must be a finite number.")
    for k in ("width", "w1", "w2", "sigma_v", "gap"):
        if k in p and p[k] <= 0:
            errs.append(f"'{k}' must be positive.")
    for k in ("V0", "V1", "V2", "depth"):
        if k in p and p[k] < 0:
            errs.append(f"'{k}' must be non-negative (use the well potential for attractive regions).")
    return errs
