"""Metadata and read-time normalization -- the only writer and only reader of
derived scalars (storage rule).

Storage rule: a field earns storage iff rebuilding it needs the network or the
points.  Pure arithmetic on stored fields is computed on read, here, never in a
plot script.  The gauge is bare 1/V everywhere:  chi(k) = <|A(k)|^2> / V.

Weight conventions (see weight_names.py and the README):
  w_eff  -- kernel weight, w_eff = w * ell (theta family: ell^theta; the V/T/D
            cell-measure schemes: the cell measure).  Tiling <=> sum w_eff = V.
  w      -- per-length mark, w = w_eff / ell.  The moment sums below are the
            <w^2 ell^n> family written as SUMS (extensive), in terms of w.

Stored (extensive, need the edge table):
  sum_wl   = sum_e w_e ell_e            (= sum w_eff)
  sum_w2l  = sum_e w_e^2 ell_e          (= sum w_eff^2 / ell)   -> Porod amplitude
  sum_w2l2 = sum_e w_e^2 ell_e^2        (= sum w_eff^2)          -> chi_self(0), sum rule
  sum_w2l3 = sum_e w_e^2 ell_e^3
  sum_w2l_kmax = sum_e w_e^2 ell_e x_e g_d(x_e) / C_d,  x_e = k_max ell_e / 2
             (finite-k_max Porod normalizer; C_2 = 1, C_3 = pi/2)
  vertex_wedge_sum = sum_v sum_{a<b} w_a w_b (pi - gamma_ab) / sin gamma_ab
  rho_w_emp = sum_wl / V                (empirical; tiling schemes give 1)
Derived on read: lambda_ell = N_ell/V, rho_ell, <w^2 ell>, <w^2 ell^2>, C_d,
K_d, and the dimensionless plot axes.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

GAUGE = "bare-1V"
C_D = {2: 1.0, 3: float(np.pi) / 2.0}     # Porod constants (K_d in the paper)

SCHEMA_IDENTITY = ("gauge", "d", "L", "V", "N_ell", "gen", "tess", "scheme", "theta",
                   "seed", "ens", "code_version", "cell")
SCHEMA_KGRID = ("dk", "k_max", "n_bins", "shell_mode")
SCHEMA_SUMS = ("sum_wl", "sum_w2l", "sum_w2l2", "sum_w2l3", "sum_w2l_kmax")
SCHEMA_EXPENSIVE = ("vertex_wedge_sum", "rho_w_emp")


def moment_sums(edge_lengths: np.ndarray, w_eff: np.ndarray, k_max: Optional[float],
                dim: int) -> dict:
    """The stored moment sums from the edge table (per-length w = w_eff / ell)."""
    ell = np.asarray(edge_lengths, dtype=float)
    w_eff = np.asarray(w_eff, dtype=float)
    w = w_eff / np.maximum(ell, 1e-300)
    out = {
        "sum_wl":   float(np.sum(w_eff)),
        "sum_w2l":  float(np.sum(w * w * ell)),
        "sum_w2l2": float(np.sum(w * w * ell ** 2)),
        "sum_w2l3": float(np.sum(w * w * ell ** 3)),
        "sum_w2l_kmax": None,
    }
    if k_max is not None and k_max > 0:
        from .small_r import g_d_form_factor
        x = 0.5 * float(k_max) * ell
        g = g_d_form_factor(x, dim=dim)
        out["sum_w2l_kmax"] = float(np.sum(w * w * ell * x * g) / C_D[dim])
    return out


def vertex_wedge_sum(net, w_eff: np.ndarray) -> float:
    """sum_v sum_{a<b} w_a w_b (pi - gamma_ab)/sin gamma_ab with per-length w."""
    from .small_r import _vertex_angle_sum
    w = np.asarray(w_eff, dtype=float) / np.maximum(net.edges_lengths, 1e-300)
    return float(_vertex_angle_sum(net, w)[0])


def build_metadata(*, net, w_eff: np.ndarray, dim: int, gen: str, tess: str, scheme: str,
                   theta: Optional[float], seed: Optional[int], ens: Optional[int],
                   k_max: Optional[float], n_bins: int, shell_mode: str,
                   code_version: str, with_wedge: bool = True,
                   extra: Optional[dict] = None) -> dict:
    """The release metadata dict (exact schema; nothing derivable is stored)."""
    cell = getattr(net, "cell", None)          # general 2D cell (None = square box)
    if cell is not None:
        L, V, dk = None, float(cell.V), 2.0 * np.pi / float(np.linalg.norm(cell.A, axis=1).max())
    else:
        L = float(net.L)
        V = L ** dim
        dk = 2.0 * np.pi / L
    meta = {
        "gauge": GAUGE, "d": int(dim), "L": L, "V": V, "N_ell": int(net.n_edges),
        "gen": gen, "tess": tess, "scheme": scheme,
        "theta": (None if theta is None else float(theta)),
        "seed": (None if seed is None else int(seed)),
        "ens": (None if ens is None else int(ens)),
        "code_version": code_version,
        "dk": dk, "k_max": (None if k_max is None else float(k_max)),
        "n_bins": int(n_bins), "shell_mode": shell_mode,
    }
    if cell is not None:
        meta["cell"] = cell.as_meta()          # rows = lattice vectors; L is None
    meta.update(moment_sums(net.edges_lengths, w_eff, k_max, dim))
    meta["vertex_wedge_sum"] = vertex_wedge_sum(net, w_eff) if with_wedge else None
    meta["rho_w_emp"] = meta["sum_wl"] / V
    if extra:
        meta.update(extra)
    return meta


def derived(meta: dict) -> dict:
    """Read-time arithmetic on stored fields (never stored)."""
    d = int(meta["d"]); V = float(meta["V"]); N = int(meta["N_ell"])
    lam = N / V
    out = {
        "lambda_ell": lam,
        "rho_w": float(meta["rho_w_emp"]),
        "mean_wl":   meta["sum_wl"] / N,
        "mean_w2l":  meta["sum_w2l"] / N,
        "mean_w2l2": meta["sum_w2l2"] / N,
        "mean_w2l3": meta["sum_w2l3"] / N,
        "mean_w2l_kmax": (None if meta.get("sum_w2l_kmax") is None else meta["sum_w2l_kmax"] / N),
        "C_d": C_D[d], "K_d": C_D[d],
        # sum-rule targets (bare gauge):  chi_self(0) = lambda <w^2 ell^2>;
        # chi_distinct(0) = -lambda <w^2 ell^2> for the tiling schemes.
        "chi_self_0": lam * meta["sum_w2l2"] / N,
        # large-k self tail:  k chi_self -> 2 C_d lambda <w^2 ell>_kmax
        "porod_tail": (None if meta.get("sum_w2l_kmax") is None
                       else 2.0 * C_D[d] * lam * meta["sum_w2l_kmax"] / N),
    }
    # dimensionless axes: see axes() below.
    return out


def axes(meta: dict, k: np.ndarray, chi: np.ndarray):
    """Scaling-regime axes for a chi(k) curve (read-time; nothing stored).

        x = k / (2 pi lambda_ell^{1/d}),   y = chi / (K_d lambda_ell^{(d-1)/d} <w^2 ell>_kmax)

    with K_d = (pi/2)^{d-2} (K_2 = 1, K_3 = pi/2), the paper's 2D and 3D axes.  The
    scaling length is the mean edge spacing lambda_ell^{-1/d}; <w^2 ell>_kmax is the
    truncated Porod moment sum_w2l_kmax / N_ell (already divided by K_d), so the
    large-k plateau of x*y is 1/pi in every dimension.  Falls back to <w^2 ell> when
    the truncated moment is absent.
    """
    d = int(meta["d"])
    lam = float(meta["N_ell"]) / float(meta["V"])
    m2 = meta.get("sum_w2l_kmax")
    m2 = (m2 if m2 is not None else meta["sum_w2l"]) / float(meta["N_ell"])
    x = np.asarray(k, dtype=float) / (2.0 * np.pi * lam ** (1.0 / d))
    y = np.asarray(chi, dtype=float) / (C_D[d] * lam ** ((d - 1.0) / d) * m2)
    return x, y
