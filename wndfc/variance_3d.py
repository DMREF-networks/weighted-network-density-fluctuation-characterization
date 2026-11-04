"""Direct edge-length variance under 3D PBC.

3D analog of ``variance.py``: random window centers in [0, L)^3, each
edge clipped against the ball B(w, R), variance taken over windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numba import njit, prange

from .network_3d import Network3D


@dataclass
class Sigma2Result3D:
    R: np.ndarray
    sigma2: np.ndarray
    n_windows: int


@njit(fastmath=True, cache=True)
def _ball_segment_clip(cx: float, cy: float, cz: float, R: float,
                       p1x: float, p1y: float, p1z: float,
                       p2x: float, p2y: float, p2z: float) -> float:
    dx = p2x - p1x
    dy = p2y - p1y
    dz = p2z - p1z
    a = dx * dx + dy * dy + dz * dz
    if a < 1e-24:
        d2 = (p1x - cx) ** 2 + (p1y - cy) ** 2 + (p1z - cz) ** 2
        return np.sqrt(a) if d2 < R * R else 0.0
    edgelen = np.sqrt(a)
    e1 = (p1x - cx) ** 2 + (p1y - cy) ** 2 + (p1z - cz) ** 2
    e2 = (p2x - cx) ** 2 + (p2y - cy) ** 2 + (p2z - cz) ** 2
    R2 = R * R
    if e1 < R2 and e2 < R2:
        return edgelen
    b = 2.0 * (dx * (p1x - cx) + dy * (p1y - cy) + dz * (p1z - cz))
    c = e1 - R2
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return 0.0
    s = np.sqrt(disc)
    t1 = (-b - s) / (2.0 * a)
    t2 = (-b + s) / (2.0 * a)
    if t1 < 0.0:
        t1 = 0.0
    if t1 > 1.0:
        t1 = 1.0
    if t2 < 0.0:
        t2 = 0.0
    if t2 > 1.0:
        t2 = 1.0
    if t2 <= t1:
        return 0.0
    return (t2 - t1) * edgelen


@njit(parallel=True, fastmath=True, cache=True)
def _accumulate_window_sums_3d(windows: np.ndarray,
                               R_grid: np.ndarray,
                               edges_v1: np.ndarray,
                               deltas: np.ndarray,
                               weights: np.ndarray,
                               L: float) -> np.ndarray:
    nW = windows.shape[0]
    nR = R_grid.shape[0]
    nE = edges_v1.shape[0]
    Rmax = 0.0
    for j in range(nR):
        if R_grid[j] > Rmax:
            Rmax = R_grid[j]
    out = np.zeros((nW, nR), dtype=np.float64)
    for iw in prange(nW):
        wx = windows[iw, 0]
        wy = windows[iw, 1]
        wz = windows[iw, 2]
        for ie in range(nE):
            dx0 = edges_v1[ie, 0] - wx
            dy0 = edges_v1[ie, 1] - wy
            dz0 = edges_v1[ie, 2] - wz
            if dx0 > 0.5 * L:
                dx0 -= L
            elif dx0 < -0.5 * L:
                dx0 += L
            if dy0 > 0.5 * L:
                dy0 -= L
            elif dy0 < -0.5 * L:
                dy0 += L
            if dz0 > 0.5 * L:
                dz0 -= L
            elif dz0 < -0.5 * L:
                dz0 += L
            v1lx = wx + dx0
            v1ly = wy + dy0
            v1lz = wz + dz0
            v2lx = v1lx + deltas[ie, 0]
            v2ly = v1ly + deltas[ie, 1]
            v2lz = v1lz + deltas[ie, 2]
            edgelen = np.sqrt(deltas[ie, 0] ** 2 + deltas[ie, 1] ** 2
                              + deltas[ie, 2] ** 2)
            mid_dx = 0.5 * (dx0 + (v2lx - wx))
            mid_dy = 0.5 * (dy0 + (v2ly - wy))
            mid_dz = 0.5 * (dz0 + (v2lz - wz))
            mid_dist = np.sqrt(mid_dx * mid_dx + mid_dy * mid_dy
                               + mid_dz * mid_dz)
            if mid_dist > Rmax + edgelen:
                continue
            for jR in range(nR):
                R = R_grid[jR]
                contrib = _ball_segment_clip(wx, wy, wz, R,
                                              v1lx, v1ly, v1lz,
                                              v2lx, v2ly, v2lz)
                if contrib > 0.0:
                    out[iw, jR] += weights[ie] * contrib
    return out


def compute_sigma2_3d(net: Network3D,
                      weights: np.ndarray,
                      R_grid: Optional[np.ndarray] = None,
                      *,
                      n_windows: int = 10000,
                      seed: Optional[int] = None) -> Sigma2Result3D:
    """Direct sigma^2(R) for 3D networks. See ``compute_sigma2`` for shape."""
    if weights.shape[0] != net.n_edges:
        raise ValueError("weights length must equal n_edges")
    L = float(net.L)
    if R_grid is None:
        R_grid = np.linspace(L / 200.0, L / 4.0, 50)
    R_grid = np.ascontiguousarray(R_grid, dtype=np.float64)
    if R_grid.max() > 0.5 * L + 1e-12:
        raise ValueError(f"R_grid.max()={R_grid.max()} exceeds L/2={0.5 * L}; "
                         "PBC min-image clip is only valid for R <= L/2")

    rng = np.random.default_rng(seed)
    windows = rng.uniform(0.0, L, size=(int(n_windows), 3))

    deltas = (net.edges_v2 - net.edges_v1).astype(np.float64, copy=False)
    edges_v1 = np.ascontiguousarray(net.edges_v1, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)

    window_sums = _accumulate_window_sums_3d(
        np.ascontiguousarray(windows, dtype=np.float64),
        R_grid, edges_v1, deltas, weights, L)

    sigma2 = window_sums.var(axis=0)
    return Sigma2Result3D(R=R_grid, sigma2=sigma2, n_windows=int(n_windows))
