"""Direct edge-length variance under 2D PBC.

For each random window center w in [0, L)^2 and each radius R, sum the
weighted clipped edge length inside the disk B(w, R), then take the
variance over windows. The geometry is handled per-edge via min-image of
``v1 - w`` so we don't materialize tiled coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numba import njit, prange

from .network import Network2D
from .pbc import Cell, sample_uniform


@dataclass
class Sigma2Result:
    R: np.ndarray
    sigma2: np.ndarray
    n_windows: int


@njit(fastmath=True, cache=True)
def _disk_segment_clip(cx: float, cy: float, R: float,
                       p1x: float, p1y: float,
                       p2x: float, p2y: float) -> float:
    """Length of segment [p1, p2] inside disk B((cx, cy), R), no PBC.

    Solves |p1 + t*(p2-p1) - c|^2 = R^2 for t in [0, 1] and returns
    edgelen * (t2 - t1). Returns 0 if segment is entirely outside.
    """
    dx = p2x - p1x
    dy = p2y - p1y
    a = dx * dx + dy * dy
    if a < 1e-24:
        d2 = (p1x - cx) ** 2 + (p1y - cy) ** 2
        return np.sqrt(a) if d2 < R * R else 0.0
    edgelen = np.sqrt(a)
    e1 = (p1x - cx) ** 2 + (p1y - cy) ** 2
    e2 = (p2x - cx) ** 2 + (p2y - cy) ** 2
    R2 = R * R
    if e1 < R2 and e2 < R2:
        return edgelen
    b = 2.0 * (dx * (p1x - cx) + dy * (p1y - cy))
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
def _accumulate_window_sums_2d(windows: np.ndarray,
                               R_grid: np.ndarray,
                               edges_v1: np.ndarray,
                               deltas: np.ndarray,
                               weights: np.ndarray,
                               L: float) -> np.ndarray:
    """For each (window, R), accumulate sum_e w_e * clip(edge_e in B(w, R)).

    PBC handled by per-edge min-image of (v1 - w): bring v1 to its closest
    image relative to the window, segment is then v1_local + t * delta.
    Valid for R <= L/2.
    """
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
        for ie in range(nE):
            dx0 = edges_v1[ie, 0] - wx
            dy0 = edges_v1[ie, 1] - wy
            if dx0 > 0.5 * L:
                dx0 -= L
            elif dx0 < -0.5 * L:
                dx0 += L
            if dy0 > 0.5 * L:
                dy0 -= L
            elif dy0 < -0.5 * L:
                dy0 += L
            v1lx = wx + dx0
            v1ly = wy + dy0
            v2lx = v1lx + deltas[ie, 0]
            v2ly = v1ly + deltas[ie, 1]
            edgelen = np.sqrt(deltas[ie, 0] ** 2 + deltas[ie, 1] ** 2)
            mid_dx = 0.5 * (dx0 + (v2lx - wx))
            mid_dy = 0.5 * (dy0 + (v2ly - wy))
            mid_dist = np.sqrt(mid_dx * mid_dx + mid_dy * mid_dy)
            if mid_dist > Rmax + edgelen:
                continue
            for jR in range(nR):
                R = R_grid[jR]
                contrib = _disk_segment_clip(wx, wy, R,
                                              v1lx, v1ly, v2lx, v2ly)
                if contrib > 0.0:
                    out[iw, jR] += weights[ie] * contrib
    return out


@njit(parallel=True, fastmath=True, cache=True)
def _accumulate_window_sums_2d_cell(windows: np.ndarray,
                                    R_grid: np.ndarray,
                                    edges_v1: np.ndarray,
                                    deltas: np.ndarray,
                                    weights: np.ndarray,
                                    A: np.ndarray,
                                    Ainv: np.ndarray) -> np.ndarray:
    """General periodic cell (rows of A = lattice vectors): the min-image of
    (v1 - w) is taken in fractional coordinates.  Valid for R <= inradius."""
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
        for ie in range(nE):
            dx = edges_v1[ie, 0] - wx
            dy = edges_v1[ie, 1] - wy
            fx = dx * Ainv[0, 0] + dy * Ainv[1, 0]
            fy = dx * Ainv[0, 1] + dy * Ainv[1, 1]
            fx -= np.round(fx)
            fy -= np.round(fy)
            dx0 = fx * A[0, 0] + fy * A[1, 0]
            dy0 = fx * A[0, 1] + fy * A[1, 1]
            v1lx = wx + dx0
            v1ly = wy + dy0
            v2lx = v1lx + deltas[ie, 0]
            v2ly = v1ly + deltas[ie, 1]
            edgelen = np.sqrt(deltas[ie, 0] ** 2 + deltas[ie, 1] ** 2)
            mid_dx = 0.5 * (dx0 + (v2lx - wx))
            mid_dy = 0.5 * (dy0 + (v2ly - wy))
            mid_dist = np.sqrt(mid_dx * mid_dx + mid_dy * mid_dy)
            if mid_dist > Rmax + edgelen:
                continue
            for jR in range(nR):
                R = R_grid[jR]
                contrib = _disk_segment_clip(wx, wy, R,
                                              v1lx, v1ly, v2lx, v2ly)
                if contrib > 0.0:
                    out[iw, jR] += weights[ie] * contrib
    return out


def compute_sigma2(net: Network2D,
                   weights: np.ndarray,
                   R_grid: Optional[np.ndarray] = None,
                   *,
                   n_windows: int = 10000,
                   seed: Optional[int] = None) -> Sigma2Result:
    """Direct sigma^2(R) by random-window sampling.

    Parameters
    ----------
    net : Network2D
    weights : (n_edges,) per-edge weights w_e (uniform = ones).
    R_grid : (n_R,) radii. Defaults to ``np.linspace(L/200, L/4, 50)``.
    n_windows : number of random window centers (default 10_000).
    seed : RNG seed (None -> non-reproducible).

    Returns
    -------
    Sigma2Result with ``sigma2(R) = Var over windows of sum_e w_e * clip_e``.
    """
    if weights.shape[0] != net.n_edges:
        raise ValueError("weights length must equal n_edges")
    cell = net.cell_like
    general = isinstance(cell, Cell) and not cell.is_square
    if general:
        side = np.sqrt(cell.V)          # nominal side for the default grid
        r_safe = cell.inradius
    else:
        side = cell.L if isinstance(cell, Cell) else float(cell)
        r_safe = 0.5 * side
    L = side
    if R_grid is None:
        R_grid = np.linspace(L / 200.0, L / 4.0, 50)
    R_grid = np.ascontiguousarray(R_grid, dtype=np.float64)
    if R_grid.max() > r_safe + 1e-12:
        raise ValueError(f"R_grid.max()={R_grid.max()} exceeds {r_safe} (L/2, or the cell "
                         "inradius); the PBC min-image clip is only valid below that")

    rng = np.random.default_rng(seed)
    windows = sample_uniform(rng, int(n_windows), cell)

    deltas = (net.edges_v2 - net.edges_v1).astype(np.float64, copy=False)
    edges_v1 = np.ascontiguousarray(net.edges_v1, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)

    if general:
        window_sums = _accumulate_window_sums_2d_cell(
            np.ascontiguousarray(windows, dtype=np.float64),
            R_grid, edges_v1, deltas, weights,
            np.ascontiguousarray(cell.A), np.ascontiguousarray(cell.Ainv))
    else:
        window_sums = _accumulate_window_sums_2d(
            np.ascontiguousarray(windows, dtype=np.float64),
            R_grid, edges_v1, deltas, weights, L)

    sigma2 = window_sums.var(axis=0)
    return Sigma2Result(R=R_grid, sigma2=sigma2, n_windows=int(n_windows))
