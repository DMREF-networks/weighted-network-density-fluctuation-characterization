"""Point pattern generators (Poisson, URL, HIP).

URL = Uniformly Random Local: each lattice site shifted independently by
a uniform draw from [-a/2, a/2)^d. ``a=0`` returns the bare lattice
(square in 2D, cubic in 3D). Lattice spacing is ``L / n_side``; centers
are placed at half-cell offsets so points land in (0, L).

HIP = Hyperplane Intersection Process (2D): the pairwise crossings of M
random lines from a Poisson line process -- an anti-hyperuniform pattern
(S(k) ~ k^{-1} as k -> 0). See Klatt & Last, arXiv:2008.10907.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def poisson_2d(N: int, L: float, *, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, L, size=(int(N), 2))


def poisson_3d(N: int, L: float, *, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, L, size=(int(N), 3))


def url_2d(n_side: int, L: float, *, a: float = 0.0,
           seed: Optional[int] = None) -> np.ndarray:
    s = L / n_side
    centers = (np.arange(n_side) + 0.5) * s
    X, Y = np.meshgrid(centers, centers, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    if a > 0.0:
        rng = np.random.default_rng(seed)
        pts = pts + (rng.random(pts.shape) - 0.5) * a
    return np.mod(pts, L)


def url_3d(n_side: int, L: float, *, a: float = 0.0,
           seed: Optional[int] = None) -> np.ndarray:
    s = L / n_side
    centers = (np.arange(n_side) + 0.5) * s
    X, Y, Z = np.meshgrid(centers, centers, centers, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    if a > 0.0:
        rng = np.random.default_rng(seed)
        pts = pts + (rng.random(pts.shape) - 0.5) * a
    return np.mod(pts, L)


def hip_2d(M: int, L: float, *, seed: Optional[int] = None,
           parallel_tol: float = 1e-10) -> np.ndarray:
    """2D hyperplane-intersection process: the in-box pairwise crossings of
    ``M`` random lines x cos(theta) + y sin(theta) = p, theta ~ U[0, pi),
    p ~ U[-L/sqrt(2), L/sqrt(2)] (every line meeting the box is representable).
    Non-periodic; N is random with E[N] = M(M-1)/(2 pi).  Returns (N, 2)
    positions in [0, L).
    """
    p_half = L / np.sqrt(2.0)
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, np.pi, size=M)
    p = rng.uniform(-p_half, p_half, size=M)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    ii, jj = np.triu_indices(M, k=1)
    det = cos_t[ii] * sin_t[jj] - cos_t[jj] * sin_t[ii]
    valid = np.abs(det) > parallel_tol
    ii, jj, d = ii[valid], jj[valid], det[valid]

    # Intersection in box-centered coords (Cramer's rule), shifted to corner.
    x_c = (p[ii] * sin_t[jj] - p[jj] * sin_t[ii]) / d
    y_c = (p[jj] * cos_t[ii] - p[ii] * cos_t[jj]) / d
    x_full = x_c + L / 2.0
    y_full = y_c + L / 2.0

    keep = (np.abs(x_c) < L / 2.0) & (np.abs(y_c) < L / 2.0)
    x = x_full[keep] % L
    y = y_full[keep] % L
    return np.column_stack([x, y])


def hip_m_for_target_n(n_target: int) -> int:
    """Smallest line count M with E[N] = M(M-1)/(2 pi) >= n_target (rho = E[N]/L^2,
    so n_target = L^2 gives rho ~ 1)."""
    return int(np.ceil((1.0 + np.sqrt(1.0 + 8.0 * np.pi * n_target)) / 2.0))
