"""Spectral -> direct bridge: sigma^2(R) from chi(k), bare 1/V gauge.

    sigma^2(R) = v_1(R) / (2 pi)^d  *  integral chi(k) alpha2~(k; R) d^d k,

    alpha2~(k; R) = |w_R~(k)|^2 / v_1(R) = 2^d pi^(d/2) Gamma(1 + d/2) J_{d/2}(kR)^2 / k^d

(the scaled window intensity; equals v_1(R) at k = 0, not 1).  With chi the
bare spectral density the formula carries no further prefactor.  Applied to
``chi.chi_self`` / ``chi.chi_distinct`` it gives the self / distinct parts of
sigma^2 termwise.  The radial integral uses the shell-mean k grid of ``chi``
and the trapezoid rule; kmax truncation is the only approximation.
"""
from __future__ import annotations

from typing import Union

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import j1

from .spectral import ChiEll2D
from .spectral_3d import ChiEll3D


def _alpha2_disk_2d(k: np.ndarray, R: float) -> np.ndarray:
    out = np.empty_like(k, dtype=float)
    v1 = np.pi * R * R
    small = np.abs(k) < 1e-12
    out[small] = v1
    ks = k[~small]
    out[~small] = 4.0 * np.pi * (j1(ks * R) ** 2) / (ks ** 2)
    return out


def _alpha2_sphere_3d(k: np.ndarray, R: float) -> np.ndarray:
    out = np.empty_like(k, dtype=float)
    v3 = (4.0 / 3.0) * np.pi * R ** 3
    small = np.abs(k) < 1e-12
    out[small] = v3
    ks = k[~small]
    kR = ks * R
    # hat w_R(k) = 4 pi (sin kR - kR cos kR) / k^3 = v3 * 3 j1(kR)/(kR), so
    # |hat w_R|^2 / v3 = 12 pi R^3 (sin kR - kR cos kR)^2 / (kR)^6 -> v3 at k->0
    # and int |hat w_R|^2 d^3k / (2 pi)^3 = v3 (Parseval; pinned in tests).
    num = np.sin(kR) - kR * np.cos(kR)
    out[~small] = 12.0 * np.pi * R ** 3 * (num ** 2) / (kR ** 6)
    return out


def parseval_sigma2(chi: Union[ChiEll2D, ChiEll3D], R_grid: np.ndarray, dim: int,
                    values: np.ndarray = None) -> np.ndarray:
    """sigma^2(R) on ``R_grid`` from ``chi`` (bare gauge).  ``values`` overrides
    the integrand (e.g. ``chi.chi_self`` or ``chi.chi_distinct``); default total."""
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    k = np.asarray(chi.k, dtype=float)
    vals = np.asarray(chi.chi_ell if values is None else values, dtype=float)
    R_grid = np.asarray(R_grid, dtype=float)
    sigma2 = np.empty(R_grid.shape, dtype=float)
    if dim == 2:
        for i, R in enumerate(R_grid):
            v_d = np.pi * R * R
            sigma2[i] = v_d / (2.0 * np.pi) * trapezoid(vals * _alpha2_disk_2d(k, R) * k, k)
    else:
        for i, R in enumerate(R_grid):
            v_d = (4.0 / 3.0) * np.pi * R ** 3
            sigma2[i] = v_d / (2.0 * np.pi ** 2) * trapezoid(vals * _alpha2_sphere_3d(k, R) * k ** 2, k)
    return sigma2
