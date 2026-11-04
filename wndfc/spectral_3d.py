"""Weighted-network spectral density chi(k) in 3D under periodic boundaries.

Same definitions as the 2D module with V = L^3 and the reciprocal lattice
k = (2 pi / L) n, n in Z^3 \\ {0}:

    chi(k)      = < |A(k)|^2 >_shell / V,
    A(k)        = sum_e w_e sinc(k . dr_e / 2) exp(i k . m_e),
    chi_self(k) = < sum_e w_e^2 sinc^2(k . dr_e / 2) >_shell / V.

The mode count grows as (kmax L / 2 pi)^3; ``chunk_size`` bounds peak memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math

import numpy as np
from numba import njit, prange

from .network_3d import Network3D
from .spectral import _bin, _mode_intensities, _bin_accumulate, _bin_finalize


@dataclass
class ChiEll3D:
    """chi(k) on radial shells: ``k``, ``chi_ell`` (total), ``chi_self``
    (binned per-mode self term), ``counts`` (modes per shell)."""

    k: np.ndarray
    chi_ell: np.ndarray
    chi_self: np.ndarray
    counts: np.ndarray

    @property
    def n_bins(self) -> int:
        return int(self.k.shape[0])

    @property
    def chi_distinct(self) -> np.ndarray:
        return self.chi_ell - self.chi_self


@njit(parallel=True, cache=True)
def _mode_intensities_3d_numba_kernel(midpoints, half, weights, kvecs):
    n_k = kvecs.shape[0]
    n_e = midpoints.shape[0]
    total = np.empty(n_k)
    self_ = np.empty(n_k)
    for ik in prange(n_k):
        kx = kvecs[ik, 0]
        ky = kvecs[ik, 1]
        kz = kvecs[ik, 2]
        re = 0.0
        im = 0.0
        s2 = 0.0
        for e in range(n_e):
            kdotr = half[e, 0] * kx + half[e, 1] * ky + half[e, 2] * kz
            if abs(kdotr) < 1e-10:
                sinc = 1.0
            else:
                sinc = math.sin(kdotr) / kdotr
            a = weights[e] * sinc
            ph = midpoints[e, 0] * kx + midpoints[e, 1] * ky + midpoints[e, 2] * kz
            re += a * math.cos(ph)
            im += a * math.sin(ph)
            s2 += a * a
        total[ik] = re * re + im * im
        self_[ik] = s2
    return total, self_


def _k_tiles_3d(L: float, kmax: float, rows_per_tile: int):
    """(kvecs, kmags) of the cubic reciprocal lattice in bands of the first index."""
    dk = 2.0 * np.pi / L
    n_max = int(np.ceil(kmax / dk)) + 1
    n_axis = np.arange(-n_max, n_max + 1)
    n2g, n3g = np.meshgrid(n_axis, n_axis, indexing="ij")
    n2g, n3g = n2g.ravel(), n3g.ravel()
    for s in range(0, n_axis.size, max(1, int(rows_per_tile))):
        band = n_axis[s:s + rows_per_tile]
        n1 = np.repeat(band, n2g.size)
        n2 = np.tile(n2g, band.size)
        n3 = np.tile(n3g, band.size)
        keep = (n1 != 0) | (n2 != 0) | (n3 != 0)
        kv = np.stack([n1[keep], n2[keep], n3[keep]], axis=1).astype(float) * dk
        km = np.linalg.norm(kv, axis=1)
        cut = km <= kmax
        if cut.any():
            yield kv[cut], km[cut]


def _reciprocal_lattice_3d(L: float, kmax: float) -> Tuple[np.ndarray, np.ndarray]:
    dk = 2.0 * np.pi / L
    n_max = int(np.ceil(kmax / dk)) + 1
    n_axis = np.arange(-n_max, n_max + 1)
    n1, n2, n3 = np.meshgrid(n_axis, n_axis, n_axis, indexing="ij")
    n1, n2, n3 = n1.ravel(), n2.ravel(), n3.ravel()
    keep = (n1 != 0) | (n2 != 0) | (n3 != 0)
    kvecs = np.stack([n1[keep], n2[keep], n3[keep]], axis=1).astype(float) * dk
    kmag = np.linalg.norm(kvecs, axis=1)
    cut = kmag <= kmax
    return kvecs[cut], kmag[cut]


def compute_chi_ell_3d(net: Network3D, weights: np.ndarray, *,
                       kmax: float = 20.0, bin_spacing: Optional[float] = None,
                       chunk_size: Optional[int] = None, shell_mode: str = "width",
                       backend: str = "numpy", rows_per_tile: int = 8) -> ChiEll3D:
    """chi(k) and chi_self(k) on radial shells, bare 1/V gauge (see module doc).
    ``backend``: ``numpy`` (reference), ``numba`` (parallel modes, tiled k-lattice),
    or ``cupy`` (optional CUDA GPU, same tiles).  ``chunk_size``: modes per batch --
    numpy default 512; cupy default and ceiling set by free device memory."""
    if backend not in ("numpy", "numba", "cupy"):
        raise ValueError(f"backend={backend!r}; expected 'numpy', 'numba' or 'cupy'")
    if shell_mode not in ("width", "exact"):
        raise ValueError(f"shell_mode={shell_mode!r}; expected 'width' or 'exact'")
    weights = np.asarray(weights, dtype=float)
    if weights.shape[0] != net.n_edges:
        raise ValueError(f"weights length {weights.shape[0]} != n_edges {net.n_edges}")
    dk = bin_spacing if bin_spacing is not None else (2.0 * np.pi / net.L)
    empty = ChiEll3D(k=np.zeros(0), chi_ell=np.zeros(0), chi_self=np.zeros(0),
                     counts=np.zeros(0, dtype=np.int64))
    if net.n_edges == 0:
        return empty
    if backend == "cupy":
        from .spectral_gpu import _compute_chi_cupy
        k, chi, chi_self, counts = _compute_chi_cupy(
            net, weights, _k_tiles_3d(net.L, kmax, rows_per_tile),
            dk, shell_mode, chunk_size)
        return ChiEll3D(k=k, chi_ell=chi, chi_self=chi_self, counts=counts)
    if backend == "numba":
        half = 0.5 * (net.edges_v2 - net.edges_v1)
        acc: dict = {}
        for kv, km in _k_tiles_3d(net.L, kmax, rows_per_tile):
            total, self_ = _mode_intensities_3d_numba_kernel(
                np.ascontiguousarray(net.edges_midpoints, dtype=np.float64),
                np.ascontiguousarray(half, dtype=np.float64),
                np.ascontiguousarray(weights, dtype=np.float64),
                np.ascontiguousarray(kv, dtype=np.float64))
            _bin_accumulate(acc, km, total, self_, dk, shell_mode)
        k_means, chi, chi_self, counts = _bin_finalize(acc, net.L ** 3)
        return ChiEll3D(k=k_means, chi_ell=chi, chi_self=chi_self, counts=counts)
    kvecs, kmags = _reciprocal_lattice_3d(net.L, kmax)
    if kvecs.shape[0] == 0:
        return empty
    half = 0.5 * (net.edges_v2 - net.edges_v1)
    total, self_ = _mode_intensities(net.edges_midpoints, half, weights, kvecs,
                                     max(1, int(chunk_size or 512)))
    V = net.L ** 3
    k_means, chi, counts = _bin(kmags, total / V, dk, shell_mode)
    _, chi_self, _ = _bin(kmags, self_ / V, dk, shell_mode)
    return ChiEll3D(k=k_means, chi_ell=chi, chi_self=chi_self, counts=counts)
