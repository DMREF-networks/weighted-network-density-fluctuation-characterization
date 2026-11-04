"""Weighted-network spectral density chi(k) in 2D under periodic boundaries.

Gauge (bare 1/V):

    chi(k)      = < |A(k)|^2 >_shell / V,
    A(k)        = sum_e w_e sinc(k . dr_e / 2) exp(i k . m_e),
    chi_self(k) = < sum_e w_e^2 sinc^2(k . dr_e / 2) >_shell / V,
    chi_distinct = chi - chi_self            (exact per mode; binning is linear)

with ``w_e`` the KERNEL weight (w_eff = w * ell: theta family ell^theta, cell-
measure schemes the cell measure), ``dr_e`` the edge vector, ``m_e`` the edge
midpoint, and V = L^2.  The kernel adds no length factor.  Modes live on the
reciprocal lattice k = (2 pi / L) n, n in Z^2 \\ {0}, |k| <= kmax, and are
averaged over radial shells (fixed-width bins, or exact lattice shells).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .network import Network2D
from .pbc import Cell
import math
from numba import njit, prange


@dataclass
class ChiEll2D:
    """chi(k) on radial shells: ``k`` (shell-mean |k|), ``chi_ell`` (total),
    ``chi_self`` (binned per-mode self term), ``counts`` (modes per shell)."""

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


def _reciprocal_lattice(L, kmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """Reciprocal-lattice vectors with 0 < |k| <= kmax (square box or Cell)."""
    if isinstance(L, Cell):
        if not L.is_square:
            return _reciprocal_lattice_cell(L, kmax)
        L = L.L
    dk = 2.0 * np.pi / L
    n_max = int(np.ceil(kmax / dk)) + 1
    n_axis = np.arange(-n_max, n_max + 1)
    n1, n2 = np.meshgrid(n_axis, n_axis, indexing="ij")
    n1 = n1.ravel()
    n2 = n2.ravel()
    keep = (n1 != 0) | (n2 != 0)
    kvecs = np.stack([n1[keep], n2[keep]], axis=1).astype(float) * dk
    kmag = np.linalg.norm(kvecs, axis=1)
    cut = kmag <= kmax
    return kvecs[cut], kmag[cut]


def _reciprocal_lattice_cell(cell: Cell, kmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """k = n @ B with B_ij = 2 pi (A^-1)^T: k . a_i = 2 pi n_i for the rows a_i of A."""
    B = cell.reciprocal()
    a_len = np.linalg.norm(cell.A, axis=1)
    n_max = int(np.ceil(kmax * a_len.max() / (2.0 * np.pi))) + 1
    n_axis = np.arange(-n_max, n_max + 1)
    n1, n2 = np.meshgrid(n_axis, n_axis, indexing="ij")
    n1 = n1.ravel()
    n2 = n2.ravel()
    keep = (n1 != 0) | (n2 != 0)
    kvecs = np.stack([n1[keep], n2[keep]], axis=1).astype(float) @ B
    kmag = np.linalg.norm(kvecs, axis=1)
    cut = kmag <= kmax
    return kvecs[cut], kmag[cut]


def default_dk(L) -> float:
    """Default radial shell width: 2 pi / L (square) or 2 pi / max|a_i| (general),
    the length of the shortest reciprocal vector -- so the first lattice shell
    never straddles the floor(k / dk) bin boundary."""
    if isinstance(L, Cell) and not L.is_square:
        return 2.0 * np.pi / float(np.linalg.norm(L.A, axis=1).max())
    side = L.L if isinstance(L, Cell) else L
    return 2.0 * np.pi / side


def _shell_bin(kmags: np.ndarray, values: np.ndarray, bin_spacing: float):
    """Fixed-width radial bins of width ``bin_spacing``: shell means of |k| and value."""
    if kmags.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int64)
    keys = np.floor(kmags / bin_spacing).astype(np.int64)
    uniq, inverse = np.unique(keys, return_inverse=True)
    counts = np.bincount(inverse, minlength=uniq.size).astype(np.int64)
    safe = np.maximum(counts, 1)
    k_means = np.bincount(inverse, weights=kmags, minlength=uniq.size) / safe
    v_means = np.bincount(inverse, weights=values, minlength=uniq.size) / safe
    order = np.argsort(k_means)
    return k_means[order], v_means[order], counts[order]


def _exact_shell_bin(kmags: np.ndarray, values: np.ndarray, dk: float):
    """One bin per exact lattice shell |n|^2 (needed for steep spectra: width
    binning merges the (1,0) and (1,1) shells, and on chi ~ k^alpha the outer
    shell outweighs the inner by 2^(alpha/2), biasing the fitted exponent low)."""
    if kmags.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int64)
    n2 = np.rint((kmags / dk) ** 2).astype(np.int64)
    uniq, inverse = np.unique(n2, return_inverse=True)
    counts = np.bincount(inverse, minlength=uniq.size).astype(np.int64)
    k_means = np.bincount(inverse, weights=kmags, minlength=uniq.size) / counts
    v_means = np.bincount(inverse, weights=values, minlength=uniq.size) / counts
    order = np.argsort(k_means)
    return k_means[order], v_means[order], counts[order]


def _bin(kmags, values, dk, shell_mode):
    if shell_mode == "exact":
        return _exact_shell_bin(kmags, values, dk)
    return _shell_bin(kmags, values, dk)


def chi_ell_modes(net: Network2D, weights: np.ndarray, kvecs: np.ndarray, *,
                  chunk_size: int = 1024) -> np.ndarray:
    """Complex amplitude A(k) at explicit k-vectors (n_k, 2), unnormalized."""
    weights = np.asarray(weights, dtype=float)
    if weights.shape[0] != net.n_edges:
        raise ValueError(f"weights length {weights.shape[0]} != n_edges {net.n_edges}")
    kvecs = np.ascontiguousarray(np.asarray(kvecs, dtype=float))
    amps = np.zeros(kvecs.shape[0], dtype=complex)
    if kvecs.shape[0] == 0 or net.n_edges == 0:
        return amps
    midpoints = net.edges_midpoints
    half = 0.5 * (net.edges_v2 - net.edges_v1)
    chunk = max(1, int(chunk_size))
    for s in range(0, kvecs.shape[0], chunk):
        kc = kvecs[s:s + chunk]
        phase = midpoints @ kc.T
        kdotr = half @ kc.T
        small = np.abs(kdotr) < 1e-10
        sinc = np.where(small, 1.0, np.sin(kdotr) / np.where(small, 1.0, kdotr))
        amps[s:s + chunk] = ((weights[:, None] * sinc) * np.exp(1j * phase)).sum(axis=0)
    return amps


def _mode_intensities(midpoints, half, weights, kvecs, chunk):
    """Per-mode |A(k)|^2 and sum_e w_e^2 sinc^2 in one pass."""
    n_k = kvecs.shape[0]
    total = np.empty(n_k)
    self_ = np.empty(n_k)
    w2 = weights ** 2
    for s in range(0, n_k, chunk):
        kc = kvecs[s:s + chunk]
        phase = midpoints @ kc.T
        kdotr = half @ kc.T
        small = np.abs(kdotr) < 1e-10
        sinc = np.where(small, 1.0, np.sin(kdotr) / np.where(small, 1.0, kdotr))
        amp = weights[:, None] * sinc
        sums = (amp * np.exp(1j * phase)).sum(axis=0)
        total[s:s + chunk] = sums.real ** 2 + sums.imag ** 2
        self_[s:s + chunk] = (w2[:, None] * sinc ** 2).sum(axis=0)
    return total, self_


def _mode_intensities_numba(midpoints, half, weights, kvecs):
    """Per-mode |A(k)|^2 and sum_e w_e^2 sinc^2 with a parallel loop over modes
    (numba); the same arithmetic as _mode_intensities, summed in a different
    order, so results agree to ~1e-13 relative, not bit-for-bit."""
    return _mode_intensities_numba_kernel(np.ascontiguousarray(midpoints, dtype=np.float64),
                                         np.ascontiguousarray(half, dtype=np.float64),
                                         np.ascontiguousarray(weights, dtype=np.float64),
                                         np.ascontiguousarray(kvecs, dtype=np.float64))


@njit(parallel=True, cache=True)
def _mode_intensities_numba_kernel(midpoints, half, weights, kvecs):
    n_k = kvecs.shape[0]
    n_e = midpoints.shape[0]
    total = np.empty(n_k)
    self_ = np.empty(n_k)
    for ik in prange(n_k):
        kx = kvecs[ik, 0]
        ky = kvecs[ik, 1]
        re = 0.0
        im = 0.0
        s2 = 0.0
        for e in range(n_e):
            kdotr = half[e, 0] * kx + half[e, 1] * ky
            if abs(kdotr) < 1e-10:
                sinc = 1.0
            else:
                sinc = math.sin(kdotr) / kdotr
            a = weights[e] * sinc
            ph = midpoints[e, 0] * kx + midpoints[e, 1] * ky
            re += a * math.cos(ph)
            im += a * math.sin(ph)
            s2 += a * a
        total[ik] = re * re + im * im
        self_[ik] = s2
    return total, self_


def _k_tiles(L, kmax: float, rows_per_tile: int):
    """Yield (kvecs, kmags) of the reciprocal lattice in bands of the first
    index, so the full k-vector array is never materialized (kmax L / 2 pi
    can reach 1e4 per axis)."""
    if isinstance(L, Cell) and not L.is_square:
        B = L.reciprocal()
        n_max = int(np.ceil(kmax * np.linalg.norm(L.A, axis=1).max() / (2.0 * np.pi))) + 1
    else:
        side = L.L if isinstance(L, Cell) else L
        dk = 2.0 * np.pi / side
        n_max = int(np.ceil(kmax / dk)) + 1
        B = np.array([[dk, 0.0], [0.0, dk]])
    n_axis = np.arange(-n_max, n_max + 1)
    for s in range(0, n_axis.size, max(1, int(rows_per_tile))):
        n1 = np.repeat(n_axis[s:s + rows_per_tile], n_axis.size)
        n2 = np.tile(n_axis, n1.size // n_axis.size)
        keep = (n1 != 0) | (n2 != 0)
        kv = np.stack([n1[keep], n2[keep]], axis=1).astype(float) @ B
        km = np.linalg.norm(kv, axis=1)
        cut = km <= kmax
        if cut.any():
            yield kv[cut], km[cut]


def _bin_accumulate(acc: dict, kmags, total, self_, dk, shell_mode):
    keys = (np.rint((kmags / dk) ** 2).astype(np.int64) if shell_mode == "exact"
            else np.floor(kmags / dk).astype(np.int64))
    uniq, inverse = np.unique(keys, return_inverse=True)
    cnt = np.bincount(inverse, minlength=uniq.size)
    sk = np.bincount(inverse, weights=kmags, minlength=uniq.size)
    st = np.bincount(inverse, weights=total, minlength=uniq.size)
    ss = np.bincount(inverse, weights=self_, minlength=uniq.size)
    for key, c, a, b, d in zip(uniq.tolist(), cnt, sk, st, ss):
        row = acc.setdefault(key, [0, 0.0, 0.0, 0.0])
        row[0] += int(c)
        row[1] += float(a)
        row[2] += float(b)
        row[3] += float(d)


def _bin_finalize(acc: dict, V: float):
    if not acc:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int64)
    rows = np.array([acc[k] for k in sorted(acc)], dtype=float)
    counts = rows[:, 0].astype(np.int64)
    k_means = rows[:, 1] / counts
    chi = rows[:, 2] / counts / V
    chi_self = rows[:, 3] / counts / V
    order = np.argsort(k_means)
    return k_means[order], chi[order], chi_self[order], counts[order]


def compute_chi_ell(net: Network2D, weights: np.ndarray, *,
                    kmax: float = 200.0, bin_spacing: Optional[float] = None,
                    chunk_size: Optional[int] = None, shell_mode: str = "width",
                    backend: str = "numpy", rows_per_tile: int = 64) -> ChiEll2D:
    """chi(k) and chi_self(k) on radial shells, bare 1/V gauge (see module doc).

    ``backend``: ``numpy`` (chunked vectorized modes; the reference path),
    ``numba`` (parallel loop over modes with the k-lattice generated in tiles
    of ``rows_per_tile`` first-index rows -- for large kmax and large N; the
    same arithmetic in a different summation order, agreement ~1e-13), or
    ``cupy`` (the same tiles on a CUDA GPU; CuPy is imported only when selected).
    ``chunk_size``: modes per batch -- numpy default 1024; cupy default and
    ceiling set by free device memory.
    """
    if shell_mode not in ("width", "exact"):
        raise ValueError(f"shell_mode={shell_mode!r}; expected 'width' or 'exact'")
    if backend not in ("numpy", "numba", "cupy"):
        raise ValueError(f"backend={backend!r}; expected 'numpy', 'numba' or 'cupy'")
    weights = np.asarray(weights, dtype=float)
    if weights.shape[0] != net.n_edges:
        raise ValueError(f"weights length {weights.shape[0]} != n_edges {net.n_edges}")
    dk = bin_spacing if bin_spacing is not None else default_dk(net.cell_like)
    empty = ChiEll2D(k=np.zeros(0), chi_ell=np.zeros(0), chi_self=np.zeros(0),
                     counts=np.zeros(0, dtype=np.int64))
    if net.n_edges == 0:
        return empty
    if backend == "cupy":
        from .spectral_gpu import _compute_chi_cupy
        k, chi, chi_self, counts = _compute_chi_cupy(
            net, weights, _k_tiles(net.cell_like, kmax, rows_per_tile),
            dk, shell_mode, chunk_size)
        return ChiEll2D(k=k, chi_ell=chi, chi_self=chi_self, counts=counts)
    if backend == "numba":
        half = 0.5 * (net.edges_v2 - net.edges_v1)
        w = np.asarray(weights, dtype=float)
        acc: dict = {}
        for kv, km in _k_tiles(net.cell_like, kmax, rows_per_tile):
            total, self_ = _mode_intensities_numba(net.edges_midpoints, half, w, kv)
            _bin_accumulate(acc, km, total, self_, dk, shell_mode)
        k_means, chi, chi_self, counts = _bin_finalize(acc, net.V)
        return ChiEll2D(k=k_means, chi_ell=chi, chi_self=chi_self, counts=counts)
    kvecs, kmags = _reciprocal_lattice(net.cell_like, kmax)
    if kvecs.shape[0] == 0:
        return empty
    half = 0.5 * (net.edges_v2 - net.edges_v1)
    total, self_ = _mode_intensities(net.edges_midpoints, half, weights, kvecs,
                                     max(1, int(chunk_size or 1024)))
    V = net.V
    k_means, chi, counts = _bin(kmags, total / V, dk, shell_mode)
    _, chi_self, _ = _bin(kmags, self_ / V, dk, shell_mode)
    return ChiEll2D(k=k_means, chi_ell=chi, chi_self=chi_self, counts=counts)
