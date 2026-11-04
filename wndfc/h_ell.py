"""Real-space edge pair correlations for 2D networks (bare 1/V gauge): one
sampled pair estimator for the self, distinct and total channels.

For an ordered pair of edges (i, j) -- i = j included -- with per-length
marks w = w_eff / ell, the weighted pair density at separation r is

    P_ij(r) dr = w_i w_j int_0^ell_i ds int_0^ell_j dt [ |x_i(s) - x_j(t)| in [r, r + dr) ].

The estimator draws ``n_samples`` independent uniform arclength pairs (s, t)
per edge pair and deposits w_eff,i w_eff,j / n_samples into the bin of each
distance; pairs are tagged self (i = j) or distinct (i != j), the total is the
untagged sum.  The distinct channel counts both orderings (i, j) and (j, i).

Stored per run (PairDeposits; everything that needs the network):
  r_edges          uniform bin edges linspace(0, r_max, n_bins + 1)
  dep_self/dist    the deposit histograms
  sq_self/dist     sums of squared deposits (counting variance per bin)
  mom_self/dist    bin-free radial moment sums  sum_{r < r_max} deposit r^n, n = 0..3
  self_overflow    self deposits at r >= r_max (edges longer than r_max)
  var_corr_*       sum over pairs of c^2 n_in^2 / n_samples (c = deposit per draw,
                   n_in = draws inside r_max): the binomial correction that turns
                   the squared-deposit sum into the sampling variance of the mass
Read-time arithmetic (real_space_curves / real_space_diagnostics):
  g_chan(r_b) = dep_chan,b / (rho_w^2 V |shell_b|),   |shell_b| = pi (e_{b+1}^2 - e_b^2)
  self      f_w = g_self         (expectation lambda_ell <w^2 (ell - r)_+> / (pi r rho_w^2))
  distinct  h_w = g_dist - 1
  total     h_tot = g_self + g_dist - 1
  M_n = rho_w^2 2 pi int_0^r_max h_w r^(n+1) dr = mom_dist[n] / V - rho_w^2 2 pi r_max^(n+2) / (n+2)
with rho_w = sum w_eff / V (empirical) and lambda_ell = N_ell / V.

Exact identities (torus): sum of all self deposits = sum w_eff^2 = V lambda_ell
<w^2 ell^2>; self + distinct over all r = (sum w_eff)^2 = rho_w^2 V^2.  Hence
the sum rule rho_w^2 int h_w d^2r = -lambda_ell <w^2 ell^2> and the total mass
rho_w^2 int (f_w + h_w) d^2r = 0 over the whole torus.  Within a disk of
radius r_max the bin-free sum M_0 measures chi_distinct(0+) = -chi_self(0) +
chi_total(0+): it closes on the sum rule for a hyperuniform measure (the
tiling schemes, chi_total(0+) = 0) and misses by chi_total(0+)/chi_self(0) for
a non-tiling one (theta = 1 on Poisson), with a realization scatter set by the
lowest modes.  The sampling variance of M_0 is (sum sq_dist - var_corr_dist) / V^2
(each pair's inside count is binomial); the per-bin sd of the curves is the
Poisson form sqrt(sq_b) / (rho_w^2 V |shell_b|), exact where a pair's hit
probability per bin is small.

Checks (never stored, never the product): the exact per-bin expectation of the
self deposits, the contact plateau h_w(0+) = vertex_wedge_sum / (pi V rho_w^2)
- 1, the Hankel inversion of the distinct spectrum chi - chi_self, and the
Gauss-Legendre pair quadrature of the distinct deposits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import j0

from .network import Network2D
from .pbc import (Cell, min_image_delta, nominal_side, periodic_pairs, safe_radius,
                  wrap_to_box)

_DEFAULT_N_BINS = 200
_DEFAULT_N_SAMPLES = 256
_DEFAULT_R_MAX_FRAC = 0.25          # r_max = L/4 of the nominal side (PBC-safe, as for sigma^2)
_N_MOMENTS = 4                      # mom_*[n], n = 0..3  ->  M_0..M_3
_VALID_TAPERS = ("none", "hann", "gaussian")


@dataclass
class PairDeposits:
    r_edges: np.ndarray
    dep_self: np.ndarray
    dep_dist: np.ndarray
    sq_self: np.ndarray
    sq_dist: np.ndarray
    mom_self: np.ndarray
    mom_dist: np.ndarray
    self_overflow: float
    var_corr_self: float
    var_corr_dist: float
    n_samples: int
    seed: int
    n_pairs_dist: int               # unordered distinct pairs sampled (midpoints within cutoff)
    cutoff: float
    cutoff_capped: bool             # cutoff hit the PBC-safe radius: deposits near r_max incomplete

    @property
    def n_bins(self) -> int:
        return int(self.r_edges.shape[0] - 1)

    @property
    def r_max(self) -> float:
        return float(self.r_edges[-1])

    @property
    def r(self) -> np.ndarray:
        """Bin centres."""
        return 0.5 * (self.r_edges[:-1] + self.r_edges[1:])

    @property
    def shell(self) -> np.ndarray:
        """Exact shell areas pi (e_{b+1}^2 - e_b^2)."""
        return np.pi * (self.r_edges[1:] ** 2 - self.r_edges[:-1] ** 2)

    def meta(self) -> Dict[str, Any]:
        """The scalar provenance stored beside the arrays (io.save_result)."""
        return {"n_bins": self.n_bins, "r_max": self.r_max, "n_samples": int(self.n_samples),
                "seed": int(self.seed), "n_pairs_dist": int(self.n_pairs_dist),
                "cutoff": float(self.cutoff), "cutoff_capped": bool(self.cutoff_capped),
                "self_overflow": float(self.self_overflow), "n_moments": int(self.mom_self.shape[0]),
                "var_corr_self": float(self.var_corr_self), "var_corr_dist": float(self.var_corr_dist)}


def _is_general(L) -> bool:
    """True for a sheared (non-square) periodic cell."""
    return isinstance(L, Cell) and not L.is_square


def default_r_max(net: Network2D) -> float:
    L = net.cell_like
    r_max = _DEFAULT_R_MAX_FRAC * nominal_side(L)
    if _is_general(L):
        r_max = min(r_max, safe_radius(L))        # sheared cell: never past the inradius
    return float(r_max)


def _pair_geometry(net: Network2D, r_max: float):
    """Distinct pairs (midpoints within r_max + ell_max, capped at the PBC-safe
    radius) and the per-edge unit vectors; pairs sorted for a fixed sample order."""
    v1 = np.asarray(net.edges_v1, dtype=np.float64)
    v2 = np.asarray(net.edges_v2, dtype=np.float64)
    ell = np.asarray(net.edges_lengths, dtype=np.float64)
    L = net.cell_like
    delta = min_image_delta(v2 - v1, L)
    hat_d = delta / np.maximum(ell[:, None], 1e-30)
    mid = wrap_to_box(v1 + 0.5 * delta, L)
    want = r_max + (float(ell.max()) if ell.size else 0.0)
    cap = safe_radius(L) - 1e-9
    cutoff = min(want, cap)
    pairs = periodic_pairs(mid, L, cutoff) if ell.size else np.zeros((0, 2), dtype=np.int64)
    if len(pairs):
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    return ell, hat_d, mid, pairs, float(cutoff), bool(want > cap)


def _pair_coefficients(i, j, ell, hat_d, mid, L):
    """|x_i(s) - x_j(t)|^2 = d0^2 + s^2 + t^2 + 2 s a_i - 2 t a_j - 2 s t B for
    arclengths s, t from the first endpoints, in the image of the midpoint pair
    (unambiguous below the PBC-safe radius, so every distance < r_max is the
    minimum-image distance)."""
    di, dj = hat_d[i], hat_d[j]
    D0 = min_image_delta(mid[i] - mid[j], L) - 0.5 * ell[i, None] * di + 0.5 * ell[j, None] * dj
    d0_sq = np.einsum("pk,pk->p", D0, D0)
    a_i = np.einsum("pk,pk->p", D0, di)
    a_j = np.einsum("pk,pk->p", D0, dj)
    B = np.einsum("pk,pk->p", di, dj)
    return d0_sq, a_i, a_j, B


class _Accumulator:
    """Deposit histogram + squared deposits + bin-free moments on uniform bins,
    plus the binomial variance correction of the inside mass."""

    def __init__(self, n_bins: int, dr: float, n_mom: int, n_samples: int):
        self.n_bins, self.inv_dr, self.n_samples = int(n_bins), 1.0 / float(dr), int(n_samples)
        self.dep = np.zeros(n_bins)
        self.sq = np.zeros(n_bins)
        self.mom = np.zeros(n_mom)
        self.overflow = 0.0
        self.var_corr = 0.0

    def add(self, d: np.ndarray, wt: np.ndarray):
        """``d`` distances (P, n), consumed in place, and ``wt`` per-row deposit weights (P,)."""
        idx = np.multiply(d, self.inv_dr, out=np.empty_like(d))
        np.floor(idx, out=idx)
        idx = idx.astype(np.int64)
        inside = idx < self.n_bins
        n_in = inside.sum(axis=1)
        self.var_corr += float(np.sum(wt * wt * n_in * n_in)) / self.n_samples
        W = np.broadcast_to(wt[:, None], d.shape)
        w_in = W[inside]
        i_in = idx[inside]
        self.dep += np.bincount(i_in, weights=w_in, minlength=self.n_bins)
        self.sq += np.bincount(i_in, weights=w_in * w_in, minlength=self.n_bins)
        d_in = d[inside]
        acc = w_in.copy()
        for n in range(self.mom.shape[0]):
            self.mom[n] += float(acc.sum())
            acc *= d_in
        self.overflow += float(W[~inside].sum())


def _pair_distances(rng, i, j, ell, d0_sq, a_i, a_j, B, n_samples: int) -> np.ndarray:
    """|x_i(s) - x_j(t)| for n_samples uniform (s, t) per pair, two temporaries.
    The 2 n_samples uniforms of a pair are one contiguous block of the stream,
    so the draws assigned to a pair do not depend on the batch boundaries."""
    blk = rng.random((i.size, 2, n_samples))
    s = blk[:, 0, :] * ell[i, None]
    t = blk[:, 1, :] * ell[j, None]
    del blk
    d2 = s + 2.0 * a_i[:, None]
    d2 *= s                                    # s^2 + 2 s a_i
    tmp = t - 2.0 * a_j[:, None]
    tmp *= t                                   # t^2 - 2 t a_j
    d2 += tmp
    tmp = np.multiply(s, t, out=tmp)
    tmp *= (-2.0 * B)[:, None]                 # -2 s t B
    d2 += tmp
    d2 += d0_sq[:, None]
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2, out=d2)


def sample_pair_deposits(net: Network2D, w_eff: Optional[np.ndarray] = None, *,
                         r_max: Optional[float] = None, n_bins: int = _DEFAULT_N_BINS,
                         n_samples: int = _DEFAULT_N_SAMPLES, seed: int = 0,
                         sample_batch: int = 1 << 22) -> PairDeposits:
    """Sampled self and distinct pair deposits on linspace(0, r_max, n_bins + 1).

    ``w_eff`` = kernel weights (None = ell, i.e. theta = 1).  Every edge pair
    within reach of r_max gets ``n_samples`` independent uniform (s, t) draws;
    the deposit per draw is w_eff,i w_eff,j / n_samples (times 2 for a
    distinct pair: both orderings).  Deterministic for a given ``seed``
    (each edge / pair owns one contiguous block of the stream, so the result is
    independent of ``sample_batch`` up to floating-point summation order);
    ``sample_batch`` only bounds the (pairs x samples) block held in memory.
    """
    ell = np.asarray(net.edges_lengths, dtype=np.float64)
    w_eff = ell if w_eff is None else np.asarray(w_eff, dtype=np.float64)
    if w_eff.shape != ell.shape:
        raise ValueError("w_eff must be one kernel weight per edge")
    L = net.cell_like
    if r_max is None:
        r_max = default_r_max(net)
    r_max = float(r_max)
    if not r_max > 0:
        raise ValueError("r_max must be positive")
    if r_max > safe_radius(L) + 1e-12:
        raise ValueError(f"r_max={r_max} exceeds the PBC-safe radius={safe_radius(L)}")
    n_bins, n_samples = int(n_bins), int(n_samples)
    if n_bins < 1 or n_samples < 1:
        raise ValueError("n_bins and n_samples must be >= 1")
    r_edges = np.linspace(0.0, r_max, n_bins + 1)
    dr = r_max / n_bins
    rng = np.random.default_rng(int(seed))
    acc_s = _Accumulator(n_bins, dr, _N_MOMENTS, n_samples)
    acc_d = _Accumulator(n_bins, dr, _N_MOMENTS, n_samples)
    rows = max(1, int(sample_batch) // n_samples)

    # self: |s - t| = ell |u - v|
    for b0 in range(0, ell.size, rows):
        le, we = ell[b0:b0 + rows], w_eff[b0:b0 + rows]
        blk = rng.random((le.size, 2, n_samples))
        u = blk[:, 0, :] - blk[:, 1, :]
        del blk
        np.abs(u, out=u)
        u *= le[:, None]
        acc_s.add(u, we * we / n_samples)

    # distinct: independent (s, t) per pair, both orderings
    ell, hat_d, mid, pairs, cutoff, capped = _pair_geometry(net, r_max)
    for b0 in range(0, len(pairs), rows):
        i, j = pairs[b0:b0 + rows, 0], pairs[b0:b0 + rows, 1]
        d0_sq, a_i, a_j, B = _pair_coefficients(i, j, ell, hat_d, mid, L)
        acc_d.add(_pair_distances(rng, i, j, ell, d0_sq, a_i, a_j, B, n_samples),
                  2.0 * w_eff[i] * w_eff[j] / n_samples)

    return PairDeposits(r_edges=r_edges, dep_self=acc_s.dep, dep_dist=acc_d.dep,
                        sq_self=acc_s.sq, sq_dist=acc_d.sq, mom_self=acc_s.mom, mom_dist=acc_d.mom,
                        self_overflow=acc_s.overflow, var_corr_self=acc_s.var_corr,
                        var_corr_dist=acc_d.var_corr, n_samples=n_samples, seed=int(seed),
                        n_pairs_dist=int(len(pairs)), cutoff=cutoff, cutoff_capped=capped)


# ---------------------------------------------------------------------------
# Read-time arithmetic
# ---------------------------------------------------------------------------

@dataclass
class RealSpaceCurves:
    r: np.ndarray
    f_self: np.ndarray              # g_self = f_w
    h_dist: np.ndarray              # g_dist - 1 = h_w
    h_total: np.ndarray             # g_self + g_dist - 1
    sd_self: np.ndarray             # Poisson-form counting sd of f_self, h_dist (per-bin hit
    sd_dist: np.ndarray             # probabilities small; a bound where they are not)


def real_space_curves(dep: PairDeposits, V: float, rho_w: float) -> RealSpaceCurves:
    """g_chan = deposits / (rho_w^2 V |shell|); self, distinct, total channels."""
    norm = rho_w ** 2 * float(V) * dep.shell
    g_s = dep.dep_self / norm
    g_d = dep.dep_dist / norm
    return RealSpaceCurves(r=dep.r, f_self=g_s, h_dist=g_d - 1.0, h_total=g_s + g_d - 1.0,
                           sd_self=np.sqrt(dep.sq_self) / norm, sd_dist=np.sqrt(dep.sq_dist) / norm)


def real_space_moments(dep: PairDeposits, V: float, rho_w: float) -> Dict[str, float]:
    """M_n = rho_w^2 2 pi int_0^r_max h_w r^(n+1) dr, bin-free, n = 0..n_moments-1."""
    out = {}
    for n in range(dep.mom_dist.shape[0]):
        bg = rho_w ** 2 * 2.0 * np.pi * dep.r_max ** (n + 2) / (n + 2)
        out[f"M_{n}"] = float(dep.mom_dist[n] / float(V) - bg)
    return out


def real_space_diagnostics(dep: PairDeposits, V: float, rho_w: float, sum_w2l2: float,
                           vertex_wedge_sum: Optional[float] = None) -> Dict[str, Any]:
    """Moments, the sum rule and the mass identities (all bin-free).

    sum rule:   M_0 = rho_w^2 int h_w d^2r   vs  -lambda_ell <w^2 ell^2> = -sum_w2l2 / V
                (closes for a tiling measure; see the module docstring);
                ``M_0_sd`` = sqrt(sum sq_dist - var_corr_dist) / V, the sampling
                standard deviation of M_0 (binomial inside counts per pair)
    total mass: rho_w^2 int (f_w + h_w) d^2r -> 0, quoted relative to sum_w2l2 / V
    self mass:  all self deposits (inside + overflow) = sum_w2l2 exactly
    h_w0:       the first-bin distinct value; ``h_w0_plateau`` the analytic
                contact plateau when vertex_wedge_sum is given.
    """
    V = float(V)
    out: Dict[str, Any] = real_space_moments(dep, V, rho_w)
    rhs = -float(sum_w2l2) / V
    lhs = out["M_0"]
    bg0 = rho_w ** 2 * np.pi * dep.r_max ** 2
    total = (dep.mom_self[0] + dep.mom_dist[0]) / V - bg0
    scale = abs(rhs) if abs(rhs) > 1e-300 else float("inf")
    c = real_space_curves(dep, V, rho_w)
    out.update({
        "sum_rule_lhs": float(lhs), "sum_rule_rhs": float(rhs),
        "sum_rule_relerr": float(abs(lhs - rhs) / scale),
        "M_0_sd": float(np.sqrt(max(dep.sq_dist.sum() - dep.var_corr_dist, 0.0)) / V),
        "total_mass": float(total), "total_mass_rel": float(abs(total) / scale),
        "self_mass_rel": float((dep.mom_self[0] + dep.self_overflow) / float(sum_w2l2) - 1.0)
        if sum_w2l2 else float("nan"),
        "h_w0": float(c.h_dist[0]), "h_w0_sd": float(c.sd_dist[0]),
        "h_w0_plateau": (None if vertex_wedge_sum is None
                         else float(vertex_wedge_sum) / (np.pi * V * rho_w ** 2) - 1.0),
    })
    return out


def deposits_from_npz(data: Dict[str, Any]) -> Optional[PairDeposits]:
    """Rebuild PairDeposits from ``io.load_result`` output (None when absent)."""
    if "h_ell_self" not in data:
        return None
    m = data["metadata"]["h_ell"]
    return PairDeposits(r_edges=np.asarray(data["h_ell_r_edges"], dtype=np.float64),
                        dep_self=np.asarray(data["h_ell_self"], dtype=np.float64),
                        dep_dist=np.asarray(data["h_ell_dist"], dtype=np.float64),
                        sq_self=np.asarray(data["h_ell_self_sq"], dtype=np.float64),
                        sq_dist=np.asarray(data["h_ell_dist_sq"], dtype=np.float64),
                        mom_self=np.asarray(data["h_ell_self_mom"], dtype=np.float64),
                        mom_dist=np.asarray(data["h_ell_dist_mom"], dtype=np.float64),
                        self_overflow=float(m["self_overflow"]), var_corr_self=float(m["var_corr_self"]),
                        var_corr_dist=float(m["var_corr_dist"]), n_samples=int(m["n_samples"]),
                        seed=int(m["seed"]), n_pairs_dist=int(m["n_pairs_dist"]),
                        cutoff=float(m["cutoff"]), cutoff_capped=bool(m["cutoff_capped"]))


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def self_expected_deposits(r_edges: np.ndarray, ell: np.ndarray, w_eff: np.ndarray) -> np.ndarray:
    """Exact expectation of the self deposits per bin:
    sum_e w_eff,e^2 [ (1 - a/ell)_+^2 - (1 - b/ell)_+^2 ]  for bin [a, b)
    (the same-edge pair density is 2 (ell - r)_+ per unit arclength squared)."""
    e = np.asarray(r_edges, dtype=np.float64)
    ell = np.asarray(ell, dtype=np.float64)
    w2 = np.asarray(w_eff, dtype=np.float64) ** 2
    F = np.clip(1.0 - e[None, :] / ell[:, None], 0.0, None) ** 2      # (N, n_bins + 1)
    return (w2[:, None] * (F[:, :-1] - F[:, 1:])).sum(axis=0)


def contact_plateau(net: Network2D, w: np.ndarray) -> float:
    """h_w(0+) = vertex_wedge_sum / (pi V rho_w^2) - 1 (per-length marks w).

    The only distinct pairs at separation r -> 0 sit on edges sharing a vertex;
    for interior angle gamma the wedge contributes (pi - gamma)/(2 pi sin gamma)
    per ordered pair, so the plateau needs the vertex table alone.
    """
    from .small_r import vertex_angle_sum
    w = np.asarray(w, dtype=np.float64)
    rho_w = float(np.sum(w * net.edges_lengths)) / net.V
    wedge = float(vertex_angle_sum(net, w)[0])
    return wedge / (np.pi * net.V * rho_w ** 2) - 1.0


def hankel_inv_2d(k: np.ndarray, f_k: np.ndarray, r_values: np.ndarray) -> np.ndarray:
    """h(r) = (1/2 pi) int_0^inf F(k) J_0(k r) k dk  (trapezoid on the k grid)."""
    k = np.asarray(k, dtype=np.float64)
    f_k = np.asarray(f_k, dtype=np.float64)
    r = np.asarray(r_values, dtype=np.float64)
    integrand = f_k[None, :] * j0(r[:, None] * k[None, :]) * k[None, :]
    return trapezoid(integrand, k, axis=1) / (2.0 * np.pi)


def _taper(k: np.ndarray, taper: str) -> np.ndarray:
    k_max = float(k.max())
    if taper == "hann":
        return np.clip(0.5 * (1.0 - np.cos(np.pi * (k_max - k) / k_max)), 0.0, 1.0)
    if taper == "gaussian":
        return np.exp(-(k / (0.5 * k_max)) ** 2 / 2.0)
    return np.ones_like(k)


def spectral_h_w(k: np.ndarray, chi: np.ndarray, chi_self: np.ndarray, rho_w: float,
                 r: np.ndarray, taper: str = "hann") -> np.ndarray:
    """Spectral check: h_w(r) = Hankel[(chi - chi_self) taper] / rho_w^2 from the
    stored bare spectrum (chi_distinct = rho_w^2 h_w~); band-limited by k_max."""
    if taper not in _VALID_TAPERS:
        raise ValueError(f"taper={taper!r} not in {_VALID_TAPERS}")
    k = np.asarray(k, dtype=np.float64)
    chi_cross = np.asarray(chi, dtype=np.float64) - np.asarray(chi_self, dtype=np.float64)
    return hankel_inv_2d(k, chi_cross * _taper(k, taper), np.asarray(r, dtype=np.float64)) / rho_w ** 2


def gl_pair_deposits(net: Network2D, w_eff: Optional[np.ndarray] = None, *,
                     r_edges: np.ndarray, n_gl: int = 24, pair_batch: int = 20000) -> np.ndarray:
    """Gauss-Legendre check of the distinct deposits: per-pair tensor
    quadrature of the two-segment distance density on the same bins (ordered
    pairs).  Aliases the delta shell for a single edge pair, so it is a
    reference for the distinct channel only, and only where many pairs
    contribute (not the first bins)."""
    e = np.asarray(r_edges, dtype=np.float64)
    ell = np.asarray(net.edges_lengths, dtype=np.float64)
    w_eff = ell if w_eff is None else np.asarray(w_eff, dtype=np.float64)
    n_bins = e.shape[0] - 1
    counts = np.zeros(n_bins)
    ell, hat_d, mid, pairs, _, _ = _pair_geometry(net, float(e[-1]))
    if len(pairs) == 0:
        return counts
    x, gw = np.polynomial.legendre.leggauss(int(n_gl))
    half = 0.5 * (x + 1.0)
    L = net.cell_like
    for b0 in range(0, len(pairs), int(pair_batch)):
        i, j = pairs[b0:b0 + int(pair_batch), 0], pairs[b0:b0 + int(pair_batch), 1]
        d0_sq, a_i, a_j, B = _pair_coefficients(i, j, ell, hat_d, mid, L)
        s = ell[i, None] * half[None, :]
        t = ell[j, None] * half[None, :]
        d2 = (d0_sq[:, None, None] + s[:, :, None] ** 2 + t[:, None, :] ** 2
              + 2.0 * s[:, :, None] * a_i[:, None, None]
              - 2.0 * t[:, None, :] * a_j[:, None, None]
              - 2.0 * s[:, :, None] * t[:, None, :] * B[:, None, None])
        d = np.sqrt(np.maximum(d2, 0.0))
        jac = 2.0 * w_eff[i] * w_eff[j] / 4.0                  # ds dt Jacobian (ell_i ell_j / 4) x w_i w_j, both orderings
        wgt = (gw[None, :, None] * gw[None, None, :]) * jac[:, None, None]
        h, _ = np.histogram(d.ravel(), bins=e, weights=wgt.ravel())
        counts += h
    return counts
