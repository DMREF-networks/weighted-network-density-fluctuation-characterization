"""Small-R closed-form predictions for sigma^2_ell(R) in 3D.

The 3D tube model (cross-validated here against
``close_pair_distinct_sigma2_3d`` and direct windows): the chord reduction
is dimension-blind and the d=3 overlap kernel is the POLYNOMIAL
ball-ball intersection volume

    v1 alpha2(r; R) = (pi/12) (2R - r)^2 (4R + r),   r <= 2R

(no bands, no arccos), so everything terminates:

  - ``G3(eta)``: 4 pi eta - 16 pi/15 (eta >= 1);
        (16 pi/3) eta^2 - (8 pi/3) eta^3 + (4 pi/15) eta^5 (eta < 1).
        Seam 44 pi/15, C^3 (branch difference O((1-eta)^4)).
  - ``unified_prediction_3d``: sigma^2_s = n_ell R^5 <w^2 G3(ell/2R)>,
        exact at all R. Long-branch (R <= min ell/2):
        2 pi n_ell <w^2 ell> R^4 - (16 pi/15) n_ell <w^2> R^5; short-branch
        (R >= max ell/2) the expansion TERMINATES (polynomial kernel).
  - ``close_pair_distinct_sigma2_3d``: GL quadrature of the polynomial
        lens over close pairs (F_3 has a C^1 cutoff like F_2 -- plain
        quadrature converges fast).
  - ``c5_exact`` / ``c5_r5_prediction``: the exact 3D termination

        sigma^2(R) = 2 pi n_ell <w^2 ell> R^4 + C5 R^5
                     - (16 pi^2/9) rho_w^2 R^6,        R <= R0,
        C5 = (16 pi/15) [ (1/V) sum_v sum_{a<b} w_a w_b
                          (pi - gamma)/sin(gamma) - n_ell <w^2> ].

    Structural d=2 vs d=3: the mean-field baseline is demoted to R^6
    (in 2D it competes at the correction order), both surviving R^5
    pieces share 16 pi/15, and <deg> ~ 15.5 gives ~115 vertex pairs per
    vertex against one -1 per edge -- C5 is large and POSITIVE, so in 3D
    the corrections to the bare 2 pi rho_ell R^4 line are not optional
    (60-160% over R/<ell> in [0.05, 0.25]).

Mirrors small_r.py (2D); shares ``_vertex_angle_sum`` and the general-d
``H_infty(B, dim=3)`` wedge formula from there.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .network_3d import Network3D
from .pbc import min_image_delta
from .small_r import _vertex_angle_sum


def _G3_unified(eta: np.ndarray) -> np.ndarray:
    """3D single-edge second-moment kernel G3(eta) = V <o^2> / R^5."""
    eta = np.asarray(eta, dtype=np.float64)
    G = np.empty_like(eta)
    m = eta >= 1.0
    G[m] = 4.0 * np.pi * eta[m] - 16.0 * np.pi / 15.0
    e = eta[~m]
    G[~m] = (16.0 * np.pi / 3.0 * e ** 2
             - 8.0 * np.pi / 3.0 * e ** 3
             + 4.0 * np.pi / 15.0 * e ** 5)
    return G


def unified_prediction_3d(net: Network3D, R_grid: np.ndarray,
                          weights: Optional[np.ndarray] = None
                          ) -> np.ndarray:
    """Unified all-R self term, 3D: sigma^2_s = n_ell R^5 <w^2 G3(ell/2R)>.

    ``weights`` are PER-LENGTH w_e (None = uniform). Spans the compound-
    Poisson regime (2 pi rho_ell R^4) through the volume law
    ((4 pi/3) n_ell <w^2 ell^2> R^3 once all edges are short) exactly.
    """
    R = np.atleast_1d(np.asarray(R_grid, dtype=np.float64))
    ell = np.asarray(net.edges_lengths, dtype=np.float64)
    n_ell = net.n_edges / float(net.L) ** 3
    out = np.empty_like(R)
    if ell.size == 0:
        out.fill(0.0)
        return out
    w2 = None
    if weights is not None:
        w2 = np.asarray(weights, dtype=np.float64) ** 2
        if w2.shape[0] != ell.shape[0]:
            raise ValueError("weights length must equal n_edges")
    for i, Ri in enumerate(R):
        if Ri <= 0:
            out[i] = 0.0
            continue
        G = _G3_unified(ell / (2.0 * Ri))
        if w2 is not None:
            G = w2 * G
        out[i] = n_ell * Ri ** 5 * float(np.mean(G))
    return out


def weighted_smallR_leading_3d(net: Network3D, weights: np.ndarray,
                               R_grid: np.ndarray) -> np.ndarray:
    """3D compound-Poisson leading order 2 pi n_ell <w^2 ell> R^4."""
    w = np.asarray(weights, dtype=float)
    R = np.atleast_1d(np.asarray(R_grid, dtype=float))
    n_ell = net.n_edges / float(net.L) ** 3
    m = float(np.mean(w ** 2 * net.edges_lengths)) if net.n_edges else 0.0
    return 2.0 * np.pi * n_ell * m * R ** 4


def _A_lens_3d(r: np.ndarray, R: float) -> np.ndarray:
    """Ball-ball intersection volume (pi/12)(2R - r)^2 (4R + r), r <= 2R."""
    r = np.asarray(r, dtype=float)
    out = np.zeros_like(r)
    m = r < 2.0 * R
    rm = r[m]
    out[m] = (np.pi / 12.0) * (2.0 * R - rm) ** 2 * (4.0 * R + rm)
    return out


def _seg_seg_dist_3d(a0: np.ndarray, d0: np.ndarray,
                     a1: np.ndarray, d1: np.ndarray) -> float:
    """Exact minimum distance between segments a0+s d0, a1+t d1 (s,t in [0,1]).

    Clamped closest-point algorithm (3D segments are generically skew, so
    the interior-interior case matters -- endpoint sampling is not enough,
    unlike planar non-crossing segments).
    """
    r = a0 - a1
    a = float(d0 @ d0)
    e = float(d1 @ d1)
    f = float(d1 @ r)
    if a <= 1e-300 and e <= 1e-300:
        return float(np.linalg.norm(r))
    if a <= 1e-300:
        s = 0.0
        t = min(max(f / e, 0.0), 1.0)
    else:
        c = float(d0 @ r)
        if e <= 1e-300:
            t = 0.0
            s = min(max(-c / a, 0.0), 1.0)
        else:
            b = float(d0 @ d1)
            den = a * e - b * b
            s = min(max((b * f - c * e) / den, 0.0), 1.0) if den > 1e-300 \
                else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = min(max(-c / a, 0.0), 1.0)
            elif t > 1.0:
                t = 1.0
                s = min(max((b - c) / a, 0.0), 1.0)
    p = a0 + s * d0
    q = a1 + t * d1
    return float(np.linalg.norm(p - q))


def _min_nonadjacent_distance_3d(net: Network3D, *, d_cut=None) -> float:
    """Min segment-segment distance over NON-vertex-adjacent 3D edge pairs.

    KDTree candidate pruning on midpoints (a pair with segment distance d
    has midpoint distance <= d + (ell_i + ell_j)/2); d_cut doubles until a
    nonadjacent pair is found. Exact skew-segment distances (see
    ``_seg_seg_dist_3d``).
    """
    from scipy.spatial import cKDTree
    L = float(net.L)
    n_e = int(net.n_edges)
    if n_e < 2:
        return 0.5 * L
    ell = np.asarray(net.edges_lengths, dtype=float)
    mids = net.edges_midpoints % L
    mids = np.clip(mids, 0.0, np.nextafter(L, 0.0))
    tree = cKDTree(mids, boxsize=L)
    left = np.asarray(net.edges_left_cell, dtype=np.int64)
    right = np.asarray(net.edges_right_cell, dtype=np.int64)
    ell_max = float(ell.max())
    d_cut = float(d_cut) if d_cut is not None else float(ell.mean())
    v1 = net.edges_v1
    dvec = net.edges_v2 - net.edges_v1
    while True:
        pairs = tree.query_pairs(min(d_cut + ell_max, 0.499 * L),
                                 output_type="ndarray")
        best = np.inf
        for ei, ej in pairs:
            ei = int(ei); ej = int(ej)
            if (left[ei] == left[ej] or left[ei] == right[ej]
                    or right[ei] == left[ej] or right[ei] == right[ej]):
                continue
            shift = min_image_delta(v1[ej] - v1[ei], L) - (v1[ej] - v1[ei])
            best = min(best, _seg_seg_dist_3d(v1[ei], dvec[ei],
                                              v1[ej] + shift, dvec[ej]))
        if best < d_cut or d_cut + ell_max >= 0.499 * L:
            return best if np.isfinite(best) else 0.499 * L
        d_cut *= 2.0


def close_pair_distinct_sigma2_3d(net: Network3D, R_grid: np.ndarray, *,
                                  weights: Optional[np.ndarray] = None,
                                  n_gauss: int = 24,
                                  diag_out: Optional[Dict] = None
                                  ) -> np.ndarray:
    """Pointwise close-pair distinct term, 3D (polynomial lens kernel):

    sigma^2_d(R) = (1/V) sum_{i != j, close} w_i w_j I_ij(R)
                   - rho_w^2 v_1(R)^2
                   + (v_1(R)^2 / V^2) sum_j w_j^2 ell_j^2,

    I_ij(R) = int int A_lens3(|u_i - u_j|, R) ds_i ds_j by tensor GL
    quadrature; v_1 = (4 pi/3) R^3. The last term is the finite-box
    diagonal remnant (O(1/V)). ``weights`` PER-LENGTH; None = uniform.
    Add ``unified_prediction_3d`` for the full sigma^2 prediction (the
    exact comparator at every R).
    """
    from scipy.spatial import cKDTree
    L = float(net.L)
    V = L ** 3
    R = np.atleast_1d(np.asarray(R_grid, dtype=float))
    R_max = float(R.max())
    n_e = int(net.n_edges)
    if n_e == 0:
        return np.zeros_like(R)
    w = (np.ones(n_e) if weights is None
         else np.asarray(weights, dtype=float))
    ell = np.asarray(net.edges_lengths, dtype=float)
    rho_w = float((w * ell).sum()) / V

    xi, om = np.polynomial.legendre.leggauss(int(n_gauss))
    tt = 0.5 * (xi + 1.0)
    om = 0.5 * om

    mids = np.clip(net.edges_midpoints % L, 0.0, np.nextafter(L, 0.0))
    tree = cKDTree(mids, boxsize=L)
    reach = float(ell.max()) + 2.0 * R_max
    pairs = tree.query_pairs(min(reach, 0.499 * L), output_type="ndarray")

    v1e = net.edges_v1
    dvec = net.edges_v2 - net.edges_v1
    acc = np.zeros_like(R)
    n_used = 0
    ww = np.outer(om, om).ravel()
    for ei, ej in pairs:
        ei = int(ei); ej = int(ej)
        shift = min_image_delta(v1e[ej] - v1e[ei], L) - (v1e[ej] - v1e[ei])
        p_i = v1e[ei] + tt[:, None] * dvec[ei]
        p_j = (v1e[ej] + shift) + tt[:, None] * dvec[ej]
        d = np.linalg.norm(p_i[:, None, :] - p_j[None, :, :], axis=2)
        if d.min() >= 2.0 * R_max:
            continue
        n_used += 1
        wfac = 2.0 * float(w[ei] * w[ej]) * float(ell[ei] * ell[ej])
        dflat = d.ravel()
        for k, Rk in enumerate(R):
            acc[k] += wfac * float(ww @ _A_lens_3d(dflat, float(Rk)))
    v1R = (4.0 * np.pi / 3.0) * R ** 3
    sigma2_d = (acc / V
                - rho_w ** 2 * v1R ** 2
                + (v1R ** 2 / V ** 2) * float((w ** 2 * ell ** 2).sum()))
    if diag_out is not None:
        diag_out["close_pair_distinct_3d"] = {
            "n_candidate_pairs": int(len(pairs)),
            "n_used_pairs": int(n_used),
            "n_gauss": int(n_gauss),
            "rho_w": rho_w,
        }
    return sigma2_d


def c5_exact(net: Network3D, *, weights: Optional[np.ndarray] = None,
             gamma_min: float = 1e-6) -> Dict:
    """Exact 3D small-R termination coefficients.

    For R <= R0,

        sigma^2(R) = 2 pi n_ell <w^2 ell> R^4 + C5 R^5
                     - (16 pi^2/9) rho_w^2 R^6   EXACTLY,

        C5 = (16 pi/15) [ (1/V) sum_v sum_{a<b at v} w_a w_b
                          (pi - gamma_ab)/sin(gamma_ab) - n_ell <w^2> ],

    with the H_inf^(3) wedge moment 8 pi/15 (shared by both R^5 pieces)
    and the mean-field baseline demoted to its own exact R^6 term
    (``baseline_R6_coeff``; in 2D it competes at the correction order).
    ``weights`` PER-LENGTH (None = uniform: C5 = (16 pi/15)(1/V)
    [angle sum - N_ell]). Also returns ``leading_R4_coeff``, the
    finite-box R^6 remnant (+(16 pi^2/9) n_ell <w^2 ell^2>/V), R0 (its
    three candidate terms; nonadjacent via exact skew-segment distances),
    and degenerate-pair counts. R0 is worst-case-order-statistics gated;
    the practical accuracy window of the limit law is much larger
    (a few % out to R ~ 0.15 <ell> at N=300).
    """
    V = float(net.L) ** 3
    n_ell = net.n_edges / V
    ell = np.asarray(net.edges_lengths, dtype=float)
    if weights is None:
        w = np.ones_like(ell)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != net.n_edges:
            raise ValueError("weights length must equal n_edges")
    if net.n_edges:
        rho_w = float((w * ell).sum()) / V
        w2_mean = float(np.mean(w ** 2))
        w2l_mean = float(np.mean(w ** 2 * ell))
        w2l2_mean = float(np.mean(w ** 2 * ell ** 2))
    else:
        rho_w = w2_mean = w2l_mean = w2l2_mean = 0.0
    pref = 16.0 * np.pi / 15.0
    out: Dict = {
        "leading_R4_coeff": 2.0 * np.pi * n_ell * w2l_mean,
        "self_residual": -pref * n_ell * w2_mean,
        "baseline_R6_coeff": -(16.0 * np.pi ** 2 / 9.0) * rho_w ** 2,
        "finite_box_remnant_R6": (16.0 * np.pi ** 2 / 9.0)
                                 * n_ell * w2l2_mean / V,
        "weighted": weights is not None,
    }
    if net.n_edges == 0:
        out.update(angle_sum=0.0, C5=0.0, R0=0.0, n_pairs=0,
                   n_collinear=0, n_parallel_clamped=0)
        return out

    angle_acc, n_pairs, n_coll, n_clamp, r0_adj = \
        _vertex_angle_sum(net, w, gamma_min=gamma_min)
    out["angle_sum"] = pref * angle_acc / V
    out["C5"] = out["angle_sum"] + out["self_residual"]
    out["n_pairs"] = int(n_pairs)
    out["n_collinear"] = int(n_coll)
    out["n_parallel_clamped"] = int(n_clamp)

    out["R0_self"] = float(ell.min()) / 2.0
    out["R0_adjacent"] = float(r0_adj)
    out["R0_nonadjacent"] = _min_nonadjacent_distance_3d(net) / 2.0
    out["R0"] = float(min(out["R0_self"], out["R0_adjacent"],
                          out["R0_nonadjacent"]))
    return out


def c5_r5_prediction(net: Network3D, R_grid: np.ndarray,
                     c5: Optional[Dict] = None,
                     weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Exact 3D termination: lead R^4 + C5 R^5 + baseline R^6, R <= R0.

    Pass a precomputed ``c5_exact(net, weights=...)`` dict to skip
    recomputation (its weighting must match ``weights``).
    """
    if c5 is None:
        c5 = c5_exact(net, weights=weights)
    R = np.atleast_1d(np.asarray(R_grid, dtype=float))
    return (c5["leading_R4_coeff"] * R ** 4
            + c5["C5"] * R ** 5
            + c5["baseline_R6_coeff"] * R ** 6)
