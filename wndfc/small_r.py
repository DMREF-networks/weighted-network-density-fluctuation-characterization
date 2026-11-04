"""Small-R closed-form predictions for sigma^2_ell(R) (2D, Convention B).

Three predictions for the edge-length variance at small windows R << <ell>:

  - ``unified_prediction(net, R)``  : single closed form valid all R,
        sigma^2_B(R) = rho_b * R^4 * <G(ell/(2R))>.
        Interpolates Regime I (16/3 rho_line R^3) and Regime II
        (pi rho_b <ell^2> R^2).
  - ``r4_prediction(net, R, vc)``   : R^3 + R^4 expansion with vertex-pair
        correction. Needs ``vc`` = ``vertex_correlations(net)``.
  - ``vertex_correlations(net)``    : gamma2_B/gamma2_A, mean coordination,
        rho_v/rho_b/rho_line, mel scalars. Used as input to r4_prediction.

Pair-covariance corrections (compound-Poisson refinements over multiple
edges) are not implemented here.

3D analogs land with goal (3b).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .network import Network2D
from .pbc import (Cell, CellLike, cell_volume, min_image_delta, nominal_side,
                  periodic_pairs, safe_radius, wrap_to_box)


_ROUND_DECIMALS = 8

# Periodic cell convention: every box-dependent quantity takes ``net.cell_like``
# (the float side for a square network, else the Cell) and goes through the
# pbc helpers, so the square-box arithmetic is the plain float path.


def _is_general_cell(L: CellLike) -> bool:
    return isinstance(L, Cell) and not L.is_square


def _pbc_cap(L: CellLike) -> float:
    """Largest periodic pair-search radius: ``0.499 L`` on a square box, the
    same 0.2% margin below the inradius (``0.998 * safe_radius``) on a general
    cell."""
    if _is_general_cell(L):
        return 0.998 * safe_radius(L)
    return 0.499 * nominal_side(L)


def _wrap_for_tree(p: np.ndarray, L: CellLike) -> np.ndarray:
    """Wrapped coordinates for a periodic pair search (square box: strictly
    inside [0, L) as the KD-tree requires; general cell: the canonical cell,
    ``periodic_pairs`` re-wraps fractionally)."""
    q = wrap_to_box(p, L)
    if _is_general_cell(L):
        return q
    side = nominal_side(L)
    return np.clip(q, 0.0, np.nextafter(side, 0.0))


def _G_unified(alpha: np.ndarray) -> np.ndarray:
    """Single-edge second-moment kernel ``G(alpha) = E[o^2] * A / R^4``.

    Piecewise closed form:

      Case A (alpha >= 1, ell >= 2R):
          G = 32 alpha / 3 - pi
      Case B (alpha <  1, ell <  2R):
          G = -pi + 32 alpha / 3
              - 2 alpha sqrt(1 - alpha^2) (2 alpha^2 + 13) / 3
              + 2 (4 alpha^2 + 1) arccos(alpha)
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    G = np.empty_like(alpha)
    mask_A = alpha >= 1.0
    G[mask_A] = 32.0 * alpha[mask_A] / 3.0 - np.pi
    a = alpha[~mask_A]
    b = np.sqrt(np.maximum(1.0 - a * a, 0.0))
    t = np.arccos(np.clip(a, -1.0, 1.0))
    G[~mask_A] = (
        -np.pi
        + 32.0 * a / 3.0
        - 2.0 * a * b * (2.0 * a * a + 13.0) / 3.0
        + 2.0 * (4.0 * a * a + 1.0) * t
    )
    return G


def g_kernel(alpha, order: int = 0) -> np.ndarray:
    """Single-edge self-overlap kernel ``G(alpha)`` and its alpha-derivatives.

    ``alpha = ell / (2R)``. ``order=0`` returns G itself (delegates to
    ``_G_unified``); ``order in {1, 2, 3, 4}`` returns ``G'``, ``G''``,
    ``G'''``, ``G''''`` from the exact closed forms (``b = sqrt(1 - alpha^2)``,
    ``t = arccos(alpha)``):

        long-edge  (alpha >= 1):  G = 32 alpha / 3 - pi
            G' = 32/3,   G'' = G''' = G'''' = 0
        short-edge (alpha <  1):
            G    = -pi + 32 alpha/3 - (2/3) alpha b (2 alpha^2 + 13)
                   + 2 (4 alpha^2 + 1) t
            G'   = 16 alpha t + 32/3 - (16/3)(alpha^2 + 2) b
            G''  = 16 t - 16 alpha b
            G''' = -32 b
            G''''= 32 alpha / b

    The branch is **C^3** at the seam ``alpha = 1`` (window diameter = edge
    length): G', G'', G''' all match the long-edge branch there (32/3, 0, 0),
    with sub-seam remainder ``(256 sqrt2 / 105)(1 - alpha)^{7/2}``; the first
    break is G'''' (Case B diverges as ``32 alpha / b`` while the long-edge
    branch is 0). Anchors: ``G(0) = 0``, ``G'(0) = 0`` (even small-alpha,
    ``G ~ 4 pi alpha^2`` so ``G''(0) = 8 pi``). The smooth kernel is what makes
    ``unified_prediction`` a kink-free exact interpolant through the Regime
    I -> II -> III crossover band.
    """
    if order == 0:
        return _G_unified(alpha)
    if order not in (1, 2, 3, 4):
        raise ValueError("order must be 0, 1, 2, 3, or 4")
    alpha = np.asarray(alpha, dtype=np.float64)
    out = np.empty_like(alpha)
    mask_A = alpha >= 1.0
    # Long-edge branch G = 32 alpha/3 - pi: slope 32/3, higher derivs vanish.
    out[mask_A] = (32.0 / 3.0) if order == 1 else 0.0
    a = alpha[~mask_A]                       # strictly < 1, so b > 0 below
    b = np.sqrt(np.maximum(1.0 - a * a, 0.0))
    t = np.arccos(np.clip(a, -1.0, 1.0))
    if order == 1:
        out[~mask_A] = (16.0 * a * t + 32.0 / 3.0
                        - (16.0 / 3.0) * (a * a + 2.0) * b)
    elif order == 2:
        out[~mask_A] = 16.0 * t - 16.0 * a * b
    elif order == 3:
        out[~mask_A] = -32.0 * b
    else:                                    # order == 4 (diverges at seam)
        out[~mask_A] = 32.0 * a / b
    return out


def _vertex_adjacency_2d(edges_v1: np.ndarray, edges_v2: np.ndarray,
                          L: CellLike) -> Dict:
    """Edge groups keyed by (wrapped, rounded) endpoint coordinates."""
    adjacency: Dict = {}
    n_edges = edges_v1.shape[0]
    wrapped_v1 = wrap_to_box(edges_v1, L)
    wrapped_v2 = wrap_to_box(edges_v2, L)
    for ei in range(n_edges):
        for endpoint in (wrapped_v1[ei], wrapped_v2[ei]):
            key = (round(float(endpoint[0]), _ROUND_DECIMALS),
                   round(float(endpoint[1]), _ROUND_DECIMALS))
            adjacency.setdefault(key, []).append(ei)
    for key in adjacency:
        adjacency[key] = sorted(set(adjacency[key]))
    return adjacency


def _gamma2_from_adjacency(edge_lengths: np.ndarray,
                            adjacency: Dict) -> Dict:
    """gamma2_B / gamma2_A vertex-adjacent edge-length covariance scalars."""
    if edge_lengths.size == 0:
        return {
            "gamma2_B": 0.0, "gamma2_A": 0.0,
            "mean_Z": 0.0, "n_vertices": 0, "n_pairs": 0,
            "pair_product_mean_B": 0.0, "pair_product_mean_A": 0.0,
            "mel": 0.0, "mel_sq": 0.0,
            "inv_mel": 0.0, "inv_mel_sq": 0.0,
            "degrees": [],
        }
    mel = float(edge_lengths.mean())
    mel_sq = mel * mel
    safe = np.maximum(edge_lengths, 1e-12)
    inv_mel = float(np.mean(1.0 / safe))
    inv_mel_sq = inv_mel * inv_mel

    acc_B = 0.0
    acc_A = 0.0
    n_pairs = 0
    n_vertices = 0
    deg_sum = 0
    degrees = []
    for edge_list in adjacency.values():
        Z_v = len(edge_list)
        if Z_v < 2:
            continue
        n_vertices += 1
        deg_sum += Z_v
        degrees.append(Z_v)
        for j in range(Z_v):
            ej = edge_list[j]
            for k in range(j + 1, Z_v):
                ek = edge_list[k]
                acc_B += float(edge_lengths[ej]) * float(edge_lengths[ek])
                acc_A += 1.0 / (float(safe[ej]) * float(safe[ek]))
                n_pairs += 1

    if n_pairs == 0:
        return {
            "gamma2_B": 0.0, "gamma2_A": 0.0,
            "mean_Z": 0.0, "n_vertices": 0, "n_pairs": 0,
            "pair_product_mean_B": mel_sq,
            "pair_product_mean_A": inv_mel_sq,
            "mel": mel, "mel_sq": mel_sq,
            "inv_mel": inv_mel, "inv_mel_sq": inv_mel_sq,
            "degrees": [],
        }

    pair_B = acc_B / n_pairs
    pair_A = acc_A / n_pairs
    mean_Z = deg_sum / n_vertices if n_vertices > 0 else 0.0
    return {
        "gamma2_B": pair_B - mel_sq,
        "gamma2_A": pair_A - inv_mel_sq,
        "mean_Z": mean_Z,
        "n_vertices": n_vertices,
        "n_pairs": n_pairs,
        "pair_product_mean_B": pair_B,
        "pair_product_mean_A": pair_A,
        "mel": mel, "mel_sq": mel_sq,
        "inv_mel": inv_mel, "inv_mel_sq": inv_mel_sq,
        "degrees": degrees,
    }


def vertex_correlations(net: Network2D) -> Dict:
    """Vertex-adjacent edge-length covariance and structural scalars.

    Returns
    -------
    dict
        Keys: ``gamma2_B``, ``gamma2_A``, ``mean_Z``, ``n_vertices``,
        ``n_pairs``, ``mel``, ``mel_sq``, ``inv_mel``, ``inv_mel_sq``,
        ``rho_v``, ``rho_b``, ``rho_line``, ``degrees``.

    Pass to ``r4_prediction`` to evaluate the R^3 + R^4 expansion.
    """
    edges_v1 = np.ascontiguousarray(net.edges_v1, dtype=np.float64)
    edges_v2 = np.ascontiguousarray(net.edges_v2, dtype=np.float64)
    L = net.cell_like
    adjacency = _vertex_adjacency_2d(edges_v1, edges_v2, L)
    res = _gamma2_from_adjacency(net.edges_lengths, adjacency)
    A = cell_volume(L)
    A = A if A > 0 else 0.0
    res["rho_v"] = res["n_vertices"] / A if A > 0 else 0.0
    res["rho_b"] = net.rho_b
    res["rho_line"] = net.rho_line
    return res


def unified_prediction(net: Network2D, R_grid: np.ndarray,
                       weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Unified all-R compound-Poisson (self-term) prediction sigma^2(R).

    Unweighted (default, ``weights=None``):

        sigma^2_B(R) = rho_b * R^4 * <G(ell/(2R))>

    spanning Regime I (small R, 16/3 rho_line R^3) and the area law
    (pi rho_b <ell^2> R^2). With ``weights`` (PER-LENGTH ``w_e``; pass
    ``w_eff / ell`` if starting from an effective weight) the weighted
    self term, exact at all R:

        sigma^2_s(R) = n_ell * R^4 * <w^2 G(ell/(2R))>
    """
    R = np.atleast_1d(R_grid).astype(np.float64)
    edge_lengths = np.asarray(net.edges_lengths, dtype=np.float64)
    rho_b = net.rho_b
    out = np.empty_like(R)
    if edge_lengths.size == 0:
        out.fill(0.0)
        return out
    if weights is not None:
        w2 = np.asarray(weights, dtype=np.float64) ** 2
        if w2.shape[0] != edge_lengths.shape[0]:
            raise ValueError("weights length must equal n_edges")
    else:
        w2 = None
    for i, Ri in enumerate(R):
        if Ri <= 0:
            out[i] = 0.0
            continue
        alpha = edge_lengths / (2.0 * Ri)
        G = _G_unified(alpha)
        if w2 is not None:
            G = w2 * G
        out[i] = rho_b * Ri ** 4 * float(np.mean(G))
    return out


def r4_prediction(net: Network2D, R_grid: np.ndarray,
                   vertex_cov: Dict) -> np.ndarray:
    """R^3 + R^4 compound-Poisson prediction with vertex-pair correction.

    LEGACY heuristic (kept for back-compat): prefer ``c4_exact`` /
    ``c4_r4_prediction`` below -- the exact quartic law, whose
    coefficient is a pure vertex-ANGLE statistic.

    sigma^2_B(R) = (16/3) rho_line R^3 + C4 R^4
        C4 = -pi rho_b
             + pi rho_v * Z(Z-1)/2 * (gamma2_B + <ell>^2) / (4 <ell>^2)

    Parameters
    ----------
    net : Network2D
    R_grid : (n_R,) window radii.
    vertex_cov : dict from ``vertex_correlations(net)``.
    """
    R = np.atleast_1d(R_grid).astype(np.float64)
    rho_b = vertex_cov["rho_b"]
    rho_line = vertex_cov["rho_line"]
    rho_v = vertex_cov["rho_v"]
    Z = vertex_cov["mean_Z"]
    mel = vertex_cov["mel"]
    gamma2_B = vertex_cov["gamma2_B"]

    sigma2_r3 = (16.0 / 3.0) * rho_line * R ** 3
    C4_single = -np.pi * rho_b
    Z_choose_2 = Z * (Z - 1.0) / 2.0
    if mel > 1e-12:
        C4_vertex = (np.pi * rho_v * Z_choose_2 * 0.25
                     * (gamma2_B + mel * mel) / (mel * mel))
    else:
        C4_vertex = 0.0
    C4 = C4_single + C4_vertex
    return sigma2_r3 + C4 * R ** 4


# ===========================================================================
# Exact small-R law
# ===========================================================================

# Radial moments int_0^2 F_d(q) q dq of the dimensionless lens kernel --
# the d-dependent prefactor of the wedge formula
# H_inf^(d) = M_d (pi - gamma)/sin(gamma).
_H_INF_MOMENT = {2: np.pi / 2.0, 3: 8.0 * np.pi / 15.0}


def H_infty(B, dim: int = 2) -> np.ndarray:
    """Vertex-adjacent pair kernel by the wedge map, general dimension:

        H_inf^(d)(gamma) = M_d (pi - gamma) / sin(gamma),
        M_2 = pi/2  ->  pi (pi - gamma) / (2 sin gamma)
        M_3 = 8 pi / 15

    ``B = cos(gamma)`` with gamma the interior angle between two edges
    parameterized outward from their shared vertex; exact (not asymptotic)
    once R <= (sin gamma / 2) min(ell_a, ell_b). Anchors (d=2):
    H_inf(0) = pi^2/4; B -> -1 (collinear continuation) -> pi/2, finite;
    B -> +1 diverges (overlapping rays, excluded by nondegeneracy).
    """
    if dim not in _H_INF_MOMENT:
        raise ValueError("dim must be 2 or 3")
    M_d = _H_INF_MOMENT[dim]
    B = np.asarray(B, dtype=float)
    gamma = np.arccos(np.clip(B, -1.0, 1.0))
    sin_g = np.sin(gamma)
    out = np.where(sin_g > 1e-300,
                   M_d * (np.pi - gamma) / np.maximum(sin_g, 1e-300),
                   np.inf)
    # Collinear limit gamma -> pi: (pi-gamma)/sin(gamma) -> 1.
    near_pi = (np.pi - gamma) < 1e-8
    out = np.where(near_pi, M_d, out)
    return out if out.ndim else float(out)


def _edge_outward_units(net: Network2D):
    """Per-edge endpoint vertex ids + outward unit vectors at each endpoint.

    Orientation of (edges_v1, edges_v2) against (edges_left_cell,
    edges_right_cell) is established per edge by minimum-image proximity of
    edges_v1 to points[left] -- exact for D/G/C builds and robust to the
    raw-ridge-coordinate ordering of V builds.
    """
    L = net.cell_like
    left = np.asarray(net.edges_left_cell, dtype=np.int64)
    right = np.asarray(net.edges_right_cell, dtype=np.int64)
    u = (net.edges_v2 - net.edges_v1)
    ell = np.maximum(np.asarray(net.edges_lengths, dtype=float), 1e-300)
    u = u / ell[:, None]
    d_left = np.linalg.norm(
        min_image_delta(net.edges_v1 - net.points[left], L), axis=1)
    d_right = np.linalg.norm(
        min_image_delta(net.edges_v1 - net.points[right], L), axis=1)
    v1_is_left = d_left <= d_right
    # Outward unit at the 'left' vertex and at the 'right' vertex.
    out_left = np.where(v1_is_left[:, None], u, -u)
    out_right = -out_left
    return left, right, out_left, out_right


def vertex_angle_sum(net, w: np.ndarray, *, gamma_min: float = 1e-6):
    """Public name of the vertex wedge sum (see _vertex_angle_sum)."""
    return _vertex_angle_sum(net, w, gamma_min=gamma_min)


def _vertex_angle_sum(net, w: np.ndarray, *, gamma_min: float = 1e-6):
    """Weighted vertex-adjacent angle statistic, dimension-blind.

    Returns ``(angle_acc, n_pairs, n_collinear, n_clamped, r0_adj)`` with
    angle_acc = sum_v sum_{a<b at v} w_a w_b (pi - gamma_ab)/sin(gamma_ab)
    and r0_adj = min over adjacent pairs of sin(gamma) min(ell_a, ell_b)/2
    (the adjacent-pair termination threshold). Works for Network2D and
    Network3D alike (outward units and dot products are dimension-blind).
    Collinear pairs (gamma -> pi) take the finite limit 1; near-parallel
    pairs are clamped at gamma_min and counted.
    """
    left, right, out_left, out_right = _edge_outward_units(net)
    n_v = int(net.points.shape[0])
    ell = np.asarray(net.edges_lengths, dtype=float)
    vcat = np.concatenate([left, right])
    ucat = np.concatenate([out_left, out_right], axis=0)
    ecat = np.concatenate([np.arange(net.n_edges, dtype=np.int64)] * 2)
    order = np.argsort(vcat, kind="stable")
    vcat = vcat[order]
    ucat = ucat[order]
    ecat = ecat[order]
    counts = np.bincount(vcat, minlength=n_v)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    angle_acc = 0.0
    n_pairs = 0
    n_collinear = 0
    n_clamped = 0
    r0_adj = np.inf
    for v in range(n_v):
        lo, hi = int(offsets[v]), int(offsets[v + 1])
        deg = hi - lo
        if deg < 2:
            continue
        U = ucat[lo:hi]
        E = ecat[lo:hi]
        cosg = np.clip(U @ U.T, -1.0, 1.0)
        iu, ju = np.triu_indices(deg, k=1)
        g = np.arccos(cosg[iu, ju])
        n_pairs += g.size
        coll = (np.pi - g) < 1e-8
        n_collinear += int(coll.sum())
        par = g < gamma_min
        n_clamped += int(par.sum())
        g = np.where(par, gamma_min, g)
        f = np.where(coll, 1.0, (np.pi - g) / np.maximum(np.sin(g), 1e-300))
        angle_acc += float((w[E[iu]] * w[E[ju]] * f).sum())
        min_ell = np.minimum(ell[E[iu]], ell[E[ju]])
        r0_here = np.sin(g) * min_ell / 2.0
        r0_here = r0_here[r0_here > 0.0]
        if r0_here.size:
            r0_adj = min(r0_adj, float(r0_here.min()))
    return angle_acc, n_pairs, n_collinear, n_clamped, r0_adj


def c4_exact(net: Network2D, *, weights: Optional[np.ndarray] = None,
             gamma_min: float = 1e-6, with_R0: bool = True) -> Dict:
    """Exact quartic coefficient of the small-R law.

    For R <= R0 the variance terminates EXACTLY:

        sigma^2_ell(R) = (16/3) n_ell <w^2 ell> R^3 + C4 R^4,

        C4 = (pi/A) sum_v sum_{a<b at v} w_a w_b (pi - gamma_ab)/sin(gamma_ab)
             - pi^2 rho_w^2 - pi n_ell <w^2>,

    with ``weights`` the PER-LENGTH w_e (None = uniform, reducing to the
    unweighted draft-B form: angle sum - pi^2 rho_ell^2 - pi rho_ell/<ell>).
    The three pieces are the vertex-adjacent H_inf pair sum (weighted by
    the incident edge weights), the mean-field baseline (rho_w =
    n_ell <w ell>), and the per-edge self residual (w_j^2 times the exact
    -pi R^4 long-edge term). R0 is pure geometry -- unchanged by weights.

    Returns a dict with the components, C4, the leading R^3 coefficient
    (``leading_R3_coeff``), the finite-box diagonal remnant coefficient
    (``finite_box_remnant`` = +pi^2 n_ell <w^2 ell^2> / A, the O(1/A) term
    the thermodynamic law drops but the close-pair route keeps), R0 (and
    its three candidate terms), and degenerate-pair counts. Collinear
    pairs (gamma -> pi) take the finite limit 1; near-parallel pairs
    (gamma < gamma_min) are clamped and counted.
    """
    A = cell_volume(net.cell_like)
    n_ell = net.rho_b
    ell = np.asarray(net.edges_lengths, dtype=float)
    if weights is None:
        w = np.ones_like(ell)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != net.n_edges:
            raise ValueError("weights length must equal n_edges")
    if net.n_edges:
        rho_w = float((w * ell).sum()) / A
        w2_mean = float(np.mean(w ** 2))
        w2l_mean = float(np.mean(w ** 2 * ell))
        w2l2_mean = float(np.mean(w ** 2 * ell ** 2))
    else:
        rho_w = w2_mean = w2l_mean = w2l2_mean = 0.0
    out: Dict = {
        "baseline": -np.pi ** 2 * rho_w ** 2,
        "self_residual": -np.pi * n_ell * w2_mean,
        "leading_R3_coeff": (16.0 / 3.0) * n_ell * w2l_mean,
        "finite_box_remnant": np.pi ** 2 * n_ell * w2l2_mean / A,
        "weighted": weights is not None,
    }
    if net.n_edges == 0:
        out.update(angle_sum=0.0, C4=0.0, R0=0.0, n_pairs=0,
                   n_collinear=0, n_parallel_clamped=0)
        return out

    angle_acc, n_pairs, n_collinear, n_clamped, r0_adj = \
        _vertex_angle_sum(net, w, gamma_min=gamma_min)

    out["angle_sum"] = np.pi / A * angle_acc
    out["C4"] = out["angle_sum"] + out["baseline"] + out["self_residual"]
    out["n_pairs"] = int(n_pairs)
    out["n_collinear"] = int(n_collinear)
    out["n_parallel_clamped"] = int(n_clamped)

    out["R0_self"] = float(ell.min()) / 2.0
    out["R0_adjacent"] = float(r0_adj)
    if with_R0:                       # the only O(N) piece -- skip if only C4 wanted
        out["R0_nonadjacent"] = _min_nonadjacent_distance(net) / 2.0
        out["R0"] = float(min(out["R0_self"], out["R0_adjacent"],
                              out["R0_nonadjacent"]))
    else:
        out["R0_nonadjacent"] = out["R0"] = None
    return out


def _segment_point_dist(p, a, d, ell2):
    """Distance of points p (n,2) to segment a + t d, t in [0,1]."""
    ap = p - a
    t = np.clip(np.einsum("ni,ni->n", ap, np.broadcast_to(d, ap.shape))
                / max(ell2, 1e-300), 0.0, 1.0)
    proj = a + t[:, None] * d
    return np.linalg.norm(p - proj, axis=1)


def _min_nonadjacent_distance(net: Network2D, *, d_cut=None) -> float:
    """Min segment-segment distance over NON-vertex-adjacent edge pairs.

    KDTree candidate pruning on midpoints: a pair with segment distance d
    has midpoint distance <= d + (ell_i + ell_j)/2, so candidates within
    d_cut + ell_max cover all pairs with d < d_cut. d_cut doubles until a
    nonadjacent pair is found (or the box scale is hit). Edges in a planar
    embedded graph do not cross, so the minimum is positive.
    """
    L = net.cell_like
    n_e = int(net.n_edges)
    if n_e < 2:
        return safe_radius(L)
    ell = np.asarray(net.edges_lengths, dtype=float)
    mids = _wrap_for_tree(net.edges_midpoints, L)
    r_cap = _pbc_cap(L)
    left = np.asarray(net.edges_left_cell, dtype=np.int64)
    right = np.asarray(net.edges_right_cell, dtype=np.int64)
    ell_max = float(ell.max())
    d_cut = float(d_cut) if d_cut is not None else float(ell.mean())
    v1 = net.edges_v1
    dvec = net.edges_v2 - net.edges_v1
    while True:
        pairs = periodic_pairs(mids, L, min(d_cut + ell_max, r_cap))
        best = np.inf
        for ei, ej in pairs:
            ei = int(ei); ej = int(ej)
            if (left[ei] == left[ej] or left[ei] == right[ej]
                    or right[ei] == left[ej] or right[ei] == right[ej]):
                continue
            # Lift edge j into edge i's frame (min-image on v1 offsets).
            shift = min_image_delta(v1[ej] - v1[ei], L) - (v1[ej] - v1[ei])
            a_j = v1[ej] + shift
            pts_j = np.array([a_j, a_j + dvec[ej],
                              a_j + 0.5 * dvec[ej]])
            d1 = _segment_point_dist(pts_j, v1[ei], dvec[ei],
                                     float(ell[ei]) ** 2).min()
            pts_i = np.array([v1[ei], v1[ei] + dvec[ei],
                              v1[ei] + 0.5 * dvec[ei]])
            d2 = _segment_point_dist(pts_i, a_j, dvec[ej],
                                     float(ell[ej]) ** 2).min()
            best = min(best, float(d1), float(d2))
        if best < d_cut or d_cut + ell_max >= r_cap:
            return best if np.isfinite(best) else r_cap
        d_cut *= 2.0


def c4_r4_prediction(net: Network2D, R_grid: np.ndarray,
                     c4: Optional[Dict] = None,
                     weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Exact small-R termination, optionally weighted (PER-LENGTH w_e):

        sigma^2(R) = (16/3) n_ell <w^2 ell> R^3 + C4 R^4,   R <= c4['R0'].

    Pass a precomputed ``c4_exact(net, weights=...)`` dict to skip
    recomputation (its weighting must match ``weights``; the dict's
    ``leading_R3_coeff`` is used when present). Uniform call unchanged:
    (16/3) rho_ell R^3 + C4 R^4.
    """
    if c4 is None:
        c4 = c4_exact(net, weights=weights)
    R = np.atleast_1d(np.asarray(R_grid, dtype=float))
    lead = c4.get("leading_R3_coeff")
    if lead is None:                       # unweighted c4 dict
        lead = (16.0 / 3.0) * net.rho_line
    return lead * R ** 3 + c4["C4"] * R ** 4


def weighted_smallR_leading(net: Network2D, weights: np.ndarray,
                            R_grid: np.ndarray) -> np.ndarray:
    """Weighted compound-Poisson leading order (16/3) n_ell <w^2 ell> R^3.

    ``weights`` are PER-LENGTH w_e.
    """
    w = np.asarray(weights, dtype=float)
    R = np.atleast_1d(np.asarray(R_grid, dtype=float))
    m = float(np.mean(w ** 2 * net.edges_lengths)) if net.n_edges else 0.0
    return (16.0 / 3.0) * net.rho_b * m * R ** 3


def crossover_Rstar(net: Network2D) -> float:
    """Regime I -> III crossover radius ``R* = (3 pi/16) <ell^2>/<ell>``.

    Where the compound-Poisson plateau ``(16/3) rho_ell R^3`` meets the area
    law ``pi rho_ell <ell^2>/<ell> R^2``. Equivalently
    ``(3 pi/16) <ell> * PDI`` with ``PDI = <ell^2>/<ell>^2 >= 1``, so
    polydispersity pushes R* outward. Weight-independent.
    """
    mel = net.mean_edge_length
    if mel <= 0:
        return 0.0
    return (3.0 * np.pi / 16.0) * net.mean_edge_length_sq / mel


def area_law_self(net: Network2D, R_grid: np.ndarray,
                  weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Self-term Regime III (large-R area law).

    The large-R asymptote of ``unified_prediction`` (small-eta expansion of
    G, eta = ell/2R: ``G = 4 pi eta^2 - (16/3) eta^3 + (4/15) eta^5 + ...``):

        sigma^2_{s,III}(R) = n_ell [ pi <w^2 ell^2> R^2
                                     - (2/3) <w^2 ell^3> R
                                     + (1/120) <w^2 ell^5> / R ],

    with ``n_ell = net.rho_b`` and PER-LENGTH ``weights`` w_e (None = uniform,
    reducing to ``pi rho_ell <ell^2>/<ell> R^2 - (2/3) rho_ell <ell^3>/<ell> R
    + (1/120) rho_ell <ell^5>/<ell> / R``). The even kernel part gives the R^2
    area term alone (no R^0 constant); corrections are odd, hence R^2, R, 1/R.
    SELF term only -- the FULL sigma^2 adds the distinct M0/M1 pieces at the
    R^2 and R orders; use ``area_law_full`` for that.
    """
    R = np.atleast_1d(np.asarray(R_grid, dtype=float))
    ell = np.asarray(net.edges_lengths, dtype=float)
    if ell.size == 0:
        return np.zeros_like(R)
    w2 = (np.ones_like(ell) if weights is None
          else np.asarray(weights, dtype=float) ** 2)
    n_ell = net.rho_b
    m2 = float(np.mean(w2 * ell ** 2))
    m3 = float(np.mean(w2 * ell ** 3))
    m5 = float(np.mean(w2 * ell ** 5))
    return n_ell * (np.pi * m2 * R ** 2 - (2.0 / 3.0) * m3 * R
                    + (1.0 / 120.0) * m5 / R)


def area_law_full(net: Network2D, R_grid: np.ndarray, *,
                  n_bins: int = 200, n_samples: int = 256, seed: int = 0,
                  r_max: Optional[float] = None,
                  diag_out: Optional[Dict] = None) -> np.ndarray:
    """Full Regime III area law (self + distinct M0/M1).

        sigma^2_ell(R) -> pi (n_ell <ell^2> + M0) R^2
                          - ((2/3) n_ell <ell^3> + 2 M1) R + O(1),

    n_ell = net.rho_b; M0, M1 the radial moments of the DISTINCT correlation
    h_ell(r) (M_n = rho_ell^2 2pi int h_ell r^{n+1} dr). The self term alone (M0=M1=0) is
    ``area_law_self``; the distinct M0 (typically < 0) is what makes this
    track the measured/full sigma^2 at large R, where the self-term law
    overshoots (~2x on Poisson). UNIFORM weight only (v1).

    h_ell comes from the sampled pair deposits (``h_ell.sample_pair_deposits``,
    ``n_samples`` uniform (s, t) draws per edge pair, bin-free moments) over
    r in [0, r_max] (r_max defaults to L/4, the PBC-safe range; M0 IS the
    sum-rule LHS rho_ell^2 int h_ell d^2r, so its accuracy is the h_ell
    integral's).
    ``diag_out`` (if given) is filled with M0, M1, A2, A1, and the self-only
    A2_self. Cost ~ n_close_pairs * n_samples for the single pass.
    """
    from .h_ell import real_space_moments, sample_pair_deposits
    R = np.atleast_1d(np.asarray(R_grid, dtype=float))
    ell = np.asarray(net.edges_lengths, dtype=float)
    if ell.size == 0:
        return np.zeros_like(R)
    if r_max is None:
        L = net.cell_like
        # Half the PBC-safe radius: L/4 on a square box, inradius/2 otherwise.
        rmax = (0.5 * safe_radius(L) if _is_general_cell(L)
                else 0.25 * nominal_side(L))
    else:
        rmax = float(r_max)
    dep = sample_pair_deposits(net, None, r_max=rmax, n_bins=int(n_bins),
                               n_samples=int(n_samples), seed=int(seed))
    A = cell_volume(net.cell_like)
    mom = real_space_moments(dep, A, float(ell.sum()) / A)     # uniform: w_eff = ell, rho_w = rho_line
    M0, M1 = float(mom["M_0"]), float(mom["M_1"])
    n_ell = net.rho_b
    l2, l3 = net.edge_moment(2.0), net.edge_moment(3.0)
    A2 = np.pi * (n_ell * l2 + M0)
    A1 = -((2.0 / 3.0) * n_ell * l3 + 2.0 * M1)
    if diag_out is not None:
        diag_out.update(M0=M0, M1=M1, A2=float(A2), A1=float(A1),
                        A2_self=float(np.pi * n_ell * l2))
    return A2 * R ** 2 + A1 * R


def _A_lens(r: np.ndarray, R: float) -> np.ndarray:
    """Lens area v1 alpha_2 = 2 R^2 arccos(r/2R) - (r/2) sqrt(4R^2 - r^2)."""
    r = np.asarray(r, dtype=float)
    out = np.zeros_like(r)
    m = r < 2.0 * R
    rm = r[m]
    out[m] = (2.0 * R * R * np.arccos(np.clip(rm / (2.0 * R), 0.0, 1.0))
              - 0.5 * rm * np.sqrt(np.maximum(4.0 * R * R - rm * rm, 0.0)))
    return out


def close_pair_distinct_sigma2(net: Network2D, R_grid: np.ndarray, *,
                               weights: Optional[np.ndarray] = None,
                               n_gauss: int = 24,
                               diag_out: Optional[Dict] = None) -> np.ndarray:
    """Pointwise close-pair distinct term.

    sigma^2_d(R) = (1/A) sum_{i != j, close} w_i w_j I_ij(R)
                   - pi^2 R^4 rho_w^2
                   + (pi^2 R^4 / A^2) sum_j w_j^2 ell_j^2,

    with I_ij(R) = int int A_lens(|u_i - u_j|, R) ds_i ds_j evaluated by
    tensor Gauss-Legendre quadrature over both arclengths -- no h_ell
    histogram, no moment truncation (the fig:small_r overlay route). The
    last term is the finite-box diagonal remnant (O(1/A)). ``weights``
    are PER-LENGTH w_e; None = uniform. Add ``unified_prediction(net,
    R_grid, weights)`` for the full sigma^2 prediction.

    Cost ~ n_close_pairs * n_gauss^2 * n_R; meant for prediction overlays
    at modest N (~1e3 edges), not production sweeps.
    """
    L = net.cell_like
    A = cell_volume(L)
    R = np.atleast_1d(np.asarray(R_grid, dtype=float))
    R_max = float(R.max())
    n_e = int(net.n_edges)
    if n_e == 0:
        return np.zeros_like(R)
    w = (np.ones(n_e) if weights is None
         else np.asarray(weights, dtype=float))
    ell = np.asarray(net.edges_lengths, dtype=float)
    rho_w = float((w * ell).sum()) / A

    # GL nodes mapped to [0, 1] per edge.
    xi, om = np.polynomial.legendre.leggauss(int(n_gauss))
    tt = 0.5 * (xi + 1.0)
    om = 0.5 * om                                   # weights on [0, 1]

    mids = _wrap_for_tree(net.edges_midpoints, L)
    reach = float(ell.max()) + 2.0 * R_max
    pairs = periodic_pairs(mids, L, min(reach, _pbc_cap(L)))

    v1 = net.edges_v1
    dvec = net.edges_v2 - net.edges_v1
    acc = np.zeros_like(R)
    n_used = 0
    for ei, ej in pairs:
        ei = int(ei); ej = int(ej)
        # Bring edge j into edge i's frame.
        shift = min_image_delta(v1[ej] - v1[ei], L) - (v1[ej] - v1[ei])
        p_i = v1[ei] + tt[:, None] * dvec[ei]       # (g, 2)
        p_j = (v1[ej] + shift) + tt[:, None] * dvec[ej]
        d = np.linalg.norm(p_i[:, None, :] - p_j[None, :, :], axis=2)
        if d.min() >= 2.0 * R_max:
            continue
        n_used += 1
        jac = float(ell[ei] * ell[ej])
        wfac = 2.0 * float(w[ei] * w[ej]) * jac     # x2: unordered pairs
        ww = np.outer(om, om).ravel()
        dflat = d.ravel()
        for k, Rk in enumerate(R):
            acc[k] += wfac * float(ww @ _A_lens(dflat, float(Rk)))
    sigma2_d = (acc / A
                - np.pi ** 2 * R ** 4 * rho_w ** 2
                + (np.pi ** 2 * R ** 4 / A ** 2)
                * float((w ** 2 * ell ** 2).sum()))
    if diag_out is not None:
        diag_out["close_pair_distinct"] = {
            "n_candidate_pairs": int(len(pairs)),
            "n_used_pairs": int(n_used),
            "n_gauss": int(n_gauss),
            "rho_w": rho_w,
        }
    return sigma2_d


def g_d_form_factor(x, dim: int = 2) -> np.ndarray:
    """Orientation-averaged squared rod form factor g_d(x), exact.

    g_d(x) = <sinc^2(x khat.ehat)>_{S^{d-1}}; closed forms

        g_2(x) = [ int_0^{2x} J_0(t) dt - J_1(2x) ] / x
        g_3(x) = [ Si(2x) - sin^2(x)/x ] / x

    with the Porod limits x g_d(x) -> C_d (C_2 = 1, C_3 = pi/2) and
    g_d(0) = 1 built in analytically.
    """
    from scipy.special import j0, j1, sici, struve
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.ones_like(x)
    m = x > 1e-8
    xm = x[m]
    if dim == 2:
        # int_0^z J_0(t) dt = z J_0(z) + (pi z / 2) [J_1(z) H_0(z) - J_0(z) H_1(z)]
        # (Struve H_nu); exact for all z, unlike scipy.special.itj0y0.
        z = 2.0 * xm
        ij0 = z * j0(z) + 0.5 * np.pi * z * (j1(z) * struve(0, z) - j0(z) * struve(1, z))
        out[m] = (ij0 - j1(z)) / xm
    elif dim == 3:
        si = sici(2.0 * xm)[0]
        out[m] = (si - np.sin(xm) ** 2 / xm) / xm
    else:
        raise ValueError("dim must be 2 or 3")
    return out


def porod_moment(net: Network2D, weights: Optional[np.ndarray],
                 kmax: float, *, dim: int = 2) -> float:
    """Finite-k corrected Porod moment.

    M_P = (1/C_d) < w^2 ell * x g_d(x) |_{x = kmax ell / 2} >
          -> <w^2 ell> as kmax -> inf,

    with g_d the orientation-averaged sinc^2 form factor and C_2 = 1,
    C_3 = pi/2. ``weights`` are PER-LENGTH w_e (None = uniform). Replaces
    <w^2 ell> in normalizations to remove the finite-kmax bias (largest
    for schemes emphasizing short edges).

    NB: the NPZ metadata key ``wk2_over_ell_porod_corrected`` (io.py) is
    the same quantity WITHOUT the 1/C_d -- in 3D it is pi/2 times this.
    """
    if net.n_edges == 0:
        return 0.0
    ell = np.asarray(net.edges_lengths, dtype=float)
    w = (np.ones_like(ell) if weights is None
         else np.asarray(weights, dtype=float))
    x = 0.5 * float(kmax) * ell
    C_d = 1.0 if dim == 2 else np.pi / 2.0
    g = g_d_form_factor(x, dim=dim)
    return float(np.mean(w ** 2 * ell * x * g) / C_d)


def self_pair_moment(r_grid: np.ndarray, ell: np.ndarray,
                     w: Optional[np.ndarray] = None) -> np.ndarray:
    """<w^2 (ell - r)_+> over the edge table (per-length marks w; None = 1).

    The self channel of the radial autocovariance, up to its density
    prefactors:  chi_self(r) = lambda_ell <w^2 (ell - r)_+> / (pi r).
    """
    ell = np.asarray(ell, dtype=float)
    w = np.ones_like(ell) if w is None else np.asarray(w, dtype=float)
    r = np.atleast_1d(np.asarray(r_grid, dtype=float))
    return np.mean(np.maximum(ell[None, :] - r[:, None], 0.0) * (w ** 2)[None, :], axis=1)


def self_pair_direct(r_centers: np.ndarray, ell: np.ndarray,
                     w: Optional[np.ndarray] = None, *, n_pairs: int = 2048,
                     seed: int = 0, edge_batch: int = 2048) -> np.ndarray:
    """Sampled same-edge estimate of <w^2 (ell - r)_+> on a uniform r grid.

    On every edge draw ``n_pairs`` independent uniform arclength pairs (s, t)
    and histogram |s - t| into the bins centred on ``r_centers`` with weight
    w^2 ell^2 / n_pairs; divide by 2 dr N_ell.  The same-edge pair density is
    2 (ell - r)_+, so the expectation is self_pair_moment exactly and the two
    differ by counting noise alone, ~ 1/sqrt(counts) per bin.  This is the
    self-channel twin of the distinct route's delta-shell histogram (same
    bins, same arclength measure).  Deterministic for a given ``seed``.
    """
    r = np.atleast_1d(np.asarray(r_centers, dtype=float))
    ell = np.asarray(ell, dtype=float)
    w = np.ones_like(ell) if w is None else np.asarray(w, dtype=float)
    if r.size < 2:
        raise ValueError("r_centers needs at least 2 points")
    dr = float(r[1] - r[0])
    edges = np.concatenate(([r[0] - 0.5 * dr], 0.5 * (r[1:] + r[:-1]), [r[-1] + 0.5 * dr]))
    rng = np.random.default_rng(seed)
    counts = np.zeros(r.size, dtype=float)
    for b0 in range(0, ell.size, int(edge_batch)):
        le = ell[b0:b0 + int(edge_batch)]
        we = w[b0:b0 + int(edge_batch)]
        u = rng.random((le.size, int(n_pairs)))
        v = rng.random((le.size, int(n_pairs)))
        d = le[:, None] * np.abs(u - v)
        wt = np.broadcast_to(((we * le) ** 2 / float(n_pairs))[:, None], d.shape)
        h, _ = np.histogram(d.ravel(), bins=edges, weights=wt.ravel())
        counts += h
    return counts / (2.0 * dr * ell.size)


def f_w_self_baseline(net: Network2D, r_grid: np.ndarray,
                      weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Closed-form weighted self baseline f_w(r).

    Each edge contributes the chord density 2 (ell - r)_+ of arclength
    pairs at separation r; pooled, weighted, and normalized as an
    isotropic pair density against rho_w^2 (2D):

        f_w(r) = lambda_ell <w^2 (ell - r)_+> / (pi r rho_w^2),

    lambda_ell = N_ell / V, rho_w = sum w ell / V, with the k = 0 identity
    rho_w^2 int f_w d^2r = lambda_ell <w^2 ell^2> = chi_self(0).
    ``weights`` are PER-LENGTH w_e (None = uniform, giving
    f_ell(r) = <(ell - r)_+> / (pi r <ell> rho_ell)).
    """
    if net.n_edges == 0:
        return np.zeros_like(np.atleast_1d(np.asarray(r_grid, dtype=float)))
    ell = np.asarray(net.edges_lengths, dtype=float)
    w = (np.ones_like(ell) if weights is None
         else np.asarray(weights, dtype=float))
    r = np.atleast_1d(np.asarray(r_grid, dtype=float))
    A = cell_volume(net.cell_like)
    rho_w = float((w * ell).sum()) / A
    lam = ell.size / A
    num = self_pair_moment(r, ell, w)
    return lam * num / (np.pi * np.maximum(r, 1e-300) * max(rho_w, 1e-300) ** 2)
