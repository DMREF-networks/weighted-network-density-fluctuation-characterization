"""Moment-matched vertex shares (2D, square periodic box): the ``d2_ring`` (D_2) weight.

The scheme hands every network vertex v a cell *v (the cells tile the box) and
spreads |*v| as shares s_{v,e} over the edges of v's ring R_v -- the edges incident
to v or to one of its graph neighbours -- so that uniform-mass rods on those edges
reproduce the cell's mass, first and second moments about x_v.  Shares are masses:
column e of the moment system holds the per-unit-mass rod moments about x_v, with
u = one endpoint - x_v and d = the edge vector (the same for either endpoint),

    mu1 = u + d/2,    mu2 = u u^T + (u d^T + d u^T)/2 + d d^T / 3,

so the mass row is all ones.  A reference split s0 (summing to |*v|) is corrected
inside the null space of the mass row by the minimum-norm least-squares solution of
the five moment rows: exact whenever the ring holds six or more edges in generic
position (|R_v| - 1 free unknowns against five rows), so every vertex's dipole and
quadrupole vanish and the scheme sits at k^6.  The weight is w_eff,e = sum_v s_{v,e}
and sum_e w_eff,e = L^2.

  d2_ring (D_2)            *v = the barycentric D cell: the union over the triangles
                           T at v of the quadrilaterals (x_v, m_a, b_T, m_b), b_T the
                           barycenter of T and m_a, m_b the midpoints of its two edges
                           at v, so |*v| = (1/3) sum_T |T|;  s0 = the D kite halves
                           |K_{v,e}| = (|T_L| + |T_R|) / 6 on the edges incident to v
                           and zero on the rest of the ring.  Without the correction
                           the weights are the D scheme's.  D network of a
                           triangulation only.

Fallbacks: a ring with fewer than two edges keeps s0 on the vertex's own edges
(``n_fallback``); an isolated vertex contributes nothing (``n_unresolved``, a
tiling deficit).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .network import Network2D
from .pbc import min_image_delta
from .planar_dual import face_polygons


def _tri_moments(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Signed area, first and second moment of the triangles (0, a, b); the
    moments carry the sign of the area, so a fan of a closed polygon sums exactly."""
    A = 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    M1 = (A / 3.0)[:, None] * (a + b)
    ab = np.einsum("ni,nj->nij", a, b)
    M2 = (A / 6.0)[:, None, None] * (np.einsum("ni,nj->nij", a, a) + np.einsum("ni,nj->nij", b, b)
                                     + 0.5 * (ab + ab.transpose(0, 2, 1)))
    return A, M1, M2


def bary_cell_moments(net: Network2D) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Barycentric D cell per vertex of a triangulated network, plus the kite halves.

    Returns ``(A, c_rel, M2, half_kite)``: area, centroid offset and second moment
    of *v about ``net.points[v]``, and |K_{v,e}| per edge (equal at both endpoints,
    half the D kite).  Pieces: at corner i of a face with lifted ring (p_0, p_1, p_2),
    edge ids e_i = (p_i -> p_{i+1}) and barycenter g, the triangles (p_i, m_i, g) and
    (p_i, g, m_{i-1}), m the edge midpoints, each of area |T|/6; the first is the
    (T, e_i) piece of the kite half.  Every face must be a triangle.
    """
    fp = face_polygons(net)
    if fp["is_homology"].any() or any(v is None or v.shape[0] != 3 for v in fp["verts"]):
        raise ValueError("d2_ring needs the D network of a triangulation (every face a triangle)")
    P = np.stack(fp["verts"])                                    # (F, 3, 2) lifted corners
    E = np.stack(fp["edge_ids"]).astype(np.int64)                # (F, 3) edge of p_i -> p_{i+1}
    left = np.asarray(net.edges_left_cell, dtype=np.int64)
    right = np.asarray(net.edges_right_cell, dtype=np.int64)
    n_v, n_e = int(net.points.shape[0]), int(net.n_edges)
    A = np.zeros(n_v)
    M1 = np.zeros((n_v, 2))
    M2 = np.zeros((n_v, 2, 2))
    half_kite = np.zeros(n_e)
    g = P.mean(axis=1)
    for i in range(3):
        e_next, e_prev = E[:, i], E[:, i - 1]
        lp, rp = left[e_prev], right[e_prev]
        vid = np.where((lp == left[e_next]) | (lp == right[e_next]), lp, rp)   # corner vertex
        p = P[:, i]
        m_next = 0.5 * (p + P[:, (i + 1) % 3]) - p
        m_prev = 0.5 * (p + P[:, i - 1]) - p
        A1, M1a, M2a = _tri_moments(m_next, g - p)
        A2, M1b, M2b = _tri_moments(g - p, m_prev)
        s = np.where(A1 + A2 >= 0.0, 1.0, -1.0)
        np.add.at(A, vid, s * (A1 + A2))
        np.add.at(M1, vid, s[:, None] * (M1a + M1b))
        np.add.at(M2, vid, s[:, None, None] * (M2a + M2b))
        np.add.at(half_kite, e_next, s * A1)
    return A, M1 / A[:, None], M2, half_kite


def _incidence(net: Network2D) -> Tuple[np.ndarray, np.ndarray]:
    """CSR incidence over the network vertices: ``ecat[offsets[v]:offsets[v+1]]``
    are the edges at vertex v (``edges_left_cell`` / ``edges_right_cell`` are the
    endpoint vertex ids for every 2D network type)."""
    n_v = int(net.points.shape[0])
    vcat = np.concatenate([net.edges_left_cell, net.edges_right_cell]).astype(np.int64)
    ecat = np.tile(np.arange(net.n_edges, dtype=np.int64), 2)
    order = np.argsort(vcat, kind="stable")
    counts = np.bincount(vcat, minlength=n_v)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return offsets, ecat[order]


def _neighbors(net: Network2D) -> List[set]:
    nbrs: List[set] = [set() for _ in range(int(net.points.shape[0]))]
    for i, j in zip(net.edges_left_cell, net.edges_right_cell):
        nbrs[int(i)].add(int(j))
        nbrs[int(j)].add(int(i))
    return nbrs


def _ring_edges(v: int, nbrs: List[set], offsets: np.ndarray, ecat: np.ndarray) -> np.ndarray:
    """Edges incident to v or to one of its graph neighbours (sorted ids)."""
    out: set = set()
    for u in {v} | nbrs[v]:
        out.update(int(e) for e in ecat[offsets[u]:offsets[u + 1]])
    return np.fromiter(sorted(out), dtype=np.int64, count=len(out))


def _solve_shares(u: np.ndarray, d: np.ndarray, s0: np.ndarray, A_v: float,
                  c_rel: np.ndarray, M2: np.ndarray):
    """Shares on one vertex's ring edges from the reference split ``s0``: mass exact
    (null space of the mass row), first and second moments least squares.  Returns
    ``(s, resid)`` with ``resid = (mass, m1_x, m1_y, m2_xx, m2_xy, m2_yy)`` target
    minus achieved."""
    n = int(u.shape[0])
    mu1 = u + 0.5 * d
    ud = np.einsum("ei,ej->eij", u, d)
    mu2 = (np.einsum("ei,ej->eij", u, u) + 0.5 * (ud + ud.transpose(0, 2, 1))
           + np.einsum("ei,ej->eij", d, d) / 3.0)
    C = np.column_stack([mu1, mu2[:, 0, 0], mu2[:, 0, 1], mu2[:, 1, 1]])       # (n, 5)
    target = np.array([A_v * c_rel[0], A_v * c_rel[1], M2[0, 0], M2[0, 1], M2[1, 1]])
    Q, _ = np.linalg.qr(np.column_stack([np.ones(n), np.eye(n)[:, :n - 1]]))
    Z = Q[:, 1:]                                                              # sum_e z_e = 0
    y, *_ = np.linalg.lstsq(C.T @ Z, target - C.T @ s0, rcond=None)
    s = s0 + Z @ y
    return s, np.concatenate([[A_v - s.sum()], target - C.T @ s])


def _ring_assembly(net: Network2D, A: np.ndarray, c_rel: np.ndarray, M2: np.ndarray,
                   prior: Callable[[int, np.ndarray, np.ndarray], np.ndarray],
                   extra: Dict) -> Tuple[np.ndarray, Dict]:
    """Shares on every vertex's ring assembled into w_eff.  ``prior(v, es, inc)`` is the
    reference split of ``A[v]`` over the edges ``es`` (``inc`` = v's own edges).
    Returns ``(W, meta)``: tiling residual, negative-weight fraction, fallback
    counters, ``extra``, residual dipole rms / max, relative second-moment residual
    mean / max."""
    L = float(net.L)
    n_v, n_e = int(net.points.shape[0]), int(net.n_edges)
    offsets, ecat = _incidence(net)
    nbrs = _neighbors(net)
    W = np.zeros(n_e)
    pv = np.zeros((n_v, 2))
    m2_rel = np.zeros(n_v)
    n_fallback = n_unresolved = 0
    for v in range(n_v):
        inc = ecat[offsets[v]:offsets[v + 1]]
        es = _ring_edges(v, nbrs, offsets, ecat)
        if es.size < 2:
            es = inc
            if es.size == 0:
                n_unresolved += 1
                continue
            n_fallback += 1
            s = prior(v, es, inc)
            np.add.at(W, es, s)
            mid = min_image_delta(net.edges_midpoints[es] - net.points[v], L)
            pv[v] = A[v] * c_rel[v] - s @ mid
            continue
        u = min_image_delta(net.edges_v1[es] - net.points[v], L)
        d = net.edges_v2[es] - net.edges_v1[es]
        s, resid = _solve_shares(u, d, prior(v, es, inc), float(A[v]), c_rel[v], M2[v])
        np.add.at(W, es, s)
        pv[v] = resid[1:3]
        m2_rel[v] = np.linalg.norm(resid[3:]) / max(np.linalg.norm(M2[v]), 1e-300)
    meta: Dict = {
        "tiling_sum": float(W.sum()),
        "tiling_resid": float(W.sum() / (L * L) - 1.0),
        "frac_negative": float((W < 0).mean()),
        "n_vertices": n_v,
        "n_fallback": int(n_fallback),
        "n_unresolved": int(n_unresolved),
        **extra,
        "pv_rms": float(np.sqrt((pv ** 2).sum(1).mean())),
        "pv_max": float(np.abs(pv).max()),
        "m2_resid_rel_mean": float(m2_rel.mean()),
        "m2_resid_rel_max": float(m2_rel.max()),
    }
    return W, meta


def d2_ring(net: Network2D, *, diag_out: Optional[dict] = None) -> np.ndarray:
    """Per-edge kernel weight w_eff of the D_2 scheme (module doc).  Diagnostics go to
    ``diag_out['d2_ring']`` (the ``_ring_assembly`` keys plus the barycentric cells' tiling
    residual)."""
    if net.cell is not None:
        raise ValueError("d2_ring needs a square periodic box")
    if net.n_edges == 0:
        return np.zeros(0)
    A, c_rel, M2, half_kite = bary_cell_moments(net)

    def prior(v, es, inc):
        s0 = np.zeros(es.size)
        own = np.isin(es, inc)
        s0[own] = half_kite[es[own]]
        return s0

    W, meta = _ring_assembly(net, A, c_rel, M2, prior,
                             extra={"cell_tiling_resid": float(A.sum() / float(net.L) ** 2 - 1.0)})
    if diag_out is not None:
        diag_out["d2_ring"] = meta
    return W
