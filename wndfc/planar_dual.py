"""Planar half-edge face-walk dual on a 2D periodic graph.

Builds the true planar-graph dual of an arbitrary 2D embedded periodic graph
(typically G or C from wndfc) by:

  1. Building directed half-edges with PBC shift labels.
  2. Cyclic-sorting outgoing half-edges at each vertex by angle.
  3. Face-walking via "predecessor of twin's incoming half-edge in the
     head's cyclic list" (yields one face per directed half-edge).
  4. Detecting and excluding torus homology cycles via non-zero shift sum.
  5. Lifting each non-homology face into the universal cover anchored at
     its first vertex, then computing area + area-weighted triangle-fan
     centroid (interior to simple polygons even when non-convex).

The resulting per-edge dual length d_e = |c_R - c_L| and the signed
sin theta_e between primal-edge and dual-edge directions feed the signed
kite-area weight w_e = (1/2) * ell_e * d_e * sin theta_e, which tiles the
torus by the diamond identity (sum of kite areas = V, the cell area, modulo
the homology faces).

Used by wndfc/weights.py:`dual_diamond_planar` to give G/C a real
planar-face dual instead of inheriting D's Voronoi-ridge dual.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .network import Network2D
from .pbc import (CellLike, cell_volume, image_shift, min_image_delta,
                  shift_vectors, wrap_to_box)


def _build_half_edges(net: Network2D):
    """Build the directed half-edge arrays from a Network2D.

    Returns
    -------
    he_tail, he_head : (2E,) int  -- source / target vertex per half-edge
    he_shift         : (2E, 2) i8 -- integer lattice coefficients s such that
                                     the head's unwrapped position is
                                     points[head] + shift_vectors(s)
                                     (square box: points[head] + s*L)
    he_twin          : (2E,) int  -- twin half-edge index
    """
    L = net.cell_like
    i = np.asarray(net.edges_left_cell, dtype=np.int64)
    j = np.asarray(net.edges_right_cell, dtype=np.int64)
    p = net.points
    delta = min_image_delta(p[j] - p[i], L)
    # j_unwrapped = p[i] + delta -> j_unwrapped = p[j] + shift_vectors(shift)
    # => shift = image_shift(p[i] + delta - p[j])  (square box: round(raw / L))
    raw = p[i] + delta - p[j]
    shift_ij = image_shift(raw, L).astype(np.int8)
    ne = i.shape[0]
    he_tail = np.concatenate([i, j])
    he_head = np.concatenate([j, i])
    he_shift = np.concatenate([shift_ij, -shift_ij])
    he_twin = np.concatenate([np.arange(ne, 2 * ne, dtype=np.int64),
                              np.arange(0, ne, dtype=np.int64)])
    return he_tail, he_head, he_shift, he_twin


def _cyclic_neighbor_lists(net: Network2D, he_tail: np.ndarray,
                           he_head: np.ndarray):
    """For each vertex v, sort outgoing half-edges by angle of (head - tail)."""
    V = net.points.shape[0]
    L = net.cell_like
    nh = he_tail.shape[0]
    delta = min_image_delta(net.points[he_head] - net.points[he_tail], L)
    ang = np.arctan2(delta[:, 1], delta[:, 0])
    # tie-break on head index for FP-degenerate angles
    order = np.lexsort((he_head, ang, he_tail))
    counts = np.bincount(he_tail, minlength=V)
    adj_start = np.zeros(V + 1, dtype=np.int64)
    adj_start[1:] = np.cumsum(counts)
    adj_he = order
    # adj_local_pos[h] = position of h within he_tail[h]'s adjacency slice.
    # Need local (per-vertex) index, not global sorted index.
    adj_local_pos = np.empty(nh, dtype=np.int64)
    global_pos = np.empty(nh, dtype=np.int64)
    global_pos[order] = np.arange(nh, dtype=np.int64)
    adj_local_pos = global_pos - adj_start[he_tail]
    return adj_start, adj_he, adj_local_pos


def _face_walk(he_tail: np.ndarray, he_head: np.ndarray,
               he_shift: np.ndarray, he_twin: np.ndarray,
               adj_start: np.ndarray, adj_he: np.ndarray,
               adj_inv: np.ndarray):
    """Trace one face per directed half-edge.

    At current half-edge h = (u -> v): take its twin t = (v -> u), find t's
    position p in v's cyclic neighbor list, and step to the cyclic
    *predecessor* h_next = adj[v][(p - 1) mod deg(v)]. Closes a cycle.

    A face's accumulated PBC shift = sum of he_shift along its half-edges.
    For interior bounded faces this is (0, 0); torus homology cycles have
    non-zero sum -- those are flagged and excluded from dual geometry.
    """
    nh = he_tail.shape[0]
    face_id = np.full(nh, -1, dtype=np.int64)
    faces: List[Dict] = []
    for h0 in range(nh):
        if face_id[h0] != -1:
            continue
        cycle = []
        total_shift = np.zeros(2, dtype=np.int64)
        h = h0
        while face_id[h] == -1:
            face_id[h] = len(faces)
            cycle.append(h)
            total_shift += he_shift[h].astype(np.int64)
            t = he_twin[h]
            v = he_head[h]
            p_t = adj_inv[t]
            deg = adj_start[v + 1] - adj_start[v]
            p_next = (p_t - 1) % deg
            h = adj_he[adj_start[v] + p_next]
        is_homology = bool((total_shift != 0).any())
        faces.append({
            "half_edges": np.asarray(cycle, dtype=np.int64),
            "total_shift": total_shift,
            "is_homology": is_homology,
        })
    return face_id, faces


def _lift_face_verts(pts: np.ndarray, hes: np.ndarray,
                     he_tail: np.ndarray, he_shift: np.ndarray,
                     L: CellLike) -> np.ndarray:
    """Lift one face cycle into the universal cover.

    Anchored at the first half-edge's tail; each subsequent vertex is
    placed by accumulating the lattice shifts of the traversed half-edges
    (``shift_vectors`` of the integer coefficients), so
    consecutive vertices are true neighbors (no wrap inside the polygon).
    The single lifting implementation shared by ``_face_geometry`` and
    ``face_polygons``.
    """
    n = hes.shape[0]
    verts = np.empty((n, 2), dtype=float)
    verts[0] = pts[he_tail[hes[0]]]
    cum_shift = np.zeros(2, dtype=float)
    for k in range(1, n):
        h_prev = hes[k - 1]
        cum_shift += shift_vectors(he_shift[h_prev].astype(float), L)
        verts[k] = pts[he_tail[hes[k]]] + cum_shift
    return verts


def _face_geometry(net: Network2D, faces: List[Dict],
                   he_tail: np.ndarray, he_shift: np.ndarray):
    """Lift each non-homology face into the universal cover (anchored on its
    first half-edge's tail) and compute polygon area + signed-shoelace
    centroid.

    The centroid uses signed triangle-fan weights so ``wsum`` is the signed
    polygon area and ``(cx, cy) / wsum`` is the true geometric polygon
    centroid; it can land outside a non-star-shaped or kernel-evading face.

    Also tracks per-face diagnostics: ``n_nonconvex_faces`` (sign-flipping
    triangle fan from ``verts[0]``) and ``n_small_faces`` (< 3 half-edges,
    skipped).
    """
    L = net.cell_like
    pts = net.points
    F = len(faces)
    face_area = np.zeros(F, dtype=float)
    face_centroid = np.zeros((F, 2), dtype=float)
    n_nonconvex = 0
    n_small_faces = 0  # n_half_edges < 3 (non-homology) -- centroid undefined,
                       # left at default (0,0). Edges incident to two such
                       # faces show up as w_e = 0 outliers.
    for fi, face in enumerate(faces):
        if face["is_homology"]:
            continue
        hes = face["half_edges"]
        n = hes.shape[0]
        if n < 3:
            n_small_faces += 1
            continue
        verts = _lift_face_verts(pts, hes, he_tail, he_shift, L)
        x = verts[:, 0]
        y = verts[:, 1]
        sa = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        face_area[fi] = abs(sa)
        if sa == 0.0:
            face_centroid[fi] = verts.mean(axis=0)
            continue
        p0 = verts[0]
        cx = 0.0
        cy = 0.0
        wsum = 0.0
        saw_pos = False
        saw_neg = False
        for k in range(1, n - 1):
            p1 = verts[k]
            p2 = verts[k + 1]
            tri_a = 0.5 * ((p1[0] - p0[0]) * (p2[1] - p0[1])
                           - (p2[0] - p0[0]) * (p1[1] - p0[1]))
            if tri_a > 0.0:
                saw_pos = True
            elif tri_a < 0.0:
                saw_neg = True
            cx += tri_a * (p0[0] + p1[0] + p2[0]) / 3.0
            cy += tri_a * (p0[1] + p1[1] + p2[1]) / 3.0
            wsum += tri_a
        if saw_pos and saw_neg:
            n_nonconvex += 1
        if abs(wsum) > 0.0:
            c = np.array([cx / wsum, cy / wsum])
        else:
            c = verts.mean(axis=0)
        face_centroid[fi] = wrap_to_box(c[None, :], L)[0]
    return face_area, face_centroid, n_nonconvex, n_small_faces


def planar_face_dual(net: Network2D, eps: float = 1e-12) -> Dict:
    """Build the planar-graph face dual under PBC.

    The dual vertex of each face is its signed-shoelace polygon centroid
    (``_face_geometry``); ``edge_sin_theta`` keeps its sign.

    Returns dict with:
      edge_face_L, edge_face_R : (E,) int   -- faces incident to each primal edge
      edge_dual_len            : (E,)       -- |c_R - c_L| (min-image)
      edge_sin_theta           : (E,)       -- sin(primal, dual), signed, in [-1, 1]
      face_area, face_centroid : (F,), (F,2)
      n_homology               : count of face cycles with non-zero shift sum
      n_bad_edges              : edges incident to a homology face (zeroed in d/sin)
      n_nonconvex_faces        : cellular faces with sign-flipping triangle fan
                                 from vertex 0 (their signed centroid may fall
                                 outside their kernel)
      n_small_faces            : non-homology face cycles with < 3 half-edges
      n_zero_kite_unmasked     : kite == 0 edges not explained by homology mask
      kite_sum, kite_resid     : Sigma_e (1/2) ell_e d_e sin theta_e and the
                                 dimensionless residual `kite_sum / V - 1`
                                 (V = cell area).
                                 Excludes bad (homology-incident) edges since
                                 their d_e, sin_theta are zeroed.
      n_faces                  : total face-cycle count from the walker (F').
                                 For a cellular embedding F' = E - V (Euler).
    """
    n_edges = int(net.n_edges)
    if n_edges == 0:
        return dict(edge_face_L=np.zeros(0, dtype=np.int64),
                    edge_face_R=np.zeros(0, dtype=np.int64),
                    edge_dual_len=np.zeros(0, dtype=float),
                    edge_sin_theta=np.zeros(0, dtype=float),
                    face_area=np.zeros(0, dtype=float),
                    face_centroid=np.zeros((0, 2), dtype=float),
                    n_homology=0, n_bad_edges=0, n_nonconvex_faces=0,
                    n_small_faces=0, n_zero_kite_unmasked=0,
                    kite_sum=0.0, kite_resid=0.0, n_faces=0)

    he_tail, he_head, he_shift, he_twin = _build_half_edges(net)
    adj_start, adj_he, adj_inv = _cyclic_neighbor_lists(net, he_tail, he_head)
    face_id, faces = _face_walk(he_tail, he_head, he_shift, he_twin,
                                adj_start, adj_he, adj_inv)
    L = net.cell_like

    edge_face_R = face_id[:n_edges]            # face right of i->j
    edge_face_L = face_id[n_edges:]            # face right of j->i = left of i->j

    face_area, face_centroid, n_nonconvex, n_small_faces = _face_geometry(
        net, faces, he_tail, he_shift)

    cL = face_centroid[edge_face_L]
    cR = face_centroid[edge_face_R]
    is_homology = np.array([f["is_homology"] for f in faces], dtype=bool)
    bad = is_homology[edge_face_L] | is_homology[edge_face_R]

    d_vec = min_image_delta(cR - cL, L)
    d_len = np.linalg.norm(d_vec, axis=1)

    p_vec = min_image_delta(
        net.points[net.edges_right_cell] - net.points[net.edges_left_cell], L)
    p_len = np.maximum(np.linalg.norm(p_vec, axis=1), eps)

    cross = p_vec[:, 0] * d_vec[:, 1] - p_vec[:, 1] * d_vec[:, 0]
    sin_theta = np.zeros_like(d_len)
    safe = d_len > eps
    # keep the sign of sin theta: escape cases give negative kite
    # contributions that cancel via the signed-shoelace identity
    # (Sum_{e in F} signed_tri = signed_area(F) regardless of c_F)
    sin_theta[safe] = cross[safe] / (p_len[safe] * d_len[safe])

    d_len[bad] = 0.0
    sin_theta[bad] = 0.0

    kite = 0.5 * np.asarray(net.edges_lengths, dtype=float) * d_len * sin_theta
    kite_sum = float(kite.sum())
    kite_resid = kite_sum / cell_volume(L) - 1.0
    n_zero_kite = int((kite == 0.0).sum() - bad.sum())  # zero kites NOT
                                                         # explained by the
                                                         # homology mask

    return dict(edge_face_L=edge_face_L,
                edge_face_R=edge_face_R,
                edge_dual_len=d_len,
                edge_sin_theta=sin_theta,
                face_area=face_area,
                face_centroid=face_centroid,
                n_homology=int(is_homology.sum()),
                n_bad_edges=int(bad.sum()),
                n_nonconvex_faces=int(n_nonconvex),
                n_small_faces=int(n_small_faces),
                n_zero_kite_unmasked=n_zero_kite,
                kite_sum=kite_sum,
                kite_resid=float(kite_resid),
                n_faces=int(len(faces)))


def face_polygons(net: Network2D) -> Dict:
    """Lifted per-face polygons of the planar face dual (read-only).

    The geometric substrate for the vector-weight machinery (Hodge sectors,
    moment-killing stencils, cone-apex weights): every non-homology face as
    an explicit polygon in the universal cover, plus the per-edge flanking
    faces. Reuses the exact half-edge build / face walk / lift of
    ``planar_face_dual``; adds no new geometry conventions.

    Returns dict with:
      verts        : list (len F) of (n_v, 2) float arrays -- each face's
                     vertex ring lifted into the universal cover (anchored
                     at its first half-edge's tail). ``None`` for homology
                     cycles (the lift does not close).
      edge_ids     : list (len F) of (n_v,) int arrays -- the network edge
                     id carrying each boundary segment
                     ``verts[k] -> verts[(k+1) % n_v]``. ``None`` for
                     homology cycles.
      area         : (F,) abs shoelace area (0 for homology / small faces)
      centroid     : (F, 2) signed-fan polygon centroid wrapped into the
                     periodic cell
      centroid_lift: (F, 2) the same centroid in each face's lifted frame
      edge_face_L / edge_face_R : (E,) face ids flanking each edge, labeled
                     relative to the ``edges_left_cell -> edges_right_cell``
                     direction (globally consistent chirality from the
                     angular half-edge sort)
      is_homology  : (F,) bool -- torus homology cycles (non-zero shift sum)
      is_small     : (F,) bool -- cycles with < 3 half-edges
      n_faces, n_homology, n_small_faces : counts
    """
    n_edges = int(net.n_edges)
    if n_edges == 0:
        return dict(verts=[], edge_ids=[],
                    area=np.zeros(0, dtype=float),
                    centroid=np.zeros((0, 2), dtype=float),
                    centroid_lift=np.zeros((0, 2), dtype=float),
                    edge_face_L=np.zeros(0, dtype=np.int64),
                    edge_face_R=np.zeros(0, dtype=np.int64),
                    is_homology=np.zeros(0, dtype=bool),
                    is_small=np.zeros(0, dtype=bool),
                    n_faces=0, n_homology=0, n_small_faces=0)

    he_tail, he_head, he_shift, he_twin = _build_half_edges(net)
    adj_start, adj_he, adj_inv = _cyclic_neighbor_lists(net, he_tail, he_head)
    face_id, faces = _face_walk(he_tail, he_head, he_shift, he_twin,
                                adj_start, adj_he, adj_inv)
    L = net.cell_like
    pts = net.points
    F = len(faces)

    verts_out: List = [None] * F
    edge_ids_out: List = [None] * F
    area = np.zeros(F, dtype=float)
    centroid = np.zeros((F, 2), dtype=float)
    centroid_lift = np.zeros((F, 2), dtype=float)
    is_homology = np.zeros(F, dtype=bool)
    is_small = np.zeros(F, dtype=bool)

    for fi, face in enumerate(faces):
        if face["is_homology"]:
            is_homology[fi] = True
            continue
        hes = face["half_edges"]
        n = hes.shape[0]
        verts = _lift_face_verts(pts, hes, he_tail, he_shift, L)
        verts_out[fi] = verts
        edge_ids_out[fi] = (hes % n_edges).astype(np.int64)
        if n < 3:
            is_small[fi] = True
            c = verts.mean(axis=0)
            centroid_lift[fi] = c
            centroid[fi] = wrap_to_box(c[None, :], L)[0]
            continue
        x = verts[:, 0]
        y = verts[:, 1]
        sa = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        area[fi] = abs(sa)
        # Signed triangle fan from verts[0]: exact polygon centroid for any
        # simple polygon (the centroid of _face_geometry).
        p0 = verts[0]
        u = verts[1:-1] - p0
        v = verts[2:] - p0
        tri_a = 0.5 * (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
        wsum = float(tri_a.sum())
        if abs(wsum) > 0.0:
            c = p0 + ((tri_a[:, None] * (u + v)).sum(axis=0) / (3.0 * wsum))
        else:
            c = verts.mean(axis=0)
        centroid_lift[fi] = c
        centroid[fi] = wrap_to_box(c[None, :], L)[0]

    return dict(verts=verts_out,
                edge_ids=edge_ids_out,
                area=area,
                centroid=centroid,
                centroid_lift=centroid_lift,
                edge_face_L=face_id[n_edges:],
                edge_face_R=face_id[:n_edges],
                is_homology=is_homology,
                is_small=is_small,
                n_faces=F,
                n_homology=int(is_homology.sum()),
                n_small_faces=int(is_small.sum()))
