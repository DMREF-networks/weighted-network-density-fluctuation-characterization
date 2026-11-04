"""3D volumetric half-face dual (signed-tetrahedron decomposition).

Builds the per-primal-edge weight ``w_e`` from the signed-tet identity over
an embedded 3D cell complex containing the network's 1-skeleton:

    w_e = sum_{(F ∋ e, C ⊃ F)} signed_vol(tet(v_a, v_b, c_F, c_C))

This is the dimensional dimension-up of the 2D ``signed_signed`` kite
identity in ``wndfc/planar_dual.py``: signed triangle -> signed
tetrahedron, planar DCEL -> Delaunay-tet walk, ``sum_e w_e -> L^3``.

``centroid_mode='signed_signed'`` is the only mode in 3D -- the signed-tet
identity is unconditional in c_F / c_C placement, so the 2D fallback modes
(``kernel``, ``signed_safe``, ``*_zero``, ``per_*``) aren't needed.

Cell complex per ntype:
  - D: Delaunay tessellation of ``net.points`` (generators). D's
    1-skeleton IS the Delaunay 1-skeleton, so tiles ``L^3`` exactly.
  - V: Voronoi tessellation of the generators -- 3-cells = Voronoi
    cells, 2-cells = Voronoi facets, 1-cells = V-edges.
    IMPLEMENTED (``_volumetric_voronoi``): each cell decomposes from its
    centroid into signed-tet kites over its facet-edges; the circumcentric
    twin of ``_volumetric_delaunay``. This is the COM-centroid placement
    of the V dual, distinct from the orthogonal ``dual_area`` bipyramid.
  - G: G's *own* embedded 3D cell complex via half-face DCEL on the
    G-graph. Walks G alone (no ambient Delaunay), so stripped D-edges
    merge their incident D-triangles into larger G-faces (the 3D analog
    of 2D's planar-face merge). NOT YET IMPLEMENTED -- the underlying
    spherical-arrangement face discovery (a 3D half-face DCEL on the
    G-1-skeleton) is unbuilt.
  - C: Poincaré dual of the generator Delaunay -- 1-cells = Delaunay
    triangles, 2-cells = Delaunay edges (tet-rings), 3-cells = generators
    (tet-stars). IMPLEMENTED (``_volumetric_c_tess``): explicit DEC walk
    of the C-3-cells with outward-CCW (right-hand about (h-g)) C-2-face
    orientation -- the dimensional lift of D's ``_TET_FACES_POS``.
    Auto-cellular (no Gabriel filter / merge / union-find / homology
    cut); centroid c_C/c_F, signed-tet kites, tiles ``L^3`` exactly.

    Note on the rejected shortcut: walking the Delaunay tessellation
    of ``net.points`` for G and dropping non-Gabriel tet edges does
    NOT tile -- the dropped contributions are missing volume, not
    redistribution. 2D ``dual_diamond_planar`` tiles for G because its
    planar face walk runs on G alone (stripping an edge merges the two
    incident D-triangles into one G-face); the 3D analog requires the
    same merge behavior, which only emerges from walking G's own DCEL.

V IS handled here via ``_volumetric_voronoi`` (the COM-centroid Voronoi-cell
dual, distinct from the orthogonal ``dual_area`` bipyramid).

Used by ``wndfc/weights_3d.py:`dual_diamond_planar```.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from .network_3d import (
    Network3D,
    _delaunay_tets_3d,
    _voropp_compute_3d,
)
from .pbc import min_image_delta


# Outward-CCW face orientations of a positively-oriented tet (g_0, g_1,
# g_2, g_3): each tuple lists the 3 face-vertex tet-local indices in
# their outward-CCW boundary order. Face F_k omits vertex k.
_TET_FACES_POS = ((1, 3, 2), (0, 2, 3), (0, 3, 1), (0, 1, 2))


def _signed_tet_vol(p0, p1, p2, p3) -> float:
    """``(1/6) det[p1-p0, p2-p0, p3-p0]`` -- signed tet volume."""
    a = p1 - p0
    b = p2 - p0
    c = p3 - p0
    return (a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0


def _face_area_centroid(loop_pts, c_vm_F: np.ndarray) -> np.ndarray:
    """True area centroid of a face via the SAME fan the kite walk uses
    (triangles about the vertex-mean ``c_vm_F``). Exact for planar faces
    (V facets); for C's generically non-planar tet-centroid rings it is
    the fan-consistent surrogate (the fan IS the surface the kites
    integrate over). A triangular face returns its vertex mean exactly
    (triangle centroid == vertex mean), so this is a no-op for D."""
    acc = np.zeros(3, dtype=float)
    a_tot = 0.0
    m = len(loop_pts)
    for i in range(m):
        pa = loop_pts[i]
        pb = loop_pts[(i + 1) % m]
        ar = 0.5 * float(np.linalg.norm(
            np.cross(pa - c_vm_F, pb - c_vm_F)))
        acc += ar * (c_vm_F + pa + pb)
        a_tot += ar
    return acc / (3.0 * a_tot) if a_tot > 1e-300 else c_vm_F


def volumetric_face_dual(net: Network3D, *,
                         centroid_mode: str = "signed_signed",
                         g_mode: str = "merge",
                         apex_mode: str = "volume_com",
                         face_anchor: str = "area_centroid",
                         collect_offset: bool = False,
                         fan_out: Optional[List[list]] = None) -> Dict:
    """Per-primal-edge volumetric dual weight via signed-tet decomposition.

    ``fan_out``, a list with one list per primal edge, receives the ``(c_C, c_F)``
    apex pair of every sub-tet accumulated onto that edge (D, V, C walks), i.e. the
    unordered centroid fan about the edge that ``wndfc.solid`` builds on.

    ``g_mode`` selects the G-tess construction (ignored for D/V/C):
      - ``merge`` (default): the tet-union-find merged-cell complex
        (``_volumetric_g_tess``). Tiles ``L^3`` exactly but the merged
        cell is one space-filling blob -> a single meaningless ``c_C``
        -> per-edge weight is gauge-degenerate. Retained as the faithful
        (but degenerate) Gabriel planar dual. Maps to weight flag
        ``dual_diamond_planar``.
      - ``dmask``: the Delaunay-tet centroid dual restricted to G-edges
        with an explicit honest deficit (``_volumetric_g_dmask``).
        Non-degenerate (compact D-tet ``c_T``); ``sum(w) < L^3`` by
        ``deficit_frac`` = stripped-edge bipyramid volume / L^3. Maps to
        weight flag ``dual_diamond_gdmask`` (alias: ``dual_diamond_dtet``).

    Returns dict with:
      per_edge_weight    : (E,) float -- ``w_e`` (signed).
      kite_sum           : float       -- ``sum(w_e)``.
      kite_resid         : float       -- ``kite_sum / L^3 - 1``.
      n_tets             : int         -- number of Delaunay tets walked.
      n_edges_unmatched  : int         -- tet-edge appearances with no
                                          matching primal edge.
      centroid_mode      : str

    ``apex_mode`` selects the placement of the cell apex ``c_C`` for the
    V / C walks (the 3D "barycentric placement"):
      - ``volume_com`` (default): true volume centroid via the SAME face-fan
        triangulation the kite walk uses (identical anchors c_F from both
        sides of every shared face, so sum_P V_P = L^3 stays exact --
        ``com_tiling_resid`` reports it). The rigorous COM placement: the
        per-cell dipole vanishes -> alpha = 4 for 3D bary V/C (draft E
        sec:mechanism iii).
      - ``vertex_mean`` (diagnostic): mean of the cell's vertices.
        For polyhedral cells with (generically nonplanar, in C) faces this
        is NOT the volume centroid, leaving a residual per-cell dipole
        ~ V_P (b_hat - b) -> alpha = 2 (the GJT relaxed-centroid regime).
        Retained for the alpha 2 -> 4 comparison; bit-identical to the
        historical output.
    D is unaffected (a simplex's vertex mean IS its volume centroid);
    G-merge cells are gauge-degenerate so the placement is moot.

    ``face_anchor`` selects the placement of the face anchor ``c_F`` for
    the V / C walks (the SECOND anchor family -- apex_mode only upgrades
    the cell family, leaving c_F at vertex-mean; this flag closes that
    gap, the "all anchors true" arm):
      - ``area_centroid`` (default): true area centroid via the same fan
        the kite walk consumes (``_face_area_centroid``). With apex_mode=
        volume_com this is the fully-true placement (Delta_1 = 0 at both
        anchor families) -- the true dual (fcom).
      - ``vertex_mean`` (historical): mean of the face's vertex loop.
        NOT the area centroid for m>3-gon faces -> a residual per-face
        dipole shared antisymmetrically by the two incident cells (the
        alpha ~ 3 channel).
    No-op for D (triangular faces: area centroid == vertex mean) and 3D
    G (gauge-degenerate / D-tet walk).

    ``collect_offset`` (opt-in, default OFF -> bit-identical output): also
    accumulate the per-edge signed-mass first moment about the edge midpoint,
    returned as ``per_edge_m1_offset`` (E, 3). Dividing by ``per_edge_weight``
    gives the carrier offset ``x_mass - x_phase`` (the sub-tet COM relative to
    the edge midpoint where compute_chi_ell places the rod) -- the per-edge
    dipole-arm diagnostic. Wired for V/D only (the dimensional-scaling test).
    """
    if centroid_mode != "signed_signed":
        raise ValueError(
            f"unknown centroid_mode={centroid_mode!r}; only 'signed_signed' "
            "is supported in 3D")
    if apex_mode not in ("vertex_mean", "volume_com"):
        raise ValueError(
            f"unknown apex_mode={apex_mode!r}; expected 'vertex_mean' or "
            "'volume_com'")
    if face_anchor not in ("vertex_mean", "area_centroid"):
        raise ValueError(
            f"unknown face_anchor={face_anchor!r}; expected 'vertex_mean' "
            "or 'area_centroid'")
    if net.ntype == "V":
        r = _volumetric_voronoi(net, apex_mode=apex_mode,
                                face_anchor=face_anchor,
                                collect_offset=collect_offset,
                                fan_out=fan_out)
    elif net.ntype == "D":
        if apex_mode == "volume_com":
            print("    [volumetric_dual] apex_mode=volume_com is a no-op "
                  "for D (simplex vertex mean == volume centroid)")
        if face_anchor == "area_centroid":
            print("    [volumetric_dual] face_anchor=area_centroid is a "
                  "no-op for D (triangular faces: centroid == vertex mean)")
        r = _volumetric_delaunay(net, collect_offset=collect_offset, fan_out=fan_out)
    elif net.ntype == "G":
        if apex_mode == "volume_com":
            print("    [volumetric_dual] apex_mode=volume_com ignored for "
                  "G (merge cells are gauge-degenerate; dmask walks D tets)")
        if face_anchor == "area_centroid":
            print("    [volumetric_dual] face_anchor=area_centroid ignored "
                  "for G (gauge-degenerate / D-tet walk)")
        if g_mode == "merge":
            r = _volumetric_g_tess(net)
        elif g_mode == "dmask":
            r = _volumetric_g_dmask(net)
        else:
            raise ValueError(
                f"unknown g_mode={g_mode!r}; expected 'merge' or 'dmask'")
    elif net.ntype == "C":
        r = _volumetric_c_tess(net, apex_mode=apex_mode,
                               face_anchor=face_anchor, fan_out=fan_out)
    else:
        raise ValueError(f"unsupported ntype={net.ntype!r}")
    r["apex_mode"] = apex_mode
    r["face_anchor"] = face_anchor
    return r


def _volumetric_g_dmask(net: Network3D) -> Dict:
    """Delaunay-tet centroid dual restricted to G-edges (honest deficit).

    The faithful, non-degenerate 3D-G ``dual_diamond`` (centroid /
    signed_signed): walk the D-tet tiling of the generators (G's points
    ARE the generators, so the tets are identical to D's), accumulate
    the signed-tet bipyramid per edge, and keep only the Gabriel-edge
    buckets. Non-Gabriel tet edges fall out of ``edge_lookup`` and are
    dropped -- that drop IS the honest deficit (the stripped edges'
    bipyramid volumes), so ``sum(w) = L^3 - deficit`` and ``kite_resid``
    is negative by construction (not an error).

    This is exactly ``_volumetric_delaunay`` evaluated on the G network
    (it is ntype-agnostic and already routes unmatched edges through the
    deficit path); we only relabel the residual as ``deficit_frac``. The
    load-bearing dual vertex is the compact D-tet centroid ``c_T`` -- the
    Gabriel-cell ``c_C`` (the degenerate blob) is never formed. ``c_F``
    cancels pairwise between the two tets sharing a face, so it carries
    no Gabriel flavor either: this is structurally "D's centroid dual
    masked to Gabriel edges", not Gabriel's own dual (the dimensional
    obstruction).
    """
    r = dict(_volumetric_delaunay(net))
    r["deficit_frac"] = -float(r["kite_resid"])  # stripped bipyramid vol / L^3
    r["g_mode"] = "dmask"
    return r


def _volumetric_delaunay(net: Network3D, *,
                         collect_offset: bool = False,
                         fan_out: Optional[List[list]] = None) -> Dict:
    """Delaunay-tet walk -- D only.

    Walks the Delaunay tessellation of ``net.points`` and accumulates the
    signed-tet contribution from each (face, cell) pair onto the matching
    primal edge. Each tet contributes 12 (edge, face) terms (4 faces x 3
    edges); each tet edge appears in 2 of the tet's faces with opposite
    CCW orientations, so the two contributions sum to a non-trivial
    per-tet, per-edge value (not a cancellation).

    Valid for D only -- D's 1-skeleton is the full Delaunay 1-skeleton,
    so every tet edge matches a primal edge and the sum tiles to ``L^3``.
    G calls into ``_volumetric_g_tess`` (different cell complex; see
    module docstring on why the Delaunay-tet walk does NOT tile for G).

    The network's canonical edge direction is ``(min, max)`` of
    ``(edges_left_cell, edges_right_cell)`` -- enforced by
    ``_unique_cell_pairs_3d`` -- so we look up sorted tet-edge keys
    directly and flip the signed-tet sign when the tet's CCW direction
    is reversed relative to canonical.
    """
    L = float(net.L)
    pts = net.points
    n_edges = int(net.n_edges)
    if n_edges == 0:
        return _empty_result()

    el = net.edges_left_cell.astype(np.int64)
    er = net.edges_right_cell.astype(np.int64)
    edge_lookup = {(int(el[ei]), int(er[ei])): ei for ei in range(n_edges)}

    cells = _voropp_compute_3d(pts, L)
    tets = _delaunay_tets_3d(cells)
    n_tets = int(tets.shape[0])

    w = np.zeros(n_edges, dtype=float)
    n_unmatched = 0

    # Per-edge sub-tet accounting. Each w[ei] is a sum of many planar
    # sub-tet (mini-kite) contributions -- one per (tet, face,
    # edge-occurrence) in the edge's bipyramid fan. These expose where in
    # each bipyramid the volume concentrates / cancels:
    #   subtet_abs[e] = sum |contrib|   (cancellation magnitude)
    #   subtet_n[e]   = # sub-tets       subtet_nneg[e] = # with contrib<0
    # |w_e| / subtet_abs[e] is the per-edge coherence in [0,1]: ~1 = clean
    # (well-centered), ~0 = heavy sign cancellation (bipyramid straddles
    # sliver tets). Dropped (non-matched, e.g. non-Gabriel in dmask) terms
    # are accumulated separately as the deficit's sub-tet structure.
    subtet_abs = np.zeros(n_edges, dtype=float)
    subtet_n = np.zeros(n_edges, dtype=np.int64)
    subtet_nneg = np.zeros(n_edges, dtype=np.int64)
    deficit_abs = 0.0
    # Opt-in carrier-offset first moment (see _volumetric_voronoi). For D the
    # sub-tet apex c_T is the simplex centroid (== vertex mean), the reference
    # baseline against which the V cell-COM offset is compared.
    m1_off = np.zeros((n_edges, 3), dtype=float) if collect_offset else None

    for ti in range(n_tets):
        tet = tets[ti]
        p_un = _lift_tet(pts, tet, L)
        sgn = _signed_tet_vol(p_un[0], p_un[1], p_un[2], p_un[3])
        if sgn == 0.0:
            continue
        c_T = p_un.mean(axis=0)
        faces = (_TET_FACES_POS if sgn > 0
                 else tuple((a, c, b) for (a, b, c) in _TET_FACES_POS))

        for (i0, i1, i2) in faces:
            c_F = (p_un[i0] + p_un[i1] + p_un[i2]) / 3.0
            for (la, lb) in ((i0, i1), (i1, i2), (i2, i0)):
                ga = int(tet[la]); gb = int(tet[lb])
                key = (ga, gb) if ga < gb else (gb, ga)
                # Face-CCW-oriented signed tet, in the consumed sign
                # convention (the cone-from-c_T sum is -vol(tet); negate
                # per term so per_edge_weight and the sub-tet diagnostics
                # share one convention -- bit-equivalent to the old
                # ``w = -w`` global flip). Re-signing to canonical edge
                # order (the old ``dir_sign``) would discard the
                # orientation the divergence telescoping needs and
                # collapse sum(w) to 0; the *bucket* is canonical, the
                # term is not.
                contrib = -_signed_tet_vol(c_T, c_F, p_un[la], p_un[lb])
                ei = edge_lookup.get(key)
                if ei is None:
                    n_unmatched += 1
                    deficit_abs += abs(contrib)
                    continue
                w[ei] += contrib
                subtet_abs[ei] += abs(contrib)
                subtet_n[ei] += 1
                if contrib < 0.0:
                    subtet_nneg[ei] += 1
                if m1_off is not None:
                    m1_off[ei] += contrib * (0.25 * (c_T + c_F
                                                     - p_un[la] - p_un[lb]))
                if fan_out is not None:
                    fan_out[ei].append((c_T, c_F))

    kite_sum = float(w.sum())
    n_subtets = int(subtet_n.sum())
    n_neg = int(subtet_nneg.sum())
    coh_mask = subtet_abs > 0.0
    if coh_mask.any():
        coh = np.abs(w[coh_mask]) / subtet_abs[coh_mask]
        coh_p01, coh_p05, coh_p50 = (
            float(np.percentile(coh, 1)),
            float(np.percentile(coh, 5)),
            float(np.percentile(coh, 50)))
    else:
        coh_p01 = coh_p05 = coh_p50 = 1.0
    out = dict(per_edge_weight=w,
               kite_sum=kite_sum,
               kite_resid=kite_sum / (L ** 3) - 1.0,
               n_tets=n_tets,
               n_edges_unmatched=int(n_unmatched),
               centroid_mode="signed_signed",
               # per-edge sub-tet arrays (programmatic/viz; NOT forwarded
               # into the JSON diag by weights_3d -- size n_edges).
               subtet_abs_sum=subtet_abs,
               subtet_n=subtet_n,
               subtet_n_neg=subtet_nneg,
               # JSON-safe scalar summaries.
               n_subtets=n_subtets,
               frac_neg_subtets=(n_neg / n_subtets
                                 if n_subtets else 0.0),
               edge_coherence_p01=coh_p01,
               edge_coherence_p05=coh_p05,
               edge_coherence_p50=coh_p50,
               n_deficit_subtets=int(n_unmatched),
               deficit_abs_sum=float(deficit_abs))
    if m1_off is not None:
        out["per_edge_m1_offset"] = m1_off
    return out


def _volumetric_voronoi(net: Network3D, *,
                        apex_mode: str = "volume_com",
                        face_anchor: str = "vertex_mean",
                        collect_offset: bool = False,
                        fan_out: Optional[List[list]] = None) -> Dict:
    """Voronoi-cell walk -- V only (the COM-centroid placed V dual).

    The circumcentric twin of ``_volumetric_delaunay``: decompose each
    generator's Voronoi cell from its centroid into signed-tet kites over
    the cell's facet-edges (V-edges), accumulating onto the matching
    primal V-edge. Each cell's outward-oriented boundary cones to
    ``+vol(cell)`` for any interior placement (signed_signed
    unconditionality), so ``sum_e w_e = sum_g vol(cell_g) = L^3``. This is
    the placed (centroid) V dual, distinct from the orthogonal ``dual_area``
    bipyramid -- the 3D analog of 2D V ``signed_signed``.

    The voro++ cells + dedup chain ``_build_V3`` ran is recomputed so each
    cell-local vertex maps to its global (deduped) V-vertex index with no
    FP matching; a drift guard trips if the V-vertex count diverges.
    """
    L = float(net.L)
    if net.generators is None:
        raise ValueError(
            "_volumetric_voronoi: net.generators is None -- expected a "
            "build_network_3d() net")
    n_edges = int(net.n_edges)
    if n_edges == 0:
        return _empty_result()

    from .network import _dedup_points_2d
    from .pbc import wrap_to_box

    gens = np.asarray(net.generators, dtype=float)
    cells = _voropp_compute_3d(gens, L)

    cell_verts: List[np.ndarray] = []
    offsets = np.empty(len(cells) + 1, dtype=np.int64)
    offsets[0] = 0
    for ci, cell in enumerate(cells):
        cv = np.asarray(cell.get("vertices", []), dtype=float)
        if cv.ndim != 2 or cv.shape[0] == 0:
            cv = np.zeros((0, 3), dtype=float)
        cell_verts.append(cv)
        offsets[ci + 1] = offsets[ci] + cv.shape[0]
    if offsets[-1] == 0:
        return _empty_result()
    all_verts = wrap_to_box(np.concatenate(cell_verts, axis=0), L)
    uniq_v, flat_to_uv = _dedup_points_2d(all_verts, L,
                                          decimals=10, eps=1e-10)
    if int(uniq_v.shape[0]) != int(net.points.shape[0]):
        raise ValueError(
            f"_volumetric_voronoi: V-vertex recompute ({uniq_v.shape[0]}) "
            f"!= net.points ({net.points.shape[0]}) -- _build_V3 chain drift")

    el = net.edges_left_cell.astype(np.int64)
    er = net.edges_right_cell.astype(np.int64)
    edge_lookup = {}
    for ei in range(n_edges):
        a = int(el[ei]); b = int(er[ei])
        edge_lookup[(a, b) if a < b else (b, a)] = ei

    w = np.zeros(n_edges, dtype=float)
    subtet_abs = np.zeros(n_edges, dtype=float)
    subtet_n = np.zeros(n_edges, dtype=np.int64)
    subtet_nneg = np.zeros(n_edges, dtype=np.int64)
    # Opt-in per-edge signed-mass first moment about the edge midpoint
    # (carrier-offset diagnostic). offset_subtet = sub-tet COM minus the
    # edge midpoint = 0.25*(c_C + c_F + v_a + v_b) - 0.5*(v_a + v_b)
    #              = 0.25*(c_C + c_F - v_a - v_b); a pure displacement, so
    # the cell-local frame == global frame (translation-invariant).
    m1_off = np.zeros((n_edges, 3), dtype=float) if collect_offset else None
    n_unmatched = 0
    deficit_abs = 0.0

    com_vol_sum = 0.0
    for ci, cell in enumerate(cells):
        verts = cell_verts[ci]
        if verts.shape[0] < 4:
            continue
        base = int(offsets[ci])
        c_vm = verts.mean(axis=0)
        # Orient every facet loop outward first (Newell normal along
        # c_F - c_vm) so the cone from any interior apex sums to the
        # cell's outward volume; the shared-facet c_F cones then cancel
        # pairwise between the two incident cells, telescoping sum over
        # cells to +L^3. Orientation is referenced to the vertex mean
        # regardless of apex_mode (interior + stable), so the
        # vertex_mean path is bit-identical to the historical output.
        faces_oriented = []
        for face in cell.get("faces", []):
            fv = face.get("vertices", [])
            m = len(fv)
            if m < 3:
                continue
            c_F = verts[fv].mean(axis=0)
            nrm = np.zeros(3, dtype=float)
            for i in range(m):
                nrm += np.cross(verts[fv[i]] - c_F,
                                verts[fv[(i + 1) % m]] - c_F)
            order = fv if float(np.dot(nrm, c_F - c_vm)) >= 0.0 else fv[::-1]
            # Orientation is decided on the vertex-mean c_F (stable +
            # historical); the anchor upgrade happens AFTER so the
            # vertex_mean path stays bit-identical. Both the volume_com
            # fan and the kite walk consume the upgraded c_F -- the same
            # anchor from both incident cells, so the shared-facet cone
            # cancellation (telescope to +L^3) is untouched.
            if face_anchor == "area_centroid":
                c_F = _face_area_centroid([verts[int(j)] for j in order],
                                          c_F)
            faces_oriented.append((c_F, order))
        if apex_mode == "volume_com":
            # Fan-COM with the SAME face anchors the kite walk uses: the
            # induced triangulation of each shared facet is identical from
            # both cells, so sum_P V_P stays exactly L^3.
            V_P = 0.0
            M1 = np.zeros(3, dtype=float)
            for c_F, order in faces_oriented:
                m = len(order)
                for i in range(m):
                    va = verts[int(order[i])]
                    vb = verts[int(order[(i + 1) % m])]
                    vt = _signed_tet_vol(c_vm, c_F, va, vb)
                    V_P += vt
                    M1 += vt * 0.25 * (c_vm + c_F + va + vb)
            com_vol_sum += V_P
            c_C = M1 / V_P if abs(V_P) > 1e-300 else c_vm
        else:
            c_C = c_vm
        for c_F, order in faces_oriented:
            m = len(order)
            for i in range(m):
                la = int(order[i])
                lb = int(order[(i + 1) % m])
                ua = int(flat_to_uv[base + la])
                ub = int(flat_to_uv[base + lb])
                if ua == ub:
                    continue
                contrib = _signed_tet_vol(c_C, c_F, verts[la], verts[lb])
                ei = edge_lookup.get((ua, ub) if ua < ub else (ub, ua))
                if ei is None:
                    n_unmatched += 1
                    deficit_abs += abs(contrib)
                    continue
                w[ei] += contrib
                subtet_abs[ei] += abs(contrib)
                subtet_n[ei] += 1
                if contrib < 0.0:
                    subtet_nneg[ei] += 1
                if m1_off is not None:
                    m1_off[ei] += contrib * (0.25 * (c_C + c_F
                                                     - verts[la] - verts[lb]))
                if fan_out is not None:
                    fan_out[ei].append((c_C, c_F))

    kite_sum = float(w.sum())
    n_subtets = int(subtet_n.sum())
    n_neg = int(subtet_nneg.sum())
    coh_mask = subtet_abs > 0.0
    if coh_mask.any():
        coh = np.abs(w[coh_mask]) / subtet_abs[coh_mask]
        coh_p01, coh_p05, coh_p50 = (
            float(np.percentile(coh, 1)),
            float(np.percentile(coh, 5)),
            float(np.percentile(coh, 50)))
    else:
        coh_p01 = coh_p05 = coh_p50 = 1.0
    out = dict(per_edge_weight=w,
               kite_sum=kite_sum,
               kite_resid=kite_sum / (L ** 3) - 1.0,
               n_tets=0,
               n_edges_unmatched=int(n_unmatched),
               centroid_mode="signed_signed",
               subtet_abs_sum=subtet_abs,
               subtet_n=subtet_n,
               subtet_n_neg=subtet_nneg,
               n_subtets=n_subtets,
               frac_neg_subtets=(n_neg / n_subtets if n_subtets else 0.0),
               edge_coherence_p01=coh_p01,
               edge_coherence_p05=coh_p05,
               edge_coherence_p50=coh_p50,
               n_deficit_subtets=int(n_unmatched),
               deficit_abs_sum=float(deficit_abs))
    if apex_mode == "volume_com":
        out["com_tiling_resid"] = com_vol_sum / (L ** 3) - 1.0
    if m1_off is not None:
        out["per_edge_m1_offset"] = m1_off
    return out


# =====================================================================
# G-tess: merged-tet cell complex (tet-union-find), no half-face DCEL.
# =====================================================================

def _tri_is_g_face(tri: Tuple[int, int, int], g_edge_set: set) -> bool:
    """A D-triangle is a G-2-face iff all 3 of its edges survive Gabriel.
    ``tri`` is a sorted generator triple; ``g_edge_set`` holds sorted
    surviving generator pairs."""
    a, b, c = tri
    return (((a, b) if a < b else (b, a)) in g_edge_set
            and ((a, c) if a < c else (c, a)) in g_edge_set
            and ((b, c) if b < c else (c, b)) in g_edge_set)


def _gens_in_frame(tet_row: np.ndarray, frame: np.ndarray,
                   tri: Tuple[int, int, int]) -> np.ndarray:
    """Coords of the 3 generators of ``tri`` in a tet's lifted ``frame``
    (4x3), ordered by ``tri`` (already sorted). Both sides of a shared
    wall call this with the same sorted ``tri`` so the rows align
    generator-to-generator."""
    out = np.empty((3, 3), dtype=float)
    for m, g in enumerate(tri):
        for k in range(4):
            if int(tet_row[k]) == g:
                out[m] = frame[k]
                break
    return out


def _g_signed_tet_build(L: float, pts: np.ndarray, tets: np.ndarray,
                        edge_lookup: dict, g_edge_set: set,
                        tri_to_tets: dict,
                        scan_order: np.ndarray) -> Tuple[np.ndarray, dict]:
    """One G-tess weight build for a given tet BFS scan order.

    Returns ``(w, diag)``. The cell complex is the connected components
    of tet-adjacency across *internal walls* (D-triangles with >= 1
    non-Gabriel edge); G-2-faces (all-Gabriel-edge D-triangles) are real
    walls and are NOT merge edges. Each cell is lifted into one
    consistent frame by a BFS spanning tree; non-tree internal walls
    whose fundamental cycle carries a nonzero PBC shift are *promoted*
    to boundary cut walls (re-added removed D-walls -- the homology
    treatment), their non-Gabriel edge contributions redistributed onto
    incident G-edges. ``scan_order`` permutes the seed order so a second
    call yields a different admissible cut set (the gauge).
    """
    n_tets = int(tets.shape[0])
    n_edges = len(edge_lookup)

    # Internal-wall adjacency (skip G-faces -- those are real walls).
    adj: List[List[Tuple[int, Tuple[int, int, int]]]] = [
        [] for _ in range(n_tets)]
    for tri, ts in tri_to_tets.items():
        if len(ts) < 2 or _tri_is_g_face(tri, g_edge_set):
            continue
        for ii in range(len(ts)):
            for jj in range(ii + 1, len(ts)):
                a, b = ts[ii], ts[jj]
                adj[a].append((b, tri))
                adj[b].append((a, tri))

    # BFS components = merged cells, with a consistent tree-lift frame.
    tet_pos: List[np.ndarray] = [None] * n_tets
    cell_of = np.full(n_tets, -1, dtype=np.int64)
    cells_tets: List[List[int]] = []
    for s in scan_order:
        s = int(s)
        if cell_of[s] != -1:
            continue
        cid = len(cells_tets)
        cells_tets.append([])
        tet_pos[s] = _lift_tet(pts, tets[s], L)
        cell_of[s] = cid
        cells_tets[cid].append(s)
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for (v, tri) in adj[u]:
                if cell_of[v] != -1:
                    continue
                qv = _lift_tet(pts, tets[v], L)
                pu = _gens_in_frame(tets[u], tet_pos[u], tri)
                qs = _gens_in_frame(tets[v], qv, tri)
                delta = (pu - qs).mean(axis=0)
                tet_pos[v] = qv + delta
                cell_of[v] = cid
                cells_tets[cid].append(v)
                dq.append(v)
    n_cells = len(cells_tets)

    # c_C per cell: volume-weighted lifted tet-centroid mean (placement
    # is identity-irrelevant under signed_signed; this is a deterministic
    # interior point).
    c_C = np.zeros((n_cells, 3), dtype=float)
    wsum = np.zeros(n_cells, dtype=float)
    for ti in range(n_tets):
        P = tet_pos[ti]
        vol = abs(_signed_tet_vol(P[0], P[1], P[2], P[3]))
        cid = int(cell_of[ti])
        c_C[cid] += vol * P.mean(axis=0)
        wsum[cid] += vol
    for cid in range(n_cells):
        if wsum[cid] > 0.0:
            c_C[cid] /= wsum[cid]
        else:
            allp = np.concatenate([tet_pos[ti] for ti in cells_tets[cid]], 0)
            c_C[cid] = allp.mean(axis=0)

    # Classify internal walls: nonzero-shift (vs the tree-lift) or
    # cross-cell -> promote to boundary cut wall. shift-0 same-cell ->
    # genuine interior chord (skipped by the face loop).
    promoted = set()
    n_internal_walls = 0
    homology_cells = set()
    for tri, ts in tri_to_tets.items():
        if len(ts) < 2 or _tri_is_g_face(tri, g_edge_set):
            continue
        n_internal_walls += 1
        for ii in range(len(ts)):
            for jj in range(ii + 1, len(ts)):
                a, b = ts[ii], ts[jj]
                if int(cell_of[a]) != int(cell_of[b]):
                    promoted.add(tri)
                    continue
                pa = _gens_in_frame(tets[a], tet_pos[a], tri)
                pb = _gens_in_frame(tets[b], tet_pos[b], tri)
                shift = np.round((pa - pb).mean(axis=0) / L)
                if np.any(shift != 0.0):
                    promoted.add(tri)
                    homology_cells.add(int(cell_of[a]))

    # Face loop. Enumerate F of tet ti iff F is a G-face (always; shift-0
    # fins cancel by opposite outward orientation + same c_C) or a
    # promoted cut wall (redistribute its non-G edge weight).
    w = np.zeros(n_edges, dtype=float)
    n_unmatched = 0
    n_g_faces = 0
    redistribute: List[Tuple[Tuple[int, int], float]] = []
    for ti in range(n_tets):
        t = tets[ti]
        P = tet_pos[ti]
        cid = int(cell_of[ti])
        sgn = _signed_tet_vol(P[0], P[1], P[2], P[3])
        if sgn == 0.0:
            continue
        faces = (_TET_FACES_POS if sgn > 0
                 else tuple((a, c, b) for (a, b, c) in _TET_FACES_POS))
        for (i0, i1, i2) in faces:
            tri = tuple(sorted((int(t[i0]), int(t[i1]), int(t[i2]))))
            is_g = _tri_is_g_face(tri, g_edge_set)
            is_prom = (not is_g) and (tri in promoted)
            if not is_g and not is_prom:
                continue
            if is_g:
                n_g_faces += 1
            c_F = (P[i0] + P[i1] + P[i2]) / 3.0
            for (la, lb) in ((i0, i1), (i1, i2), (i2, i0)):
                ga = int(t[la])
                gb = int(t[lb])
                key = (ga, gb) if ga < gb else (gb, ga)
                # Face-CCW-oriented signed tet (no canonical re-signing --
                # see _volumetric_delaunay; the global -1 is undone below).
                val = _signed_tet_vol(c_C[cid], c_F, P[la], P[lb])
                ei = edge_lookup.get(key)
                if ei is not None:
                    w[ei] += val
                else:
                    redistribute.append((key, val))
                    if not is_prom:
                        n_unmatched += 1

    # Redistribute promoted-wall non-G edge weight onto incident G-edges
    # (split evenly among G-edges sharing an endpoint generator). Keeps
    # sum(w) = L^3 exactly with per_edge_weight at length n_edges (3.B.2).
    if redistribute:
        gen_to_eidx: dict = defaultdict(set)
        for (gg, hh), ei in edge_lookup.items():
            gen_to_eidx[gg].add(ei)
            gen_to_eidx[hh].add(ei)
        for (key, val) in redistribute:
            ga, gb = key
            targets = sorted(gen_to_eidx.get(ga, set())
                             | gen_to_eidx.get(gb, set()))
            if targets:
                share = val / len(targets)
                for ei in targets:
                    w[ei] += share

    # Global cone-sign (-1), same as _volumetric_delaunay, so the tiling
    # identity reads sum(w) = +L^3.
    w = -w
    kite_sum = float(w.sum())
    diag = dict(
        kite_sum=kite_sum,
        kite_resid=kite_sum / (L ** 3) - 1.0,
        n_tets=n_tets,
        n_merged_cells=n_cells,
        n_g_faces=int(n_g_faces),
        n_internal_walls=int(n_internal_walls),
        n_homology_cells=int(len(homology_cells)),
        n_synthetic_walls=int(len(promoted)),
        n_edges_unmatched=int(n_unmatched),
    )
    return w, diag


def _volumetric_g_tess(net: Network3D) -> Dict:
    """G-tess volumetric dual via the merged-tet cell complex.

    Substrate = voro++ Delaunay tets of ``net.points``. Codim-1 walls are
    D-triangles; a wall is a real G-2-face iff all 3 of its edges survive
    Gabriel, else it is *internal* and the two tets it separates merge.
    Merged cells are the connected components of tet-adjacency across
    internal walls; each lifts into one consistent frame, with
    torus-wrapping (nonzero-shift) cycles cut by re-adding a removed
    D-wall (the homology treatment). The signed-tet identity then
    tiles ``sum_e w_e = L^3`` for any c_C / c_F placement (signed_signed
    is unconditional). This is the 3D analog of 2D ``planar_face_dual``
    and uses the same substrate-from-voro++ data flow as
    ``_volumetric_delaunay``; V is never consulted.

    ``gauge_spread`` rebuilds with a reversed seed order (a different
    admissible cut set) and reports the max relative per-edge weight
    deviation -- it *measures* that the homology-cut choice is below
    signal rather than assuming it. Zero (up to FP) when there are no
    homology cells.
    """
    L = float(net.L)
    pts = net.points
    n_edges = int(net.n_edges)
    if n_edges == 0:
        return _empty_result_g()

    el = net.edges_left_cell.astype(np.int64)
    er = net.edges_right_cell.astype(np.int64)
    edge_lookup = {(int(el[ei]), int(er[ei])): ei for ei in range(n_edges)}
    g_edge_set = set(edge_lookup.keys())

    cells = _voropp_compute_3d(pts, L)
    tets = _delaunay_tets_3d(cells)
    n_tets = int(tets.shape[0])
    if n_tets == 0:
        return _empty_result_g()

    tri_to_tets: dict = {}
    for ti in range(n_tets):
        t = tets[ti]
        for omit in range(4):
            tri = tuple(sorted(int(t[k]) for k in range(4) if k != omit))
            tri_to_tets.setdefault(tri, []).append(ti)

    fwd = np.arange(n_tets, dtype=np.int64)
    w, diag = _g_signed_tet_build(L, pts, tets, edge_lookup, g_edge_set,
                                  tri_to_tets, fwd)

    # Gauge probe: a reversed seed order gives different spanning trees ->
    # different promoted (cut-wall) representatives for homology cells.
    if diag["n_homology_cells"] > 0:
        w2, _ = _g_signed_tet_build(L, pts, tets, edge_lookup, g_edge_set,
                                    tri_to_tets, fwd[::-1].copy())
        denom = max(float(np.mean(np.abs(w))), 1e-300)
        gauge_spread = float(np.max(np.abs(w - w2)) / denom)
    else:
        gauge_spread = 0.0

    return dict(per_edge_weight=w,
                kite_sum=diag["kite_sum"],
                kite_resid=diag["kite_resid"],
                n_tets=diag["n_tets"],
                n_edges_unmatched=diag["n_edges_unmatched"],
                n_merged_cells=diag["n_merged_cells"],
                n_g_faces=diag["n_g_faces"],
                n_internal_walls=diag["n_internal_walls"],
                n_homology_cells=diag["n_homology_cells"],
                n_synthetic_walls=diag["n_synthetic_walls"],
                gauge_spread=gauge_spread,
                centroid_mode="signed_signed")


def _empty_result_g() -> Dict:
    return dict(per_edge_weight=np.zeros(0, dtype=float),
                kite_sum=0.0, kite_resid=0.0, n_tets=0,
                n_edges_unmatched=0, n_merged_cells=0, n_g_faces=0,
                n_internal_walls=0, n_homology_cells=0,
                n_synthetic_walls=0, gauge_spread=0.0,
                centroid_mode="signed_signed")


def _volumetric_c_tess(net: Network3D, *,
                       apex_mode: str = "volume_com",
                       face_anchor: str = "vertex_mean",
                       fan_out: Optional[List[list]] = None) -> Dict:
    """C-tess Poincare-dual walk -- explicit DEC, centroid, signed-tet kites.

    C is the barycentric (Poincare) dual of the generator Delaunay and is
    *already* a genuine cell complex (no Gabriel filter, no merge, no
    union-find): C-vertices = tet centroids, C-edges <-> Delaunay
    triangles, C-2-faces <-> Delaunay edges (tet-rings), C-3-cells <->
    generators (tet-stars). The per-primal-edge weight is the signed-tet
    bipyramid

        w_e = sum_{d in d*e} sum_{g in d*d} -signed_tet(c_C(g), c_F(d),
                                                        c_T1, c_T2)

    where e is a C-edge (= Delaunay triangle, endpoints c_T1, c_T2 = the
    two tet centroids sharing it), d ranges over the 3 Delaunay-edge
    sides of that triangle (C-2-faces), and g over the 2 endpoint
    generators of each side (C-3-cells) -- 6 sub-kites per edge.

    The "open orientation-sign question" is resolved geometrically, not
    as an abstract eps(D in dd)*eta(d in dg): we walk each C-3-cell g
    (generator tet-star) with its boundary C-2-faces oriented *outward*
    and each face's ring traversed CCW about the outward normal (h-g)
    via the right-hand rule -- exactly ``_volumetric_delaunay``'s
    outward-CCW ``_TET_FACES_POS`` convention lifted one level up the
    complex. Each C-2-face is shared by exactly 2 C-3-cells (its
    Delaunay-edge's 2 endpoint generators); traversed oppositely from
    each, the c_F-cone telescopes per cell so sum over a cell's boundary
    = vol(cell) for *any* c_C/c_F (signed_signed unconditionality), hence
    sum_e w_e = sum_g vol(g) = L^3. Centroid (mean of incident tet
    centroids) is the chosen interior c_C/c_F (placement is
    identity-irrelevant; the mean is a clean interior point).

    PBC: every point entering one signed-tet term is the centroid of a
    tet containing generator g, so a single g-anchored minimum-image
    frame (the C analog of ``_lift_tet``'s tet[0] anchor) keeps each
    term translation-clean; signed_tet is translation-invariant so a
    per-3-cell frame is all the identity needs.

    C is auto-cellular for non-degenerate (e.g. Poisson) point sets --
    every generator's tet-star is a topological ball, so there is no
    deficit, no Gabriel masking, and no homology cut. A tet-star that
    wraps the torus (non-ball / nonzero PBC class) is detected and
    raised loudly.
    """
    L = float(net.L)
    if net.generators is None:
        raise ValueError(
            "_volumetric_c_tess: net.generators is None -- expected a "
            "build_network_3d() net (it sets .generators), not a raw "
            "builder result")
    gens = np.asarray(net.generators, dtype=float)
    n_edges = int(net.n_edges)
    if n_edges == 0:
        return _empty_result()

    # Recompute the exact deterministic chain _build_C3 ran so that tet
    # index ti <-> net.points[ti] (the C-vertex = tet centroid) with no
    # FP matching: both call _delaunay_tets_3d(_voropp_compute_3d(gens)),
    # which returns np.asarray(sorted(set)), on the same net.generators.
    cells = _voropp_compute_3d(gens, L)
    tets = _delaunay_tets_3d(cells)
    n_tets = int(tets.shape[0])
    if n_tets != int(net.points.shape[0]):
        raise ValueError(
            f"_volumetric_c_tess: tet recompute ({n_tets}) != net.points "
            f"({net.points.shape[0]}) -- _build_C3 chain drift")

    # C-edge lookup: sorted *tet*-index pair -> primal C-edge index. For
    # C, edges_{left,right}_cell hold tet indices (the triangle-adjacency
    # from _tet_adjacency), not generators.
    el = net.edges_left_cell.astype(np.int64)
    er = net.edges_right_cell.astype(np.int64)
    edge_lookup = {}
    for ei in range(n_edges):
        a = int(el[ei]); b = int(er[ei])
        edge_lookup[(a, b) if a < b else (b, a)] = ei

    # Generator -> its tet-star (the C-3-cell). Tet membership set per tet
    # for O(1) "does tet ti contain h".
    gen_to_tets: dict = defaultdict(list)
    tet_set: List[frozenset] = [None] * n_tets
    for ti in range(n_tets):
        s = frozenset(int(tets[ti, k]) for k in range(4))
        tet_set[ti] = s
        for gg in s:
            gen_to_tets[gg].append(ti)

    w = np.zeros(n_edges, dtype=float)
    subtet_abs = np.zeros(n_edges, dtype=float)
    subtet_n = np.zeros(n_edges, dtype=np.int64)
    subtet_nneg = np.zeros(n_edges, dtype=np.int64)
    n_unmatched = 0
    deficit_abs = 0.0
    n_nonsimple_rings = 0
    n_c3_cells = 0
    com_vol_sum = 0.0

    span_tol = 0.5 * L  # a non-wrapping local tet-star is compact (< L/2)

    for g, star in gen_to_tets.items():
        if not star:
            continue
        anchor = gens[g]
        # Lift every star tet's centroid into g's minimum-image frame
        # (anchor at generator g; every star tet contains g so this is
        # unambiguous and local).
        cent: dict = {}
        for ti in star:
            acc = np.zeros(3, dtype=float)
            for k in range(4):
                acc += min_image_delta(gens[int(tets[ti, k])] - anchor, L)
            cent[ti] = anchor + acc / 4.0
        cent_arr = np.array([cent[ti] for ti in star])
        span = cent_arr.max(axis=0) - cent_arr.min(axis=0)
        if np.any(span >= span_tol):
            raise ValueError(
                f"_volumetric_c_tess: generator {g} tet-star spans {span} "
                f">= L/2 ({span_tol:.6g}) -- the C-3-cell wraps the torus "
                f"(homology / non-ball). v1 is auto-cellular only; the "
                f"promote/cut machinery is not ported.")
        c_vm = cent_arr.mean(axis=0)
        n_c3_cells += 1

        # Neighbor generators h: a C-2-face (g,h) exists for every
        # Delaunay edge incident to g (every other generator co-occurring
        # with g in some star tet).
        neighbors = set()
        for ti in star:
            neighbors |= tet_set[ti]
        neighbors.discard(g)

        faces_g = []  # (c_F, oriented cycle) per C-2-face of this 3-cell
        for h in neighbors:
            # Tet-ring of Delaunay edge (g,h) = the C-2-face dual to it
            # (the star tets that also contain h).
            ring = [ti for ti in star if h in tet_set[ti]]
            m = len(ring)
            if m < 3:
                # A genuine interior Delaunay-edge ring is a closed cycle
                # of >= 3 tets; fewer => degenerate / boundary artifact.
                n_nonsimple_rings += 1
                continue

            # Build the C-2-face boundary as the *triangle-adjacency*
            # cycle, not an azimuth sort: consecutive ring tets share a
            # (g,h,x) triangle = a real C-edge, so every consumed pair is
            # guaranteed in edge_lookup (exact ddface; azimuth ordering
            # mis-seams near-degenerate rings). Each ring tet has exactly
            # two (g,h)-triangles -> two ring neighbours.
            ringset = set(ring)
            tri_rt: dict = defaultdict(list)
            for ti in ring:
                for x in tet_set[ti]:
                    if x != g and x != h:
                        tri_rt[x].append(ti)
            adj: dict = defaultdict(list)
            simple = True
            for x, ts in tri_rt.items():
                if len(ts) == 2:
                    a, b = ts
                    adj[a].append(b)
                    adj[b].append(a)
                else:
                    simple = False
                    break
            if not simple or any(len(adj[t]) != 2 for t in ring):
                n_nonsimple_rings += 1
                continue
            cyc = [ring[0]]
            prev = -1
            cur = ring[0]
            while True:
                nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
                if nxt == ring[0]:
                    break
                cyc.append(nxt)
                prev, cur = cur, nxt
                if len(cyc) > m:
                    break
            if len(cyc) != m or set(cyc) != ringset:
                n_nonsimple_rings += 1
                continue

            c_F = np.array([cent[ti] for ti in ring]).mean(axis=0)
            # Orient the cycle CCW about the outward normal (h-g) (D's
            # outward-CCW _TET_FACES_POS convention lifted one level --
            # the explicit DEC orientation sign). The adjacency walk
            # fixes the cycle; its *direction* is arbitrary, so reverse
            # it when its (area-weighted) Newell normal opposes (h-g).
            # Newell is used, not an atan2 winding sum: the winding sum
            # is fragile for near-degenerate rings (a ~collinear
            # projection flips its sign and mis-orients one small face);
            # the Newell normal is area-weighted and stable because a
            # C-2-face dual to edge (g,h) is ~perpendicular to (h-g), so
            # its projected area along u_hat is robustly nonzero. From
            # cell h the same face is edge (h,g) -> normal (g-h) -> the
            # opposite traversal with the same c_F, which is exactly the
            # pairwise c_F-cone cancellation that makes the per-cell cone
            # sum telescope to vol(g).
            u = min_image_delta(gens[h] - anchor, L)
            nu = float(np.linalg.norm(u))
            if nu == 0.0:
                n_nonsimple_rings += 1
                continue
            u_hat = u / nu
            nrm = np.zeros(3, dtype=float)
            for i in range(m):
                pa = cent[cyc[i]] - c_F
                pb = cent[cyc[(i + 1) % m]] - c_F
                nrm += np.cross(pa, pb)
            if float(np.dot(nrm, u_hat)) < 0.0:
                cyc.reverse()
            # Anchor upgrade AFTER orientation (Newell stays referenced
            # to the vertex-mean c_F). The ring cents live in g's lifted
            # frame; from cell h the same ring differs by a pure lattice
            # translation and the area centroid commutes with it, so the
            # pairwise c_F-cone cancellation is untouched.
            if face_anchor == "area_centroid":
                c_F = _face_area_centroid([cent[t] for t in cyc], c_F)
            faces_g.append((c_F, cyc))

        if apex_mode == "volume_com":
            # True volume centroid of the C-3-cell via the SAME face fans
            # the kite walk consumes (anchors c_F shared with the h-side
            # walk of every face, so sum_g V_g stays exactly L^3). The
            # decisive 3D bary V/C placement (draft E mechanism iii).
            V_P = 0.0
            M1 = np.zeros(3, dtype=float)
            for c_F, cyc in faces_g:
                mc = len(cyc)
                for i in range(mc):
                    ca = cent[cyc[i]]
                    cb = cent[cyc[(i + 1) % mc]]
                    vt = _signed_tet_vol(c_vm, c_F, ca, cb)
                    V_P += vt
                    M1 += vt * 0.25 * (c_vm + c_F + ca + cb)
            com_vol_sum += V_P
            c_C = M1 / V_P if abs(V_P) > 1e-300 else c_vm
        else:
            c_C = c_vm

        for c_F, cyc in faces_g:
            mc = len(cyc)
            for i in range(mc):
                ta = cyc[i]
                tb = cyc[(i + 1) % mc]
                key = (ta, tb) if ta < tb else (tb, ta)
                ei = edge_lookup.get(key)
                # Sign convention fixed empirically so the closed
                # outward boundary cones to +vol(g) (the C lift of D's
                # leading '-': RH-about-(h-g) is opposite-handed to D's
                # _TET_FACES_POS, so C un-negates). Verified: D unchanged
                # (+L^3), C -> +L^3 exactly.
                contrib = _signed_tet_vol(c_C, c_F, cent[ta], cent[tb])
                if ei is None:
                    n_unmatched += 1
                    deficit_abs += abs(contrib)
                    continue
                w[ei] += contrib
                subtet_abs[ei] += abs(contrib)
                subtet_n[ei] += 1
                if contrib < 0.0:
                    subtet_nneg[ei] += 1
                if fan_out is not None:
                    fan_out[ei].append((c_C, c_F))

    kite_sum = float(w.sum())
    n_subtets = int(subtet_n.sum())
    n_neg = int(subtet_nneg.sum())
    coh_mask = subtet_abs > 0.0
    if coh_mask.any():
        coh = np.abs(w[coh_mask]) / subtet_abs[coh_mask]
        coh_p01, coh_p05, coh_p50 = (
            float(np.percentile(coh, 1)),
            float(np.percentile(coh, 5)),
            float(np.percentile(coh, 50)))
    else:
        coh_p01 = coh_p05 = coh_p50 = 1.0
    return dict(per_edge_weight=w,
                kite_sum=kite_sum,
                kite_resid=kite_sum / (L ** 3) - 1.0,
                n_tets=n_tets,
                n_edges_unmatched=int(n_unmatched),
                centroid_mode="signed_signed",
                subtet_abs_sum=subtet_abs,
                subtet_n=subtet_n,
                subtet_n_neg=subtet_nneg,
                n_subtets=n_subtets,
                frac_neg_subtets=(n_neg / n_subtets
                                  if n_subtets else 0.0),
                edge_coherence_p01=coh_p01,
                edge_coherence_p05=coh_p05,
                edge_coherence_p50=coh_p50,
                n_deficit_subtets=int(n_unmatched),
                deficit_abs_sum=float(deficit_abs),
                n_c3_cells=int(n_c3_cells),
                n_nonsimple_rings=int(n_nonsimple_rings),
                **({"com_tiling_resid": com_vol_sum / (L ** 3) - 1.0}
                   if apex_mode == "volume_com" else {}))


def _lift_tet(pts: np.ndarray, tet: np.ndarray, L: float) -> np.ndarray:
    """Lift a tet's 4 vertices into a single unwrapped frame anchored on
    vertex 0. All downstream centroids / signed-tet calls run in this
    per-tet local frame, so PBC wraps stay invisible to the signed-tet
    identity (which is translation-invariant)."""
    p_un = np.empty((4, 3), dtype=float)
    p0 = pts[int(tet[0])]
    p_un[0] = p0
    for k in range(1, 4):
        p_un[k] = p0 + min_image_delta(pts[int(tet[k])] - p0, L)
    return p_un


def _empty_result() -> Dict:
    return dict(per_edge_weight=np.zeros(0, dtype=float),
                kite_sum=0.0, kite_resid=0.0, n_tets=0,
                n_edges_unmatched=0, centroid_mode="signed_signed")
