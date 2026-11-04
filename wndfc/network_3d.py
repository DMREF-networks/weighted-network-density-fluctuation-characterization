from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .pbc import min_image_delta, wrap_to_box


@dataclass
class Network3D:
    """Per-edge / per-cell geometry for chi_ell(k) computation under 3D PBC.

    Conventions mirror Network2D but in (x, y, z):
      - Coordinates live in ``[0, L)^3``.
      - ``edges_v1`` is wrapped to ``[0, L)^3``; ``edges_v2 - edges_v1`` is the
        canonical (minimum-image) displacement, with ``edges_v2`` possibly
        outside the box by at most one image so the segment is straight.
      - ``cell_volumes`` is the 3D measure (analog of 2D ``cell_areas``).
      - ``cell_star_lengths`` is sum of incident edge lengths per cell.
      - ``points`` are the network vertices (generators for D/V/G; tet
        centroids for C). ``generators`` always retains the input generators.

    Field name ``cell_volumes`` (not ``cell_areas``) marks the dimensional
    distinction; ``weights_3d.py`` reads this field.
    """

    L: float
    ntype: str
    points: np.ndarray
    edges_v1: np.ndarray
    edges_v2: np.ndarray
    edges_lengths: np.ndarray
    edges_left_cell: np.ndarray
    edges_right_cell: np.ndarray
    cell_volumes: np.ndarray
    cell_star_lengths: np.ndarray
    generators: np.ndarray = None
    edges_dual_area: np.ndarray = None  # area of dual Delaunay triangle per V-edge (3 generators). None = not populated for this ntype.
    # Native 3-cell rho_line_inv (V-tess): per-edge incident generator triple,
    # their primary cell volumes, and the two generator-star definitions.
    edges_gen_triple: np.ndarray = None
    gen_volumes: np.ndarray = None
    gen_star_vedge: np.ndarray = None
    gen_star_delaunay: np.ndarray = None
    # Orphaned: raw voro++ ``cells`` (pyvoro dicts) from the primary pass,
    # kept when ``retain_voropp_cells=True``. No current consumer; held for
    # future face-discovery / cell-shape diagnostics.
    voropp_cells: list = None
    # Per-build integrity diagnostic. Populated by
    # _build_V3: circumcenter-merge / spurious-edge counts that quantify the
    # near-lattice (URL) Voronoi-vertex degeneracy. None for builders that
    # don't emit it (D/G/C).
    build_diag: dict = None
    # Delaunay tets (n_tets, 4) backing C: points[i] = centroid(tets[i]) and
    # edges_left/right_cell index into it. Populated by _build_C3; None else.
    tets: np.ndarray = None

    @property
    def cell_like(self):
        """The periodic cell as the pbc helpers take it: the cubic box side (3D has
        no general cell)."""
        return self.L

    @property
    def V(self) -> float:
        return float(self.L) ** 3

    @property
    def n_edges(self) -> int:
        return int(self.edges_lengths.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.cell_volumes.shape[0])

    @property
    def mean_edge_length(self) -> float:
        return float(self.edges_lengths.mean()) if self.n_edges else 0.0

    @property
    def mean_edge_length_sq(self) -> float:
        return float((self.edges_lengths ** 2).mean()) if self.n_edges else 0.0

    def edge_moment(self, theta: float) -> float:
        """Return ``<ell^theta>`` over PBC edges (0.0 on empty network)."""
        if not self.n_edges:
            return 0.0
        return float((self.edges_lengths ** float(theta)).mean())

    @property
    def edges_midpoints(self) -> np.ndarray:
        return 0.5 * (self.edges_v1 + self.edges_v2)

    @property
    def rho_line(self) -> float:
        """Edge-length density ``n_edges * length / L^3``"""
        return float(self.edges_lengths.sum()) / (self.L ** 3)

    @property
    def rho_b(self) -> float:
        """Edge-count density ``n_edges / L^3`` (a.k.a. bond density)."""
        if self.L <= 0:
            return 0.0
        return float(self.n_edges) / (self.L ** 3)

    @property
    def coordination(self) -> float:
        """Mean vertex coordination ``Z = 2 * n_edges / n_unique_vertices``.

        Vertices are grouped by wrapped (mod L) 3D endpoint coordinates
        rounded to 8 decimals.
        """
        if not self.n_edges:
            return 0.0
        L = float(self.L)
        keys = set()
        v1 = self.edges_v1
        v2 = self.edges_v2
        for ei in range(self.n_edges):
            keys.add((round(float(v1[ei, 0]) % L, 8),
                      round(float(v1[ei, 1]) % L, 8),
                      round(float(v1[ei, 2]) % L, 8)))
            keys.add((round(float(v2[ei, 0]) % L, 8),
                      round(float(v2[ei, 1]) % L, 8),
                      round(float(v2[ei, 2]) % L, 8)))
        n_v = len(keys)
        return 2.0 * self.n_edges / n_v if n_v > 0 else 0.0


def _voropp_compute_3d(points: np.ndarray, L: float):
    import pyvoro

    n = int(points.shape[0])
    dispersion = (L * L * L / max(n, 1)) ** (1.0 / 3.0)
    return pyvoro.compute_voronoi(
        points.tolist(),
        limits=[[0.0, float(L)], [0.0, float(L)], [0.0, float(L)]],
        dispersion=float(dispersion),
        periodic=[True, True, True],
    )


def _voropp_cell_volumes(cells) -> np.ndarray:
    return np.asarray([float(c.get("volume", np.nan)) for c in cells], dtype=float)


def _cell_volumes_streaming(points: np.ndarray, L: float) -> np.ndarray:
    n = int(points.shape[0])
    if n == 0:
        return np.zeros(0, dtype=float)
    from .network import _dedup_points_2d
    uniq, inv = _dedup_points_2d(points, L)
    m = int(uniq.shape[0])
    # split a dedup-merged cell's volume across its members (sum rule)
    mult = np.bincount(inv, minlength=m)
    from .streaming import stream_volumes_periodic, AVAILABLE
    if AVAILABLE:
        vols, n_done = stream_volumes_periodic(uniq, L)
        if n_done == m and np.all(np.isfinite(vols)) and np.all(vols > 0.0):
            return (vols / mult)[inv]
    return (_voropp_cell_volumes(_voropp_compute_3d(uniq, L)) / mult)[inv]


def _star_lengths_from_edges(n_cells: int,
                             edges_left: np.ndarray,
                             edges_right: np.ndarray,
                             edges_lengths: np.ndarray) -> np.ndarray:
    star = np.zeros(n_cells, dtype=float)
    np.add.at(star, edges_left, edges_lengths)
    np.add.at(star, edges_right, edges_lengths)
    return star


def _unique_cell_pairs_3d(cells) -> Tuple[np.ndarray, np.ndarray]:
    seen = set()
    pairs = []
    for ci, cell in enumerate(cells):
        for face in cell.get("faces", []):
            nbr = face.get("adjacent_cell")
            if nbr is None or nbr < 0:
                continue
            key = (ci, nbr) if ci < nbr else (nbr, ci)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    if not pairs:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    pairs.sort()
    arr = np.asarray(pairs, dtype=np.int64)
    return arr[:, 0], arr[:, 1]


def _polygon_area_3d(verts: np.ndarray) -> float:
    """Area of a planar convex polygon (ordered 3D vertex loop) via a
    triangle fan from the first vertex. Translation-invariant, so the
    owning cell's frame is fine."""
    n = int(verts.shape[0])
    if n < 3:
        return 0.0
    e1 = verts[1:-1] - verts[0]
    e2 = verts[2:] - verts[0]
    nrm = np.cross(e1, e2).sum(axis=0)
    return 0.5 * float(np.linalg.norm(nrm))


def _unique_cell_pairs_area_3d(cells):
    """``_unique_cell_pairs_3d`` plus the Voronoi facet area per cell-pair
    (the shared polygon's area, the dual face of the Delaunay edge). Facet
    vertices are read in the owning cell's frame (PBC-consistent within one
    cell). Returns ``(left, right, facet_area)``, sorted/aligned as the
    pair-only twin."""
    seen = {}
    for ci, cell in enumerate(cells):
        verts = np.asarray(cell.get("vertices", []), dtype=float)
        for face in cell.get("faces", []):
            nbr = face.get("adjacent_cell")
            if nbr is None or nbr < 0:
                continue
            key = (ci, nbr) if ci < nbr else (nbr, ci)
            if key in seen:
                continue
            fv = face.get("vertices", [])
            seen[key] = (_polygon_area_3d(verts[fv])
                         if len(fv) >= 3 and verts.shape[0] > 0 else 0.0)
    if not seen:
        return (np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=float))
    pairs = sorted(seen.keys())
    arr = np.asarray(pairs, dtype=np.int64)
    areas = np.asarray([seen[k] for k in pairs], dtype=float)
    return arr[:, 0], arr[:, 1], areas


def _build_D3(points: np.ndarray, L: float,
              *, retain_voropp_cells: bool = False) -> Network3D:
    cells = _voropp_compute_3d(points, L)
    left, right, facet_areas = _unique_cell_pairs_area_3d(cells)
    cell_volumes = _voropp_cell_volumes(cells)
    n_edges = int(left.shape[0])
    keep_cells = cells if retain_voropp_cells else None
    if n_edges == 0:
        zeros3 = np.zeros((0, 3), dtype=float)
        return Network3D(L=L, ntype="D", points=points,
                         edges_v1=zeros3, edges_v2=zeros3,
                         edges_lengths=np.zeros(0),
                         edges_left_cell=left, edges_right_cell=right,
                         cell_volumes=cell_volumes,
                         cell_star_lengths=np.zeros(len(cells)),
                         edges_dual_area=np.zeros(0),
                         voropp_cells=keep_cells)
    p_left = points[left]
    p_right = points[right]
    delta = min_image_delta(p_right - p_left, L)
    v2 = p_left + delta
    lengths = np.linalg.norm(delta, axis=1)
    star = _star_lengths_from_edges(len(cells), left, right, lengths)
    return Network3D(L=L, ntype="D", points=points,
                     edges_v1=p_left.copy(), edges_v2=v2,
                     edges_lengths=lengths,
                     edges_left_cell=left, edges_right_cell=right,
                     cell_volumes=cell_volumes, cell_star_lengths=star,
                     edges_dual_area=facet_areas,
                     voropp_cells=keep_cells)


def _gabriel_filter_3d(points: np.ndarray, midpoints: np.ndarray,
                       radii: np.ndarray, L: float) -> np.ndarray:
    """Keep edges whose diametral sphere contains no other generator."""
    from scipy.spatial import cKDTree

    shifts = np.array([(dx, dy, dz)
                       for dx in (-1, 0, 1)
                       for dy in (-1, 0, 1)
                       for dz in (-1, 0, 1)], dtype=float) * L
    tiled = np.concatenate([points + s for s in shifts], axis=0)
    tree = cKDTree(tiled)
    n_orig = points.shape[0]
    keep = np.ones(midpoints.shape[0], dtype=bool)
    eps = 1e-9
    for k_idx, (m, r) in enumerate(zip(midpoints, radii)):
        idxs = tree.query_ball_point(m, r - eps)
        idxs_orig = {idx % n_orig for idx in idxs}
        if len(idxs_orig) > 0:
            keep[k_idx] = False
    return keep


def _build_G3(points: np.ndarray, L: float,
              *, retain_voropp_cells: bool = False) -> Network3D:
    base = _build_D3(points, L, retain_voropp_cells=retain_voropp_cells)
    if base.n_edges == 0:
        return Network3D(L=L, ntype="G", **{**base.__dict__, "ntype": "G"})
    midpoints = wrap_to_box(0.5 * (base.edges_v1 + base.edges_v2), L)
    radii = base.edges_lengths / 2.0
    keep = _gabriel_filter_3d(points, midpoints, radii, L)
    sel = np.where(keep)[0]
    star = _star_lengths_from_edges(base.n_cells,
                                    base.edges_left_cell[sel],
                                    base.edges_right_cell[sel],
                                    base.edges_lengths[sel])
    return Network3D(L=L, ntype="G", points=points,
                     edges_v1=base.edges_v1[sel],
                     edges_v2=base.edges_v2[sel],
                     edges_lengths=base.edges_lengths[sel],
                     edges_left_cell=base.edges_left_cell[sel],
                     edges_right_cell=base.edges_right_cell[sel],
                     cell_volumes=base.cell_volumes, cell_star_lengths=star,
                     edges_dual_area=base.edges_dual_area[sel],
                     voropp_cells=base.voropp_cells)


def _enumerate_voronoi_edges_3d(cells):
    """Walk every cell's face polygons and emit one record per Voronoi edge.

    A Voronoi edge in 3D is shared by exactly 3 generic Voronoi cells. Within
    a single cell ``ci`` the edge appears as a polygon side shared by two of
    ``ci``'s faces; the neighbors of those two faces are the other two cells.
    Dedup key = sorted 3-cell tuple ``(ci, nj, nk)``.

    Yields tuples ``(triple, ci, v_a_local, v_b_local, nj, nk)`` so the caller
    can pull the endpoint coordinates from ``cells[ci]['vertices']`` directly
    (vertex coords within one cell are PBC-consistent).
    """
    seen = set()
    for ci, cell in enumerate(cells):
        faces = cell.get("faces", [])
        v_to_faces: dict = {}
        for fi, face in enumerate(faces):
            for v_local in face.get("vertices", []):
                v_to_faces.setdefault(int(v_local), []).append(fi)
        for fi, face in enumerate(faces):
            fv = face.get("vertices", [])
            m = len(fv)
            if m < 3:
                continue
            for i in range(m):
                va = int(fv[i])
                vb = int(fv[(i + 1) % m])
                shared = set(v_to_faces.get(va, ())) & set(v_to_faces.get(vb, ()))
                shared.discard(fi)
                if len(shared) != 1:
                    continue
                fj = next(iter(shared))
                nj = faces[fi].get("adjacent_cell")
                nk = faces[fj].get("adjacent_cell")
                if nj is None or nk is None or nj < 0 or nk < 0:
                    continue
                if nj == nk or nj == ci or nk == ci:
                    continue
                triple = tuple(sorted((ci, int(nj), int(nk))))
                if triple in seen:
                    continue
                seen.add(triple)
                yield triple, ci, va, vb, int(nj), int(nk)


def _triangle_area_pbc(generators: np.ndarray, triple, L: float) -> float:
    """Area of the Delaunay triangle (3 generators) under PBC.

    Anchored on the first generator and unwrapped via min-image deltas to the
    other two; area = |(p1 - p0) x (p2 - p0)| / 2. Used as the dual-face
    quantity for a 3D Voronoi edge (which is shared by exactly 3 cells).
    """
    p0 = generators[triple[0]]
    d1 = min_image_delta(generators[triple[1]] - p0, L)
    d2 = min_image_delta(generators[triple[2]] - p0, L)
    return 0.5 * float(np.linalg.norm(np.cross(d1, d2)))


def _tet_circumcenter_pbc(generators: np.ndarray, tet: np.ndarray,
                          L: float) -> np.ndarray:
    """Circumcenter of a Delaunay tet under PBC, anchored on its first generator.

    Solves |c - p_i|^2 = |c - p_0|^2 for i = 1..3, which expands to a 3x3
    linear system A c = b where rows of A are 2*(p_i - p_0) and the
    right-hand side is |p_i|^2 - |p_0|^2 (computed in unwrapped coords).
    """
    p0 = generators[tet[0]]
    p_un = np.empty((4, 3), dtype=float)
    p_un[0] = p0
    for k in range(1, 4):
        p_un[k] = p0 + min_image_delta(generators[tet[k]] - p0, L)
    A = 2.0 * (p_un[1:] - p_un[0])
    b = np.einsum("ij,ij->i", p_un[1:], p_un[1:]) - np.dot(p_un[0], p_un[0])
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c = p_un.mean(axis=0)
    return c


def _all_tet_circumcenters_pbc(generators: np.ndarray, tets: np.ndarray,
                               L: float) -> np.ndarray:
    if tets.shape[0] == 0:
        return np.zeros((0, 3), dtype=float)
    out = np.empty((tets.shape[0], 3), dtype=float)
    for i, tet in enumerate(tets):
        out[i] = _tet_circumcenter_pbc(generators, tet, L)
    return wrap_to_box(out, L)


def _triangle_to_tets(tets: np.ndarray) -> dict:
    """Map each Delaunay triangle (sorted 3-cell tuple) -> list of incident tets."""
    tri_map: dict = {}
    for ti, tet in enumerate(tets):
        a, b, c, d = (int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3]))
        for face in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
            key = tuple(sorted(face))
            tri_map.setdefault(key, []).append(ti)
    return tri_map


def _build_V3(points: np.ndarray, L: float,
              *, compute_cell_volumes: bool = True) -> Network3D:
    """V-tessellation under the net_vor convention.

    Each Voronoi edge in 3D borders 3 generator cells, so the per-generator
    "native" path has no canonical 2-of-3 cell pair for rli-family weights.
    We instead use the *secondary* tessellation that always assigns each
    V-edge a unique 2-cell pair: build a periodic Voronoi on the tet
    circumcenters (= the actual Voronoi vertices), and indexes per-cell
    quantities by tet rather than by generator. Each V-edge then connects
    exactly two tets -- the two sharing the V-edge's dual Delaunay triangle.

    This is the convention-free choice in 3D (the same secondary tessellation
    for all four ntypes).

    The single-source voro++ verts + dedup are the always-on structural
    anti-spike here; residual near-lattice degeneracy fails the pipeline's
    tiling check (TilingError).

    ``compute_cell_volumes``: when False, skip the secondary voro++ pass on
    the tet circumcenters (~5x as many points as ``points``). The resulting
    ``cell_volumes`` is filled with zeros instead. Only safe for downstream
    weights that don't read ``cell_volumes`` -- ``uniform``, ``length_power``,
    ``star_inv``, ``dual_area``. Saves the bulk of the V-build memory at large
    N (the secondary tess is the OOM bottleneck, not the primary).
    """
    cells = _voropp_compute_3d(points, L)

    # --- V-vertices from voro++ verts, single source ---
    # Identity AND geometry both come from the voro++ Voronoi vertices
    # (periodically deduped), never re-solved circumcenters. Re-solving
    # circumcenters from _delaunay_tets_3d + _tet_circumcenter_pbc
    # (np.linalg.solve) returns garbage on the near-flat tets that URL's
    # lattice-derived connectivity yields even at FULL displacement (a=1) ->
    # phase e^{ikr} (circumcenter) and edge-delta (voro++ vert) disagree ->
    # broadband low-k spike that vertex dedup cannot catch (garbage
    # scatters, doesn't collide). Sourcing identity from the voro++ verts
    # too gives geometry==identity, no solve, no _delaunay_tets_3d. Poisson:
    # well-separated verts -> _dedup_points_2d is a no-op (np.unique
    # identity, `inverse` identity) -> bit-identical.
    from .network import _dedup_points_2d

    # Gather every per-cell voro++ vertex (global box coords) + flat offsets
    # so (ci, v_local) -> flat row -> deduped global V-vertex id.
    cell_vert_arrs = []
    offsets = np.empty(len(cells) + 1, dtype=np.int64)
    offsets[0] = 0
    for ci, cell in enumerate(cells):
        cv = np.asarray(cell.get("vertices", []), dtype=float)
        if cv.ndim != 2 or cv.shape[0] == 0:
            cv = np.zeros((0, 3), dtype=float)
        cell_vert_arrs.append(cv)
        offsets[ci + 1] = offsets[ci] + cv.shape[0]
    if offsets[-1] == 0:
        all_verts = np.zeros((0, 3), dtype=float)
    else:
        all_verts = wrap_to_box(np.concatenate(cell_vert_arrs, axis=0), L)
    # Tight tolerances at the V-vertex site only: near-coincident Voronoi
    # vertices can be ~1e-9·L apart (~5e-8 at L=50), so the global
    # midpoint-tuned defaults (1e-6 Stage-1, 1e-4 Stage-2) would merge them.
    # 1e-10 is above FP coordinate noise (~1e-13·L) yet below that separation.
    uniq_v, flat_to_uv = _dedup_points_2d(all_verts, L,
                                          decimals=10, eps=1e-10)
    n_uv = int(uniq_v.shape[0])

    edge_records = []   # (triple, ci, va, vb)
    for triple, ci, va, vb, _nj, _nk in _enumerate_voronoi_edges_3d(cells):
        if cell_vert_arrs[ci].shape[0] == 0:
            continue
        edge_records.append((triple, ci, int(va), int(vb)))
    n_edges = len(edge_records)

    if n_edges == 0 or n_uv == 0:
        zeros3 = np.zeros((0, 3), dtype=float)
        net = Network3D(L=L, ntype="V", points=uniq_v,
                        edges_v1=zeros3, edges_v2=zeros3,
                        edges_lengths=np.zeros(0),
                        edges_left_cell=np.zeros(0, dtype=np.int64),
                        edges_right_cell=np.zeros(0, dtype=np.int64),
                        cell_volumes=np.zeros(n_uv),
                        cell_star_lengths=np.zeros(n_uv),
                        edges_dual_area=np.zeros(0))
        net.build_diag = {
            "build": "pyvoro_vert",
            "n_vverts_raw": int(offsets[-1]), "n_uniq_vverts": n_uv,
            "n_v_edges_raw": 0, "n_v_edges": 0,
            "n_selfloop_dropped": 0,
            "frac_selfloop": 0.0, "frac_zero_len_edges": 0.0,
        }
        return net

    edge_left_uv = np.empty(n_edges, dtype=np.int64)
    edge_right_uv = np.empty(n_edges, dtype=np.int64)
    raw_v1 = np.empty((n_edges, 3), dtype=float)
    raw_v2 = np.empty((n_edges, 3), dtype=float)
    edges_dual_area = np.empty(n_edges, dtype=float)
    edge_gen_triple = np.empty((n_edges, 3), dtype=np.int64)
    for ei, (triple, ci, va, vb) in enumerate(edge_records):
        cv = cell_vert_arrs[ci]
        raw_v1[ei] = cv[va]            # same-cell frame -> PBC-consistent
        raw_v2[ei] = cv[vb]
        edge_left_uv[ei] = flat_to_uv[offsets[ci] + va]
        edge_right_uv[ei] = flat_to_uv[offsets[ci] + vb]
        edges_dual_area[ei] = _triangle_area_pbc(points, triple, L)
        edge_gen_triple[ei] = triple

    edges_v1 = wrap_to_box(raw_v1, L)
    delta = min_image_delta(raw_v2 - raw_v1, L)
    edges_v2 = edges_v1 + delta
    lengths = np.linalg.norm(delta, axis=1)

    # Drop spurious edges whose two endpoints deduped onto the SAME global
    # V-vertex (true zero-length / degenerate). Poisson: none -> identity.
    _tol = 1e-9 * max(float(L), 1.0)
    selfloop = (edge_left_uv == edge_right_uv)
    valid = (~selfloop) & (lengths > _tol)
    n_v_edges_raw = int(n_edges)
    if not np.all(valid):
        sel = np.where(valid)[0]
        edges_v1 = edges_v1[sel]
        edges_v2 = edges_v2[sel]
        lengths = lengths[sel]
        edge_left_uv = edge_left_uv[sel]
        edge_right_uv = edge_right_uv[sel]
        edges_dual_area = edges_dual_area[sel]
        edge_gen_triple = edge_gen_triple[sel]

    if compute_cell_volumes:
        v_volumes = _cell_volumes_streaming(uniq_v, L)
    else:
        v_volumes = np.zeros(n_uv, dtype=float)

    star = _star_lengths_from_edges(n_uv, edge_left_uv, edge_right_uv,
                                    lengths)

    # Native 3-cell rho_line_inv ingredients (primary generator cells; no
    # secondary pass). vedge star = incident V-edge length per generator;
    # delaunay star = generator Delaunay-bond length.
    n_gen = len(cells)
    gen_volumes = _voropp_cell_volumes(cells)
    gen_star_vedge = np.zeros(n_gen, dtype=float)
    for _c in range(3):
        np.add.at(gen_star_vedge, edge_gen_triple[:, _c], lengths)
    gen_star_delaunay = np.zeros(n_gen, dtype=float)
    _gl, _gr = _unique_cell_pairs_3d(cells)
    if _gl.shape[0] > 0:
        _dl = np.linalg.norm(min_image_delta(points[_gr] - points[_gl], L),
                             axis=1)
        np.add.at(gen_star_delaunay, _gl, _dl)
        np.add.at(gen_star_delaunay, _gr, _dl)

    net = Network3D(L=L, ntype="V", points=uniq_v,
                    edges_v1=edges_v1, edges_v2=edges_v2,
                    edges_lengths=lengths,
                    edges_left_cell=edge_left_uv,
                    edges_right_cell=edge_right_uv,
                    cell_volumes=v_volumes,
                    cell_star_lengths=star,
                    edges_dual_area=edges_dual_area,
                    edges_gen_triple=edge_gen_triple,
                    gen_volumes=gen_volumes,
                    gen_star_vedge=gen_star_vedge,
                    gen_star_delaunay=gen_star_delaunay)
    # --- V-build integrity diagnostic (schema v2) ------------------
    # Counts are ~0 on Poisson (single-source build is exact there) and
    # quantify the residual on URL. No circumcenter/tet stage anymore.
    n_v_edges = int(lengths.shape[0])
    n_selfloop = int(selfloop.sum())
    net.build_diag = {
        "build": "pyvoro_vert",
        "n_vverts_raw": int(offsets[-1]),
        "n_uniq_vverts": int(n_uv),
        "n_v_edges_raw": n_v_edges_raw,
        "n_v_edges": n_v_edges,
        "n_selfloop_dropped": n_selfloop,
        "frac_selfloop": float(n_selfloop / max(n_v_edges_raw, 1)),
        "frac_zero_len_edges": (float(np.mean(lengths < _tol))
                                if n_v_edges else 0.0),
    }
    return net


def _delaunay_tets_3d(cells, *, diag: dict | None = None) -> np.ndarray:
    """Derive Delaunay tetrahedra from voro++ output.

    Each Voronoi vertex in 3D is generically the meet of 4 cells; equivalently
    every cell-local vertex sits on (typically) 3 faces of that cell, whose
    neighbors give the other 3 generators of one Delaunay tetrahedron.

    ``diag`` (optional out-dict): records ``n_total_vverts`` (cell-local
    vertex incidences scanned) and ``n_degenerate_vverts`` (those on
    != 3 faces and hence skipped -- the cubic 8-cell degeneracy that
    substrate symmetrization removes; 0 on Poisson). Default None keeps
    every existing caller (V/D/G) bit-identical (returned tets unchanged
    either way; the counters only feed the C build_diag).
    """
    tets = set()
    n_total_vv = 0
    n_degen_vv = 0
    for ci, cell in enumerate(cells):
        faces = cell.get("faces", [])
        v_to_faces: dict = {}
        for fi, face in enumerate(faces):
            for v_local in face.get("vertices", []):
                v_to_faces.setdefault(int(v_local), []).append(fi)
        for v_local, face_idxs in v_to_faces.items():
            n_total_vv += 1
            if len(face_idxs) != 3:
                n_degen_vv += 1
                continue
            n1 = faces[face_idxs[0]].get("adjacent_cell")
            n2 = faces[face_idxs[1]].get("adjacent_cell")
            n3 = faces[face_idxs[2]].get("adjacent_cell")
            if any(x is None or x < 0 for x in (n1, n2, n3)):
                continue
            quad = tuple(sorted((ci, int(n1), int(n2), int(n3))))
            if len(set(quad)) != 4:
                continue
            tets.add(quad)
    if diag is not None:
        diag["n_total_vverts"] = int(n_total_vv)
        diag["n_degenerate_vverts"] = int(n_degen_vv)
        diag["frac_degenerate_vverts"] = (
            float(n_degen_vv / n_total_vv) if n_total_vv else 0.0)
    if not tets:
        return np.zeros((0, 4), dtype=np.int64)
    return np.asarray(sorted(tets), dtype=np.int64)


def _tet_centroids_pbc(tets: np.ndarray,
                       points: np.ndarray, L: float) -> np.ndarray:
    if tets.shape[0] == 0:
        return np.zeros((0, 3), dtype=float)
    a = points[tets[:, 0]]
    b = points[tets[:, 1]]
    c = points[tets[:, 2]]
    d = points[tets[:, 3]]
    b_un = a + min_image_delta(b - a, L)
    c_un = a + min_image_delta(c - a, L)
    d_un = a + min_image_delta(d - a, L)
    cents = 0.25 * (a + b_un + c_un + d_un)
    return wrap_to_box(cents, L)


def _tet_adjacency(tets: np.ndarray, *,
                   diag: dict | None = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Two tets are adjacent iff they share 3 of 4 generators (a triangle).

    ``diag`` (optional out-dict): ``n_overshared_tris`` (Delaunay
    triangles in >2 tets -- the cospherical degeneracy) and
    ``n_clique_edges_added`` (pairwise C-edges the clique branch emits
    for them, the spurious-edge count). Both 0 on Poisson; default None
    keeps existing callers bit-identical.
    """
    if tets.shape[0] == 0:
        if diag is not None:
            diag["n_overshared_tris"] = 0
            diag["n_clique_edges_added"] = 0
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    face_to_tet: dict = {}
    for ti, tet in enumerate(tets):
        for omit in range(4):
            face = tuple(sorted(int(tet[k]) for k in range(4) if k != omit))
            face_to_tet.setdefault(face, []).append(ti)
    pairs_set = set()
    n_overshared = 0
    n_clique_added = 0
    for tris in face_to_tet.values():
        if len(tris) == 2:
            t0, t1 = sorted(tris)
            pairs_set.add((t0, t1))
        elif len(tris) > 2:
            n_overshared += 1
            tris_sorted = sorted(tris)
            for i in range(len(tris_sorted)):
                for j in range(i + 1, len(tris_sorted)):
                    pairs_set.add((tris_sorted[i], tris_sorted[j]))
            m = len(tris_sorted)
            n_clique_added += m * (m - 1) // 2
    if diag is not None:
        diag["n_overshared_tris"] = int(n_overshared)
        diag["n_clique_edges_added"] = int(n_clique_added)
    if not pairs_set:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    arr = np.asarray(sorted(pairs_set), dtype=np.int64)
    return arr[:, 0], arr[:, 1]


def _build_C3(points: np.ndarray, L: float,
              *, compute_cell_volumes: bool = True) -> Network3D:
    # The cospherical / cubic 8-cell Voronoi-vertex degeneracy lives in the
    # combinatorics (_delaunay_tets_3d's != 3-face skip, _tet_adjacency's
    # > 2-tet clique); an input that trips it fails the pipeline's tiling
    # check (TilingError).
    _dtet_diag: dict = {}
    _adj_diag: dict = {}
    cells = _voropp_compute_3d(points, L)
    tets = _delaunay_tets_3d(cells, diag=_dtet_diag)
    centroids = _tet_centroids_pbc(tets, points, L)

    left, right = _tet_adjacency(tets, diag=_adj_diag)
    n_edges = int(left.shape[0])

    bdiag = {
        "kind": "C_symdel",
        "n_tets": int(tets.shape[0]),
        "n_c_edges": int(n_edges),
        **_dtet_diag,
        **_adj_diag,
    }

    # The secondary voro++ on the ~d*N tet centroids is the OOM/time
    # bottleneck at large N (the primary pass on the generators is not).
    # ``dual_diamond_planar`` -- and anything not in
    # ``_WEIGHTS_3D_NEED_CELL_VOLUMES`` -- never reads ``cell_volumes``,
    # so ``__main__.py`` passes ``compute_cell_volumes=False`` and this
    # pass is skipped (zeros). Mirrors ``_build_V3``.
    secondary_volumes = np.zeros(centroids.shape[0], dtype=float)
    if compute_cell_volumes and centroids.shape[0] > 0:
        secondary_volumes = _cell_volumes_streaming(centroids, L)

    if n_edges == 0:
        zeros3 = np.zeros((0, 3), dtype=float)
        net = Network3D(L=L, ntype="C", points=centroids,
                        edges_v1=zeros3, edges_v2=zeros3,
                        edges_lengths=np.zeros(0),
                        edges_left_cell=left, edges_right_cell=right,
                        cell_volumes=secondary_volumes,
                        cell_star_lengths=np.zeros(centroids.shape[0]),
                        tets=tets)
        net.build_diag = bdiag
        return net

    p_left = centroids[left]
    p_right = centroids[right]
    delta = min_image_delta(p_right - p_left, L)
    v2 = p_left + delta
    lengths = np.linalg.norm(delta, axis=1)

    star = _star_lengths_from_edges(centroids.shape[0], left, right, lengths)
    net = Network3D(L=L, ntype="C", points=centroids,
                    edges_v1=p_left.copy(), edges_v2=v2,
                    edges_lengths=lengths,
                    edges_left_cell=left, edges_right_cell=right,
                    cell_volumes=secondary_volumes, cell_star_lengths=star,
                    tets=tets)
    net.build_diag = bdiag
    return net


_BUILDERS_3D = {"D": _build_D3, "V": _build_V3, "G": _build_G3, "C": _build_C3}


def build_network_3d(points: np.ndarray, L: float, ntype: str,
                     *, compute_cell_volumes: bool = True,
                     retain_voropp_cells: bool = False) -> Network3D:
    """Construct a Network3D for the requested tessellation type.

    Args:
      points: ``(N, 3)`` array of generator coordinates. Wrapped into ``[0, L)^3``.
      L: side length of the cubic periodic box.
      ntype: one of ``D``, ``V``, ``G``, ``C``.
      compute_cell_volumes: if False, skip the secondary voro++ pass that
        populates ``cell_volumes`` for ntypes that build a secondary tess
        (V on tet circumcenters; C on tet centroids). Set False for runs
        whose weight does not read ``cell_volumes`` (uniform /
        length_power / star_inv / dual_area / dual_diamond_planar) to
        avoid the OOM-prone secondary tess at large N.
      retain_voropp_cells: orphaned. Keeps the raw voro++ ``cells`` output
        on ``Network3D.voropp_cells`` (D/G only). No current consumer.

    Returns:
      Network3D with edge geometry, cell volumes, and star lengths populated.
    """
    if ntype not in _BUILDERS_3D:
        raise ValueError(f"unknown ntype: {ntype!r}; expected one of {sorted(_BUILDERS_3D)}")
    pts = wrap_to_box(np.asarray(points, dtype=float), L)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3); got shape {pts.shape}")
    builder_kwargs = {}
    if ntype in ("V", "C"):
        builder_kwargs["compute_cell_volumes"] = compute_cell_volumes
    if ntype in ("D", "G"):
        builder_kwargs["retain_voropp_cells"] = retain_voropp_cells
    net = _BUILDERS_3D[ntype](pts, float(L), **builder_kwargs)
    net.generators = pts.copy()
    return net
