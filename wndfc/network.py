from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .pbc import (Cell, CellLike, cell_volume, min_image_delta, periodic_pairs,
                  shift_vectors, wrap_to_box)


@dataclass
class Network2D:
    """Per-edge / per-cell geometry for chi_ell(k) computation under 2D PBC.

    Conventions:
      - All coordinates live in the periodic box ``[0, L)^2``.
      - ``edges_v1`` lies in ``[0, L)^2``; ``edges_v2`` is the minimum-image
        partner relative to ``edges_v1`` and may lie outside ``[0, L)`` by at
        most one image so that ``edges_v2 - edges_v1`` is the canonical
        displacement (no wrap inside it).
      - ``cell_areas`` and ``cell_star_lengths`` are indexed by the cell index
        used in ``edges_left_cell`` / ``edges_right_cell``.
      - ``points`` is the network-vertex array (generators for D/V/G;
        centroids for C). ``generators`` is always the input generator
        positions used to build the primary tessellation.
    """

    L: float
    ntype: str
    points: np.ndarray
    edges_v1: np.ndarray
    edges_v2: np.ndarray
    edges_lengths: np.ndarray
    edges_left_cell: np.ndarray
    edges_right_cell: np.ndarray
    cell_areas: np.ndarray
    cell_star_lengths: np.ndarray
    generators: np.ndarray = None
    edges_dual_length: np.ndarray = None  # length of dual Voronoi edge per Delaunay edge (or vice-versa). None = not populated for this ntype.
    triangle_areas: np.ndarray = None  # per-Delaunay-triangle area, indexed by triangle id used in C's edges_left_cell/right_cell. Populated for C; None elsewhere.
    edges_dual_v1: np.ndarray = None  # dual-edge (Voronoi-ridge) endpoint A per primal edge, wrapped to [0, L)^2 like edges_v1. Populated for D/G; None otherwise. IN-MEMORY DIAGNOSTIC ONLY -- io.save_chi_ell never serializes the network, so this adds zero bytes to prod runs.
    edges_dual_v2: np.ndarray = None  # ridge endpoint B = edges_dual_v1 + minimum-image delta (straight dual segment, no wrap inside it). Same convention/lifetime as edges_dual_v1.
    # Native (per-generator) rho_line_inv for V: incident generator pair per
    # V-edge, primary generator cell areas, incident-V-edge generator star.
    edges_gen_pair: np.ndarray = None
    gen_areas: np.ndarray = None
    gen_star_vedge: np.ndarray = None
    # Delaunay triangles (n_tri, 3) backing C: points[i] = centroid(triangles[i])
    # and edges_left/right_cell index into it. Populated by _build_C; None else.
    triangles: np.ndarray = None
    # Builder diagnostics (counts, multiplicities, the degenerate flag read by
    # pipeline.build_weighted_network); in-memory only, never serialized.
    build_diag: dict = None
    # Periodic cell: None = the square box of side L (every existing code path);
    # a general Cell sets L = NaN so that any square-box arithmetic left on that
    # path surfaces instead of silently using the wrong box.
    cell: Cell = None

    @property
    def cell_like(self):
        """The periodic cell as the pbc helpers take it: the float side for a
        square network (so square-box code paths are unchanged), else the Cell."""
        return self.L if self.cell is None else self.cell

    @property
    def V(self) -> float:
        return cell_volume(self.cell_like)

    @property
    def n_edges(self) -> int:
        return int(self.edges_lengths.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.cell_areas.shape[0])

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
        """Edge-length density ``sum of edge lengths / V``"""
        return float(self.edges_lengths.sum()) / self.V

    @property
    def rho_b(self) -> float:
        """Edge-count density ``n_edges / V`` (a.k.a. bond density)."""
        if not self.V > 0:
            return 0.0
        return float(self.n_edges) / self.V

    @property
    def coordination(self) -> float:
        """Mean vertex coordination ``Z = 2 * n_edges / n_unique_vertices``.

        Vertices are grouped by wrapped (mod L) endpoint coordinates rounded
        to 8 decimals.
        """
        if not self.n_edges:
            return 0.0
        keys = set()
        v1 = wrap_to_box(self.edges_v1, self.cell_like)
        v2 = wrap_to_box(self.edges_v2, self.cell_like)
        for ei in range(self.n_edges):
            keys.add((round(float(v1[ei, 0]), 8),
                      round(float(v1[ei, 1]), 8)))
            keys.add((round(float(v2[ei, 0]), 8),
                      round(float(v2[ei, 1]), 8)))
        n_v = len(keys)
        return 2.0 * self.n_edges / n_v if n_v > 0 else 0.0


def _voropp_compute(points: np.ndarray, L: float):
    import pyvoro

    n = int(points.shape[0])
    dispersion = (L * L / max(n, 1)) ** 0.5
    # pyvoro's 2D call runs voro++ on a slab of fixed absolute half-height
    # (z_height = 0.5).  When the mean spacing is far below that, the block
    # decomposition degenerates and the build becomes super-linear (a unit
    # box with 8192 points takes hours).  Tie the slab to the spacing in that
    # regime only; O(1)-spacing inputs keep their exact floating-point path.
    kw = {"z_height": 0.5 * float(dispersion)} if dispersion < 0.5 else {}
    return pyvoro.compute_2d_voronoi(
        points.tolist(),
        limits=[[0.0, float(L)], [0.0, float(L)]],
        dispersion=float(dispersion),
        periodic=[True, True],
        **kw,
    )


def _polygon_area(verts: np.ndarray) -> float:
    if verts.shape[0] < 3:
        return 0.0
    x = verts[:, 0]
    y = verts[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _voropp_cell_areas(cells) -> np.ndarray:
    return np.asarray([float(c.get("volume", np.nan)) for c in cells], dtype=float)


def _polygon_centroid(verts: np.ndarray) -> np.ndarray:
    """Signed-area centroid of a 2D polygon (vertices in CW or CCW order).

    voro++ returns cell vertices in unwrapped absolute coords forming a
    single continuous polygon, so this is well-defined even for cells that
    cross the periodic boundary.
    """
    if verts.shape[0] < 3:
        return verts.mean(axis=0) if verts.shape[0] > 0 else np.zeros(2)
    x = verts[:, 0]; y = verts[:, 1]
    x1 = np.roll(x, -1); y1 = np.roll(y, -1)
    cross = x * y1 - x1 * y
    a2 = float(cross.sum())                       # 2 * signed area
    if abs(a2) < 1e-30:
        return verts.mean(axis=0)
    cx = float(((x + x1) * cross).sum()) / (3.0 * a2)
    cy = float(((y + y1) * cross).sum()) / (3.0 * a2)
    return np.array([cx, cy], dtype=float)


def midpt_secondary_voronoi(points: np.ndarray, L: float):
    """Build the secondary Voronoi of a set of (already-deduped) points and
    return ``(areas, centroids)``.

    Used for the segment-Voronoi-cell HU diagnostic: per-edge offset
    Delta_e = midpt_e - centroid(Omega_e) controls the Class-I exponent
    (alpha=4 if Delta=0, else alpha=2).
    """
    cells = _voropp_compute(points, L)
    areas = np.asarray([float(c.get("volume", np.nan)) for c in cells],
                       dtype=float)
    cents = np.empty((len(cells), 2), dtype=float)
    for i, c in enumerate(cells):
        v = np.asarray(c.get("vertices", []), dtype=float)
        cents[i] = _polygon_centroid(v) if v.shape[0] >= 3 else points[i]
    return areas, cents


def _star_lengths_from_edges(n_cells: int,
                             edges_left: np.ndarray,
                             edges_right: np.ndarray,
                             edges_lengths: np.ndarray) -> np.ndarray:
    star = np.zeros(n_cells, dtype=float)
    np.add.at(star, edges_left, edges_lengths)
    np.add.at(star, edges_right, edges_lengths)
    return star


def _unique_cell_pairs(cells, L: float = None):
    """Return (left, right, dual_lengths, dual_v1, dual_v2) per unique
    cell-pair edge.

    ``dual_v1``/``dual_v2`` are the two Voronoi-ridge endpoint coords (the
    same ``cell_verts[fv[0/1]]`` already used for the length), raw in cell
    ``ci``'s frame -- the caller wraps them. NaN rows for faces with < 2
    vertices (dual_len 0 there too). These are computed regardless; we just
    no longer discard them. Never serialized (see Network2D field docs)."""
    seen = {}
    for ci, cell in enumerate(cells):
        cell_verts = np.asarray(cell.get("vertices", []), dtype=float)
        for face in cell.get("faces", []):
            nbr = face.get("adjacent_cell")
            if nbr is None or nbr < 0:
                continue
            key = (ci, nbr) if ci < nbr else (nbr, ci)
            if key in seen:
                continue
            fv = face.get("vertices", [])
            if len(fv) >= 2 and cell_verts.shape[0] > 0 and L is not None:
                v1 = cell_verts[fv[0]]
                v2 = cell_verts[fv[1]]
                d = min_image_delta(v2 - v1, float(L))
                dual_len = float(np.linalg.norm(d))
            else:
                dual_len = 0.0
                v1 = np.full(2, np.nan)
                v2 = np.full(2, np.nan)
            seen[key] = (dual_len, np.asarray(v1, dtype=float),
                         np.asarray(v2, dtype=float))
    pairs = sorted(seen.keys())
    if not pairs:
        return (np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=float),
                np.zeros((0, 2), dtype=float),
                np.zeros((0, 2), dtype=float))
    arr = np.asarray(pairs, dtype=np.int64)
    duals = np.asarray([seen[k][0] for k in pairs], dtype=float)
    dv1 = np.asarray([seen[k][1] for k in pairs], dtype=float)
    dv2 = np.asarray([seen[k][2] for k in pairs], dtype=float)
    return arr[:, 0], arr[:, 1], duals, dv1, dv2


def _stream_unique_pairs(points: np.ndarray, L: float):
    """Streaming voro++ 2D path -> (cell_areas, left, right, dual_lengths).

    Returns ``None`` if the streaming extension isn't built (caller falls
    back to pyvoro). On success, ``left`` and ``right`` are the deduplicated
    sorted (left < right) generator-id pairs spanning every Voronoi edge,
    and ``dual_lengths[i]`` is the length of the Voronoi face (perpendicular
    bisector segment) shared by ``left[i]`` and ``right[i]``.
    """
    try:
        from .streaming import stream_network_periodic_2d, AVAILABLE_2D
    except ImportError:
        return None
    if not AVAILABLE_2D:
        return None
    areas, e_cell, e_nbr, v1, v2, _ = stream_network_periodic_2d(
        points, L)
    if e_cell.shape[0] == 0:
        return (areas,
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=float),
                np.zeros((0, 2), dtype=float),
                np.zeros((0, 2), dtype=float))
    keep = (e_cell < e_nbr) & (e_nbr >= 0)
    left = e_cell[keep].astype(np.int64, copy=False)
    right = e_nbr[keep].astype(np.int64, copy=False)
    v1k = np.asarray(v1, dtype=float)[keep]
    v2k = np.asarray(v2, dtype=float)[keep]
    delta = min_image_delta(v2k - v1k, L)
    dual = np.linalg.norm(delta, axis=1)
    order = np.lexsort((right, left))
    # v1/v2 are the Voronoi-ridge endpoints already produced by the
    # streaming extension -- returned, not recomputed (zero added cost).
    return (areas, left[order], right[order], dual[order],
            v1k[order], v2k[order])


def _dedup_points_2d(points: np.ndarray, L: CellLike, decimals: int = 6,
                     eps: float | None = None):
    """Wrap into the box and dedupe near-coincident points (2 stages).

    voro++ aborts ("Duplicate:") when two input particles share a block at
    sub-grid spacing. Cocircular Delaunay triangles produce identical
    triangle centroids in C-tess, degenerate Voronoi vertices produce
    identical circumcenters in V-tess, and dense edge-midpoint sets
    (midpt_sec_area at N>=1e5: ~1.7M midpoints) produce pairs a few 1e-5
    apart. Returns ``(unique_points, inverse)`` where ``inverse`` maps each
    input row to its unique-row index, so a secondary scalar (areas /
    volumes) computed on ``unique_points`` can be expanded back to the
    original ordering via ``scalar[inverse]``.

    Stage 1 -- round to ``decimals`` (absolute, 1e-6 box units) + np.unique.
    Fast, removes exact / grid-coincident dups, but CANNOT merge a pair
    straddling a rounding boundary, so near-coincident survivors slip
    through and voro++ still fatal-exits at scale.

    Stage 2 -- residual periodic KDTree merge: union-find any Stage-1
    survivors within ``eps`` (absolute box units, default 1e-4 >> voro++'s
    ~1e-5 duplicate tolerance, << any real edge length). No-op when no
    pair is that close, so well-separated inputs (typical 2D D/V/G/C) are
    unaffected; only sets that would otherwise abort voro++ change.

    Env overrides: ``WNDFC_DEDUP_DECIMALS`` (Stage 1; raise for tighter
    pyvoro parity), ``WNDFC_DEDUP_EPS`` (Stage 2; 0 disables Stage 2).
    """
    if points.shape[0] == 0:
        return points, np.zeros(0, dtype=np.int64)
    import os as _os
    # Explicit kwargs (caller-supplied via the function signature) win over
    # env vars — site-specific tolerances (e.g. tight 1e-10 at the V-vertex
    # dedup, which keeps near-coincident vertices apart) must not be erased by a global env
    # override tuned for a different site (midpoint dedup wants ~1e-4).
    _caller_decimals = decimals != 6
    _override = _os.environ.get("WNDFC_DEDUP_DECIMALS")
    if _override is not None and not _caller_decimals:
        try:
            decimals = int(_override)
        except ValueError:
            pass
    # --- Stage 1: round + unique (unchanged; preserves prior parity) ------
    keys = np.round(wrap_to_box(points, L), decimals=decimals)
    _, uniq_idx, inverse = np.unique(keys, axis=0,
                                     return_index=True, return_inverse=True)
    order = np.argsort(uniq_idx)
    remap = np.empty_like(order)
    remap[order] = np.arange(order.size)
    inv1 = remap[inverse].astype(np.int64, copy=False)
    pts1 = points[uniq_idx[order]]

    # --- Stage 2: residual tolerance merge (periodic KDTree union-find) ---
    _caller_eps = eps is not None
    if eps is None:
        eps = 1e-4
    _epso = _os.environ.get("WNDFC_DEDUP_EPS")
    if _epso is not None and not _caller_eps:
        try:
            eps = float(_epso)
        except ValueError:
            pass
    n1 = int(pts1.shape[0])
    if eps <= 0.0 or n1 < 2:
        return pts1, inv1
    from scipy.spatial import cKDTree
    if isinstance(L, Cell) and not L.is_square:
        pairs = periodic_pairs(pts1, L, eps)
    else:
        side = L.L if isinstance(L, Cell) else float(L)
        wp = wrap_to_box(pts1, side)
        # cKDTree(boxsize=L) needs coords strictly in [0, L); clamp the
        # wrap_to_box == L float edge (np.mod can return exactly L).
        wp = np.clip(wp, 0.0, np.nextafter(side, 0.0))
        try:
            pairs = cKDTree(wp, boxsize=side).query_pairs(
                eps, output_type='ndarray')
        except Exception:
            pairs = cKDTree(wp).query_pairs(eps, output_type='ndarray')
    if len(pairs) == 0:
        return pts1, inv1
    parent = np.arange(n1, dtype=np.int64)

    def _find(a: int) -> int:
        r = a
        while parent[r] != r:
            r = parent[r]
        while parent[a] != r:
            parent[a], a = r, parent[a]
        return r

    for a, b in pairs:
        ra, rb = _find(int(a)), _find(int(b))
        if ra != rb:
            parent[rb] = ra
    # Vectorized full path-compression to roots (1-2 passes; mostly self).
    while True:
        nxt = parent[parent]
        if np.array_equal(nxt, parent):
            break
        parent = nxt
    _uroots, inv2 = np.unique(parent, return_inverse=True)
    inv2 = inv2.astype(np.int64, copy=False)
    # Representative = first Stage-1 row landing in each cluster (stable).
    first = np.full(_uroots.size, -1, dtype=np.int64)
    for i in range(n1):
        c = inv2[i]
        if first[c] < 0:
            first[c] = i
    return pts1[first], inv2[inv1].astype(np.int64, copy=False)


def _stream_cell_areas_2d(points: np.ndarray, L: float):
    try:
        from .streaming import stream_network_periodic_2d, AVAILABLE_2D
    except ImportError:
        return None
    if not AVAILABLE_2D:
        return None
    areas, _ec, _en, _v1, _v2, _ = stream_network_periodic_2d(points, L)
    return areas


def _unique_sorted_triangles(tri: np.ndarray) -> np.ndarray:
    """Distinct rows of an ``(m, 3)`` int array in lexicographic order, int64."""
    if tri.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int64)
    tri = tri.astype(np.int64, copy=False)
    tri = tri[np.lexsort((tri[:, 2], tri[:, 1], tri[:, 0]))]
    keep = np.ones(tri.shape[0], dtype=bool)
    keep[1:] = np.any(tri[1:] != tri[:-1], axis=1)
    return tri[keep]


def _stream_triangles_from_records(e_cell: np.ndarray, e_nbr: np.ndarray,
                                   v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Delaunay triangles from the streamed cell-edge records, by topology only.

    The helper emits each cell's edges contiguously in boundary order, so two
    consecutive records of one cell meet at a Voronoi vertex iff the second
    starts where the first ends (``v2[j] == v1[j+1]``, the same stored double;
    the last record wraps to the first).  That vertex is the circumcenter of
    the triangle (cell, nbr_j, nbr_j+1).  Same rule as the pyvoro path
    (``_delaunay_triangles``): no coordinate rounding, so two distinct voro++
    vertices stay distinct however close they are, and a vertex voro++ itself
    merged (k > 3 cocircular generators) yields k triangles on both paths --
    the ``n_tri != 2N`` condition the pipeline's degeneracy check reads.
    Rows sorted, unique, int64.
    """
    e_cell = e_cell.astype(np.int64, copy=False)
    e_nbr = e_nbr.astype(np.int64, copy=False)
    n = int(e_cell.shape[0])
    if n == 0:
        return np.zeros((0, 3), dtype=np.int64)
    nxt = np.arange(1, n + 1, dtype=np.int64)
    new_block = np.flatnonzero(e_cell[1:] != e_cell[:-1]) + 1
    starts = np.concatenate(([0], new_block))
    ends = np.concatenate((new_block, [n]))
    nxt[ends - 1] = starts
    shared = (v2[:, 0] == v1[nxt, 0]) & (v2[:, 1] == v1[nxt, 1])
    na = e_nbr
    nb = e_nbr[nxt]
    ok = shared & (na >= 0) & (nb >= 0) & (na != nb)
    tri = np.sort(np.stack([e_cell[ok], na[ok], nb[ok]], axis=1), axis=1)
    tri = tri[(tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2])]
    return _unique_sorted_triangles(tri)


def _stream_delaunay_triangles_2d(points: np.ndarray, L: float):
    """Delaunay triangles + per-generator cell areas from streaming voro++
    (``None`` if the 2D extension is unbuilt).  Triangles follow the exact
    topological rule of ``_stream_triangles_from_records``, identical to the
    pyvoro path; a near-cocircular quad whose two Voronoi vertices are closer
    than any coordinate tolerance is still two triangles.
    """
    try:
        from .streaming import stream_network_periodic_2d, AVAILABLE_2D
    except ImportError:
        return None
    if not AVAILABLE_2D:
        return None
    areas, e_cell, e_nbr, v1, v2, _ = stream_network_periodic_2d(points, L)
    return _stream_triangles_from_records(e_cell, e_nbr, v1, v2), areas


def _build_D(points: np.ndarray, L: float) -> Network2D:
    streamed = _stream_unique_pairs(points, L)
    if streamed is not None:
        cell_areas, left, right, dual, dv1, dv2 = streamed
    else:
        cells = _voropp_compute(points, L)
        left, right, dual, dv1, dv2 = _unique_cell_pairs(cells, L=L)
        cell_areas = _voropp_cell_areas(cells)
    n_cells = int(cell_areas.shape[0])
    n_edges = int(left.shape[0])
    if n_edges == 0:
        zeros = np.zeros((0, 2), dtype=float)
        return Network2D(L=L, ntype="D", points=points,
                         edges_v1=zeros, edges_v2=zeros,
                         edges_lengths=np.zeros(0),
                         edges_left_cell=left, edges_right_cell=right,
                         cell_areas=cell_areas,
                         cell_star_lengths=np.zeros(n_cells),
                         edges_dual_length=np.zeros(0),
                         edges_dual_v1=np.zeros((0, 2)),
                         edges_dual_v2=np.zeros((0, 2)))
    p_left = points[left]
    p_right = points[right]
    delta = min_image_delta(p_right - p_left, L)
    v2 = p_left + delta
    lengths = np.linalg.norm(delta, axis=1)
    star = _star_lengths_from_edges(n_cells, left, right, lengths)
    # Voronoi-ridge endpoints in the edges_v1/edges_v2 convention: v1
    # wrapped to the box, v2 = v1 + min-image delta (straight dual segment,
    # PBC already resolved -- so kite quads built from these are simple).
    d_v1 = wrap_to_box(dv1, L)
    d_v2 = d_v1 + min_image_delta(dv2 - dv1, L)
    if streamed is not None:
        # Ridge sanity check: some streaming binaries emit vertex pairs in
        # voro++ STORAGE order, not ring order, so |v2 - v1| is NOT the
        # ridge length for most cells. A true Voronoi ridge is perpendicular
        # to its generator bond -- cheap to verify in bulk. Warn loudly;
        # data left intact (rebuild wndfc/streaming to fix at the source).
        t = d_v2 - d_v1
        t_len = np.linalg.norm(t, axis=1)
        ok_len = t_len > 1e-12
        if ok_len.any():
            cosang = np.abs(np.einsum("ei,ei->e", t, delta)) \
                / np.maximum(t_len * lengths, 1e-300)
            frac_bad = float(np.mean((cosang > 1e-6) & ok_len))
            if frac_bad > 1e-3:
                print(f"    [wndfc D stream] WARNING: {frac_bad:.1%} "
                      "of streamed Voronoi ridges are NOT perpendicular to "
                      "their generator bonds -- stale streaming .so with "
                      "storage-order vertex pairs. edges_dual_length / "
                      "edges_dual_v1/v2 (dual_area, dual_diamond_tiled, "
                      "cell_area_per_gen, kite diagnostics) are UNRELIABLE "
                      "on this build. Rebuild the streaming extension "
                      "(wndfc/streaming/setup.py build_ext --inplace) "
                      "or set WNDFC_NO_STREAM=1.")
    return Network2D(L=L, ntype="D", points=points,
                     edges_v1=p_left.copy(), edges_v2=v2,
                     edges_lengths=lengths,
                     edges_left_cell=left, edges_right_cell=right,
                     cell_areas=cell_areas, cell_star_lengths=star,
                     edges_dual_length=dual,
                     edges_dual_v1=d_v1, edges_dual_v2=d_v2)


def _build_G(points: np.ndarray, L: float) -> Network2D:
    base = _build_D(points, L)
    if base.n_edges == 0:
        return Network2D(L=L, ntype="G", **{**base.__dict__, "ntype": "G"})
    midpoints = wrap_to_box(0.5 * (base.edges_v1 + base.edges_v2), L)
    radii = base.edges_lengths / 2.0
    keep = _gabriel_filter(points, midpoints, radii, L)
    sel = np.where(keep)[0]
    star = _star_lengths_from_edges(base.n_cells,
                                    base.edges_left_cell[sel],
                                    base.edges_right_cell[sel],
                                    base.edges_lengths[sel])
    base_dual = base.edges_dual_length
    g_dual = base_dual[sel] if base_dual is not None else None
    g_dv1 = base.edges_dual_v1[sel] if base.edges_dual_v1 is not None else None
    g_dv2 = base.edges_dual_v2[sel] if base.edges_dual_v2 is not None else None
    return Network2D(L=L, ntype="G", points=points,
                     edges_v1=base.edges_v1[sel],
                     edges_v2=base.edges_v2[sel],
                     edges_lengths=base.edges_lengths[sel],
                     edges_left_cell=base.edges_left_cell[sel],
                     edges_right_cell=base.edges_right_cell[sel],
                     cell_areas=base.cell_areas, cell_star_lengths=star,
                     edges_dual_length=g_dual,
                     edges_dual_v1=g_dv1, edges_dual_v2=g_dv2)


def _gabriel_filter(points: np.ndarray, midpoints: np.ndarray,
                    radii: np.ndarray, L: CellLike) -> np.ndarray:
    from scipy.spatial import cKDTree

    shifts = shift_vectors(np.array([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)],
                                    dtype=float), L)
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


def _build_V(points: np.ndarray, L: float,
             *, compute_cell_areas: bool = True) -> Network2D:
    """V-tessellation under the net_vor convention.

    Each Voronoi edge in 2D borders 2 generator cells, but for rli-family
    weights we use the *secondary* tessellation indexed per-triangle (V-vertex)
    rather than per-generator, mirroring the 3D _build_V3 construction.

    Steps:
      1. Build primary voro++ on generators, collect V-edges.
      2. Re-derive Delaunay triangles from voro++ output.
      3. Compute each triangle's circumcenter under PBC (= V-vertex).
      4. Run voro++ on the circumcenters -> per-triangle cell areas.
      5. Map each V-edge to its 2 incident triangles via the Delaunay edge
         connecting the V-edge's two original generator cells.
      6. Star lengths computed per-triangle from incident V-edge half-lengths.
      7. cell_areas / edges_left_cell / edges_right_cell are all per-triangle
         indexed (Network2D.cell_areas size = n_triangles, not n_generators).

    ``compute_cell_areas=False`` skips step 4 (the secondary voro++ pass on
    circumcenters) and returns ``cell_areas = zeros(n_tri)`` -- for weights
    that never read ``cell_areas`` (uniform / length_power / dual_area /
    dual_diamond_planar). 2D mirror of build_network_3d's
    ``compute_cell_volumes``; ``cell_star_lengths`` is built from edges and
    is unaffected, so star-based weights still work when this is off.
    """
    cells = _voropp_compute(points, L)

    edge_records = []
    for ci, cell in enumerate(cells):
        cell_verts = np.asarray(cell.get("vertices", []), dtype=float)
        for face in cell.get("faces", []):
            nbr = face.get("adjacent_cell")
            if nbr is None or nbr < 0:
                continue
            if nbr <= ci:
                continue
            fv = face.get("vertices", [])
            if len(fv) < 2:
                continue
            v1 = cell_verts[fv[0]]
            v2 = cell_verts[fv[1]]
            edge_records.append((ci, nbr, v1, v2))

    triangles = _delaunay_triangles(cells)
    n_tri = int(triangles.shape[0])
    n_edges = len(edge_records)

    if n_edges == 0 or n_tri == 0:
        zeros = np.zeros((0, 2), dtype=float)
        return Network2D(L=L, ntype="V", points=np.zeros((0, 2)),
                         edges_v1=zeros, edges_v2=zeros,
                         edges_lengths=np.zeros(0),
                         edges_left_cell=np.zeros(0, dtype=np.int64),
                         edges_right_cell=np.zeros(0, dtype=np.int64),
                         cell_areas=np.zeros(n_tri),
                         cell_star_lengths=np.zeros(n_tri))

    circumcenters = _all_triangle_circumcenters_pbc_2d(points, triangles, L)
    # Ancillary sites = the distinct Voronoi vertices.  A generic vertex is the
    # circumcenter of exactly one Delaunay triangle; a k-fold degenerate vertex
    # (k > 3 cocircular generators) is one site carried by several coincident
    # circumcenters, merged here with its multiplicity.  Every V-edge is keyed
    # by its two endpoint sites, so rho_line_inv tiles (sum_e w_e ell_e = L^2)
    # on degenerate inputs exactly as on generic ones.
    # Topology merge: coincident vertices only (FP noise), so a short V-edge
    # between two nearby vertices stays an edge.
    sites, inverse = _dedup_points_2d(circumcenters, L, decimals=12, eps=1e-9 * L)
    n_sites = int(sites.shape[0])
    mult = np.bincount(inverse, minlength=n_sites)
    if compute_cell_areas:
        # The secondary voro++ needs its input free of near-duplicates (its
        # duplicate tolerance): merge at the standard radius and split each
        # merged cell's area back over its sites.
        uniq2, inv2 = _dedup_points_2d(sites, L)
        mult2 = np.bincount(inv2, minlength=uniq2.shape[0])
        streamed = _stream_cell_areas_2d(uniq2, L)
        if streamed is not None:
            areas2 = np.asarray(streamed, dtype=float)
        else:
            areas2 = _voropp_cell_areas(_voropp_compute(uniq2, L))
        site_areas = (areas2 / mult2)[inv2]
    else:
        # Gated off: the chosen weight does not read cell_areas.  Star lengths
        # are built from edges below, so star-based weights are unaffected.
        site_areas = np.zeros(n_sites, dtype=float)

    raw_v1 = np.empty((n_edges, 2), dtype=float)
    raw_v2 = np.empty((n_edges, 2), dtype=float)
    raw_dual = np.empty(n_edges, dtype=float)  # |gen[ci] - gen[nbr]| (PBC)
    edge_gen_pair = np.empty((n_edges, 2), dtype=np.int64)
    for ei, (ci, nbr, v1, v2) in enumerate(edge_records):
        edge_gen_pair[ei] = (ci, nbr)
        raw_dual[ei] = float(np.linalg.norm(
            min_image_delta(points[nbr] - points[ci], L)))
        raw_v1[ei] = v1
        raw_v2[ei] = v2

    # Endpoint -> site by position (voro++ vertices and analytic circumcenters
    # agree to FP noise; the dedup radius bounds the residual).
    from scipy.spatial import cKDTree
    tree = cKDTree(wrap_to_box(sites, L), boxsize=L)
    d1, s1 = tree.query(wrap_to_box(raw_v1, L))
    d2, s2 = tree.query(wrap_to_box(raw_v2, L))
    tol = max(2e-4, 1e-9 * L)
    n_unmapped = int(((d1 > tol) | (d2 > tol)).sum())
    if n_unmapped:
        raise RuntimeError(f"V build: {n_unmapped}/{n_edges} V-edge endpoints are not "
                           f"within {tol:g} of any Voronoi-vertex site")
    edge_left = np.minimum(s1, s2).astype(np.int64)
    edge_right = np.maximum(s1, s2).astype(np.int64)

    edges_v1 = wrap_to_box(raw_v1, L)
    delta = min_image_delta(raw_v2 - raw_v1, L)
    edges_v2 = edges_v1 + delta
    lengths = np.linalg.norm(delta, axis=1)

    # Self-loops (both endpoints one site: two coincident vertices voro++ kept
    # distinct) carry no measure and would only give undefined unit vectors.
    keep = edge_left != edge_right
    n_dropped = int((~keep).sum())
    if n_dropped:
        sel = np.where(keep)[0]
        edges_v1, edges_v2, lengths = edges_v1[sel], edges_v2[sel], lengths[sel]
        edge_left, edge_right = edge_left[sel], edge_right[sel]
        raw_dual, edge_gen_pair = raw_dual[sel], edge_gen_pair[sel]

    star = _star_lengths_from_edges(n_sites, edge_left, edge_right, lengths)

    # Native (per-generator) rho_line_inv ingredients (cf. weights.py).
    gen_areas = _voropp_cell_areas(cells)
    gen_star_vedge = np.zeros(len(cells), dtype=float)
    np.add.at(gen_star_vedge, edge_gen_pair[:, 0], lengths)
    np.add.at(gen_star_vedge, edge_gen_pair[:, 1], lengths)

    net = Network2D(L=L, ntype="V", points=sites,
                    edges_v1=edges_v1, edges_v2=edges_v2,
                    edges_lengths=lengths,
                    edges_left_cell=edge_left,
                    edges_right_cell=edge_right,
                    cell_areas=site_areas, cell_star_lengths=star,
                    edges_dual_length=raw_dual,
                    edges_gen_pair=edge_gen_pair,
                    gen_areas=gen_areas,
                    gen_star_vedge=gen_star_vedge)
    net.build_diag = {"n_tri": n_tri, "n_tri_expected": 2 * int(points.shape[0]),
                      "n_sites": n_sites, "max_multiplicity": int(mult.max()),
                      "n_self_loops_dropped": n_dropped, "degenerate": False}
    return net


def _triangle_circumcenter_pbc_2d(generators: np.ndarray, tri: np.ndarray,
                                  L: float) -> np.ndarray:
    """2D Delaunay triangle circumcenter under PBC, anchored on first generator."""
    p0 = generators[tri[0]]
    p1 = p0 + min_image_delta(generators[tri[1]] - p0, L)
    p2 = p0 + min_image_delta(generators[tri[2]] - p0, L)
    A = 2.0 * np.array([p1 - p0, p2 - p0])
    b = np.array([np.dot(p1, p1) - np.dot(p0, p0),
                  np.dot(p2, p2) - np.dot(p0, p0)])
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return (p0 + p1 + p2) / 3.0


def _all_triangle_circumcenters_pbc_2d(generators: np.ndarray,
                                       triangles: np.ndarray,
                                       L: float) -> np.ndarray:
    if triangles.shape[0] == 0:
        return np.zeros((0, 2), dtype=float)
    out = np.empty((triangles.shape[0], 2), dtype=float)
    for i, tri in enumerate(triangles):
        out[i] = _triangle_circumcenter_pbc_2d(generators, tri, L)
    return wrap_to_box(out, L)


def _delaunay_triangles(cells) -> np.ndarray:
    triangles = set()
    for ci, cell in enumerate(cells):
        faces = cell.get("faces", [])
        v_to_faces = {}
        for fi, face in enumerate(faces):
            for v_local in face.get("vertices", []):
                v_to_faces.setdefault(v_local, []).append(fi)
        for v_local, face_idxs in v_to_faces.items():
            if len(face_idxs) != 2:
                continue
            f1, f2 = face_idxs
            n1 = faces[f1].get("adjacent_cell")
            n2 = faces[f2].get("adjacent_cell")
            if n1 is None or n2 is None or n1 < 0 or n2 < 0:
                continue
            if n1 == n2:
                continue
            tri = tuple(sorted((ci, int(n1), int(n2))))
            if len(set(tri)) != 3:
                continue
            triangles.add(tri)
    if not triangles:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(sorted(triangles), dtype=np.int64)


def _triangle_centroids_pbc(triangles: np.ndarray,
                            points: np.ndarray, L: float) -> np.ndarray:
    if triangles.shape[0] == 0:
        return np.zeros((0, 2), dtype=float)
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    b_un = a + min_image_delta(b - a, L)
    c_un = a + min_image_delta(c - a, L)
    cents = (a + b_un + c_un) / 3.0
    return wrap_to_box(cents, L)


def _triangle_areas_pbc(triangles: np.ndarray,
                        points: np.ndarray, L: float) -> np.ndarray:
    if triangles.shape[0] == 0:
        return np.zeros(0, dtype=float)
    a = points[triangles[:, 0]]
    ab = min_image_delta(points[triangles[:, 1]] - a, L)
    ac = min_image_delta(points[triangles[:, 2]] - a, L)
    return 0.5 * np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])


def _triangle_adjacency(triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if triangles.shape[0] == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    T = triangles.shape[0]
    a = triangles[:, [0, 1, 0]].ravel()
    b = triangles[:, [1, 2, 2]].ravel()
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    tri_id = np.repeat(np.arange(T, dtype=np.int64), 3)
    order = np.lexsort((hi, lo))
    lo = lo[order]; hi = hi[order]; tri_id = tri_id[order]
    same = (lo[1:] == lo[:-1]) & (hi[1:] == hi[:-1])
    if not np.any(same):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    i = np.flatnonzero(same)
    t0 = np.minimum(tri_id[i], tri_id[i + 1])
    t1 = np.maximum(tri_id[i], tri_id[i + 1])
    key = t0.astype(np.int64) * (T + 1) + t1.astype(np.int64)
    uniq = np.unique(key)
    t0u = (uniq // (T + 1)).astype(np.int64)
    t1u = (uniq %  (T + 1)).astype(np.int64)
    return t0u, t1u


def _build_C(points: np.ndarray, L: float,
             c_area_mode: str = "sec_vor") -> Network2D:
    streamed_primary = _stream_delaunay_triangles_2d(points, L)
    if streamed_primary is not None:
        triangles, _ = streamed_primary
    else:
        cells = _voropp_compute(points, L)
        triangles = _delaunay_triangles(cells)
    centroids = _triangle_centroids_pbc(triangles, points, L)

    left, right = _triangle_adjacency(triangles)
    n_edges = int(left.shape[0])

    cell_areas = np.zeros(centroids.shape[0], dtype=float)
    edges_dual_length = None
    inverse = None
    uniq_left_arr = uniq_right_arr = uniq_dual_arr = None
    # Always populate per-triangle areas (cheap; needed by `triangle_kite` weight).
    tri_areas_full = _triangle_areas_pbc(triangles, points, L) \
        if triangles.shape[0] > 0 else np.zeros(0, dtype=float)
    if centroids.shape[0] > 0:
        if c_area_mode == "triangle":
            cell_areas = tri_areas_full
            # No centroid Voronoi here, so no dual-ridge geometry available.
            # edges_dual_length stays None -> dual_area weight will raise a
            # clean error if requested under c_area_mode='triangle'.
        else:
            uniq_pts, inverse = _dedup_points_2d(centroids, L)
            # split dedup-merged cells across members (sum rule)
            mult = np.bincount(inverse, minlength=uniq_pts.shape[0])
            # _stream_unique_pairs returns areas + dual ridge lengths from a
            # single voro++ pass on the deduped centroids; same cost as the
            # cell-areas-only streaming call (no extra Voronoi build).
            streamed = _stream_unique_pairs(uniq_pts, L)
            if streamed is not None:
                (areas_uniq, uniq_left_arr, uniq_right_arr,
                 uniq_dual_arr, _, _) = streamed
                cell_areas = (areas_uniq / mult)[inverse]
            else:
                secondary = _voropp_compute(uniq_pts, L)
                cell_areas = (_voropp_cell_areas(secondary) / mult)[inverse]
                (uniq_left_arr, uniq_right_arr, uniq_dual_arr,
                 _, _) = _unique_cell_pairs(secondary, L=L)

    if n_edges == 0:
        zeros = np.zeros((0, 2), dtype=float)
        return Network2D(L=L, ntype="C", points=centroids,
                         edges_v1=zeros, edges_v2=zeros,
                         edges_lengths=np.zeros(0),
                         edges_left_cell=left, edges_right_cell=right,
                         cell_areas=cell_areas,
                         cell_star_lengths=np.zeros(centroids.shape[0]),
                         edges_dual_length=(np.zeros(0)
                                            if c_area_mode != "triangle"
                                            else None),
                         triangle_areas=tri_areas_full,
                         triangles=triangles)

    p_left = centroids[left]
    p_right = centroids[right]
    delta = min_image_delta(p_right - p_left, L)
    v2 = p_left + delta
    lengths = np.linalg.norm(delta, axis=1)

    star = _star_lengths_from_edges(centroids.shape[0], left, right, lengths)

    # Map each C-edge (centroid_left, centroid_right) to its dual ridge in
    # the centroid Voronoi. C-edges come from triangle adjacency; centroid
    # Voronoi adjacency is a related-but-distinct structure -- in the typical
    # case every C-edge corresponds to a centroid-Voronoi face, but a small
    # fraction of edges (degenerate centroid placement, dedup collisions)
    # may not. Missing entries default to 0.0; a single warn line summarizes.
    if c_area_mode != "triangle" and uniq_dual_arr is not None:
        dual_by_pair: dict = {}
        for ul, ur, ud in zip(uniq_left_arr, uniq_right_arr, uniq_dual_arr):
            ui, uj = (int(ul), int(ur)) if ul < ur else (int(ur), int(ul))
            dual_by_pair[(ui, uj)] = float(ud)
        edges_dual_length = np.zeros(n_edges, dtype=float)
        n_missing = 0
        for ei in range(n_edges):
            ui = int(inverse[int(left[ei])])
            uj = int(inverse[int(right[ei])])
            if ui == uj:
                n_missing += 1
                continue
            key = (ui, uj) if ui < uj else (uj, ui)
            d = dual_by_pair.get(key)
            if d is None:
                n_missing += 1
            else:
                edges_dual_length[ei] = d
        if n_missing:
            print(f"    [wndfc C dual] {n_missing}/{n_edges} C-edges had "
                  f"no centroid-Voronoi face (dual=0 for those)")

    net = Network2D(L=L, ntype="C", points=centroids,
                    edges_v1=p_left.copy(), edges_v2=v2,
                    edges_lengths=lengths,
                    edges_left_cell=left, edges_right_cell=right,
                    cell_areas=cell_areas,
                    cell_star_lengths=star,
                    edges_dual_length=edges_dual_length,
                    triangle_areas=tri_areas_full,
                    triangles=triangles)
    net.build_diag = _delaunay_diag(int(triangles.shape[0]), int(points.shape[0]))
    return net


def _delaunay_diag(n_tri: int, n_gen: int) -> dict:
    """A periodic Delaunay TRIANGULATION of n generators has exactly 2n
    triangles (Euler on the torus); voro++ output with k-valent vertices
    (k > 3 cocircular generators) yields more, and the C tessellation built
    from it is wrong -- the flag makes the pipeline raise TilingError."""
    return {"n_tri": int(n_tri), "n_tri_expected": 2 * int(n_gen),
            "degenerate": int(n_tri) != 2 * int(n_gen)}


_BUILDERS = {"D": _build_D, "V": _build_V, "G": _build_G, "C": _build_C}


def build_network(points: np.ndarray, L: float, ntype: str,
                  c_area_mode: str = "sec_vor",
                  compute_cell_areas: bool = True) -> Network2D:
    """Construct a Network2D for the requested tessellation type.

    Args:
      points: ``(N, 2)`` array of generator coordinates. Wrapped into ``[0, L)``.
      L: side length of the periodic box.
      ntype: one of ``D``, ``V``, ``G``, ``C``.
      c_area_mode: only used when ``ntype="C"``. ``"sec_vor"`` (default)
        uses secondary Voronoi cell areas around each centroid.
        ``"triangle"`` uses the area of the Delaunay simplex the centroid
        sits in -- skips the secondary voro pass entirely.
      compute_cell_areas: only used when ``ntype="V"``. ``True`` (default)
        runs the secondary voro++ pass on circumcenters for per-triangle
        cell areas. ``False`` skips it (``cell_areas = zeros(n_tri)``) for
        weights that never read ``cell_areas`` -- 2D mirror of
        ``build_network_3d(compute_cell_volumes=...)``.

    Returns:
      Network2D with edge geometry, cell areas, and star lengths populated.
    """
    if ntype not in _BUILDERS:
        raise ValueError(f"unknown ntype: {ntype!r}; expected one of {sorted(_BUILDERS)}")
    if isinstance(L, Cell):
        if not L.is_square:
            raise ValueError("build_network tessellates a point pattern with voro++, which "
                             "needs an orthogonal cell; for a general cell build the network "
                             "from a triangulation (build_from_triangulation)")
        L = L.L
    pts = wrap_to_box(np.asarray(points, dtype=float), L)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be (N, 2); got shape {pts.shape}")
    if ntype == "C":
        net = _build_C(pts, float(L), c_area_mode=c_area_mode)
    elif ntype == "V":
        net = _build_V(pts, float(L), compute_cell_areas=compute_cell_areas)
    else:
        net = _BUILDERS[ntype](pts, float(L))
    net.generators = pts.copy()
    return net


# --- General periodic cell: Voronoi shim and networks from a triangulation -----

def voronoi_cells_general(points: np.ndarray, cell, radii: np.ndarray = None,
                          *, margin_spacings: float = 4.0, max_tries: int = 4):
    """Periodic Voronoi (or power) cells of ``points`` in a general cell.

    voro++ only takes orthogonal boxes, so the points within ``margin`` of the
    cell boundary are replicated into the neighbouring images and a
    NON-periodic voro++ pass is run on the enlarged cloud inside its bounding
    box; the cells of the original points are kept.  A kept cell touching the
    bounding box means the margin was too small: it is doubled and the pass
    repeated.  Returns ``(areas, pairs)``: per-point cell areas and the (i < j)
    adjacencies among the original points (the radical Delaunay when ``radii``
    is given, the Delaunay otherwise).
    """
    import pyvoro
    from .pbc import as_cell, wrap_fractional

    cell = as_cell(cell)
    pts = np.asarray(points, dtype=float)
    n = int(pts.shape[0])
    f = wrap_fractional(pts, cell)
    base = f @ cell.A
    spacing = np.sqrt(cell.V / max(n, 1))
    margin = float(margin_spacings) * spacing
    rad = None if radii is None else np.asarray(radii, dtype=float)
    shifts = [(sx, sy) for sx in (-1, 0, 1) for sy in (-1, 0, 1) if (sx, sy) != (0, 0)]
    for _attempt in range(int(max_tries)):
        dfrac = margin / cell.sigma_min                     # conservative fractional margin
        near = ((f < dfrac) | (f > 1.0 - dfrac)).any(axis=1)
        idx_near = np.flatnonzero(near)
        cloud = [base]
        owner = [np.arange(n, dtype=np.int64)]
        for sx, sy in shifts:
            cloud.append(base[idx_near] + np.array([sx, sy], dtype=float) @ cell.A)
            owner.append(idx_near)
        cloud = np.concatenate(cloud, axis=0)
        owner = np.concatenate(owner)
        rad_cloud = None if rad is None else rad[owner]
        pad = 0.5 * margin
        lo = cloud.min(axis=0) - pad
        hi = cloud.max(axis=0) + pad
        disp = float(np.sqrt((hi - lo).prod() / cloud.shape[0]))
        kw = {"z_height": 0.5 * disp} if disp < 0.5 else {}
        if rad_cloud is not None:
            kw["radii"] = rad_cloud.tolist()
        cells = pyvoro.compute_2d_voronoi(
            cloud.tolist(), limits=[[float(lo[0]), float(hi[0])], [float(lo[1]), float(hi[1])]],
            dispersion=disp, periodic=[False, False], **kw)
        areas = np.empty(n, dtype=float)
        touches_wall = False
        pairs = set()
        for i in range(n):
            c = cells[i]
            areas[i] = float(c.get("volume", np.nan))
            for face in c.get("faces", []):
                nbr = face.get("adjacent_cell")
                if nbr is None or nbr < 0:
                    touches_wall = True
                    break
                j = int(owner[nbr])
                if j != i:
                    pairs.add((min(i, j), max(i, j)))
            if touches_wall:
                break
        if not touches_wall:
            pairs = np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)
            return areas, pairs
        margin *= 2.0
    raise RuntimeError("voronoi_cells_general: a kept cell still touches the bounding box after "
                       f"{max_tries} margin doublings (margin = {margin:g})")


def _validate_edges(n_points: int, edges: np.ndarray) -> np.ndarray:
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if e.size and (e.min() < 0 or e.max() >= n_points):
        raise ValueError("edge index out of range")
    lo = np.minimum(e[:, 0], e[:, 1])
    hi = np.maximum(e[:, 0], e[:, 1])
    if np.any(lo == hi):
        raise ValueError("self-loop in the edge list")
    return np.unique(np.stack([lo, hi], axis=1), axis=0)


def _power_centres(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                   r1: np.ndarray, r2: np.ndarray, r3: np.ndarray) -> np.ndarray:
    """Point of equal power |x - p_i|^2 - r_i^2 to three disks (unwrapped
    positions); the circumcentre when the radii are equal."""
    M = np.stack([2.0 * (p2 - p1), 2.0 * (p3 - p1)], axis=1)          # (T, 2, 2)
    rhs = np.stack([(p2 ** 2).sum(1) - (p1 ** 2).sum(1) - (r2 ** 2 - r1 ** 2),
                    (p3 ** 2).sum(1) - (p1 ** 2).sum(1) - (r3 ** 2 - r1 ** 2)], axis=1)
    return np.linalg.solve(M, rhs[..., None])[..., 0]


def _face_adjacency(faces: list) -> Tuple[np.ndarray, np.ndarray]:
    """(left, right) face pairs sharing an edge, from face vertex cycles."""
    key_to_faces: dict = {}
    for fi, cyc in enumerate(faces):
        n = len(cyc)
        for k in range(n):
            a, b = int(cyc[k]), int(cyc[(k + 1) % n])
            key_to_faces.setdefault((min(a, b), max(a, b)), []).append(fi)
    left, right = [], []
    for fs in key_to_faces.values():
        if len(fs) == 2 and fs[0] != fs[1]:
            left.append(min(fs)); right.append(max(fs))
    if not left:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    pairs = np.unique(np.stack([np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)], axis=1), axis=0)
    return pairs[:, 0], pairs[:, 1]


def _polygon_centroid_area(verts: np.ndarray):
    """Signed-fan area centroid and |area| of a simple polygon (unwrapped verts)."""
    x, y = verts[:, 0], verts[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area2 = cross.sum()
    if abs(area2) < 1e-300:
        return verts.mean(axis=0), 0.0
    cx = ((x + xn) * cross).sum() / (3.0 * area2)
    cy = ((y + yn) * cross).sum() / (3.0 * area2)
    return np.array([cx, cy]), 0.5 * abs(area2)


def _orthogonal_dual_lengths(pts, cell_arg, faces, he_tail, he_shift, e, radii):
    """Per primal edge (rows of ``e``, sorted pairs) of a triangulation: the length
    of the dual edge joining the power centres of its two faces (circumcentres
    without radii) -- the radical axis / Voronoi ridge, orthogonal to the edge."""
    from .planar_dual import _lift_face_verts
    tri = np.asarray([he_tail[f["half_edges"]] for f in faces], dtype=np.int64)
    xy = np.asarray([_lift_face_verts(pts, f["half_edges"], he_tail, he_shift, cell_arg) for f in faces],
                    dtype=float)
    r = np.zeros(tri.shape, dtype=float) if radii is None else np.asarray(radii, dtype=float)[tri]
    nodes = wrap_to_box(_power_centres(xy[:, 0], xy[:, 1], xy[:, 2], r[:, 0], r[:, 1], r[:, 2]), cell_arg)
    face_of: dict = {}
    for fi, t in enumerate(tri):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            face_of.setdefault((min(a, b), max(a, b)), []).append(fi)
    pair = np.full((e.shape[0], 2), -1, dtype=np.int64)
    for k, (a, b) in enumerate(e):
        fs = face_of.get((int(a), int(b)), ())
        if len(fs) == 2:
            pair[k] = fs
    ok = pair[:, 0] >= 0
    d = np.zeros(e.shape[0], dtype=float)
    d[ok] = np.linalg.norm(min_image_delta(nodes[pair[ok, 1]] - nodes[pair[ok, 0]], cell_arg), axis=1)
    return d


def build_from_graph(points: np.ndarray, cell, edges: np.ndarray, ntype: str,
                     *, radii: np.ndarray = None,
                     compute_cell_areas: bool = True) -> Network2D:
    """D / C / V / G networks from a GIVEN periodic planar graph on ``points``
    (an edge list: a packing's contact network, a strut layout, ...) in a
    general periodic cell.

      D  the graph itself -- any closed periodic planar graph; on a
         triangulation ``edges_dual_length`` holds the orthogonal (power) dual
         edge per primal edge
      G  its Gabriel subgraph (empty diametral disk on the node positions)
      C  the centroid of each face, joined across shared edges (area centroid
         for polygons; the vertex mean for triangles)
      V  the power (Laguerre) dual: the power centre of each triangle -- the
         circumcentre when ``radii`` is None -- joined across shared edges;
         requires a triangulation (every face a triangle); each V-edge is
         perpendicular to its primal edge (radical axis), the maximal deviation
         is reported in ``build_diag``

    ``build_diag`` reports the face census (Euler characteristic, face sizes,
    torus-homology faces); ``degenerate`` is set only when the requested
    network needs a triangulation and the graph is not one.  The per-vertex
    ``cell_areas`` used by the V weight scheme are the Euclidean Voronoi cells
    of the network vertices (``voronoi_cells_general``).
    """
    from .pbc import as_cell
    from .planar_dual import _build_half_edges, _cyclic_neighbor_lists, _face_walk, _lift_face_verts

    cell = as_cell(cell)
    cell_arg = cell.L if cell.is_square else cell
    net_cell = None if cell.is_square else cell
    L_field = cell.L                                   # NaN for a general cell
    pts = wrap_to_box(np.asarray(points, dtype=float), cell_arg)
    n = int(pts.shape[0])
    e = _validate_edges(n, edges)
    i, j = e[:, 0], e[:, 1]
    delta = min_image_delta(pts[j] - pts[i], cell_arg)
    v1 = pts[i]
    v2 = v1 + delta
    lengths = np.linalg.norm(delta, axis=1)
    if np.any(lengths <= 0):
        raise ValueError("zero-length edge in the edge list")
    star = _star_lengths_from_edges(n, i, j, lengths)
    vertex_areas = np.zeros(n, dtype=float)
    if compute_cell_areas:
        vertex_areas, _ = voronoi_cells_general(pts, cell)
    netD = Network2D(L=L_field, ntype="D", points=pts, edges_v1=v1, edges_v2=v2,
                     edges_lengths=lengths, edges_left_cell=i, edges_right_cell=j,
                     cell_areas=vertex_areas, cell_star_lengths=star, generators=pts.copy(),
                     cell=net_cell)
    # faces of the graph on the torus
    he_tail, he_head, he_shift, he_twin = _build_half_edges(netD)
    adj_start, adj_he, adj_inv = _cyclic_neighbor_lists(netD, he_tail, he_head)
    _, faces = _face_walk(he_tail, he_head, he_shift, he_twin, adj_start, adj_he, adj_inv)
    n_edges = int(e.shape[0])
    n_faces = len(faces)
    face_sizes = np.array([f["half_edges"].shape[0] for f in faces], dtype=np.int64)
    n_homology = int(sum(1 for f in faces if f["is_homology"]))
    is_tri = bool(n_edges == 3 * n and n_faces == 2 * n and (face_sizes == 3).all() and n_homology == 0)
    diag = {"n_vertices": n, "n_edges": n_edges, "n_faces": n_faces, "euler": int(n - n_edges + n_faces),
            "n_homology": n_homology, "is_triangulation": is_tri,
            "face_sizes": {int(s): int(c) for s, c in zip(*np.unique(face_sizes, return_counts=True))},
            "n_tri": int((face_sizes == 3).sum()), "n_tri_expected": 2 * n,
            "degenerate": False}
    if ntype == "D":
        if is_tri:
            netD.edges_dual_length = _orthogonal_dual_lengths(pts, cell_arg, faces, he_tail, he_shift, e, radii)
        netD.build_diag = diag
        return netD
    if ntype == "G":
        mid = wrap_to_box(v1 + 0.5 * delta, cell_arg)
        keep = _gabriel_filter(pts, mid, 0.5 * lengths, cell_arg)
        sel = np.flatnonzero(keep)
        starG = _star_lengths_from_edges(n, i[sel], j[sel], lengths[sel])
        netG = Network2D(L=L_field, ntype="G", points=pts, edges_v1=v1[sel], edges_v2=v2[sel],
                         edges_lengths=lengths[sel], edges_left_cell=i[sel], edges_right_cell=j[sel],
                         cell_areas=vertex_areas, cell_star_lengths=starG, generators=pts.copy(),
                         cell=net_cell)
        netG.build_diag = {**diag, "n_gabriel_edges": int(sel.size)}
        return netG
    if ntype not in ("C", "V"):
        raise ValueError(f"unknown ntype: {ntype!r}")
    if n_homology:
        diag["degenerate"] = True
        raise ValueError(f"the graph has {n_homology} torus-homology faces (not closed in periodicity); "
                         f"C and V need a closed periodic planar graph ({diag})")
    if ntype == "V" and not is_tri:
        diag["degenerate"] = True
        raise ValueError(f"the graph is not a triangulation ({diag}); V needs one (C, G and D do not)")
    # faces: vertex cycles and unwrapped positions (first half-edge's tail anchors)
    cycles = [he_tail[f["half_edges"]] for f in faces]
    lifted = [_lift_face_verts(pts, f["half_edges"], he_tail, he_shift, cell_arg) for f in faces]
    face_area = np.empty(n_faces, dtype=float)
    nodes = np.empty((n_faces, 2), dtype=float)
    if ntype == "C":
        for fi, verts in enumerate(lifted):
            if verts.shape[0] == 3:
                nodes[fi] = verts.mean(axis=0)
                face_area[fi] = 0.5 * abs((verts[1, 0] - verts[0, 0]) * (verts[2, 1] - verts[0, 1])
                                          - (verts[1, 1] - verts[0, 1]) * (verts[2, 0] - verts[0, 0]))
            else:
                nodes[fi], face_area[fi] = _polygon_centroid_area(verts)
        tri = None
    else:
        tri = np.asarray(cycles, dtype=np.int64)
        tri_xy = np.asarray(lifted, dtype=float)
        face_area = 0.5 * np.abs((tri_xy[:, 1, 0] - tri_xy[:, 0, 0]) * (tri_xy[:, 2, 1] - tri_xy[:, 0, 1])
                                 - (tri_xy[:, 1, 1] - tri_xy[:, 0, 1]) * (tri_xy[:, 2, 0] - tri_xy[:, 0, 0]))
        r = np.zeros((n_faces, 3), dtype=float) if radii is None else np.asarray(radii, dtype=float)[tri]
        nodes = _power_centres(tri_xy[:, 0], tri_xy[:, 1], tri_xy[:, 2], r[:, 0], r[:, 1], r[:, 2])
    nodes = wrap_to_box(nodes, cell_arg)
    left, right = _face_adjacency(cycles)
    d = min_image_delta(nodes[right] - nodes[left], cell_arg)
    e_v1 = nodes[left]
    e_v2 = e_v1 + d
    e_len = np.linalg.norm(d, axis=1)
    node_areas = np.zeros(n_faces, dtype=float)
    if compute_cell_areas:
        node_areas, _ = voronoi_cells_general(nodes, cell)
    starN = _star_lengths_from_edges(n_faces, left, right, e_len)
    if tri is None:
        tri = np.asarray(cycles, dtype=object) if any(len(c) != 3 for c in cycles) \
            else np.asarray(cycles, dtype=np.int64)
    net = Network2D(L=L_field, ntype=ntype, points=nodes, edges_v1=e_v1, edges_v2=e_v2,
                    edges_lengths=e_len, edges_left_cell=left, edges_right_cell=right,
                    cell_areas=node_areas, cell_star_lengths=starN, generators=pts.copy(),
                    triangle_areas=face_area, triangles=(tri if not isinstance(tri, np.ndarray) or tri.dtype != object else None),
                    cell=net_cell)
    extra = {}
    if ntype == "V":
        # dual edge (left -> right) against the shared primal edge: |cos| of the angle
        # between them; 0 for an exact radical axis
        shared = np.empty((left.size, 2), dtype=np.int64)
        for k, (a, b) in enumerate(zip(left, right)):
            s = np.intersect1d(tri[a], tri[b])
            shared[k] = s[:2]
        pd = min_image_delta(pts[shared[:, 1]] - pts[shared[:, 0]], cell_arg)
        cosang = (np.abs(np.einsum("ei,ei->e", d, pd))
                  / np.maximum(e_len * np.linalg.norm(pd, axis=1), 1e-300))
        extra["max_dual_primal_cos"] = float(cosang.max()) if cosang.size else 0.0
        extra["n_zero_length_dual"] = int((e_len <= 1e-12 * np.sqrt(cell.V)).sum())
    net.build_diag = {**diag, **extra}
    return net


# name kept for callers written against the triangulation-only builder
build_from_triangulation = build_from_graph
