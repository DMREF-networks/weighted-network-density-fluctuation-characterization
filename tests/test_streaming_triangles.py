"""Streamed Delaunay-triangle extraction: the topological ring-walk on the cell-edge
records (no coordinate rounding), identical to the pyvoro path.

The record-level test runs everywhere; the backend parity tests need the 2D streaming
extension and are skipped where it is not built."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.environ.get("WNDFC_TEST_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wndfc.network import _stream_triangles_from_records   # noqa: E402


def _streaming_available() -> bool:
    try:
        from wndfc.streaming import AVAILABLE_2D
    except ImportError:
        return False
    return bool(AVAILABLE_2D)


def test_ring_walk_on_synthetic_records():
    """Cell 0 is a triangle-shaped cell with neighbours 1, 2, 3 (ring A->B->C->A), cell 1
    has a broken chain (its second record does not start where the first ends) and a
    self-image face (neighbour -1): only cell 0's three corners give triangles, the
    wrap-around corner included; rows come back sorted and unique."""
    A, B, C = np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])
    e_cell = np.array([0, 0, 0, 1, 1, 1])
    e_nbr = np.array([1, 2, 3, 0, 2, -1])
    v1 = np.array([A, B, C, A, C + 1e-12, B])
    v2 = np.array([B, C, A, B, A, C])
    tri = _stream_triangles_from_records(e_cell, e_nbr, v1, v2)
    assert tri.tolist() == [[0, 1, 2], [0, 1, 3], [0, 2, 3]]


def test_near_coincident_vertices_stay_distinct():
    """Two consecutive records whose shared vertex differs in the last bit are NOT a
    shared vertex (no rounding, no tolerance): the rule is exact identity."""
    A, B, C = np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])
    e_cell = np.array([0, 0, 0])
    e_nbr = np.array([1, 2, 3])
    v1 = np.array([A, np.nextafter(B, B + 1.0), C])
    v2 = np.array([B, C, A])
    tri = _stream_triangles_from_records(e_cell, e_nbr, v1, v2)
    assert tri.tolist() == [[0, 1, 3], [0, 2, 3]]        # the A->B / B'->C corner is broken


def _cocircular_quad(delta: float, L: float = 20.0):
    base = np.random.default_rng(1).uniform(0.0, L, (400, 2))
    c0 = np.array([10.3, 9.7])
    r = 0.35
    base = base[np.linalg.norm(base - c0, axis=1) > 1.2]
    ang = np.array([0.3, 1.9, 3.4, 5.1])
    circ = c0 + r * np.stack([np.cos(ang), np.sin(ang)], 1)
    circ[0] = c0 + (r + delta) * np.array([np.cos(ang[0]), np.sin(ang[0])])
    return np.vstack([base, circ])


@pytest.mark.skipif(not _streaming_available(), reason="2D streaming extension not built")
@pytest.mark.parametrize("case", ["poisson400", "poisson4000", "cocircular_1e-9",
                                  "cocircular_1e-7", "cocircular_1e-6", "cocircular_0"])
def test_streamed_C_triangles_match_pyvoro(case):
    """Streamed vs pyvoro triangle sets: identical on generic input and on a quad
    cocircular to delta >= 1e-9; at an exact tie the two voro++ builds may break it
    differently, so there only the triangles outside the quad must agree and none may
    be lost (n_tri >= 2N)."""
    from wndfc.network import (_delaunay_triangles, _stream_delaunay_triangles_2d,
                                                _voropp_compute)
    L = 20.0
    if case == "poisson400":
        pts = np.random.default_rng(1).uniform(0.0, L, (400, 2))
    elif case == "poisson4000":
        L = 63.0
        pts = np.random.default_rng(5).uniform(0.0, L, (4000, 2))
    else:
        pts = _cocircular_quad(float(case.split("_")[1]), L)
    N = pts.shape[0]
    streamed = _stream_delaunay_triangles_2d(pts, L)
    assert streamed is not None
    tri_s, areas = streamed
    tri_p = _delaunay_triangles(_voropp_compute(pts, L))
    assert abs(areas.sum() / (L * L) - 1.0) < 1e-10
    if case != "cocircular_0":
        assert tri_s.shape[0] == 2 * N
        assert np.array_equal(tri_s, tri_p)
    else:
        assert tri_s.shape[0] >= 2 * N
        quad = set(range(N - 4, N))
        differing = {tuple(t) for t in tri_s.tolist()} ^ {tuple(t) for t in tri_p.tolist()}
        assert all(set(t) <= quad for t in differing), differing
