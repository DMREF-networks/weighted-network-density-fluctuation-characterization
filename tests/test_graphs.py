"""Arbitrary periodic planar graphs through the D slot (no triangulation
needed), the derived C / G networks on them, and V's triangulation
requirement."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.environ.get("WNDFC_TEST_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wndfc.network import build_from_graph                         # noqa: E402
from wndfc.pipeline import build_weighted_network, TILING_TOL      # noqa: E402


def square_grid(n: int, L: float):
    """n x n periodic square grid: 4-regular, all faces squares (not a triangulation)."""
    g = (np.arange(n) + 0.5) * (L / n)
    pts = np.array([(x, y) for x in g for y in g], dtype=float)
    idx = lambda i, j: (i % n) * n + (j % n)
    edges = np.array([(idx(i, j), idx(i + 1, j)) for i in range(n) for j in range(n)]
                     + [(idx(i, j), idx(i, j + 1)) for i in range(n) for j in range(n)])
    return pts, edges


@pytest.mark.parametrize("weight", ["length_power", "rho_line_inv", "midpt_sec_area", "dual_diamond_planar"])
def test_square_grid_D_all_weights(weight):
    L = 12.0
    pts, edges = square_grid(12, L)
    net, w, info = build_weighted_network(pts, L, "D", weight, network_source="edges", edges=edges)
    d = net.build_diag
    assert d["n_edges"] == 2 * pts.shape[0] and d["face_sizes"] == {4: pts.shape[0]} and d["euler"] == 0
    assert not d["is_triangulation"] and not d["degenerate"]
    if weight != "length_power":
        assert info["tiling_resid"] <= TILING_TOL


def test_square_grid_C_is_the_dual_grid():
    L = 12.0
    pts, edges = square_grid(12, L)
    net = build_from_graph(pts, L, edges, "C")
    assert net.n_cells == pts.shape[0] and net.n_edges == 2 * pts.shape[0]     # dual square grid
    assert np.allclose(net.edges_lengths, 1.0)


def test_square_grid_G_keeps_the_grid():
    L = 12.0
    pts, edges = square_grid(12, L)
    net = build_from_graph(pts, L, edges, "G", compute_cell_areas=False)
    assert net.n_edges == edges.shape[0]


def test_square_grid_V_refused():
    L = 12.0
    pts, edges = square_grid(12, L)
    with pytest.raises(ValueError, match="triangulation"):
        build_from_graph(pts, L, edges, "V", compute_cell_areas=False)
