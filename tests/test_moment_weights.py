"""Moment-matched vertex shares (d2_ring): the cells tile, the shares tile and match
every moment through second order (D_2 on D), the D_2 kite halves are the D scheme,
and the scheme is 2D square-box only (D network only)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.environ.get("WNDFC_TEST_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wndfc.network import build_network, voronoi_cells_general     # noqa: E402
from wndfc.network_3d import build_network_3d                      # noqa: E402
from wndfc.generators import poisson_2d, poisson_3d                # noqa: E402
from wndfc.moment_weights import bary_cell_moments, d2_ring        # noqa: E402
from wndfc.pbc import Cell                                         # noqa: E402
from wndfc.pipeline import build_weighted_network, TILING_TOL      # noqa: E402
from wndfc.weights import compute_weights                          # noqa: E402
from wndfc.weights_3d import compute_weights_3d                    # noqa: E402

L = 20.0


@pytest.fixture(scope="module")
def pts():
    return poisson_2d(400, L, seed=11)


def test_bary_cells_tile_and_kite_halves_are_the_d_scheme(pts):
    net = build_network(pts, L, "D", compute_cell_areas=False)
    A, c_rel, M2, half_kite = bary_cell_moments(net)
    assert abs(A.sum() / L ** 2 - 1.0) < 1e-12 and (A > 0).all()
    assert np.allclose(M2, M2.transpose(0, 2, 1))
    own = np.zeros(A.shape[0])                     # the kite halves at a vertex make up its cell
    np.add.at(own, net.edges_left_cell, half_kite)
    np.add.at(own, net.edges_right_cell, half_kite)
    assert np.abs(own - A).max() < 1e-12 * A.mean()
    kite = compute_weights(net, "dual_diamond_planar")
    assert np.abs(2.0 * half_kite - kite).max() < 1e-12 * kite.mean()


def test_d2_ring_exact_through_second_order(pts):
    net = build_network(pts, L, "D", compute_cell_areas=False)
    diag: dict = {}
    w = d2_ring(net, diag_out=diag)
    m = diag["d2_ring"]
    cell = L ** 2 / pts.shape[0]
    assert abs(m["tiling_resid"]) < 1e-9 and abs(m["cell_tiling_resid"]) < 1e-12
    assert m["n_fallback"] == 0 and m["n_unresolved"] == 0
    assert m["pv_max"] < 1e-9 * cell ** 1.5
    assert m["m2_resid_rel_max"] < 1e-6
    assert w.shape == (net.n_edges,) and np.isfinite(w).all()
    assert 0.0 < (w < 0).mean() < 0.05


def test_d2_ring_through_the_pipeline(pts):
    _, pairs = voronoi_cells_general(pts, L)                       # the Delaunay edges of the points
    for kw in ({"network_source": "tessellation"}, {"network_source": "edges", "edges": pairs}):
        net, w, info = build_weighted_network(pts, L, "D", "d2_ring", **kw)
        assert info["tiling_resid"] <= TILING_TOL


def test_d2_ring_rejects_other_networks_3d_and_general_cells(pts):
    for nt in ("V", "G", "C"):
        with pytest.raises(ValueError):
            build_weighted_network(pts, L, nt, "d2_ring")
    net3 = build_network_3d(poisson_3d(60, 6.0, seed=1), 6.0, "D", compute_cell_volumes=False)
    with pytest.raises(ValueError):
        compute_weights_3d(net3, "d2_ring")
    cell = Cell(np.array([[L, 0.0], [0.3 * L, L]]))
    with pytest.raises(ValueError):
        build_weighted_network(pts, cell, "D", "d2_ring", network_source="edges",
                               edges=voronoi_cells_general(pts, cell)[1])
