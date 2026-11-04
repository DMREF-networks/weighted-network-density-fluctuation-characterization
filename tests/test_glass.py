"""Ideal-glass packings (Bolton-Lum library): the contact network under the
packing's own sheared periodic cell.  Needs the packing library, which is
cited, not vendored: set WNDFC_GLASS_DIR to a packing directory in the
idealPackingLibrary layout (positions.dat, radii.dat, latticeVectors.dat) with
the authors' <name>edges.npy reachable above it; otherwise the module skips.

  7.3  the edge list is a torus triangulation with 3N edges and 2N faces
  7.5  the power dual is the radical-axis network (every V-edge perpendicular
       to its primal edge) and coincides with the radical Delaunay of the disks
  7.7  tiling of the V/T/D weight schemes on the four derived networks
  7.8  edge list == radical Delaunay == contacts (gap statistics)
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.environ.get("WNDFC_TEST_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GLASS = os.environ.get("WNDFC_GLASS_DIR")
pytestmark = pytest.mark.skipif(not GLASS or not os.path.isdir(GLASS),
                                reason="WNDFC_GLASS_DIR not set to a packing directory")

from wndfc.loaders import load_packing                             # noqa: E402
from wndfc.network import build_from_triangulation, voronoi_cells_general  # noqa: E402
from wndfc.pipeline import build_weighted_network, contact_gap_stats, TILING_TOL  # noqa: E402


@pytest.fixture(scope="module")
def packing():
    pk = load_packing(GLASS)
    if pk["edges"] is None or pk["radii"] is None:
        pytest.skip("packing has no edge list / radii next to it")
    return pk


def test_edge_list_is_a_torus_triangulation(packing):
    net = build_from_triangulation(packing["points"], packing["cell"], packing["edges"], "D",
                                   compute_cell_areas=False)
    d = net.build_diag
    n = packing["points"].shape[0]
    assert d["n_edges"] == 3 * n and d["n_faces"] == 2 * n and d["euler"] == 0
    assert d["is_triangulation"] and d["n_homology"] == 0 and not d["degenerate"]


def test_contacts_radical_and_edges_agree(packing):
    stats = contact_gap_stats(packing["points"], packing["cell"], packing["edges"], packing["radii"], 1e-3)
    assert stats["n_gap_gt_tol"] == 0 and stats["max_gap"] < 1e-2
    _, radical = voronoi_cells_general(packing["points"], packing["cell"], packing["radii"])
    e = packing["edges"]
    file_set = {(int(min(a, b)), int(max(a, b))) for a, b in e}
    rad_set = {(int(a), int(b)) for a, b in radical}
    assert file_set == rad_set


def test_power_dual_is_the_radical_axis_network(packing):
    net = build_from_triangulation(packing["points"], packing["cell"], packing["edges"], "V",
                                   radii=packing["radii"], compute_cell_areas=False)
    assert net.build_diag["max_dual_primal_cos"] < 1e-9
    assert net.build_diag["n_zero_length_dual"] == 0


@pytest.mark.parametrize("nt", ["D", "C", "V", "G"])
@pytest.mark.parametrize("weight", ["rho_line_inv", "midpt_sec_area", "dual_diamond_planar"])
def test_tiling_on_the_contact_networks(packing, nt, weight):
    if nt == "G" and weight == "dual_diamond_planar":
        pytest.skip("the Gabriel subgraph has non-triangular faces; the kite dual is defined but slow")
    net, w, info = build_weighted_network(packing["points"], packing["cell"], nt, weight,
                                          network_source="edges", edges=packing["edges"],
                                          radii=packing["radii"])
    assert info["tiling_resid"] <= TILING_TOL
    assert abs(net.cell_areas.sum() / net.V - 1.0) < 1e-9 or weight == "dual_diamond_planar"
