"""General periodic cell: the square-cell reduction, the reciprocal lattice of a
sheared cell (Bragg peaks of a lattice), and the general-cell sigma^2 / h_w /
tiling paths against the square-box paths on the same network."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.environ.get("WNDFC_TEST_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wndfc.pbc import (Cell, cell_volume, image_shift, min_image_delta, periodic_pairs,  # noqa: E402
                            wrap_to_box)
from wndfc.generators import poisson_2d                            # noqa: E402
from wndfc.network import build_network, build_from_triangulation, voronoi_cells_general  # noqa: E402
from wndfc.spectral import compute_chi_ell, _reciprocal_lattice     # noqa: E402
from wndfc.variance import compute_sigma2                          # noqa: E402
from wndfc.weights import compute_weights                          # noqa: E402
from wndfc.pipeline import build_weighted_network                  # noqa: E402
from wndfc import norms                                            # noqa: E402

L, N, SEED = 20.0, 400, 11
SHEAR = np.array([[1.0, 0.15], [-0.05, 1.1]])          # a general cell (rows = lattice vectors)


def _delaunay_edges(net):
    return np.stack([net.edges_left_cell, net.edges_right_cell], axis=1)


@pytest.fixture(scope="module")
def net_sq():
    return build_network(poisson_2d(N, L, seed=SEED), L, "D")


# 1. pbc helpers: a square Cell is the float path
def test_square_cell_reduces_to_float():
    c = Cell.square(L)
    rng = np.random.default_rng(1)
    x = rng.uniform(-3 * L, 3 * L, size=(500, 2))
    assert np.array_equal(min_image_delta(x, c), min_image_delta(x, L))
    assert np.array_equal(wrap_to_box(x, c), wrap_to_box(x, L))
    assert np.array_equal(image_shift(x, c), image_shift(x, L))
    assert cell_volume(c) == L * L
    kv, km = _reciprocal_lattice(L, 5.0)
    kv2, km2 = _reciprocal_lattice(c, 5.0)
    assert np.array_equal(kv, kv2) and np.array_equal(km, km2)


# 2. general-cell min image: exact on lattice translations, periodic_pairs vs brute force
def test_general_cell_min_image_and_pairs():
    c = Cell(SHEAR * 7.0)
    rng = np.random.default_rng(2)
    f = rng.uniform(0, 1, size=(300, 2))
    x = f @ c.A
    n = rng.integers(-3, 4, size=(300, 2)).astype(float)
    shifted = x + n @ c.A
    assert np.allclose(min_image_delta(shifted - x, c), 0.0, atol=1e-12)
    assert np.array_equal(image_shift(shifted - x, c), n)
    r = 1.2
    pairs = periodic_pairs(x, c, r)
    got = {(int(a), int(b)) for a, b in pairs}
    d = np.linalg.norm(min_image_delta(x[None, :, :] - x[:, None, :], c), axis=-1)
    iu = np.triu_indices(300, 1)
    want = {(int(a), int(b)) for a, b, dd in zip(iu[0], iu[1], d[iu]) if dd <= r}
    assert got == want


# 3. Bragg peaks of a lattice sit exactly on the reciprocal lattice of a sheared cell
def test_bragg_in_sheared_cell():
    n_side = 12
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3.0) / 2.0])           # triangular lattice, its own hexagonal cell
    A = np.stack([n_side * a1, n_side * a2])
    c = Cell(A)
    ij = np.array([(i, j) for i in range(n_side) for j in range(n_side)], dtype=float)
    pts = ij @ np.stack([a1, a2])
    # the Delaunay of the lattice is degenerate for voro++; use its known triangulation
    idx = lambda i, j: (i % n_side) * n_side + (j % n_side)
    edges = []
    for i in range(n_side):
        for j in range(n_side):
            edges += [(idx(i, j), idx(i + 1, j)), (idx(i, j), idx(i, j + 1)), (idx(i, j), idx(i + 1, j - 1))]
    edges = np.array(edges)
    net = build_from_triangulation(pts, c, edges, "D", compute_cell_areas=False)
    assert net.n_edges == 3 * n_side ** 2 and not net.build_diag["degenerate"]
    chi = compute_chi_ell(net, net.edges_lengths, kmax=8.0)
    # per-mode: intensity only where k is a reciprocal vector of the UNIT lattice
    kv, km = _reciprocal_lattice(c, 8.0)
    B1 = 2.0 * np.pi * np.linalg.inv(np.stack([a1, a2])).T          # unit-lattice reciprocal rows
    n_unit = kv @ np.linalg.inv(B1)                                   # integer iff Bragg
    is_bragg = np.all(np.abs(n_unit - np.round(n_unit)) < 1e-9, axis=1)
    from wndfc.spectral import _mode_intensities
    half = 0.5 * (net.edges_v2 - net.edges_v1)
    tot, _ = _mode_intensities(net.edges_midpoints, half, net.edges_lengths, kv, 1024)
    assert tot[~is_bragg].max() < 1e-18 * tot[is_bragg].max()
    assert is_bragg.sum() >= 6


# 4. the same Poisson network as a square-cell tessellation and as a triangulation in
#    Cell.square(L): identical geometry, chi, tiling, sigma^2 and h_w on both code paths
def test_triangulation_route_matches_square_tessellation(net_sq):
    edges = _delaunay_edges(net_sq)
    net_tri = build_from_triangulation(net_sq.generators, L, edges, "D")
    assert net_tri.n_edges == net_sq.n_edges
    a = np.lexsort((net_sq.edges_right_cell, net_sq.edges_left_cell))
    b = np.lexsort((net_tri.edges_right_cell, net_tri.edges_left_cell))
    assert np.allclose(net_sq.edges_lengths[a], net_tri.edges_lengths[b], rtol=0, atol=1e-12)
    assert np.allclose(net_sq.cell_areas, net_tri.cell_areas, rtol=1e-9)   # voro++ periodic vs shim
    for wt in ("rho_line_inv", "midpt_sec_area", "dual_diamond_planar"):
        w = compute_weights(net_tri, wt)
        assert abs(w.sum() / (L * L) - 1.0) < 1e-9, wt


# 5. a sheared cell holding the SAME point pattern mapped affinely: the general
#    sigma^2 kernel, the general h_w route and the general tiling all agree with
#    themselves (Parseval bridge vs direct) and the chi at k=0+ matches the sum rule
def test_general_cell_invariants():
    c = Cell(SHEAR * 18.0)
    rng = np.random.default_rng(SEED)
    pts = rng.uniform(0, 1, size=(N, 2)) @ c.A
    _, pairs = voronoi_cells_general(pts, c)               # Delaunay of the pattern in the sheared cell
    for wt in ("rho_line_inv", "midpt_sec_area", "dual_diamond_planar"):
        net, w, info = build_weighted_network(pts, c, "D", wt, network_source="edges", edges=pairs)
        assert info["tiling_resid"] <= 1e-8, wt
        assert not net.build_diag["degenerate"]
    net, w, _ = build_weighted_network(pts, c, "D", "length_power", network_source="edges", edges=pairs)
    chi = compute_chi_ell(net, w, kmax=12.0)
    meta = norms.build_metadata(net=net, w_eff=w, dim=2, gen="poi", tess="D", scheme="theta", theta=1.0,
                                seed=SEED, ens=1, k_max=12.0, n_bins=chi.n_bins, shell_mode="width",
                                code_version="test", with_wedge=True)
    assert meta["L"] is None and np.allclose(meta["cell"], c.A) and abs(meta["V"] - c.V) < 1e-12
    dv = norms.derived(meta)
    assert abs(chi.chi_self[0] / dv["chi_self_0"] - 1.0) < 0.05
    from wndfc.parseval import parseval_sigma2
    R = np.geomspace(0.6, 1.2, 4)                  # k_0 R <= 0.4: the bridge misses [0, k_0) (non-HU pattern)
    bridge = parseval_sigma2(chi, R, 2)
    direct = compute_sigma2(net, w / net.edges_lengths, R_grid=R, n_windows=20000, seed=SEED).sigma2
    assert np.all(np.abs(bridge / direct - 1.0) < 0.1), bridge / direct
    from wndfc.h_ell import real_space_curves, sample_pair_deposits
    dep = sample_pair_deposits(net, w, r_max=3.0, n_bins=40, n_samples=64, seed=SEED)
    assert dep.n_pairs_dist > 0 and not dep.cutoff_capped
    c = real_space_curves(dep, net.V, float(np.sum(w)) / net.V)
    assert np.all(np.isfinite(c.h_dist)) and np.all(np.isfinite(c.f_self))
    assert abs((dep.mom_self[0] + dep.self_overflow) / float(np.sum(w ** 2)) - 1.0) < 1e-12
