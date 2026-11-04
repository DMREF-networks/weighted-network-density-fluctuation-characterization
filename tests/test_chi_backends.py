"""The two chi(k) backends compute the same arithmetic in a different summation
order: identical shells and mode counts, values equal to ~1e-13 relative."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.environ.get("WNDFC_TEST_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wndfc.generators import poisson_2d, poisson_3d                # noqa: E402
from wndfc.network import build_network                            # noqa: E402
from wndfc.network_3d import build_network_3d                      # noqa: E402
from wndfc.pbc import Cell                                         # noqa: E402
from wndfc.network import voronoi_cells_general, build_from_triangulation  # noqa: E402
from wndfc.spectral import compute_chi_ell                         # noqa: E402
from wndfc.spectral_3d import compute_chi_ell_3d                   # noqa: E402


def _same(a, b, rtol=1e-12):
    assert a.k.size == b.k.size and np.array_equal(a.counts, b.counts)
    assert np.allclose(a.k, b.k, rtol=0, atol=1e-12)
    assert np.allclose(a.chi_ell, b.chi_ell, rtol=rtol, atol=0)
    assert np.allclose(a.chi_self, b.chi_self, rtol=rtol, atol=0)


@pytest.mark.parametrize("shell_mode", ["width", "exact"])
def test_backends_agree_2d(shell_mode):
    net = build_network(poisson_2d(400, 20.0, seed=11), 20.0, "D")
    w = net.edges_lengths ** 2
    a = compute_chi_ell(net, w, kmax=12.0, shell_mode=shell_mode)
    b = compute_chi_ell(net, w, kmax=12.0, shell_mode=shell_mode, backend="numba", rows_per_tile=7)
    _same(a, b)


def test_backends_agree_general_cell():
    c = Cell(np.array([[1.0, 0.15], [-0.05, 1.1]]) * 16.0)
    rng = np.random.default_rng(5)
    pts = rng.uniform(0, 1, size=(300, 2)) @ c.A
    _, pairs = voronoi_cells_general(pts, c)
    net = build_from_triangulation(pts, c, pairs, "D", compute_cell_areas=False)
    a = compute_chi_ell(net, net.edges_lengths, kmax=10.0)
    b = compute_chi_ell(net, net.edges_lengths, kmax=10.0, backend="numba", rows_per_tile=5)
    _same(a, b)


def test_backends_agree_3d():
    net = build_network_3d(poisson_3d(300, 7.0, seed=3), 7.0, "D")
    a = compute_chi_ell_3d(net, net.edges_lengths, kmax=5.0)
    b = compute_chi_ell_3d(net, net.edges_lengths, kmax=5.0, backend="numba", rows_per_tile=3)
    _same(a, b)
