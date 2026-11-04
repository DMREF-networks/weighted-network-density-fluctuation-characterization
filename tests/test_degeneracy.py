"""Degenerate generator sets: the V scheme must tile on an exact lattice (its
ancillary sites are the distinct Voronoi vertices), the pipeline must reject the
degenerate C tessellation of a lattice (TilingError; CLI exit 3), generic inputs
must tile, and small boxes must build (pyvoro block sizing)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.environ.get("WNDFC_TEST_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wndfc.network import build_network                            # noqa: E402
from wndfc.weights import compute_weights                          # noqa: E402
from wndfc.generators import poisson_2d, url_3d                    # noqa: E402
from wndfc.pipeline import build_weighted_network, TILING_TOL, TilingError  # noqa: E402


def square_lattice(n: int, L: float, jitter: float = 0.0, seed: int = 3) -> np.ndarray:
    g = (np.arange(n) + 0.5) * (L / n)
    p = np.array([(x, y) for x in g for y in g], dtype=float)
    if jitter:
        p = p + jitter * L * np.random.default_rng(seed).standard_normal(p.shape)
    return p


@pytest.mark.parametrize("L", [1.0, 20.0])
def test_v_scheme_square_lattice_tiles_without_jitter(L):
    pts = square_lattice(20, L)
    net = build_network(pts, L, "V")
    w = compute_weights(net, "rho_line_inv")
    assert net.n_edges == 2 * pts.shape[0]                 # 4-valent vertices: E = 2N
    assert net.build_diag["max_multiplicity"] == 4
    assert abs(w.sum() / L ** 2 - 1.0) < 1e-9


def test_v_scheme_near_degenerate_lattice_tiles():
    L = 1.0
    pts = square_lattice(20, L, jitter=1e-9)
    net = build_network(pts, L, "V")
    w = compute_weights(net, "rho_line_inv")
    assert abs(w.sum() / L ** 2 - 1.0) < 1e-9


@pytest.mark.parametrize("weight", ["midpt_sec_area", "rho_line_inv", "length_power"])
def test_pipeline_rejects_C_on_lattice(weight):
    L = 1.0
    pts = square_lattice(20, L)
    with pytest.raises(TilingError, match="degenerate=True"):
        build_weighted_network(pts, L, "C", weight)


def test_cli_exits_3_on_lattice_C(tmp_path):
    import contextlib
    import io
    from wndfc.__main__ import main
    out = tmp_path / "lat.npz"
    err = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
        rc = main(["--url", "10", "--url-a", "0", "-L", "10", "--ntype", "C", "--weight", "rho_line_inv",
                   "--kmax", "4", "--out", str(out)])
    assert rc == 3 and "degenerate=True" in err.getvalue()
    assert not out.exists()


@pytest.mark.parametrize("nt", ["D", "V", "G", "C"])
@pytest.mark.parametrize("weight", ["rho_line_inv", "midpt_sec_area", "dual_diamond_planar"])
def test_pipeline_generic_tiles(nt, weight):
    L = 20.0
    pts = poisson_2d(400, L, seed=11)
    net, w, info = build_weighted_network(pts, L, nt, weight)
    assert info["tiling_resid"] <= 1e-9


def test_small_box_builds_and_tiles():
    L = 1.0
    pts = poisson_2d(512, L, seed=5)
    for nt in ("D", "V"):
        net = build_network(pts, L, nt)
        assert abs(net.cell_areas.sum() / L ** 2 - 1.0) < 1e-9


def test_pipeline_3d_cubic_lattice_V():
    L = 6.0
    pts = url_3d(6, L, a=0.0, seed=0)                      # perfect cubic lattice
    net, w, info = build_weighted_network(pts, L, "V", "rho_line_inv", dim=3)
    assert info["tiling_resid"] <= TILING_TOL
