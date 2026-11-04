"""Release invariant tests -- the gauge-independent half (tiling, moment sums,
analytic-library identities).

  2. tiling:      sum_e w_eff = V  (=> rho_w_emp = 1)  for the V/T/D schemes
  5. self collapse: y = <w^2 2(ell-r)_+>/<w^2 ell> vs x = r <w^2 ell>/<w^2 ell^2>
                  has intercept 2 and unit area (pure function of the ell/w table)
  6. rho_w empirical = sum w_eff / V, never nominal
  +  Porod limits: x g_d(x) -> C_d, and norms.sum_w2l_kmax == N * small_r.porod_moment
  +  analytic-library closed forms are finite and monotone where they must be

Run:  python -m pytest tests -q     (from voro_clean_release/, .venv-voro)
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.integrate import trapezoid
import pytest

# WNDFC_TEST_ROOT overrides the package root (e.g. a staged copy)
ROOT = os.environ.get("WNDFC_TEST_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wndfc.generators import poisson_2d, poisson_3d, url_2d          # noqa: E402
from wndfc.network import build_network                               # noqa: E402
from wndfc.network_3d import build_network_3d                         # noqa: E402
from wndfc.weights import compute_weights                             # noqa: E402
from wndfc.weights_3d import compute_weights_3d                       # noqa: E402
from wndfc import small_r as sr                                        # noqa: E402
from wndfc import norms                                                # noqa: E402

L2, N2 = 20.0, 400
L3, N3 = 10.0, 1000
SEED = 11


@pytest.fixture(scope="module")
def nets2d():
    pts = poisson_2d(N2, L2, seed=SEED)
    return {nt: build_network(pts, L2, nt) for nt in ("D", "V", "G", "C")}


@pytest.fixture(scope="module")
def nets2d_url():
    pts = url_2d(20, L2, a=1.0, seed=SEED)
    return {nt: build_network(pts, L2, nt) for nt in ("D", "V", "G", "C")}


@pytest.fixture(scope="module")
def nets3d():
    pts = poisson_3d(N3, L3, seed=SEED)
    return {nt: build_network_3d(pts, L3, nt) for nt in ("D", "V", "C")}


# --------------------------------------------------------------------------
# 2. Tiling: sum w_eff = V for the cell-measure schemes (2D and 3D)
# --------------------------------------------------------------------------
SCHEMES_2D = {
    "V": dict(weight="rho_line_inv"),
    "T": dict(weight="midpt_sec_area"),
    "D": dict(weight="dual_diamond_planar"),
}


@pytest.mark.parametrize("nt", ["D", "V", "G", "C"])
@pytest.mark.parametrize("scheme", ["V", "T", "D"])
def test_tiling_2d(nets2d, nt, scheme):
    net = nets2d[nt]
    kw = dict(SCHEMES_2D[scheme])
    w_eff = compute_weights(net, kw.pop("weight"), **kw)
    assert np.all(np.isfinite(w_eff))
    assert abs(float(np.sum(w_eff)) / (L2 * L2) - 1.0) < 1e-9, scheme
    assert abs(norms.moment_sums(net.edges_lengths, w_eff, None, 2)["sum_wl"] / (L2 * L2) - 1.0) < 1e-9


@pytest.mark.parametrize("nt", ["D", "V", "G", "C"])
@pytest.mark.parametrize("scheme", ["V", "T", "D"])
def test_tiling_2d_url(nets2d_url, nt, scheme):
    net = nets2d_url[nt]
    kw = dict(SCHEMES_2D[scheme])
    w_eff = compute_weights(net, kw.pop("weight"), **kw)
    assert abs(float(np.sum(w_eff)) / (L2 * L2) - 1.0) < 1e-9, scheme


SCHEMES_3D = {
    "V": dict(weight="rho_line_inv"),
    "T": dict(weight="midpt_sec_area"),
    "D": dict(weight="dual_diamond_planar"),
}


@pytest.mark.parametrize("nt", ["D", "V", "C"])
@pytest.mark.parametrize("scheme", ["V", "T", "D"])
def test_tiling_3d(nets3d, nt, scheme):
    net = nets3d[nt]
    kw = dict(SCHEMES_3D[scheme])
    w_eff = compute_weights_3d(net, kw.pop("weight"), **kw)
    assert np.all(np.isfinite(w_eff))
    assert abs(float(np.sum(w_eff)) / (L3 ** 3) - 1.0) < 1e-9, scheme


# --------------------------------------------------------------------------
# 6. rho_w empirical, and theta family conventions
# --------------------------------------------------------------------------
@pytest.mark.parametrize("theta", [0.0, 1.0, 2.0])
def test_theta_family_is_ell_power(nets2d, theta):
    net = nets2d["D"]
    w_eff = compute_weights(net, "length_power", theta=theta)
    assert np.allclose(w_eff, net.edges_lengths ** theta, rtol=1e-14, atol=0)
    sums = norms.moment_sums(net.edges_lengths, w_eff, None, 2)
    assert abs(sums["sum_wl"] / (L2 * L2) - float(np.sum(w_eff)) / (L2 * L2)) < 1e-15
    # per-length w = ell^(theta-1): <w^2 ell> = <ell^(2 theta - 1)>
    assert np.isclose(sums["sum_w2l"], float(np.sum(net.edges_lengths ** (2 * theta - 1))), rtol=1e-12)


def test_theta1_is_unweighted(nets2d):
    """theta = 1 is the unweighted kernel (w_eff = ell); no separate 'uniform' token."""
    net = nets2d["C"]
    assert np.allclose(compute_weights(net, "length_power", theta=1.0), net.edges_lengths, rtol=1e-14, atol=0)
    with pytest.raises(ValueError):
        compute_weights(net, "uniform")


# --------------------------------------------------------------------------
# 5. Self collapse (pure function of the edge table)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("theta", [0.0, 1.0, 2.0])
def test_self_collapse(nets2d, theta):
    net = nets2d["D"]
    ell = net.edges_lengths
    w = ell ** (theta - 1.0)                      # per-length mark
    w2l, w2l2 = float(np.mean(w * w * ell)), float(np.mean(w * w * ell ** 2))
    x = np.linspace(0.0, 12.0, 4001)
    r = x * w2l2 / w2l
    y = np.array([np.mean(w * w * 2.0 * np.clip(ell - ri, 0.0, None)) for ri in r]) / w2l
    assert abs(y[0] - 2.0) < 1e-12                      # intercept 2
    assert abs(trapezoid(y, x) - 1.0) < 2e-3             # unit area (grid-limited)


# --------------------------------------------------------------------------
# Porod / truncated-moment consistency between norms.py and small_r.py
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 3])
def test_porod_limit(dim):
    x = np.array([50.0, 200.0, 1000.0])
    g = sr.g_d_form_factor(x, dim=dim)
    assert np.allclose(x * g, norms.C_D[dim], rtol=2e-2)
    assert abs(float(sr.g_d_form_factor(np.array([0.0]), dim=dim)[0]) - 1.0) < 1e-9


@pytest.mark.parametrize("nt", ["D", "V"])
def test_truncated_moment_matches_small_r(nets2d, nt):
    net = nets2d[nt]
    ell = net.edges_lengths
    w = ell                                       # per-length w = ell (theta = 2)
    w_eff = w * ell
    kmax = 30.0
    s = norms.moment_sums(ell, w_eff, kmax, 2)["sum_w2l_kmax"]
    assert np.isclose(s, net.n_edges * sr.porod_moment(net, w, kmax, dim=2), rtol=1e-10)
    # x g_2(x) -> 1 from both sides (oscillatory), so only the limit is asserted
    s_big = norms.moment_sums(ell, w_eff, 1e5, 2)["sum_w2l_kmax"]
    assert np.isclose(s_big, norms.moment_sums(ell, w_eff, None, 2)["sum_w2l"], rtol=1e-3)


def test_vertex_wedge_sum_finite(nets2d):
    for nt, net in nets2d.items():
        v = norms.vertex_wedge_sum(net, net.edges_lengths)   # w_eff = ell -> w = 1
        assert np.isfinite(v) and v > 0, nt


def test_analytic_closed_forms():
    a = np.array([0.1, 0.5, 1.0, 2.0, 10.0])
    g2 = sr.g_kernel(a, order=0)
    assert np.all(np.isfinite(g2)) and np.all(np.diff(g2) > 0)
    B = np.array([-0.9, 0.0, 0.9])
    assert np.all(np.isfinite(sr.H_infty(B, dim=2))) and np.all(np.isfinite(sr.H_infty(B, dim=3)))
    assert np.isclose(sr.crossover_Rstar.__doc__ is not None, True)
