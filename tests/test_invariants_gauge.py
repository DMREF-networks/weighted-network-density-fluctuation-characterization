"""Release invariant tests -- the gauge-DEPENDENT half (bare 1/V API).

Requires the Phase 2 API: ``compute_chi_ell`` returns ``chi_self`` (binned
per-mode empirical self), ``parseval_sigma2(chi, R_grid, dim)`` has no
<w ell> argument, and ``norms.build_metadata`` exists.  The module skips itself
until that API is present.

  1. self tail:      k chi_self(k) -> 2 C_d lambda <w^2 ell>_kmax at the top shells;
                     total/self -> 1 there (theta >= 1; V theta=0 excluded, see triage)
  2. sum rule (0):   chi_self(k_min) ~ lambda <w^2 ell^2>   (sinc^2 -> 1)
  3. small R:        direct sigma^2 == exact comparator (close-pair + self) at any R
  4. Parseval:       bridge(chi_bare) vs direct sigma^2, and exact linearity in the gauge
  6. theta pin:      theta in {0,1,2}: spectral (bridge) vs real-space (direct) agree,
                     and the w_eff / per-length API boundary is consistent
  7. rho_w empirical from norms == sum w_eff / V
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# WNDFC_TEST_ROOT overrides the package root (e.g. a staged copy)
ROOT = os.environ.get("WNDFC_TEST_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wndfc.spectral import ChiEll2D, compute_chi_ell            # noqa: E402

if "chi_self" not in getattr(ChiEll2D, "__dataclass_fields__", {}):
    pytest.skip("bare-gauge API (chi_self on ChiEll2D) not present yet -- Phase 2", allow_module_level=True)

from wndfc.generators import poisson_2d                            # noqa: E402
from wndfc.network import build_network                            # noqa: E402
from wndfc.weights import compute_weights                          # noqa: E402
from wndfc.variance import compute_sigma2                          # noqa: E402
from wndfc.parseval import parseval_sigma2                         # noqa: E402
from wndfc import small_r as sr                                    # noqa: E402
from wndfc import norms                                            # noqa: E402

L, N, SEED = 30.0, 900, 11
KMAX = 20.0          # self tail / metadata k_max
KMAX_PIN = 40.0      # Parseval pin (theta = 0 needs k R >~ 30 at the smallest R)


@pytest.fixture(scope="module")
def net():
    return build_network(poisson_2d(N, L, seed=SEED), L, "D")


def _w_eff(net, theta):
    return net.edges_lengths ** theta


def _meta(net, w_eff, theta, n_bins):
    return norms.build_metadata(net=net, w_eff=w_eff, dim=2, gen="poisson", tess="D",
                                scheme="theta", theta=theta, seed=SEED, ens=1,
                                k_max=KMAX, n_bins=n_bins, shell_mode="width",
                                code_version="test", with_wedge=False)


# 1. self tail + total/self -> 1
@pytest.mark.parametrize("theta", [1.0, 2.0])
def test_self_tail(net, theta):
    w_eff = _w_eff(net, theta)
    chi = compute_chi_ell(net, w_eff, kmax=KMAX)
    dv = norms.derived(_meta(net, w_eff, theta, chi.k.size))
    n_top = max(5, chi.k.size // 10)
    top = slice(chi.k.size - n_top, None)
    tail = np.median(chi.k[top] * chi.chi_self[top])
    assert abs(tail / dv["porod_tail"] - 1.0) < 3e-2
    assert abs(np.median(chi.chi_ell[top] / chi.chi_self[top]) - 1.0) < 0.15


# 2. chi_self(k -> 0) = lambda <w^2 ell^2>
@pytest.mark.parametrize("theta", [0.0, 1.0, 2.0])
def test_chi_self_zero(net, theta):
    w_eff = _w_eff(net, theta)
    chi = compute_chi_ell(net, w_eff, kmax=5.0)
    dv = norms.derived(_meta(net, w_eff, theta, chi.k.size))
    r = chi.chi_self[0] / dv["chi_self_0"]
    assert 0.97 < r <= 1.0 + 1e-12          # sinc^2 <= 1, and ~1 at k = dk


# 7. rho_w empirical
def test_rho_w_empirical(net):
    for wtok in ("rho_line_inv", "midpt_sec_area", "dual_diamond_planar"):
        w_eff = compute_weights(net, wtok)
        m = _meta(net, w_eff, None, 1)
        assert abs(m["rho_w_emp"] - float(np.sum(w_eff)) / (L * L)) < 1e-14
        assert abs(m["rho_w_emp"] - 1.0) < 1e-9, wtok       # tiling schemes


# 3. small R: direct == exact comparator (close-pair + self) at any R
def test_small_R_exact_comparator(net):
    w_eff = _w_eff(net, 1.0)
    w_pl = w_eff / net.edges_lengths
    R = np.array([0.4, 0.6, 0.9, 1.2])
    direct = compute_sigma2(net, w_pl, R_grid=R, n_windows=40000, seed=SEED).sigma2
    exact = sr.unified_prediction(net, R, weights=w_pl) + sr.close_pair_distinct_sigma2(net, R, weights=w_pl, n_gauss=16)
    assert np.all(np.abs(direct / exact - 1.0) < 0.06), (direct / exact)
    c4 = sr.c4_exact(net, weights=w_pl)
    lead = c4["leading_R3_coeff"] * R[0] ** 3 + c4["C4"] * R[0] ** 4
    if c4["R0"] is not None and R[0] < c4["R0"]:
        assert abs(direct[0] / lead - 1.0) < 0.05


# 4 + 6. Parseval bridge vs direct, per theta (the theta pin), and gauge linearity
@pytest.mark.parametrize("theta", [0.0, 1.0, 2.0])
def test_parseval_theta_pin(net, theta):
    w_eff = _w_eff(net, theta)
    R = np.geomspace(0.8, 1.5, 4)                      # R <= L/20: >= 100 independent windows
    chi = compute_chi_ell(net, w_eff, kmax=KMAX_PIN)
    bridge = parseval_sigma2(chi, R, 2)
    direct = compute_sigma2(net, w_eff / net.edges_lengths, R_grid=R, n_windows=30000, seed=SEED).sigma2
    assert np.all(np.abs(bridge / direct - 1.0) < 0.06), (bridge / direct)
    # gauge linearity: a dressed array fed to the bare bridge is off by exactly the factor
    c = 1.7
    chi_c = ChiEll2D(k=chi.k, chi_ell=chi.chi_ell * c, counts=chi.counts, chi_self=chi.chi_self * c)
    assert np.allclose(parseval_sigma2(chi_c, R, 2), c * bridge, rtol=1e-12)
    # API boundary: norms (w_eff) and small_r (per-length) agree on the truncated moment
    s = norms.moment_sums(net.edges_lengths, w_eff, KMAX_PIN, 2)["sum_w2l_kmax"]
    assert np.isclose(s, net.n_edges * sr.porod_moment(net, w_eff / net.edges_lengths, KMAX_PIN, dim=2), rtol=1e-10)
