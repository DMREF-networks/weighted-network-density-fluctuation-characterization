"""Certification of the sampled pair estimator (h_ell.sample_pair_deposits)
on fixed-seed cells.

  self      the self deposits equal their exact per-bin expectation to counting
            noise (z = (dep - E)/sqrt(sq): rms ~ 1), and their mass is
            sum w_eff^2 exactly
  distinct  the distinct deposits equal the Gauss-Legendre pair quadrature on
            the same bins to counting noise (r >= 2 dr, where the GL check is
            itself reliable); the two bin-free sums M_0 agree
  plateau   the first fine bin of h_w reaches the analytic contact plateau
            (V tessellation; the D plateau is sliver-dominated)
  spectral  the Hankel inversion of the stored distinct spectrum agrees with the
            sampled h_w above the band limit
  mass      the sum rule as the bin-free sum and the total mass (self + distinct
            - background) close to the realization noise of one small cell
  moments   the bin-free M_0 equals the bin sum of h_w at rho_w != 1 exactly, the
            sampling sd of M_0 is the binomial value (below the squared-deposit bound),
            and small_r.area_law_full consumes the moments consistently
  storage   deterministic for a seed and for the batch size; NPZ round trip rebuilds
            the deposits and every read-time number

Run:  python -m pytest tests/test_h_ell_sampled.py -q     (from voro_clean_release/, .venv-voro)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.environ.get("WNDFC_TEST_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wndfc.generators import poisson_2d                                   # noqa: E402
from wndfc.network import build_network                                   # noqa: E402
from wndfc.pipeline import build_weighted_network                         # noqa: E402
from wndfc.spectral import compute_chi_ell                                # noqa: E402
from wndfc import norms                                                   # noqa: E402
from wndfc.io import load_result, save_result                             # noqa: E402
from wndfc.h_ell import (contact_plateau, deposits_from_npz, gl_pair_deposits,  # noqa: E402
                              real_space_curves, real_space_diagnostics, real_space_moments,
                              sample_pair_deposits, self_expected_deposits, spectral_h_w)
from wndfc.small_r import area_law_full                                   # noqa: E402

L, N, SEED = 20.0, 400, 11
R_MAX, N_BINS = 6.0, 30            # r_max + ell_max < L/2: no cutoff cap on this cell
KMAX = 30.0


@pytest.fixture(scope="module")
def cell():
    """D tessellation, theta = 1 (w_eff = ell) and the D scheme (tiling)."""
    pts = poisson_2d(N, L, seed=SEED)
    net = build_network(pts, L, "D")
    _, w_planar, info = build_weighted_network(pts, L, "D", "dual_diamond_planar", dim=2)
    assert info["tiling_resid"] <= 1e-8
    return net, {"th1": np.asarray(net.edges_lengths, float), "planar": np.asarray(w_planar, float)}


def _scalars(net, w_eff):
    V = net.V
    return V, float(w_eff.sum()) / V, float((w_eff ** 2).sum())


def _z(dep, ref, sq):
    ok = sq > 0
    return (dep[ok] - ref[ok]) / np.sqrt(sq[ok])


@pytest.mark.parametrize("wk", ["th1", "planar"])
def test_self_matches_expectation(cell, wk):
    net, W = cell
    w = W[wk]
    dep = sample_pair_deposits(net, w, r_max=R_MAX, n_bins=N_BINS, n_samples=64, seed=SEED)
    E = self_expected_deposits(dep.r_edges, net.edges_lengths, w)
    z = _z(dep.dep_self, E, dep.sq_self)
    assert np.sqrt(np.mean(z ** 2)) < 1.6 and np.abs(z).max() < 5.0, (np.sqrt(np.mean(z ** 2)), np.abs(z).max())
    # exact mass: every self draw deposits w_eff^2 / n_samples somewhere
    assert abs((dep.mom_self[0] + dep.self_overflow) / float((w ** 2).sum()) - 1.0) < 1e-12
    assert dep.self_overflow == 0.0                      # ell_max < r_max on this cell


@pytest.mark.parametrize("wk", ["th1", "planar"])
def test_distinct_matches_gauss_legendre(cell, wk):
    net, W = cell
    w = W[wk]
    dep = sample_pair_deposits(net, w, r_max=R_MAX, n_bins=N_BINS, n_samples=144, seed=SEED)
    assert not dep.cutoff_capped
    gl = gl_pair_deposits(net, w, r_edges=dep.r_edges, n_gl=12)
    dr = R_MAX / N_BINS
    m = dep.r >= 2.0 * dr
    z = _z(dep.dep_dist[m], gl[m], dep.sq_dist[m])
    assert np.sqrt(np.mean(z ** 2)) < 1.6 and np.abs(z).max() < 5.0, (np.sqrt(np.mean(z ** 2)), np.abs(z).max())
    # the bin-free sums agree to the sampling noise of the total (sum of the per-bin variances)
    V = net.V
    sd = np.sqrt(dep.sq_dist.sum()) / V
    assert abs(dep.mom_dist[0] - gl.sum()) / V < 4.0 * sd + 1e-9


def test_contact_plateau_first_bin():
    """V tessellation: h_w on [0, dr] with dr = 0.02 sits on the analytic plateau."""
    net = build_network(poisson_2d(N, L, seed=SEED), L, "V")
    w = np.asarray(net.edges_lengths, float)
    dep = sample_pair_deposits(net, w, r_max=1.0, n_bins=50, n_samples=1024, seed=SEED)
    V, rho_w, _ = _scalars(net, w)
    c = real_space_curves(dep, V, rho_w)
    h0 = contact_plateau(net, np.ones_like(w))          # per-length marks = 1 at theta = 1
    assert abs(h0 + 0.36) < 0.1, h0                      # the V plateau (stable across seeds)
    assert abs(c.h_dist[0] - h0) < max(4.0 * c.sd_dist[0], 0.06), (c.h_dist[0], c.sd_dist[0], h0)


def test_spectral_check_above_band_limit(cell):
    net, W = cell
    w = W["th1"]
    chi = compute_chi_ell(net, w, kmax=KMAX)
    dep = sample_pair_deposits(net, w, r_max=R_MAX, n_bins=N_BINS, n_samples=256, seed=SEED)
    V, rho_w, _ = _scalars(net, w)
    c = real_space_curves(dep, V, rho_w)
    h_spec = spectral_h_w(chi.k, chi.chi_ell, chi.chi_self, rho_w, c.r)
    m = c.r >= 2.0 * np.pi / KMAX * 2.0                  # twice the band limit
    rms = np.sqrt(np.mean((h_spec[m] - c.h_dist[m]) ** 2))
    assert rms < 6e-3, rms


def test_sum_rule_and_total_mass(cell):
    """Tiling scheme: M_0 (bin-free) vs -lambda <w^2 ell^2> and the total mass,
    both to the realization noise of one N = 400 cell (the ensemble closes)."""
    net, W = cell
    w = W["planar"]
    V, rho_w, s2 = _scalars(net, w)
    assert abs(rho_w - 1.0) < 1e-8
    dep = sample_pair_deposits(net, w, r_max=R_MAX, n_bins=N_BINS, n_samples=64, seed=SEED)
    dg = real_space_diagnostics(dep, V, rho_w, s2, norms.vertex_wedge_sum(net, w))
    assert dg["sum_rule_rhs"] == pytest.approx(-s2 / V)
    assert dg["sum_rule_lhs"] == pytest.approx(dg["M_0"])
    assert abs(dg["self_mass_rel"]) < 1e-12
    assert dg["sum_rule_relerr"] < 0.5, dg
    assert dg["total_mass_rel"] == pytest.approx(dg["sum_rule_relerr"], rel=1e-9)   # exact self mass
    assert dg["h_w0_plateau"] == pytest.approx(contact_plateau(net, w / net.edges_lengths))


def test_moments_at_rho_w_not_one(cell):
    """theta = 1 (rho_w = 3.4): M_0 is exactly the bin sum rho_w^2 sum_b h_b |shell_b|;
    M_n for n >= 1 agrees with the r_b^n bin sum to O(dr); the binomial sampling sd of
    M_0 lies below the squared-deposit bound and is reproduced by the seed scatter."""
    net, W = cell
    w = W["th1"]
    V, rho_w, s2 = _scalars(net, w)
    assert rho_w > 2.0
    dep = sample_pair_deposits(net, w, r_max=R_MAX, n_bins=N_BINS, n_samples=32, seed=SEED)
    c = real_space_curves(dep, V, rho_w)
    mom = real_space_moments(dep, V, rho_w)
    M0_bins = float(rho_w ** 2 * np.sum(c.h_dist * dep.shell))
    assert mom["M_0"] == pytest.approx(M0_bins, rel=1e-10, abs=1e-10)
    for n in (1, 2):
        Mn_bins = float(rho_w ** 2 * np.sum(c.h_dist * dep.shell * dep.r ** n))
        assert abs(mom[f"M_{n}"] - Mn_bins) < 0.05 * abs(Mn_bins) + 0.05, (n, mom[f"M_{n}"], Mn_bins)
    dg = real_space_diagnostics(dep, V, rho_w, s2)
    bound = float(np.sqrt(dep.sq_dist.sum()) / V)
    assert 0.0 < dg["M_0_sd"] < 0.7 * bound
    m0 = [real_space_moments(sample_pair_deposits(net, w, r_max=R_MAX, n_bins=N_BINS, n_samples=32,
                                                  seed=s), V, rho_w)["M_0"] for s in range(6)]
    sd_emp = float(np.std(m0, ddof=1))
    assert 0.4 * dg["M_0_sd"] < sd_emp < 2.5 * dg["M_0_sd"], (sd_emp, dg["M_0_sd"], bound)


def test_area_law_full_consumes_moments(cell):
    net, _ = cell
    R = np.array([1.0, 2.0, 3.0])
    diag = {}
    out = area_law_full(net, R, n_bins=N_BINS, n_samples=16, seed=SEED, r_max=R_MAX, diag_out=diag)
    assert np.all(np.isfinite(out)) and out.shape == R.shape
    assert diag["M0"] < 0 and diag["A2"] < diag["A2_self"]
    assert diag["A2"] == pytest.approx(np.pi * (net.rho_b * net.edge_moment(2.0) + diag["M0"]))
    assert out[-1] == pytest.approx(diag["A2"] * 9.0 + diag["A1"] * 3.0)


def test_deterministic_and_npz_round_trip(cell, tmp_path):
    net, W = cell
    w = W["th1"]
    a = sample_pair_deposits(net, w, r_max=R_MAX, n_bins=N_BINS, n_samples=16, seed=3)
    b = sample_pair_deposits(net, w, r_max=R_MAX, n_bins=N_BINS, n_samples=16, seed=3)
    for k in ("dep_self", "dep_dist", "sq_self", "sq_dist", "mom_self", "mom_dist"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k
    # batch-size independence (each edge / pair owns one block of the stream)
    b2 = sample_pair_deposits(net, w, r_max=R_MAX, n_bins=N_BINS, n_samples=16, seed=3, sample_batch=1 << 9)
    for k in ("dep_self", "dep_dist", "sq_self", "sq_dist", "mom_self", "mom_dist"):
        assert np.allclose(getattr(a, k), getattr(b2, k), rtol=1e-12, atol=1e-9), k
    assert b2.var_corr_dist == pytest.approx(a.var_corr_dist, rel=1e-12)
    c = sample_pair_deposits(net, w, r_max=R_MAX, n_bins=N_BINS, n_samples=16, seed=4)
    assert not np.array_equal(a.dep_dist, c.dep_dist)
    chi = compute_chi_ell(net, w, kmax=KMAX)
    meta = norms.build_metadata(net=net, w_eff=w, dim=2, gen="poi", tess="D", scheme="theta", theta=1.0,
                                seed=SEED, ens=1, k_max=KMAX, n_bins=chi.n_bins, shell_mode="width",
                                code_version="test")
    path = str(tmp_path / "round.npz")
    save_result(path, chi=chi, net=net, w_eff=w, meta=meta, h_ell=a)
    d = load_result(path)
    r = deposits_from_npz(d)
    for k in ("r_edges", "dep_self", "dep_dist", "sq_self", "sq_dist", "mom_self", "mom_dist"):
        assert np.array_equal(getattr(a, k), getattr(r, k)), k
    assert r.meta() == a.meta()
    m = d["metadata"]
    dg_run = real_space_diagnostics(a, net.V, float(w.sum()) / net.V, float((w ** 2).sum()), m["vertex_wedge_sum"])
    dg_read = real_space_diagnostics(r, m["V"], m["rho_w_emp"], m["sum_w2l2"], m["vertex_wedge_sum"])
    for k, v in dg_run.items():
        assert dg_read[k] == pytest.approx(v, rel=1e-12, abs=1e-14), k
    assert "h_ell_direct" not in d and "h_ell_spectral" not in d and "diagnostics" not in m["h_ell"]
