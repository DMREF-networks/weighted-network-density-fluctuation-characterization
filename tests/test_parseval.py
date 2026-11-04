"""Window-kernel invariants for parseval.py.

Both window kernels tilde_alpha2(k;R) = |hat w_R(k)|^2 / v_d(R) must satisfy

    tilde_alpha2(k -> 0; R) = v_d(R)                       (continuity with the k = 0 branch)
    int tilde_alpha2(k;R) d^d k / (2 pi)^d = 1             (|hat w_R|^2 integrates to v_d)

and parseval_sigma2 on a flat chi_ell = 1 must return sigma^2(R) = v_d(R).
A kernel wrong by any R-dependent factor fails all three.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
from scipy.integrate import trapezoid
import pytest

ROOT = os.environ.get("WNDFC_TEST_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wndfc.parseval import (_alpha2_disk_2d, _alpha2_sphere_3d,  # noqa: E402
                                             parseval_sigma2)


def _v_d(R, dim):
    return np.pi * R * R if dim == 2 else (4.0 / 3.0) * np.pi * R ** 3


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("R", [0.05, 1.0, 7.0])
def test_kernel_parseval_identity_and_k0_limit(dim, R):
    kern = _alpha2_disk_2d if dim == 2 else _alpha2_sphere_3d
    # k -> 0: |hat w_R|^2 / v_d -> v_d, and the k > 0 branch must join the k = 0 branch
    assert kern(np.array([0.0]), R)[0] == pytest.approx(_v_d(R, dim))
    # kR = 1e-3: series deviation O((kR)^2) ~ 2e-7; smaller kR hits cancellation in sin t - t cos t
    assert kern(np.array([1e-3 / R]), R)[0] == pytest.approx(_v_d(R, dim), rel=1e-6)
    # int tilde_alpha2 d^d k / (2 pi)^d = 1
    k = np.linspace(1e-6, 4000.0 / R, 4_000_001)
    jac = k / (2.0 * np.pi) if dim == 2 else k ** 2 / (2.0 * np.pi ** 2)
    val = trapezoid(kern(k, R) * jac, k)
    assert val == pytest.approx(1.0, rel=2e-3)


@pytest.mark.parametrize("dim", [2, 3])
def test_parseval_sigma2_flat_chi_gives_window_volume(dim):
    R_grid = np.array([0.1, 0.5, 2.0])
    k = np.linspace(1e-6, 4000.0 / R_grid.min(), 4_000_001)
    chi = SimpleNamespace(k=k, chi_ell=np.ones_like(k))
    s2 = parseval_sigma2(chi, R_grid, dim)
    assert s2 == pytest.approx(_v_d(R_grid, dim), rel=2e-3)
