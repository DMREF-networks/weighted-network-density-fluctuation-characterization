"""The paper's scheme symbols (V, T, D, D2, zeta) and the MC* names resolve to the
weight tokens on the CLI and in the API; the tokens themselves pass through."""
import os
import sys

import pytest

ROOT = os.environ.get("WNDFC_TEST_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from wndfc import SCHEME, WEIGHTS, resolve_weight  # noqa: E402
from wndfc.__main__ import _parse_args  # noqa: E402

ALIASES = {"V": "rho_line_inv", "T": "midpt_sec_area", "D": "dual_diamond_planar", "D2": "d2_ring",
           "MCV": "rho_line_inv", "MCT": "midpt_sec_area", "MCD": "dual_diamond_planar", "MCD2": "d2_ring",
           "theta": "length_power", "zeta": "length_power"}


@pytest.mark.parametrize("alias,token", sorted(ALIASES.items()))
def test_alias_resolves(alias, token):
    assert resolve_weight(alias) == token
    assert resolve_weight(alias.lower()) == token
    assert SCHEME[token] in ("V", "T", "D", "D2", "theta")


@pytest.mark.parametrize("token", WEIGHTS)
def test_token_passes_through(token):
    assert resolve_weight(token) == token


def test_unknown_raises():
    with pytest.raises(ValueError):
        resolve_weight("kite")


def test_cli_accepts_alias_and_zeta():
    a = _parse_args(["--poisson", "10", "-L", "5", "--weight", "V"])
    assert a.weight == "rho_line_inv"
    a = _parse_args(["--poisson", "10", "-L", "5", "--weight", "length_power", "--zeta", "2"])
    assert a.weight == "length_power" and a.theta == 2.0
