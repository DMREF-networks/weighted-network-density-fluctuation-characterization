"""Fig. 3 -- small-R laws vs the measured edge-length variance, 2D Poisson, theta = 1:
(a) exact self term (`unified`), (b) truncated area law (`regime_III_full`),
(c) exact close-pair comparator (`exact`, on its own R grid).  Lines = analytical,
markers = direct window sampling.  Same scaled axes as Fig. 2.

    python figures/fig3.py --data <root> --out <dir>

<root>/<T>.npz from legacy.convert_regimes_json (sigma2_R / sigma2_direct plus the
analytic curves; exact on exact_R).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sigma2_panels import Sigma2Curve, plot_sigma2_panel  # noqa: E402

# Per-panel legend and axis text, all in one place.  Every string is rendered
# through text.usetex when latex is on the PATH (chi_panels._RC), so: raw
# strings, math inside $...$, escape % & # _ outside math, no unicode.
#   headers    -> legend column headers, (line column, marker column)
#   xlabel / ylabel -> axis labels
#   legend_loc -> anchored-legend location
PANELS = {
    "unified": {                                   # (a) exact self term
        "headers": (r"$\sigma^{2^{(s)}}_\ell$", r"$\sigma^2_\ell$"),
        "xlabel": r"$R \lambda_{\ell}^{1/2}$",
        "ylabel": r"$\hat{\sigma}^2_\ell(R) \, \lambda_{\ell}^{1/2} \, / \langle w^2 \ell \rangle$",
        "legend_loc": "lower right",
    },
    "regime_III_full": {                           # (b) truncated area law
        "headers": (r"Area Law", r"$\sigma^2_\ell$"),
        "xlabel": r"$R \lambda_{\ell}^{1/2}$",
        "ylabel": r"$\hat{\sigma}^2_\ell(R) \, \lambda_{\ell}^{1/2} \, / \langle w^2 \ell \rangle$",
        "legend_loc": "lower right",
    },
    "exact": {                                     # (c) close-pair comparator
        "headers": (r"Analytical", r"Measured"),
        "xlabel": r"$R \lambda_{\ell}^{1/2}$",
        "ylabel": r"$\sigma^2_\ell(R) \, \lambda_{\ell}^{1/2} \, / \langle w^2 \ell \rangle$",
        "legend_loc": "lower right",
    },
}

# C:\Users\runes\OneDrive\Personal files pre 12072024\Desktop\Research\elv_up\voro_clean_release\plots\fig3ext\notes.md

def _scales(meta: dict):
    lam = float(meta["N_ell"]) / float(meta["V"])
    return np.sqrt(lam), float(meta["sum_w2l"]) / float(meta["N_ell"])


def load_curve(path: str, curve: str, direct_path: str = None) -> Sigma2Curve:
    """Analytic curves from ``path``; the measured markers from ``direct_path``
    when given (a different dataset is fine: each side is scaled by its own
    metadata, and the scaled axes are universal per tessellation)."""
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["metadata_json"]))
        R_a = d["exact_R"] if curve == "exact" else d["sigma2_R"]
        y_a, R_d, y_d = d[curve], d["sigma2_R"], d["sigma2_direct"]
    if curve == "exact":                      # drop the finite-size top: R/L > 0.1
        keep = R_a <= 0.18 * float(meta["L"] or 32.0)
        R_a, y_a = R_a[keep], y_a[keep]
    dmeta = meta
    if direct_path is not None:
        with np.load(direct_path, allow_pickle=False) as d:
            dmeta = json.loads(str(d["metadata_json"]))
            R_d, y_d = d["sigma2_R"], d["sigma2_direct"]
    s, m2 = _scales(meta)
    ds, dm2 = _scales(dmeta)
    x_spec, x_dir = R_a * s, R_d * ds
    m = x_dir <= x_spec.max() * (1.0 + 1e-9)    # markers only where the analytic line reaches
    return Sigma2Curve(ntype=meta["tess"], x_spec=x_spec, y_spec=y_a * s / m2,
                       x_dir=x_dir[m], y_dir=y_d[m] * ds / dm2, meta=meta)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--direct-data", default=None, metavar="DIR",
                    help="take the measured markers (sigma2_R / sigma2_direct) from DIR/<T>.npz; "
                         "analytic curves still come from --data")
    a = ap.parse_args(argv)
    for curve, p in PANELS.items():
        curves = [load_curve(os.path.join(a.data, f"{t}.npz"), curve,
                             os.path.join(a.direct_data, f"{t}.npz") if a.direct_data else None)
                  for t in "CDGV"]
        out = plot_sigma2_panel(curves, os.path.join(a.out, f"{curve}.png"), legend_loc=p["legend_loc"],
                                headers=p["headers"], xlabel=p["xlabel"], ylabel=p["ylabel"],
                                line_first=True,legend_scale=.9)
        print(out, " ".join(f"{c.ntype}:lam={c.meta['N_ell']:.4f}" for c in curves))


if __name__ == "__main__":
    main()
