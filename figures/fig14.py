"""Fig. 14 -- loaded systems (2D ideal glass, stealthy hyperuniform, HIP) under the
V / T / D weight schemes, per tessellation.

    python figures/fig14.py --data <root> --out <dir> [--systems stealthy hip]

<root>/<system>/<T>_<weight>/*.npz with system in {ideal, stealthy, hip}.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chi_panels import PAPER_STYLE, curves_from_roots, guide, plot_chi_panel  # noqa: E402

WEIGHT = {"V": "rho_line_inv", "T": "midpt_sec_area", "D": "dual_diamond_planar"}
ANNOT = {"V": r"$\mathcal{V}$", "T": r"$\mathcal{T}$", "D": r"$\mathcal{D}$"}
REFS = {
    ("stealthy", "V"): [guide(2, (0.001, 0.4), 0.53), guide(4, (0.001, 0.4), 0.20)],
    ("stealthy", "T"): [guide(3, (0.001, 0.5), 0.35), guide(4, (0.001, 0.5), 0.08)],
    ("stealthy", "D"): [guide(3, (0.001, 0.5), 0.41), guide(4, (0.001, 0.5), 0.20)],
    ("hip", "V"): [guide(2, (0.001, 0.4), 0.53), guide(3, (0.001, 0.4), 0.20)],
    ("hip", "T"): [guide(2, (0.001, 0.4), 0.43), guide(3, (0.001, 0.4), 0.15)],
    ("hip", "D"): [guide(3, (0.001, 0.4), 0.29), guide(4, (0.001, 0.4), 0.06)],
    ("ideal", "V"): [guide(2, (0.001, 0.53), 0.63), guide(3, (0.001, 0.6), 0.24)],
    ("ideal", "T"): [guide(3, (0.001, 0.53), 0.47), guide(4, (0.001, 0.53), 0.24)],
    ("ideal", "D"): [guide(3, (0.001, 0.53), 0.38), guide(4, (0.001, 0.53), 0.24)],
}
# rebin: shells per decade (0 = native shells on the pooled grid)
LAYOUT = {
    "stealthy": dict(xlim=(10 ** -2, 10 ** -0.03), ylim=(10 ** -7.5, 10 ** 0.6), legend_layout="refs_left",
                     legend_pad=0.5, rebin=0),
    "hip":      dict(xlim=(10 ** -3, 0.8), ylim=(10 ** -6, 10 ** 1), legend_layout="rows", legend_pad=None, rebin=20),
    "ideal":    dict(xlim=(10 ** -3.1, 10 ** -0.03), ylim=(10 ** -7.5, 10 ** 0.6), legend_layout="refs_left",
                     legend_pad=0.5, rebin=0),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--systems", nargs="*", default=["ideal", "stealthy", "hip"])
    a = ap.parse_args(argv)
    for system in a.systems:
        lay = LAYOUT[system]
        for scheme in ("V", "T", "D"):
            roots = {nt: os.path.join(a.data, system, f"{nt}_{WEIGHT[scheme]}", "*.npz") for nt in "CDGV"}
            curves = curves_from_roots(roots, rebin=lay["rebin"])
            out = plot_chi_panel(curves, os.path.join(a.out, f"{system}_{scheme}.png"), refs=REFS[(system, scheme)],
                                 annot=ANNOT[scheme], xlim=lay["xlim"], ylim=lay["ylim"], legend_bbox=None,
                                 legend_layout=lay["legend_layout"], legend_pad=lay["legend_pad"], style=PAPER_STYLE)
            print(out, " ".join(f"{c.ntype}:n={c.meta['n_files']},rho_w={c.meta['rho_w_emp']:.4f}" for c in curves))


if __name__ == "__main__":
    main()
