"""chi(k) of loaded 2D systems (stealthy hyperuniform, HIP) at theta = 1, per
tessellation -- the length-weighted companions of the contact-network panel.

    python figures/fig_loaded_th1.py --data <root> --out <dir> [--systems stealthy hip]

<root>/<system>/<T>_length_power_th1/*.npz.  HIP curves are rebinned to 20 shells
per decade and carry a k^{-1} guide (anti-hyperuniform).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chi_panels import PAPER_STYLE, curves_from_roots, guide, plot_chi_panel  # noqa: E402

LAYOUT = {  # rebin: shells per decade (0 = native shells on the pooled grid)
    "stealthy": dict(xlim=(10 ** -2, 10 ** -0.03), ylim=(10 ** -3.3, 10 ** 0.6), rebin=0, refs=(),
                     legend_loc="upper left", legend_layout="cols", legend_pad=1.2,
                     style=dict(PAPER_STYLE, legend_kw=dict(PAPER_STYLE["legend_kw"], handlelength=2.9, columnspacing=1.65))),
    "hip":      dict(xlim=(10 ** -3, 0.8), ylim=(10 ** -1.5, 10 ** 2.8), rebin=20, refs=(guide(-1, (0.0001, 0.5), 0.7),),
                     legend_loc="lower left", legend_layout="rows", legend_pad=0.8, style=PAPER_STYLE),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--systems", nargs="*", default=list(LAYOUT))
    a = ap.parse_args(argv)
    for system in a.systems:
        lay = LAYOUT[system]
        roots = {T: os.path.join(a.data, system, f"{T}_length_power_th1", "*.npz") for T in "CDGV"}
        curves = curves_from_roots(roots, rebin=lay["rebin"])
        out = plot_chi_panel(curves, os.path.join(a.out, f"{system}_th1.png"), refs=lay["refs"], xlim=lay["xlim"],
                             ylim=lay["ylim"], legend_loc=lay["legend_loc"], legend_bbox=None,
                             legend_layout=lay["legend_layout"], legend_pad=lay["legend_pad"], style=lay["style"])
        print(out, " ".join(f"{c.ntype}:n={c.meta['n_files']},rho_w={c.meta['rho_w_emp']:.4f}" for c in curves))


if __name__ == "__main__":
    main()
