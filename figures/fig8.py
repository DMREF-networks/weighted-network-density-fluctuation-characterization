"""Fig. 8 -- 2D chi(k) under the theta family (w_eff = ell^theta; the paper's zeta) per
tessellation, Poisson and URL substrates; theta = 0, 0.5, ..., 2.5 on the per-tessellation ramp.

    python figures/fig8.py --data <root> --out <dir>

<root> holds release-schema NPZs in <T>_<ctype>_length_power_th<X>p<Y>/ subdirs.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chi_panels import PAPER_STYLE, guide, plot_theta_panel, theta_curves  # noqa: E402

STYLE = dict(PAPER_STYLE, ystride=1, cbar_label_fs=12, guide_legend_kw=dict(handlelength=3.2, borderaxespad=0.01))
XLIM, YLIM = (10 ** -2.4, 0.95), (10 ** -3.3, 10 ** 0.88)
# k^2 guide on the URL panels: y anchor per tessellation
REFS = {("C", "url"): [guide(2, (0.0001, 0.53), 0.35)],
        ("D", "url"): [guide(2, (0.0001, 0.53), 0.40)],
        ("V", "url"): [guide(2, (0.0001, 0.53), 0.30)]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    for ctype in ("poi", "url"):
        for nt in "CDGV":
            curves = theta_curves(a.data, nt, ctype)
            out = plot_theta_panel(curves, nt, os.path.join(a.out, f"{nt}_vc_{ctype}_lp.png"),
                                   refs=REFS.get((nt, ctype), ()), xlim=XLIM, ylim=YLIM,
                                   legend_bbox=(0.986, 0.024), cbar_label=r"$\zeta$", style=STYLE)
            print(out, " ".join(f"th{th:g}:n={c.meta['n_files']}" for th, c in sorted(curves.items())))


if __name__ == "__main__":
    main()
