"""chi(k) on the D tessellation of a Poisson substrate under theta = 1, the D scheme
and the moment-matched D_2 scheme, with k^4 and k^6 guides.

    python figures/fig_d2_compare.py --data <root> --out <dir>

<root>/length_power_th1/*.npz, <root>/dual_diamond_planar/*.npz, <root>/d2_ring/*.npz.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chi_panels  # noqa: E402
from chi_panels import GUIDE_LS, PAPER_STYLE, Curve, RefLine, load_group_pooled, plot_chi_panel, scaled_axes  # noqa: E402

CURVES = (  # (subdir, label, color key), legend order
    ("dual_diamond_planar", r"$\mathcal{D}$", "Dbary"),
    ("d2_ring", r"$\mathcal{D}_{2}$", "D2"),
    ("length_power_th1", r"$\zeta{=}1$", "zeta1"),
)
chi_panels.TESS_COLORS.update({"Dbary": "#0072B2", "D2": "#56B4E9", "zeta1": "#009E73"})   # extra keys only
REFS = [RefLine(4, (0.04, 0.43), 0.433, GUIDE_LS[4], r"$k^{4}$", color="0.6", lw=3),
        RefLine(6, (0.01, 0.50), 0.215, GUIDE_LS[6], r"$k^{6}$", color="0.6", lw=3)]
XLIM, YLIM = (10 ** -2.8, 0.78), (10 ** -12.3, 10.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    curves = []
    for sub, label, key in CURVES:
        k, chi, meta, n = load_group_pooled([os.path.join(a.data, sub, "*.npz")])
        x, y = scaled_axes(k, chi, meta)
        curves.append(Curve(label=label, ntype=key, x=x, y=y, meta={"n_files": n, "rho_w_emp": meta["rho_w_emp"]}))
    out = plot_chi_panel(curves, os.path.join(a.out, "vc_poi_com_comparison.png"), refs=REFS, xlim=XLIM, ylim=YLIM,
                         legend_ncol=1, legend_bbox=(0.98, 0.02), legend_pad=0.2, style=PAPER_STYLE)
    print(out, " ".join(f"{c.label}:n={c.meta['n_files']},rho_w={c.meta['rho_w_emp']:.4f}" for c in curves))


if __name__ == "__main__":
    main()
