"""Figs. 9 and 11 -- self / distinct / total decomposition of chi(k), one panel
per tessellation.  Fig. 9: theta = 0, 2 (Poisson) and theta = 0, 1 (URL);
Fig. 11: the D scheme (dual_bary) on Poisson and URL.

    python figures/fig9.py --data <root> --out <dir>

<root>/<stem>/<T>/*.npz, stem in {th0_poi, th2_poi, th0_url, th1_url,
dual_bary_poi, dual_bary_url}; each NPZ carries chi_ell and chi_self.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chi_panels import RefLine  # noqa: E402
from decomp_panels import load_decomp, plot_decomp_panel  # noqa: E402

STEMS = ("th0_poi", "th1_poi", "th2_poi", "th0_url", "th1_url", "th2_url", "dual_bary_poi", "dual_bary_url")
XLIM = (10 ** -2.4, 2.8)
YLIM = (10 ** -9, 10 ** 2)
# inset placement (axes fractions: x0, y0, w, h): default plus per-(stem, tess) overrides
INSET_RECT = (0.15, 0.13, 0.23, 0.30)
INSET_RECTS = {
    # ("th0_poi", "C"): (0.15, 0.13, 0.35, 0.28),
    # ("th2_poi", "C"): (0.15, 0.13, 0.35, 0.28),
    # ("th2_url", "C"): (0.15, 0.13, 0.35, 0.28),
    # ("th1_poi", "C"): (0.15, 0.13, 0.35, 0.28),
    # ("th0_url", "C"): (0.35, 0.1, 0.35, 0.28),
    # ("th1_url", "C"): (0.35, 0.1, 0.35, 0.28),
    ("th0_poi", "C"): (0.13, 0.3, 0.4, 0.33),
    ("th2_poi", "C"): (0.25, 0.15, 0.4, 0.33),
    ("th2_url", "C"): (0.25, 0.15, 0.4, 0.33),
    ("th1_poi", "C"): (0.25, 0.15, 0.4, 0.33),
    ("th0_url", "C"): (0.25, 0.15, 0.4, 0.33),
    ("th1_url", "C"): (0.25, 0.15, 0.4, 0.33),
    ("dual_bary_poi", "C"): (0.39, 0.1, 0.4, 0.33),
    ("dual_bary_url", "C"): (0.5, 0.09, 0.4, 0.33),
}
# inset indicator box: minimum height as a fraction of the axes height (the inset y window is
# sub-pixel on the log axis and would draw as a line); unlisted panels keep the exact window.
IND_MIN_H = {
    ("dual_bary_url", "C"): 0.03,
    ("th0_url", "C"): 0.03,
    ("th1_url", "C"): 0.03,
    ("th0_poi", "C"): .03,
    ("th2_poi", "C"): .03,
    ("th2_url", "C"): .03,
    ("th1_poi", "C"): .03,
    ("th2_poi", "C"): .03,
    ("dual_bary_poi", "C"): .03 
    }
# legend layout per (stem, tess): [(loc, keys), ...] or [(loc, keys, legend_kwargs), ...],
# keys from "self", "dist", "total", "" (blank spacer cell); one legend per group in the listed
# order; legend_kwargs override the shared style for that group, e.g. {"borderaxespad": 1.2},
# {"bbox_to_anchor": (0.98, 0.25)} or {"ncol": 1}.  Cells fill column by column (left column
# first), so ("self", "", "dist", "total") = self alone top-left, dist/total stacked right;
# ("", "self", "dist", "total") = self bottom-left.  Unlisted panels use the single default legend.
# LEGEND_ON switches the main legend per (stem, tess); unlisted = off.
LEGEND_ON = {("th0_poi", "C"): True}
LEGENDS = {
    ("th0_poi", "C"): [("lower right", ("", "self", "dist", "total"),
                    {"bbox_to_anchor": (0.95, 0.05), "borderaxespad": 0})],
    ("th2_poi", "C"): [("lower right", ("self", "dist", "total",),{"bbox_to_anchor": (0.88,0.08)})],
    ("th2_url", "C"): [("lower right", ("self", "dist", "total",),{"bbox_to_anchor": (0.88,0.08)})],
    ("th1_poi", "C"): [("lower right", ("self", "dist", "total",),{"bbox_to_anchor": (0.88,0.08)})],
    ("th0_url", "C"): [("lower right", ("self", "dist", "total",),{"bbox_to_anchor": (0.88,0.08)})],
    ("th1_url", "C"): [("lower right", ("self", "dist", "total",),{"bbox_to_anchor": (0.88,0.08)})],
    ("th1_poi", "C"): [("lower right", ("self", "dist", "total",),{"bbox_to_anchor": (0.88,0.08)})],
    ("dual_bary_poi", "C"):  [("lower right", ("self", "dist", "total",),{"bbox_to_anchor": (0.37,0.4)})],
    ("dual_bary_url", "C"):  [("lower right", ("self", "dist", "total",),{"bbox_to_anchor": (0.37,0.4)})],
}
# power-law guide lines per (stem, tess): _ref(exp, y_frac, (x0_frac, x1_frac)); y_frac is the
# log fraction of YLIM at which the line crosses the middle of its x region, the x region is in
# log fractions of XLIM (0 = left edge, 1 = right edge).  Lower y_frac = lower line; a shorter
# region ends the line earlier.  k^4 on the D scheme (dual_bary), k^2 on th0 URL.  The labels sit in
# their own legend: REF_LEGENDS per (stem, tess) as loc or (loc, legend_kwargs), else REF_LEGEND.
REF_LS = {2: (0, (1.0, 1.6)), 3: (0, (4.0, 1.4, 1.0, 1.4)), 4: (0, (5.0, 2.0))}


def _ref(exp, y_frac, region=(0.05, 0.45)):
    return [RefLine(exp, region, y_frac, REF_LS[exp], f"$k^{{{exp}}}$")]


REF_LEGEND = "upper right"
REF_LEGENDS = {
    # ("dual_bary_url", "C"): ("upper left", {"bbox_to_anchor": (0.02, 0.98)}),
}


# per-component style, keys "self" / "dist" / "total".  Colour: int = rung of TESS_RAMPS[tess]
# (0 light .. 10 dark), "base" = TESS_COLORS[tess], else any matplotlib colour.  Linestyle: any
# matplotlib ls incl. dash tuples.  Omit a key to keep the default (grey -- self, base - dist, k : total).
COLORS = {"self": "#452f00", "dist": "#56B4E9", "total": "#E69F00"}
LINESTYLES = {"self": (0, (1, 3)),          # dash-dot
              "dist": "-",  # dash-dot-dot
              "total": "-"}

REFS = {
    ("th0_url", "C"): _ref(2, 0.6),
    ("th0_url", "D"): _ref(2, 0.55),
    ("th0_url", "G"): _ref(2, 0.55),
    ("th0_url", "V"): _ref(2, 0.55),
    ("dual_bary_poi", "C"): _ref(4, 0.48),
    ("dual_bary_poi", "D"): _ref(4, 0.45),
    ("dual_bary_poi", "G"): _ref(4, 0.45),
    ("dual_bary_poi", "V"): _ref(4, 0.36),
    ("dual_bary_url", "C"): _ref(4, 0.39),
    ("dual_bary_url", "D"): _ref(4, 0.45),
    ("dual_bary_url", "G"): _ref(4, 0.45),
    ("dual_bary_url", "V"): _ref(4, 0.45),
}
# export ZOOM_dual_bary_poi="--inset-rect 0.15 0.38 0.23 0.28"
# export ZOOM_dual_bary_url="--inset-rect 0.15 0.38 0.23 0.28"
# export ZOOM_th0_poi="--inset-rect-t V 0.15 0.65 0.23 0.28"
# export ZOOM_th0_url="--inset-rect-t V 0.25 0.6 0.23 0.28 --inset-rect-t D 0.6 0.1 0.23 0.28 --inset-rect-t C 0.6 0.1 0.23 0.28 --inset-rect-t G 0.6 0.1 0.23 0.28"
# export ZOOM_th1_url="--inset-rect-t V 0.15 0.1 0.23 0.28 --inset-rect-t D 0.3 0.1 0.23 0.28 --inset-rect-t C 0.42 0.1 0.23 0.28 --inset-rect-t G 0.15 0.1 0.23 0.28"

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stems", nargs="*", default=list(STEMS))
    ap.add_argument("--tess", default="CDGV", help="tessellations to render, e.g. --tess C")
    ap.add_argument("--inset-xlim", nargs=2, type=float, default=None,
                    help="pin the inset x window (default: auto from the lowest bins)")
    ap.add_argument("--inset-ylim", nargs=2, type=float, default=None,
                    help="pin the inset y window (default: auto from the data)")
    a = ap.parse_args(argv)
    for stem in a.stems:
        for t in a.tess:
            c = load_decomp(os.path.join(a.data, stem, t, "*.npz"), t)
            out = plot_decomp_panel(c, os.path.join(a.out, f"{stem}_{t}.png"), xlim=XLIM, ylim=YLIM,
                                    inset_rect=INSET_RECTS.get((stem, t), INSET_RECT),
                                    inset_tick_fs=9,
                                    inset_xlim=a.inset_xlim, inset_ylim=a.inset_ylim,
                                    ind_min_h=IND_MIN_H.get((stem, t), 0.0),
                                    legend=LEGEND_ON.get((stem, t), False), legend_groups=LEGENDS.get((stem, t)),
                                    refs=REFS.get((stem, t), ()), ref_legend=REF_LEGENDS.get((stem, t), REF_LEGEND),
                                    colors=COLORS, linestyles=LINESTYLES)
            print(out, f"n={c.meta['n_files']} rho_w={c.meta['rho_w_emp']:.4f} "
                       f"self(k0)={c.self_[0]:.4f} dist(k0)={c.dist[0]:.4f} tot(k0)={c.total[0]:.3e}")


if __name__ == "__main__":
    main()
