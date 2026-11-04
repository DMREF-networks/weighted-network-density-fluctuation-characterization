"""Large-wavenumber decay of chi(k) at theta = 1, 2D (solid) and 3D (dotted) on one
panel, Poisson substrate, C/D/G/V.

    python figures/fig_highk_2d3d.py --data <root> --out <dir>

<root>/2d_K600/<T>_poi_length_power_th1/*.npz and <root>/2d_K80/... are pooled per
tessellation (union shell grid); <root>/3d_K300/<T>_poi_length_power_th1/*.npz is
the 3D set.  Axes x = k / (2 pi lambda_ell^{1/d}),
y = chi / [lambda_ell^{(d-1)/d} (pi/2)^{d-2} <w^2 ell>], so both dimensions decay
as 1/(pi x) at large k; curves are rebinned to 20 shells per decade.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chi_panels import (PAPER_STYLE, TESS_COLORS, _RC, _finish_axes, _power_ref, guide,  # noqa: E402
                        load_group_pooled, scaled_axes)
from paired_legend import STYLE as LEGEND_STYLE, paired_table_legend  # noqa: E402

import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.colors import to_hex, to_rgb           # noqa: E402

XLABEL = r"$k / (2\pi \lambda_\ell^{1/d})$"
YLABEL = r"$\tilde\chi_{_{\ell,w}}(k) / [ \lambda_\ell^{^{\frac{d-1}{d}}} (\frac{\pi}{2})^{^{d-2}} \langle w^2\ell \rangle ]$"
XLIM, YLIM = (0.3, 30.0), (0.013, 8.0)
REBIN = 20
COLOR_3D = {T: to_hex(0.5 * (np.array(to_rgb(c)) + 1.0)) for T, c in TESS_COLORS.items()}   # half-way to white
REF = guide(-1, (0.3, 0.999), 0.15)
STYLE = dict(PAPER_STYLE, ystride=1)
# the 2D | 3D table legend in the paper's geometry
TABLE = dict(LEGEND_STYLE, fontsize=10, handlelength=3, line_span_frac=0.79, rowsep_em=0.155, colsep_em=0.3,
             cellpad_em=0.05, headerpad_em=0.05, borderpad=0.2, borderaxespad=0.7, frameon=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    curves = {}
    for T in "CDGV":
        k, chi, meta, n2 = load_group_pooled([os.path.join(a.data, d, f"{T}_poi_length_power_th1", "*.npz")
                                              for d in ("2d_K600", "2d_K80")], REBIN)
        curves[(T, 2)] = scaled_axes(k, chi, meta) + (n2,)
        k, chi, meta, n3 = load_group_pooled([os.path.join(a.data, "3d_K300", f"{T}_poi_length_power_th1", "*.npz")], REBIN)
        curves[(T, 3)] = scaled_axes(k, chi, meta) + (n3,)
    with matplotlib.rc_context({**_RC, **STYLE["rc"]}):
        fig, ax = plt.subplots(figsize=(3.375, 2.8))
        ax.set_xscale("log"); ax.set_yscale("log")
        for (T, d), (x, y, _) in curves.items():
            m = y > 0
            if d == 2:
                ax.plot(x[m], y[m], lw=2.8, color=TESS_COLORS[T], zorder=2)
            else:
                ax.plot(x[m], y[m], lw=2.8, color=COLOR_3D[T], ls=":", zorder=3)   # 3D over 2D
        ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
        ref = _power_ref(ax, REF, XLIM, YLIM)
        _finish_axes(ax, STYLE, XLABEL, YLABEL)
        paired_table_legend(ax, [(T, TESS_COLORS[T]) for T in "CDGV"],
                            [("2D", "line"), ("3D", "line", {"linestyle": ":", "row_colors":
                                                             COLOR_3D})],
                            loc="upper right", bbox_to_anchor=(0.999, 1.006), style=TABLE)
        ax.legend([ref], [REF.legend_label], loc="center right", bbox_to_anchor=(0.98, 0.42), fontsize=10,
                  frameon=False, handlelength=3.5, borderaxespad=0.01)
        fig.tight_layout()
        os.makedirs(a.out, exist_ok=True)
        out = os.path.join(a.out, "all4_2d3d.png")
        fig.savefig(out, dpi=600, bbox_inches="tight"); fig.savefig(out[:-4] + ".eps", bbox_inches="tight")
        plt.close(fig)
    print(out, " ".join(f"{T}{d}D:n={n},xy_tail={np.nanmean((x * y)[-5:]):.3f}" for (T, d), (x, y, n) in curves.items()))


if __name__ == "__main__":
    main()
