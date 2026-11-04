"""Fig. 3 (16/3) -- measured edge-length variance against its small-R law, 2D
Poisson, theta = 1.  Solid lines = direct window sampling, one per tessellation;
dotted line = the small-R asymptote

    sigma^2(R) -> (16/3) lambda_ell <w^2 ell> R^3        (d = 2).

On the scaled axes of Fig. 2 (x = R lambda_ell^{1/2},
y = sigma^2 lambda_ell^{1/2} / <w^2 ell>) the asymptote is the single curve
y = (16/3) x^3 for every tessellation.

    python figures/fig3_163.py --data <root> --out <dir>

<root>/<T>.npz as for fig3.py (sigma2_R / sigma2_direct and the stored sums).
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chi_panels import PAPER_STYLE, TESS_COLORS, _RC, _finish_axes  # noqa: E402
from sigma2_panels import load_sigma2  # noqa: E402

import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.legend import Legend                   # noqa: E402

# Panel text, windows and styles, all in one place.  Strings are rendered through
# text.usetex when latex is on the PATH (chi_panels._RC): raw strings, math in $...$.
PANEL = {
    "xlabel": r"$R\,\lambda_\ell^{1/2}$",
    "ylabel": r"$\sigma^2_{\ell,w}(R)\,\lambda_\ell^{1/2} / \langle w^2\ell\rangle$",
    "xlim": (10 ** -1.7, 10 ** 1.4),
    "ylim": (10 ** -4.6, 10 ** 3.5),
    "legend_loc": "lower right",       # tessellation legend; None -> off
}
LINE = dict(lw=2.8, ls="-")                            # the sampled curves
ZORDER = {"C": 2, "D": 3, "G": 4, "V": 5}              # higher draws on top
ASYMPTOTE = {
    "prefactor": 16.0 / 3.0,
    "exp": 3,
    "x_range": (None, None),           # extent in plotted x; None -> axis edge
    "style": dict(color="black", ls=(0, (1.0, 1.6)), lw=2.5, zorder=6),
    "label": r"$\frac{16}{3}\lambda_\ell\langle w^2\ell\rangle R^3$",
    "legend_loc": "upper left",        # its own legend; None -> off
}
LEGEND_KW = dict(fontsize=10, frameon=False, handlelength=3.0)


def plot_panel(curves, out_png: str, *, figsize=(3.375, 2.8), dpi: int = 600) -> str:
    with matplotlib.rc_context({**_RC, **PAPER_STYLE["rc"]}):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xscale("log"); ax.set_yscale("log")
        handles = {}
        for c in curves:
            m = np.isfinite(c.y_dir) & (c.y_dir > 0)
            handles[c.ntype], = ax.plot(c.x_dir[m], c.y_dir[m], color=TESS_COLORS[c.ntype], label=c.ntype,
                                        zorder=ZORDER[c.ntype], **LINE)
        xa, xb = (lim if v is None else v for v, lim in zip(ASYMPTOTE["x_range"], PANEL["xlim"]))
        xg = np.geomspace(xa, xb, 200)
        ref, = ax.plot(xg, ASYMPTOTE["prefactor"] * xg ** ASYMPTOTE["exp"], **ASYMPTOTE["style"])
        ax.set_xlim(*PANEL["xlim"]); ax.set_ylim(*PANEL["ylim"])
        _finish_axes(ax, PAPER_STYLE, PANEL["xlabel"], PANEL["ylabel"])
        if PANEL["legend_loc"]:
            ax.legend(handles=[handles[t] for t in "CDGV" if t in handles], loc=PANEL["legend_loc"], **LEGEND_KW)
        if ASYMPTOTE["legend_loc"] and ASYMPTOTE["label"]:      # second legend beside the first
            ax.add_artist(Legend(ax, [ref], [ASYMPTOTE["label"]], loc=ASYMPTOTE["legend_loc"], **LEGEND_KW))
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        fig.savefig(os.path.splitext(out_png)[0] + ".eps", bbox_inches="tight")
        plt.close(fig)
    return out_png


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    curves = [load_sigma2(os.path.join(a.data, f"{t}.npz")) for t in "DVGC"]    # drawing order
    out = plot_panel(curves, os.path.join(a.out, "sigma2_163.png"))
    print(out, " ".join(f"{c.ntype}:x=[{c.x_dir.min():.3g},{c.x_dir.max():.3g}]" for c in curves))


if __name__ == "__main__":
    main()
