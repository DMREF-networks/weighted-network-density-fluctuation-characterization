"""Paper-style sigma^2(R) panels: direct window sampling (markers) vs the
Parseval bridge (lines), per tessellation.

Axes (2D):  x = R lambda_ell^{1/2},   y = sigma^2(R) lambda_ell^{1/2} / <w^2 ell>
with lambda_ell = N_ell / V and <w^2 ell> = sum_w2l / N_ell (stored sums).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

from chi_panels import LABEL_FS, LEGEND_FS, TESS_COLORS, TICK_FS, _RC  # noqa: E402
from paired_legend import paired_table_legend  # noqa: E402

TESS_MARKERS = {"C": "^", "D": "s", "G": "D", "V": "o"}
MARK_OFFSET = {"D": 0.00, "V": 0.025, "G": 0.05, "C": 0.075}     # de-stack the marker trains
XLABEL_2D = r"$R \lambda_{\ell}^{1/2}$"
YLABEL_2D = r"$\sigma^2_{\ell,w}(R) \, \lambda_{\ell}^{1/2} \, / \langle w^2 \ell \rangle$"


@dataclass
class Sigma2Curve:
    ntype: str
    x_spec: np.ndarray
    y_spec: np.ndarray
    x_dir: Optional[np.ndarray] = None
    y_dir: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)


def load_sigma2(path: str) -> Sigma2Curve:
    """One release NPZ (or converted legacy file) -> scaled curves."""
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["metadata_json"]))
        R_s = d["sigma2_R"]
        s_spec = d["sigma2_spectral"] if "sigma2_spectral" in d.files else None
        s_dir = d["sigma2_direct"] if "sigma2_direct" in d.files else None
        R_d = d["sigma2_R_direct"] if "sigma2_R_direct" in d.files else R_s
    dim = int(meta["d"])
    lam = float(meta["N_ell"]) / float(meta["V"])
    m2 = float(meta["sum_w2l"]) / float(meta["N_ell"])
    s = lam ** (1.0 / dim)
    xs = lambda R: np.asarray(R, dtype=float) * s          # noqa: E731
    ys = lambda v: np.asarray(v, dtype=float) * s / m2      # noqa: E731
    return Sigma2Curve(ntype=meta["tess"],
                       x_spec=xs(R_s), y_spec=(ys(s_spec) if s_spec is not None else None),
                       x_dir=(xs(R_d) if s_dir is not None else None), y_dir=(ys(s_dir) if s_dir is not None else None),
                       meta=meta)


def plot_sigma2_panel(curves: Sequence[Sigma2Curve], out_png: str, *, dim: int = 2,
                      xlim=(10 ** -1.7, 10 ** 1.4), ylim=(10 ** -4.6, 10 ** 3.5), x_range=(0.001, 16.0),
                      lw: float = 2.8, legend_scale: float = 0.7, legend_loc: str = "lower right",
                      headers=("Direct", "Spectral"), line_first: bool = False, legend: bool = True,
                      xlabel: str = XLABEL_2D, ylabel: str = YLABEL_2D, legend_style: Optional[dict] = None,
                      figsize=(3.375, 2.8), dpi: int = 600) -> str:
    with matplotlib.rc_context(_RC):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xscale("log"); ax.set_yscale("log")
        for c in sorted(curves, key=lambda c: c.ntype):
            col = TESS_COLORS[c.ntype]
            if c.y_spec is not None:
                m = (c.x_spec >= x_range[0]) & (c.x_spec <= x_range[1]) & (c.y_spec > 0)
                ax.plot(c.x_spec[m], c.y_spec[m], color=col, lw=lw, ls="-", zorder=2)
            if c.y_dir is not None:
                m = (c.x_dir >= x_range[0]) & (c.x_dir <= x_range[1]) & (c.y_dir > 0)
                ax.plot(c.x_dir[m], c.y_dir[m], ls="none", marker=TESS_MARKERS[c.ntype], markersize=4,
                        markerfacecolor=col, markeredgecolor="black", markeredgewidth=0.2,
                        markevery=(MARK_OFFSET[c.ntype], 0.1), zorder=5)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel, fontsize=LABEL_FS)
        ax.set_ylabel(ylabel, fontsize=LABEL_FS)
        ax.tick_params(which="both", labelsize=TICK_FS, direction="in", top=True, right=True)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(base=10, numticks=999))
            axis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10)), numticks=999))
            axis.set_minor_formatter(NullFormatter())
        # Row x column table legend: rows C D G V, one column per route.
        # `line_first` puts the line column first (Analytical | Measured);
        # otherwise markers lead (Direct | Spectral).  legend=False renders the
        # panel bare, for figure sets that carry one legend on a single panel.
        if legend:
            present = [nt for nt in "CDGV" if any(c.ntype == nt for c in curves)]
            rows = [(nt, TESS_COLORS[nt], TESS_MARKERS[nt]) for nt in present]
            kinds = ("line", "marker") if line_first else ("marker", "line")
            paired_table_legend(ax, rows, list(zip(headers, kinds)), loc=legend_loc,
                                fontsize=LEGEND_FS, scale=legend_scale, frameon=False,
                                marker_edgecolor="black", marker_edgewidth=0.8, style=legend_style)
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        fig.savefig(os.path.splitext(out_png)[0] + ".eps", bbox_inches="tight")
        plt.close(fig)
    return out_png
