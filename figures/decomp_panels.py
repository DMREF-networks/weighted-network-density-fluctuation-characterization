"""Paper-style self / distinct / total decomposition panels, one tessellation per
panel, on the scaled 2D axes of chi_panels (x = k/(2 pi lambda^{1/2}),
y = chi / (lambda^{1/2} <w^2 ell>)).  chi_distinct = chi_ell - chi_self is
computed from the two stored spectra; |chi_distinct| is drawn.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.ticker import FixedLocator, LogLocator, MaxNLocator, NullFormatter  # noqa: E402

from chi_panels import (LABEL_FS, LEGEND_FS, TESS_RAMPS, TESS_COLORS, TICK_FS, XLABEL_2D,  # noqa: E402
                        YLABEL_2D, _RC, _power_ref, load_group, scaled_axes)

LBL_SELF = r"$\tilde\chi^{(s)}_{_{\ell,w}}(k)$"
LBL_DIST = r"$|\tilde\chi^{(d)}_{_{\ell,w}}(k)|$"
LBL_TOTAL = r"$\tilde\chi_{_{\ell,w}}(k)$"
PANEL_LEGEND_FS = 9   # self / dist / total legend entries
REF_LEGEND_FS = 12     # k^n guide-line legend labels (per-panel: REF_LEGENDS kwargs {"fontsize": ..})
DEFAULT_COLORS = {"self": "0.45", "dist": "base", "total": "k"}
DEFAULT_LS = {"self": "--", "dist": "-", "total": ":"}


def _resolve_color(spec, ntype):
    """int -> rung of TESS_RAMPS[ntype] (0 light .. 10 dark); "base" -> TESS_COLORS[ntype];
    anything else is passed to matplotlib as is."""
    if isinstance(spec, (int, np.integer)):
        return TESS_RAMPS[ntype][spec]
    return TESS_COLORS[ntype] if spec == "base" else spec


def _three_round_ticks(lo: float, hi: float) -> np.ndarray:
    """Exactly three round ticks inside [lo, hi].  MaxNLocator(3)'s own ticks when
    it yields three; else the coarsest step m*10^e from MaxNLocator's default step
    family that fits three multiples, the triple centred in the window."""
    t = MaxNLocator(3).tick_values(lo, hi)
    t = t[(t >= lo) & (t <= hi)]
    if t.size == 3:
        return t
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return t
    e0 = int(np.floor(np.log10(span)))
    mant = (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8)             # MaxNLocator default steps (10 = next decade's 1)
    for s in sorted((m * 10.0 ** e for e in (e0 - 2, e0 - 1, e0) for m in mant), reverse=True):
        first = np.ceil(lo / s) * s
        n = int(np.floor((hi - first) / s)) + 1        # multiples of s inside the window
        if n >= 3:
            k = int(round((0.5 * (lo + hi) - first) / s)) - 1
            k = min(max(k, 0), n - 3)
            return first + s * np.arange(k, k + 3)
    return np.linspace(lo, hi, 3)


@dataclass
class Decomp:
    ntype: str
    x: np.ndarray
    total: np.ndarray
    self_: np.ndarray
    dist: np.ndarray
    meta: dict = field(default_factory=dict)


def load_decomp(pattern: str, ntype: str) -> Decomp:
    """Ensemble-mean total and self spectra from one group of release NPZs."""
    k, chi, meta, n = load_group(pattern)
    selfs = []
    for f in sorted(glob.glob(pattern)):
        with np.load(f, allow_pickle=False) as d:
            selfs.append(d["chi_self"][:k.size].copy())
    chi_s = np.mean(selfs, axis=0)
    x, y_tot = scaled_axes(k, chi, meta)
    _, y_self = scaled_axes(k, chi_s, meta)
    meta["n_files"] = n
    return Decomp(ntype, x, y_tot, y_self, y_tot - y_self, meta)


def plot_decomp_panel(c: Decomp, out_png: str, *, xlim=None, ylim=None, lw: float = 2.8, legend: bool = False,
                      legend_loc: str = "upper right", inset: bool = True, inset_bins: int = 4,
                      inset_rect=(0.15, 0.13, 0.23, 0.30), inset_xlim=None, inset_ylim=None,
                      inset_tick_fs=None, ind_min_h: float = 0.0, xlabel: str = XLABEL_2D, ylabel: str = YLABEL_2D,
                      y_decade_stride: int = 2, legend_groups=None, refs=(), ref_legend=None, colors=None, linestyles=None,
                      figsize=(3.375, 2.8), dpi: int = 600) -> str:
    """legend_groups: [(loc, keys), ...] or [(loc, keys, legend_kwargs), ...] with keys
    from "self", "dist", "total" -- one legend per group, entries in the listed order;
    "" is an invisible spacer cell.  Cells fill column by column, the extra cell of an
    odd count landing in the LEFT column, so ("self", "", "dist", "total") with ncol 2
    puts self alone top-left and dist/total stacked on the right.  legend_kwargs
    override the shared legend style for that group (e.g. borderaxespad,
    bbox_to_anchor).  None = the single legend at legend_loc.
    refs: chi_panels.RefLine sequence, drawn by _power_ref.
    ref_legend: their own legend, independent of `legend`: a loc string or (loc, legend_kwargs);
    None = no guide-line legend.
    ind_min_h: minimum height of the inset indicator box as a fraction of the (log) axes height;
    a thinner inset y window is padded symmetrically about its log centre.  0 = exact window.
    colors / linestyles: per-component overrides keyed "self", "dist", "total" on top of
    DEFAULT_COLORS / DEFAULT_LS; colour specs resolve via _resolve_color."""
    cols = {**DEFAULT_COLORS, **(colors or {})}
    lss = {**DEFAULT_LS, **(linestyles or {})}
    c_self, c_dist, c_tot = (_resolve_color(cols[k], c.ntype) for k in ("self", "dist", "total"))
    ls_self, ls_dist, ls_tot = (lss[k] for k in ("self", "dist", "total"))
    with matplotlib.rc_context(_RC):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xscale("log"); ax.set_yscale("log")
        x, ys, yd, yt = c.x, np.abs(c.self_), np.abs(c.dist), np.abs(c.total)
        h_self = ax.plot(x, ys, color=c_self, ls=ls_self, lw=0.90 * lw, zorder=10, label=LBL_SELF)[0]
        h_dist = ax.plot(x, yd, color=c_dist, ls=ls_dist, lw=0.70 * lw, zorder=9, label=LBL_DIST)[0]
        h_tot = ax.plot(x, yt, color=c_tot, ls=ls_tot, lw=lw, zorder=5, label=LBL_TOTAL)[0]
        if xlim: ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
        else:   # floor one decade under the data, fixed headroom for the legend
            v = np.concatenate([ys, yd, yt]); v = v[np.isfinite(v) & (v > 0)]
            ax.set_ylim(10 ** np.floor(np.log10(v.min())), 10 ** 2.7)
        ref_handles = [_power_ref(ax, r, ax.get_xlim(), ax.get_ylim()) for r in refs]
        ax.set_xlabel(xlabel, fontsize=LABEL_FS)
        ax.set_ylabel(ylabel, fontsize=LABEL_FS)
        ax.tick_params(which="both", labelsize=TICK_FS, direction="in", top=True, right=True)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(base=10, numticks=999))
            axis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10)), numticks=999))
            axis.set_minor_formatter(NullFormatter())
        if y_decade_stride > 1:   # label every n-th decade (exponents divisible by n); skipped decades stay as minor ticks
            ylo_, yhi_ = ax.get_ylim()
            ex = np.arange(np.floor(np.log10(ylo_)), np.ceil(np.log10(yhi_)) + 1)
            ax.yaxis.set_major_locator(FixedLocator(10.0 ** ex[ex % y_decade_stride == 0]))
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(1, 10)), numticks=999))
        # 2-col legend, last (odd) entry tucked under the right column
        from matplotlib.lines import Line2D
        leg_kw = dict(fontsize=PANEL_LEGEND_FS, frameon=False, columnspacing=1, handlelength=1.7, borderpad=.4, borderaxespad=.4,labelspacing=.8)
        if refs and ref_legend is not None:   # guide-line legend; kept as an extra artist so the main legend can follow
            loc, kw = (ref_legend, {}) if isinstance(ref_legend, str) else ref_legend
            lg = ax.legend(ref_handles, [h.get_label() for h in ref_handles], loc=loc,
                           **{**leg_kw, "fontsize": REF_LEGEND_FS, "handlelength": 3.5, **kw})
            ax.add_artist(lg)
        if legend and legend_groups:
            by_key = {"self": h_self, "dist": h_dist, "total": h_tot,
                      "": Line2D([], [], ls="none")}   # spacer cell
            for i, grp in enumerate(legend_groups):
                loc, keys = grp[0], grp[1]
                kw = {"ncol": 2, **leg_kw, **(grp[2] if len(grp) > 2 else {})}
                hs = [by_key[k] for k in keys]
                lg = ax.legend(hs, [h.get_label() for h in hs], loc=loc, **kw)
                if i < len(legend_groups) - 1:
                    ax.add_artist(lg)   # keep earlier groups when the next legend is drawn
        elif legend:
            hs = [h_self, Line2D([], [], ls="none"), h_dist, h_tot]
            ax.legend(hs, [h.get_label() for h in hs], loc=legend_loc, ncol=3, **leg_kw)
        if inset:
            # linear zoom of the lowest bins: self vs distinct, same y scaling as the panel
            n = min(inset_bins, x.size)
            ins = ax.inset_axes(list(inset_rect))
            ins.set_xscale("linear"); ins.set_yscale("linear") # adjust for linear
            ins.plot(x[:n + 3], ys[:n + 3], color=c_self, ls=ls_self, lw=lw, zorder=4)
            ins.plot(x[:n + 3], yd[:n + 3], color=c_dist, ls=ls_dist, lw=lw * 0.5, zorder=3)
            xlo, xhi = (0.7 * x[0], x[n - 1] * 1.4) if inset_xlim is None else inset_xlim
            # self distinct
            v = np.concatenate([ys[:n], yd[:n]])
            lo, hi = float(v.min()), float(v.max()); span = (hi - lo) or abs(hi) or 1.0
            ## self
            # lo, hi = float(ys[:n].min()), float(ys[:n].max()); span = (hi - lo) or abs(hi) or 1.0
            ## distinct
            # lo, hi = float(yd[:n].min()), float(yd[:n].max()); span = (hi - lo) or abs(hi) or 1.0
            ylo, yhi = (lo - 0.3 * span, hi + 0.2 * span) if inset_ylim is None else inset_ylim # linear
            # ylo, yhi = (lo / span, hi * span) if inset_ylim is None else inset_ylim
            span_y_lo = 1.0; span_y_hi = 1.0; ins.set_xlim(xlo, xhi); ins.set_ylim(ylo * span_y_lo, yhi * span_y_hi)
            ins.tick_params(which="both", direction="in", top=True, right=True,
                            labelsize=inset_tick_fs if inset_tick_fs is not None else max(4.0, TICK_FS * 0.5),
                            length=2.0, width=0.6, pad=1.5)
            # ins.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=4))
            ins.xaxis.set_major_locator(FixedLocator(_three_round_ticks(xlo, xhi)))
            ins.yaxis.set_major_locator(FixedLocator(_three_round_ticks(*ins.get_ylim())))
            ind_col = "black"
            # TESS_RAMPS[c.ntype][5]
            for sp in ins.spines.values():
                sp.set_linewidth(0.6); sp.set_edgecolor(ind_col)
            ins.set_facecolor("white"); ins.set_zorder(10)
            ix0, ix1 = ins.get_xlim(); iy0, iy1 = ins.get_ylim()
            px0, px1 = ax.get_xlim(); py0, py1 = ax.get_ylim()
            cx0, cx1, cy0, cy1 = max(ix0, px0), min(ix1, px1), max(iy0, py0), min(iy1, py1)
            if ind_min_h > 0 and cy1 > cy0:   # sub-pixel y window on the log axis -> visible box
                lo, hi = np.log10(cy0), np.log10(cy1)
                min_h = ind_min_h * (np.log10(py1) - np.log10(py0))
                if hi - lo < min_h:
                    mid = 0.5 * (lo + hi)
                    cy0, cy1 = 10 ** (mid - 0.5 * min_h), 10 ** (mid + 0.5 * min_h)
            if cx1 > cx0 and cy1 > cy0:
                ind = ax.indicate_inset([cx0, cy0, cx1 - cx0, cy1 - cy0], ins, edgecolor=ind_col, lw=1, alpha=1.0)
                conns = ind.connectors if hasattr(ind, "connectors") else ind[1]
                for cl in conns:
                    cl.set_visible(False)
                # ax.indicate_inset([cx0, cy0, cx1 - cx0, cy1 - cy0], ins, edgecolor="0.4", lw=0.6, alpha=1.0)
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        fig.savefig(os.path.splitext(out_png)[0] + ".eps", bbox_inches="tight")
        plt.close(fig)
    return out_png
