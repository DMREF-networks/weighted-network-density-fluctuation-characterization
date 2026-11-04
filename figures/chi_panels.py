"""Paper-style chi(k) panels from release NPZs (storage rule: no arithmetic here
beyond the read-time axes of norms.derived).

Axes:   x = k / (2 pi lambda_ell^{1/d}),
        y = chi(k) / (K_d lambda_ell^{(d-1)/d} <w^2 ell>_kmax),   K_d = (pi/2)^{d-2}
(the scaling length is the mean edge spacing lambda_ell^{-1/d}; <w^2 ell>_kmax
is the finite-kmax Porod moment in the paper's convention, i.e. divided by K_d,
so the large-k plateau of x*y is 1/pi in 2D and 3D)
with lambda_ell = N_ell / V and the finite-kmax Porod moment <w^2 ell>_kmax from
the stored moment sums.  Curves are ensemble means over seeds of chi on the
shared shell grid; the moment sums are ensemble means as well.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from wndfc import norms                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter  # noqa: E402

TESS_COLORS = {"C": "#DE9800", "D": "#005897", "G": "#A274CB", "V": "#CE0043"}
# XLABEL_2D = r"$k / (2\pi\,\lambda_\ell^{1/2})$"
# YLABEL_2D = r"$\tilde\chi_{{_\ell}\!{_,}\!{_w}}(k) / (\lambda_\ell^{1/2}\,\langle w^2\ell\rangle)$"
# XLABEL_3D = r"$k / (2\pi\,\lambda_\ell^{1/3})$"
# YLABEL_3D = r"$\tilde\chi_{{_\ell}\!{_,}\!{_w}}(k) / (\lambda_\ell^{2/3}\,\langle w^2\ell\rangle)$"

XLABEL_2D = r"$k / (2\pi\,\lambda_\ell^{1/2})$"
YLABEL_2D = r"$\tilde\chi_{_{\ell,w}}(k) / (\lambda_\ell^{1/2}\,\langle w^2\ell\rangle)$"
XLABEL_3D = r"$k / (2\pi\,\lambda_\ell^{1/3})$"
YLABEL_3D = r"$\tilde\chi_{_{\ell,w}}(k) / ((\pi/2) \lambda_\ell^{2/3}\,\langle w^2\ell\rangle)$"

_RC = {
    "text.usetex": shutil.which("latex") is not None and os.environ.get("MPL_NO_TEX") is None,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "legend.frameon": False, "legend.columnspacing": 0.5, "legend.handlelength": 1.3,
    "legend.borderpad": 0.6, "legend.borderaxespad": 0.5,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
}
LABEL_FS, TICK_FS, LEGEND_FS = 12, 11, 14


@dataclass
class RefLine:
    """Power-law guide line k^exp through (sqrt(x0 x1), y_anchor); region/y_frac are
    log fractions of the axes span."""
    exp: float
    region: Tuple[float, float]
    y_frac: float
    linestyle: object
    legend_label: str
    color: str = "0.37"
    lw: float = 2.4


DASH_SHORT = (0, (1.2, 1.0, 0.3, 1.0))
DASH_LONG = (0, (3.5, 2.2))
# the paper's guide pattern per exponent: k^2 dotted, k^3 dash-dot, k^4 dashed
GUIDE_LS = {2: (0, (1.0, 1.6)), 3: (0, (4.0, 1.4, 1.0, 1.4)), 4: (0, (5.0, 2.0)),
            6: (0, (1.3, 1.3, 2.2, 2.2)), -1: (0, (4.0, 1.2, 1.0, 1.2, 1.0, 1.2))}


def guide(exp, region, y_frac, color="0.37", lw=2.4) -> RefLine:
    return RefLine(exp, region, y_frac, GUIDE_LS[exp], rf"$k^{{{exp}}}$", color=color, lw=lw)


# Text / legend style: the release default, and the paper's chi(k) panels (12 / 11 / 10 pt,
# long legend samples, y labels on even decades; font.size 13 sets the tight_layout margins).
RELEASE_STYLE = dict(label_fs=LABEL_FS, tick_fs=TICK_FS, legend_fs=LEGEND_FS, ystride=1, legend_kw={}, rc={})
PAPER_STYLE = dict(label_fs=12, tick_fs=11, legend_fs=10, annot_fs=16, ystride=2, rc={"font.size": 13},
                   legend_kw=dict(handlelength=2.6, columnspacing=0.5, borderpad=0.1, borderaxespad=0.3))


class _StridedLog(LogFormatterSciNotation):
    """Decade labels on every ``stride``-th exponent only (10^0 anchored)."""

    def __init__(self, stride: int):
        super().__init__(base=10.0)
        self.stride = stride

    def __call__(self, x, pos=None):
        return "" if x <= 0 or round(np.log10(x)) % self.stride else super().__call__(x, pos)


def _finish_axes(ax, st: dict, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=st["label_fs"])
    ax.set_ylabel(ylabel, fontsize=st["label_fs"])
    ax.tick_params(which="both", labelsize=st["tick_fs"], direction="in", top=True, right=True)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, numticks=999))
        axis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10)), numticks=999))
        axis.set_minor_formatter(NullFormatter())
    if st["ystride"] > 1:
        ax.yaxis.set_major_formatter(_StridedLog(st["ystride"]))


@dataclass
class Curve:
    label: str
    ntype: str
    x: np.ndarray
    y: np.ndarray
    meta: dict = field(default_factory=dict)


def check_edge_count(meta: dict, path: str = "") -> None:
    """A periodic 2D D, V or C network of N generators has exactly 3N edges; a converted
    file carries N as ``legacy.n_generators``.  Fewer edges = a lost near-cocircular
    quad, more = an over-counted k-fold vertex: the file is defective, refuse it."""
    n_gen = (meta.get("legacy") or {}).get("n_generators")
    if not n_gen or meta.get("d") != 2 or meta.get("tess") not in ("D", "V", "C"):
        return
    if int(meta["N_ell"]) != 3 * int(n_gen):
        raise ValueError(f"{path}: N_ell={meta['N_ell']} != 3N={3 * int(n_gen)} -- "
                         f"lost or duplicated edges in the {meta['tess']} build; regenerate this seed")


def load_group(pattern: str) -> Tuple[np.ndarray, np.ndarray, dict, int]:
    """Ensemble-mean chi on the shared shell grid + ensemble-mean metadata sums."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    ks, chis, metas = [], [], []
    for f in files:
        with np.load(f, allow_pickle=False) as d:
            ks.append(d["k"].copy()); chis.append(d["chi_ell"].copy())
            metas.append(json.loads(str(d["metadata_json"])))
            check_edge_count(metas[-1], f)
    n = min(k.size for k in ks)
    if len({k.size for k in ks}) > 1:       # mixed k_max: the shorter grids end on a partial shell
        n -= 1
    k0 = ks[0][:n]
    for k in ks:
        if not np.allclose(k[:n], k0):
            raise ValueError(f"k grid mismatch in {pattern}")
    chi = np.mean([c[:n] for c in chis], axis=0)
    meta = dict(metas[0])
    for key in ("N_ell", "V", "sum_wl", "sum_w2l", "sum_w2l2", "sum_w2l3", "sum_w2l_kmax", "rho_w_emp"):
        vals = [m.get(key) for m in metas if m.get(key) is not None]   # mean over files carrying the key
        meta[key] = float(np.mean(vals)) if vals else None
    return k0, chi, meta, len(files)


def load_group_pooled(patterns: Sequence[str], rebin_per_decade: int = 0):
    """As load_group over several globs pooled on the union shell grid (files of
    different k_max contribute where they have shells); with ``rebin_per_decade``
    the mean is rebinned onto log-spaced shells weighted by the stored mode counts.
    Returns (k, chi, meta, n_files)."""
    files = sorted(f for p in patterns for f in glob.glob(p))
    if not files:
        raise FileNotFoundError(" ".join(patterns))
    ks, chis, cnts, metas = [], [], [], []
    for f in files:
        with np.load(f, allow_pickle=False) as d:
            ks.append(d["k"].copy()); chis.append(d["chi_ell"].copy())
            cnts.append(d["counts"].copy() if "counts" in d.files else np.ones_like(ks[-1]))
            metas.append(json.loads(str(d["metadata_json"])))
            check_edge_count(metas[-1], f)
    n = max(k.size for k in ks)
    k0 = max(ks, key=lambda k: k.size)
    for k in ks:
        m = k.size if k.size == n else k.size - 1     # a shorter grid ends on a partial shell
        if not np.allclose(k[:m], k0[:m]):
            raise ValueError(f"k grid mismatch in {patterns}")
    def pad(a):
        a = a if a.size == n else a[:-1]
        return np.concatenate([a, np.full(n - a.size, np.nan)])
    chi = np.nanmean([pad(c) for c in chis], axis=0)
    cnt = np.nanmean([pad(c) for c in cnts], axis=0)
    meta = dict(metas[0])
    for key in ("N_ell", "V", "sum_wl", "sum_w2l", "sum_w2l2", "sum_w2l3", "sum_w2l_kmax", "rho_w_emp"):
        vals = [m.get(key) for m in metas if m.get(key) is not None]
        meta[key] = float(np.mean(vals)) if vals else None
    if rebin_per_decade:
        ok = np.isfinite(chi) & (k0 > 0) & (cnt > 0)
        k0, chi, cnt = k0[ok], chi[ok], cnt[ok]
        edges = np.logspace(np.log10(k0.min()), np.log10(k0.max()),
                            int(np.ceil(np.log10(k0.max() / k0.min()) * rebin_per_decade)) + 2)
        idx = np.clip(np.searchsorted(edges, k0, side="right") - 1, 0, edges.size - 2)
        kb = np.bincount(idx, cnt * k0, edges.size - 1); cb = np.bincount(idx, cnt * chi, edges.size - 1)
        wb = np.bincount(idx, cnt, edges.size - 1)
        hit = wb > 0
        k0, chi = kb[hit] / wb[hit], cb[hit] / wb[hit]
    return k0, chi, meta, len(files)


def scaled_axes(k: np.ndarray, chi: np.ndarray, meta: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Paper scaling-regime axes; the arithmetic lives in norms.axes."""
    return norms.axes(meta, k, chi)


axes_2d = scaled_axes


def _power_ref(ax, r: RefLine, xlim, ylim, n=200):
    f0, f1 = r.region
    x0 = xlim[0] * (xlim[1] / xlim[0]) ** f0
    x1 = xlim[0] * (xlim[1] / xlim[0]) ** f1
    y_anchor = ylim[0] * (ylim[1] / ylim[0]) ** r.y_frac
    xs = np.geomspace(x0, x1, n)
    ys = y_anchor * (xs / np.sqrt(x0 * x1)) ** r.exp
    return ax.plot(xs, ys, color=r.color, lw=r.lw, ls=r.linestyle, alpha=0.95, zorder=0,
                   label=r.legend_label)[0]


def _refs_left(handles: Sequence, refs: Sequence) -> List:
    """Two legend columns: the guides bottom-aligned on the left, the curves on the right."""
    n = max(len(handles), len(refs))
    blank = Line2D([], [], color="none", label="")
    return [blank] * (n - len(refs)) + list(refs) + list(handles) + [blank] * (n - len(handles))


def _row_major(handles: Sequence, ncol: int) -> List:
    """Reorder handles so a column-filled legend reads row by row."""
    nrow = int(np.ceil(len(handles) / ncol))
    grid = [handles[i:i + ncol] for i in range(0, len(handles), ncol)]
    out = []
    for c in range(ncol):
        for r in range(nrow):
            if c < len(grid[r]):
                out.append(grid[r][c])
    return out


def plot_chi_panel(curves: Sequence[Curve], out_png: str, *, dim: int = 2,
                   refs: Sequence[RefLine] = (), annot: Optional[str] = None,
                   annot_loc=(0.10, 0.90), xlim=(10 ** -3.1, 1.0), ylim=(10 ** -10.5, 20.0),
                   lw: float = 2.8, legend: bool = True, legend_ncol: int = 2, legend_bbox=(0.99, 0.04),
                   legend_loc: str = "lower right", legend_layout: str = "rows",
                   legend_pad: Optional[float] = None, ylabel: Optional[str] = None,
                   style: Optional[dict] = None,
                   figsize=(3.375, 2.8), dpi: int = 600) -> str:
    """legend_layout: "rows" (curves then guides, read row by row over ``legend_ncol``
    columns), "cols" (the same entries filled column by column) or "refs_left" (guides in a
    bottom-aligned left column, curves on the right).  legend_pad overrides the style's
    borderaxespad.  style: RELEASE_STYLE (default) or PAPER_STYLE."""
    st = style or RELEASE_STYLE
    with matplotlib.rc_context({**_RC, **st.get("rc", {})}):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xscale("log"); ax.set_yscale("log")
        handles = []
        for c in curves:
            m = c.y > 0
            handles.append(ax.plot(c.x[m], c.y[m], lw=lw, color=TESS_COLORS[c.ntype], label=c.label)[0])
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ref_handles = [_power_ref(ax, r, xlim, ylim) for r in refs]
        _finish_axes(ax, st, XLABEL_2D if dim == 2 else XLABEL_3D,
                     ylabel or (YLABEL_2D if dim == 2 else YLABEL_3D))
        if annot:
            ax.text(annot_loc[0], annot_loc[1], annot, transform=ax.transAxes, ha="left", va="center",
                    fontsize=st.get("annot_fs", st["legend_fs"]), zorder=600)
        if legend:   # NB this legend also carries the k^n reference-line labels
            if legend_layout == "refs_left":
                hs, ncol = _refs_left(handles, ref_handles), 2
            elif legend_layout == "cols":
                hs, ncol = handles + ref_handles, legend_ncol
            else:
                hs, ncol = _row_major(handles + ref_handles, legend_ncol), legend_ncol
            legkw = dict(st["legend_kw"])
            if legend_bbox is not None:
                legkw["bbox_to_anchor"] = legend_bbox
            if legend_pad is not None:
                legkw["borderaxespad"] = legend_pad
            ax.legend(hs, [h.get_label() for h in hs], loc=legend_loc, ncol=ncol,
                      fontsize=st["legend_fs"], frameon=False, **legkw)
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        fig.savefig(os.path.splitext(out_png)[0] + ".eps", bbox_inches="tight")
        plt.close(fig)
    return out_png


def curves_from_roots(root_by_ntype: Dict[str, str], ntypes: Sequence[str] = ("C", "D", "G", "V"),
                      rebin: Optional[int] = None):
    """One curve per tessellation from NPZ globs keyed by ntype (sorted order = legend order).
    rebin: None -> load_group (shared grid); an int -> load_group_pooled (union grid,
    rebinned to that many shells per decade when > 0)."""
    out = []
    for nt in sorted(ntypes):
        k, chi, meta, n = (load_group(root_by_ntype[nt]) if rebin is None
                           else load_group_pooled([root_by_ntype[nt]], rebin))
        x, y = scaled_axes(k, chi, meta)
        out.append(Curve(label=nt, ntype=nt, x=x, y=y, meta={"n_files": n, **{kk: meta.get(kk) for kk in ("N_ell", "V", "rho_w_emp", "sum_w2l_kmax")}}))
    return out


# ---------------------------------------------------------------------------
# theta-sweep panels: one tessellation, curves colored by theta on a discrete
# per-tessellation ramp (11 equal-lightness steps over theta = 0 .. 2.5)
# ---------------------------------------------------------------------------
THETA_RAMP_THETAS = np.arange(0.0, 2.5 + 1e-9, 0.25)
TESS_RAMPS = {
    "C": ["#E0D3B5", "#DEBC7E", "#DEA43D", "#D49100", "#BF8200", "#AC7400", "#996700",
          "#875A00", "#754D00", "#644000", "#533300"],
    "D": ["#C6D8E5", "#B1C6D8", "#9CB6CC", "#88A6C1", "#7497B7", "#6088AE", "#4A79A6",
          "#336B9F", "#135D99", "#134E7F", "#1A405F"],
    "G": ["#DAD1E2", "#CBBCD9", "#BEA8D2", "#B294CE", "#A77FCB", "#996EC0", "#8861AA",
          "#775596", "#674982", "#573D6E", "#47315C"],
    "V": ["#F1C8D5", "#ECAFBE", "#E896A6", "#E37D8E", "#DE6276", "#D8435F", "#D01848",
          "#BA083E", "#A00C37", "#870B2F", "#700827"],
}


def theta_curves(root: str, ntype: str, ctype: str, thetas=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5)):
    """theta -> Curve for <root>/<T>_<ctype>_length_power_th<X>p<Y>/*.npz."""
    out = {}
    for th in thetas:
        tag = f"{ntype}_{ctype}_length_power_th{int(th)}p{int(round((th % 1) * 10))}"
        k, chi, meta, n = load_group(os.path.join(root, tag, "*.npz"))
        x, y = scaled_axes(k, chi, meta)
        out[th] = Curve(label=f"{th:g}", ntype=ntype, x=x, y=y, meta={"n_files": n, "theta": th})
    return out


def plot_theta_panel(curves: Dict[float, Curve], ntype: str, out_png: str, *, dim: int = 2,
                     refs: Sequence[RefLine] = (), xlim=(10 ** -2.4, 0.95), ylim=(10 ** -3.3, 10 ** 1.2),
                     lw: float = 2.8, legend_loc: str = "lower right", legend_bbox=None,
                     cbar_label: str = r"$\theta$", style: Optional[dict] = None,
                     figsize=(3.375, 2.8), dpi: int = 600) -> str:
    st = style or RELEASE_STYLE
    from matplotlib.colors import BoundaryNorm, ListedColormap
    hexes = TESS_RAMPS[ntype]
    step = float(THETA_RAMP_THETAS[1] - THETA_RAMP_THETAS[0])
    bounds = np.append(THETA_RAMP_THETAS - step / 2, THETA_RAMP_THETAS[-1] + step / 2)
    cmap = ListedColormap(hexes, name="tess_ramp")
    norm = BoundaryNorm(bounds, len(hexes))
    with matplotlib.rc_context({**_RC, **st.get("rc", {})}):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xscale("log"); ax.set_yscale("log")
        for th in sorted(curves):                       # theta = 2.5 drawn last (on top)
            c = curves[th]
            m = c.y > 0
            ax.plot(c.x[m], c.y[m], lw=lw, color=cmap(norm(th)))
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ref_handles = [_power_ref(ax, r, xlim, ylim) for r in refs]
        _finish_axes(ax, st, XLABEL_2D if dim == 2 else XLABEL_3D, YLABEL_2D if dim == 2 else YLABEL_3D)
        if ref_handles and legend_loc:    # only ref-line labels; the theta ramp stays on the colorbar
            legkw = {"bbox_to_anchor": legend_bbox} if legend_bbox is not None else {}
            ax.legend(ref_handles, [h.get_label() for h in ref_handles], loc=legend_loc,
                      fontsize=st["legend_fs"], frameon=False, **legkw, **st.get("guide_legend_kw", {}))
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, aspect=30)
        cb.set_label(cbar_label, fontsize=st.get("cbar_label_fs", st["legend_fs"]), rotation=0, labelpad=8)
        cb.set_ticks([t for t in THETA_RAMP_THETAS if float(t) % 0.5 == 0])
        cb.ax.tick_params(labelsize=st["tick_fs"])
        cb.outline.set_linewidth(0.6)
        fig.tight_layout()
        fig.tight_layout()                # second pass settles the colorbar axes
        os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        fig.savefig(os.path.splitext(out_png)[0] + ".eps", bbox_inches="tight")
        plt.close(fig)
    return out_png
