"""Portable row x column "paired" legend: row headers on the LEFT, real
glyph columns under real column headers.

    ┌───────────────────────────┐
    │        Direct   Spectral  │
    │   C      ●        ───     │
    │   D      ■        ───     │
    │   G      ▲        ───     │
    │   V      ◆        ───     │
    └───────────────────────────┘

Why this exists: ``ax.legend`` cannot do this layout. Labels are hard-wired
to the right of handles, and there is one centered title string -- the
existing ``--paired-legend`` look fakes its two columns with HandlerTuple
padding plus a hand-tuned ``\\hspace`` inside the title. That construction
is left untouched; this module is the opt-in replacement built from
offsetboxes (a real table), so column headers sit over their glyphs by
construction, the row letters go on the left, and nothing depends on usetex
spacing.

Design rules:
  * Pure matplotlib, no imports from any plotting script, so every panel
    module can call it.
  * One style source: the STYLE dict below. Geometry is in em (font-size
    units), so `scale` grows fonts, glyphs, and every pad together.
  * Column glyph kinds are declared per column, because the pairing is not
    fixed: chi/sigma^2 panels are marker=Direct / line=Spectral, the small-R
    overlay is line=Analytical / marker=Measured.
  * Rows render in the order given -- callers state C D G V explicitly
    instead of inheriting plot order.

Usage:
    from paired_legend import paired_table_legend
    paired_table_legend(
        ax,
        rows=[("C", "#DE9800", "^"), ("D", "#005897", "s"),
              ("G", "#A274CB", "D"), ("V", "#CE0043", "o")],
        columns=[("Direct", "marker"), ("Spectral", "line")],
        loc="lower right")

Harness (renders demo PNGs so the legend can be iterated in isolation):
    python figures/paired_legend.py [out_dir]
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple, Union

from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from matplotlib.offsetbox import (AnchoredOffsetbox, DrawingArea, HPacker,
                                  VPacker)
from matplotlib.text import Text

# Row spec: (label, color) or (label, color, marker). Column spec:
# (header, kind) or (header, kind, overrides) with kind in {"marker","line"}
# and overrides e.g. {"linestyle": "--", "linewidth": 2.0, "row_colors": {"C": "#a67200"}}.
RowSpec = Union[Tuple[str, str], Tuple[str, str, str]]
ColSpec = Union[Tuple[str, str], Tuple[str, str, dict]]

# The single style source. Tune here to restyle every caller; per-call kwargs
# (fontsize / marker_edge* / scale / frameon) and `style=` override it.
STYLE = {
    "marker_size": 8,
    "marker_edgecolor": "none",
    "marker_edgewidth": 0.6,
    "line_width": 2.5,
    "handlelength": 3,         # line glyph length, in font-size units
    "line_span_frac": 0.5,    # line glyph spans this fraction of its cell
    "fontsize": 9,
    "framealpha": 1.0,
    "edgecolor": "black",
    "facecolor": "white",
    "frameon": True,
    # table geometry, in font-size units (em)
    "rowsep_em": 0.05,         # vertical gap between rows
    "colsep_em": 0.6,          # horizontal gap between columns
    "cellpad_em": 0.1,        # horizontal padding inside each cell
    "headerpad_em": 0.1,      # extra height around the header text
    "borderpad": 0.2,          # frame padding, font units (legend borderpad)
    "borderaxespad": .7,      # frame <-> axes gap  (legend borderaxespad)
}


def _base_style() -> dict:
    """A copy of STYLE (callers may mutate their copy freely)."""
    return dict(STYLE)


def _text_wh_pts(fig, s: str, fontsize: float) -> Tuple[float, float]:
    """Exact rendered (width, height) of ``s`` in points -- measured through
    the figure's renderer (usetex-safe), no per-font heuristics."""
    if not s:
        return 0.0, 0.0
    t = fig.text(0.5, 0.5, s, fontsize=fontsize)
    try:
        try:
            renderer = fig.canvas.get_renderer()
        except AttributeError:          # non-Agg canvas
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
        bb = t.get_window_extent(renderer=renderer)
        k = 72.0 / fig.dpi
        return bb.width * k, bb.height * k
    except Exception:                    # last resort: em-box heuristic
        return 0.62 * fontsize * len(s), 1.1 * fontsize
    finally:
        t.remove()


def _text_cell(W: float, H: float, s: str, fontsize: float,
               color: str = "black") -> DrawingArea:
    da = DrawingArea(W, H, 0, 0)
    if s:
        da.add_artist(Text(W / 2.0, H / 2.0, s, ha="center", va="center",
                           fontsize=fontsize, color=color))
    return da


def _glyph_cell(W: float, H: float, kind: str, color: str, marker: str,
                st: dict, overrides: dict) -> DrawingArea:
    da = DrawingArea(W, H, 0, 0)
    if kind == "marker":
        da.add_artist(Line2D(
            [W / 2.0], [H / 2.0], linestyle="None",
            marker=overrides.get("marker", marker),
            markersize=overrides.get("markersize", st["marker_size"]),
            markerfacecolor=color,
            markeredgecolor=overrides.get("markeredgecolor",
                                          st["marker_edgecolor"]),
            markeredgewidth=overrides.get("markeredgewidth",
                                          st["marker_edgewidth"])))
    elif kind == "line":
        inset = st["cellpad_em"] * st["fontsize"] * 0.5
        span = (W - 2 * inset) * st.get("line_span_frac", 1.0)
        da.add_artist(Line2D(
            [W / 2.0 - span / 2.0, W / 2.0 + span / 2.0],
            [H / 2.0, H / 2.0], color=color,
            linewidth=overrides.get("linewidth", st["line_width"]),
            linestyle=overrides.get("linestyle", "-"),
            solid_capstyle="butt", dash_capstyle="butt"))
    else:
        raise ValueError(f"column kind {kind!r}: expected 'marker' or 'line'")
    return da


def paired_table_legend(ax,
                        rows: Sequence[RowSpec],
                        columns: Sequence[ColSpec],
                        *,
                        markers: Optional[dict] = None,
                        loc: str = "lower right",
                        bbox_to_anchor=None,
                        fontsize: Optional[float] = None,
                        marker_edgecolor: Optional[str] = None,
                        marker_edgewidth: Optional[float] = None,
                        scale: float = 1.0,
                        frameon: Optional[bool] = None,
                        style: Optional[dict] = None) -> AnchoredOffsetbox:
    """Add the row x column paired legend to ``ax``; returns the artist.

    rows     : ordered (label, color[, marker]); marker falls back to
               ``markers[label]`` then 'o'.
    columns  : ordered (header, kind[, overrides]); kind 'marker' | 'line'.
    loc      : legend-style loc string ('best' is not a thing for anchored
               boxes and maps to 'lower right').
    fontsize / marker_edge* : per-call overrides of the shared style.
    scale    : uniform grow/shrink of the whole legend (fonts, glyphs, pads).
    style    : full style-dict override (see STYLE for keys).
    """
    st = dict(style) if style else _base_style()
    if fontsize is not None:
        st["fontsize"] = fontsize
    if marker_edgecolor is not None:
        st["marker_edgecolor"] = marker_edgecolor
    if marker_edgewidth is not None:
        st["marker_edgewidth"] = marker_edgewidth
    # Uniform shrink/grow of the WHOLE legend: fonts, glyphs, and (via the
    # em-based geometry) every pad/sep follow the scaled fontsize.
    if scale != 1.0:
        st["fontsize"] = float(st["fontsize"]) * scale
        st["marker_size"] = float(st["marker_size"]) * scale
        st["line_width"] = float(st["line_width"]) * scale
        st["marker_edgewidth"] = float(st["marker_edgewidth"] or 0) * scale
    fs = float(st["fontsize"])
    fig = ax.figure

    # -- normalize specs ---------------------------------------------------
    R: List[Tuple[str, str, str]] = []
    for r in rows:
        lab, col = r[0], r[1]
        mk = r[2] if len(r) > 2 else (markers or {}).get(lab, "o")
        R.append((str(lab), col, mk))
    C: List[Tuple[str, str, dict]] = []
    for c in columns:
        hdr, kind = c[0], c[1]
        C.append((str(hdr), str(kind), dict(c[2]) if len(c) > 2 else {}))

    # -- uniform cell geometry (points). Fixed-size cells are what make the
    # table exact: every column's rows stack to identical heights, so the
    # columns can be packed independently and still align row-for-row.
    pad = st["cellpad_em"] * fs
    lab_ws, lab_hs = zip(*[_text_wh_pts(fig, lab, fs) for lab, _, _ in R])
    hdr_ws, hdr_hs = zip(*[_text_wh_pts(fig, hdr, fs) for hdr, _, _ in C])
    H_row = max(max(lab_hs), st["marker_size"]
                + 2 * float(st["marker_edgewidth"] or 0) + 2.0) \
        + st["rowsep_em"] * fs * 0.4
    H_hdr = max(hdr_hs) + 2 * st["headerpad_em"] * fs
    W_lab = max(lab_ws) + 2 * pad
    glyph_min = {"line": st["handlelength"] * fs,
                 "marker": st["marker_size"] * 1.7}
    W_cols = [max(hdr_ws[i], glyph_min[C[i][1]]) + 2 * pad
              for i in range(len(C))]

    rowsep = st["rowsep_em"] * fs

    # -- build columns as VPackers of fixed cells --------------------------
    lab_children = [_text_cell(W_lab, H_hdr, "", fs)]
    lab_children += [_text_cell(W_lab, H_row, lab, fs) for lab, _, _ in R]
    packed = [VPacker(pad=0, sep=rowsep, align="center",
                      children=lab_children)]
    for (hdr, kind, over), W in zip(C, W_cols):
        kids = [_text_cell(W, H_hdr, hdr, fs)]
        row_colors = over.get("row_colors") or {}          # per-row colour override for this column
        kids += [_glyph_cell(W, H_row, kind, row_colors.get(lab, col), mk, st, over)
                 for lab, col, mk in R]
        packed.append(VPacker(pad=0, sep=rowsep, align="center",
                              children=kids))
    table = HPacker(pad=0, sep=st["colsep_em"] * fs, align="bottom",
                    children=packed)

    if loc in (None, "best", 0):
        loc = "lower right"      # anchored boxes have no 'best'
    if frameon is None:          # explicit kwarg wins over the style/JSON
        frameon = bool(st.get("frameon", True))
    box = AnchoredOffsetbox(
        loc=loc, child=table, frameon=frameon,
        pad=st["borderpad"], borderpad=st["borderaxespad"],
        prop=FontProperties(size=fs),
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=(ax.transAxes if bbox_to_anchor is not None else None))
    # frame styled like the scripts' legends: rounded box, black edge, white
    # face; framealpha fades the FACE only (a legend's edge stays opaque).
    box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    box.patch.set_facecolor(to_rgba(st["facecolor"], st["framealpha"]))
    box.patch.set_edgecolor(st["edgecolor"])
    box.set_clip_on(False)
    ax.add_artist(box)
    return box


# ---------------------------------------------------------------------------
# Harness: render the legend on dummy data so it can be iterated in seconds.
#   python figures/paired_legend.py [out_dir]
# ---------------------------------------------------------------------------
def _demo(out_dir: str = ".") -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    TESS_COLORS = {"C": "#DE9800", "D": "#005897",
                   "G": "#A274CB", "V": "#CE0043"}
    TESS_MARKERS = {"D": "s", "V": "o", "G": "D", "C": "^"}

    rows = [(t, TESS_COLORS[t], TESS_MARKERS.get(t, "o"))
            for t in ("C", "D", "G", "V")]
    cases = [
        ("demo_direct_spectral",
         [("Direct", "marker"), ("Spectral", "line")], "lower right"),
        ("demo_analytical_measured",
         [("Analytical", "line"), ("Measured", "marker")], "upper left"),
    ]
    out = []
    x = np.logspace(-1, 1.2, 120)
    os.makedirs(out_dir, exist_ok=True)
    for name, columns, loc in cases:
        fig, ax = plt.subplots(figsize=(3.375, 2.8))
        for i, (lab, col, mk) in enumerate(rows):
            y = (0.4 + 0.25 * i) * x ** (3 - 0.3 * i)
            ax.loglog(x, y, color=col, lw=2.5)
            ax.loglog(x[::12], y[::12], linestyle="none", marker=mk,
                      markersize=4.5, color=col, markeredgecolor="black",
                      markeredgewidth=0.5)
        ax.set_xlabel(r"$R\,\rho_\ell$", fontsize=16)
        ax.set_ylabel(r"$\sigma^2_\ell(R)$", fontsize=16)
        ax.tick_params(which="both", labelsize=12)
        paired_table_legend(ax, rows, columns, loc=loc,
                            marker_edgecolor="black", marker_edgewidth=0.8)
        p = os.path.join(out_dir, f"paired_legend_{name}.png")
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  wrote {p}")
        out.append(p)
    return out


if __name__ == "__main__":
    import sys
    _demo(sys.argv[1] if len(sys.argv) > 1 else ".")
