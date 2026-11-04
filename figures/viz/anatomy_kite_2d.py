"""Fig. 1: windows of the four base tessellations (C/D/G/V) on a Poisson
and a URL generator pattern (seed 16, N = 100, L = 10).

Each panel is a borderless render of the primal cells filled in the
tessellation hue with white cell edges; the generator pattern is overlaid
as black dots. The generators are the same input points on every ntype
(``build_network`` stamps them on D/V/G/C alike), so the dots are the
visual bridge between the four windows.

Cells are recovered with the same periodic half-edge walk
(``wndfc.planar_dual._face_walk``) that the kite / dual-area weight
construction uses, so a cell here is exactly a face there -- no second
triangulation. Homology cycles and degenerate (< 3-gon) faces are dropped.

Framing: cells whose min-image polygon (re-anchored to the box centre by
an integer-L shift) meets the collection box ``WIN_FRAC * L`` are drawn;
the displayed half-window is ``VIEW_FACTOR * WIN_FRAC * L`` (an interior
inset, so no boundary sliver bleeds into the crop). Generators use the
identical integer-L re-anchoring, so every dot lands in its own cell.

Profiles:
  clean      Poisson generators              -> <VIZ_OUT>/clean/
  cleanurl   URL lattice, n_side = 10, a = 1  -> <VIZ_OUT>/cleanurl/

Outputs, per profile and ntype:
  <VIZ_OUT>/<profile>/anatomy_2d_<ntype>_tess_clean.png
  <VIZ_OUT>/<profile>/eps/anatomy_2d_<ntype>_tess_clean.eps
VIZ_OUT defaults to figures/viz/outputs/anatomy.

Running with no environment variables renders the paper figure; the
defaults below are the paper values. Optional overrides:
  VIZ_OUT             output root
  RUN_PROFILES        comma-separated subset of {clean, cleanurl}
  CLEAN_WIN_FRAC      collection box   = WIN_FRAC * L               (0.40)
  CLEAN_VIEW_FACTOR   half-window      = VIEW_FACTOR * WIN_FRAC * L  (0.78)
  CLEAN_LW_SCALE      multiplier on every drawn linewidth            (2.5)
  CLEAN_BLOCK_DOT     generator marker size (scatter ``s``)          (80)
RUN_PANELS is accepted for compatibility; ``tess`` is the only panel.

The sibling schematics import ``TESS_COLORS``, ``_shade`` and
``_face_polygons`` from this module.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))          # release root
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from wndfc.generators import poisson_2d, url_2d
from wndfc.network import build_network
# Internal walker reused so the panels use the *exact* same face
# definition as the kite construction (no second triangulation path).
from wndfc.planar_dual import (
    _build_half_edges,
    _cyclic_neighbor_lists,
    _face_walk,
)

# Tessellation palette shared with the chi panels (figures/chi_panels.py).
TESS_COLORS = {"C": "#DE9800", "D": "#005897", "G": "#A274CB", "V": "#CE0043"}

# Cell fill tone: 0.0 = pure tess hue, >0 lightens toward white, <0 darkens
# (passed straight to _shade).
FILL_SHADE = 0.0

OUTDIR = os.environ.get("VIZ_OUT", os.path.join(_HERE, "outputs", "anatomy"))

SEED, L, N = 16, 10.0, 100

# Framing knobs (paper values as defaults). ``TESS_VIEW_FRAC`` is the
# half-window as a fraction of the collection box, so that
# ``view = TESS_VIEW_FRAC * win`` with ``win = WIN_FRAC * L``.
WIN_FRAC = float(os.environ.get("CLEAN_WIN_FRAC", "0.40"))
_VIEW = float(os.environ.get("CLEAN_VIEW_FACTOR", "0.78")) * WIN_FRAC
TESS_VIEW_FRAC = _VIEW / WIN_FRAC
LW_SCALE = float(os.environ.get("CLEAN_LW_SCALE", "2.5"))
BLOCK_DOT_S = float(os.environ.get("CLEAN_BLOCK_DOT", "80"))


@dataclass(frozen=True)
class Profile:
    name: str
    ctype: str      # "poi" = Poisson (N pts); "URL" = url_2d lattice,
                    # n_side = round(sqrt(N)) so N stays the point budget
    a: float        # URL perturbation amplitude (absolute units)
    clean_dir: str


PROFILES = [
    Profile("clean", "poi", 0.0, os.path.join(OUTDIR, "clean")),
    Profile("cleanurl", "URL", 1.0, os.path.join(OUTDIR, "cleanurl")),
]


def _scale_linewidths(fig, s):
    """Multiply every drawn line / patch / collection-edge width in the
    figure by ``s`` (no-op when s == 1)."""
    if s == 1.0:
        return
    for ax in fig.axes:
        for ln in ax.lines:
            ln.set_linewidth(ln.get_linewidth() * s)
        for col in ax.collections:
            col.set_linewidth(np.asarray(col.get_linewidth(),
                                         dtype=float) * s)
        for pat in ax.patches:
            pat.set_linewidth(pat.get_linewidth() * s)


def _save_eps_sibling(fig, png_path, **savekw):
    """Save a vector .eps sibling of ``png_path`` into ``<png_dir>/eps/``.

    Wrapped in try/except because matplotlib's PostScript backend chokes
    on usetex / RGBA content -- the vector sibling must never break the
    PNG output. ``dpi`` is dropped (vector-only) and ``format='eps'`` is
    forced."""
    if not png_path.lower().endswith(".png"):
        return
    eps_dir = os.path.join(os.path.dirname(png_path), "eps")
    eps_path = os.path.join(eps_dir,
                            os.path.basename(png_path)[:-4] + ".eps")
    try:
        os.makedirs(eps_dir, exist_ok=True)
        kw = {k: v for k, v in savekw.items() if k != "dpi"}
        fig.savefig(eps_path, format="eps", **kw)
    except Exception as exc:  # PS backend / usetex / RGBA fallout
        print(f"  [eps skip] {eps_path}: {exc}")


def _save_clean_only(fig, clean_dir, name):
    """Save the text-free PNG ``<clean_dir>/<name>_clean.png`` (+ eps
    sibling). Borderless: axes off, zero pad, white facecolor, so the
    geometry bleeds to the image edge with no frame or whitespace."""
    out = os.path.join(clean_dir, name + "_clean.png")
    for ax in fig.axes:
        ax.set_axis_off()
    _scale_linewidths(fig, LW_SCALE)
    save_kw = dict(dpi=175, bbox_inches="tight", pad_inches=0,
                   facecolor="white")
    fig.savefig(out, **save_kw)
    _save_eps_sibling(fig, out, **save_kw)
    plt.close(fig)
    print(f"  saved {out}")


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def _shade(hex_color: str, t: float) -> tuple:
    """Blend ``hex_color`` toward white (t>0) or black (t<0) by |t|."""
    from matplotlib.colors import to_rgb
    base = np.array(to_rgb(hex_color))
    if t >= 0:
        mix = base * (1 - t) + np.array([1.0, 1.0, 1.0]) * t
    else:
        mix = base * (1 + t)
    return tuple(mix)


def _scatter_generators(ax, net, center, half):
    """Overlay the generator pattern as black dots.

    Each generator is taken to its minimum image nearest ``center`` -- the
    identical integer-L re-anchoring ``panel_tess_patch`` uses for its
    faces, so dots land inside the cells they belong to -- then those
    within 1.2*half of ``center`` are drawn (axes clipping trims the
    overflow)."""
    if net.generators is None:
        return
    g = np.asarray(net.generators, dtype=float)
    if g.size == 0:
        return
    c = np.asarray(center, dtype=float)
    g = g - L * np.round((g - c) / L)
    m = ((np.abs(g[:, 0] - c[0]) <= 1.2 * half)
         & (np.abs(g[:, 1] - c[1]) <= 1.2 * half))
    if not m.any():
        return
    ax.scatter(g[m, 0], g[m, 1], s=BLOCK_DOT_S, c="black",
               edgecolors="none", linewidths=0.0, zorder=9)


def _face_polygons(net):
    """Every cellular face of the base tessellation as a polygon.

    Reuses the *same* half-edge walk the kite construction uses
    (``_face_walk``), so a face here is exactly a face there. Each
    polygon is lifted into the universal cover anchored on its first
    half-edge's tail (identical to ``planar_dual._face_geometry``), so its
    shape and area are exact; callers re-anchor by integer-L shifts for
    framing.

    Homology cycles and degenerate (< 3-gon) faces are dropped.
    Returns a list of dicts ``{fi, verts (n,2), centroid (2,), area}``.
    """
    he_tail, he_head, he_shift, he_twin = _build_half_edges(net)
    adj_start, adj_he, adj_inv = _cyclic_neighbor_lists(net, he_tail, he_head)
    _, faces = _face_walk(he_tail, he_head, he_shift, he_twin,
                          adj_start, adj_he, adj_inv)
    pts = np.asarray(net.points, dtype=float)
    Lf = float(net.L)
    out = []
    for fi, face in enumerate(faces):
        if face["is_homology"]:
            continue
        hes = face["half_edges"]
        n = hes.shape[0]
        if n < 3:
            continue
        verts = np.empty((n, 2), dtype=float)
        verts[0] = pts[he_tail[hes[0]]]
        cum = np.zeros(2, dtype=float)
        for kk in range(1, n):
            cum += he_shift[hes[kk - 1]].astype(float) * Lf
            verts[kk] = pts[he_tail[hes[kk]]] + cum
        x, y = verts[:, 0], verts[:, 1]
        area = 0.5 * abs(float(np.sum(x * np.roll(y, -1)
                                      - np.roll(x, -1) * y)))
        out.append(dict(fi=fi, verts=verts,
                         centroid=verts.mean(axis=0), area=area))
    return out


def _frame(ax, center, half):
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


# --------------------------------------------------------------------------
# panel
# --------------------------------------------------------------------------
def panel_tess_patch(ntype, net, color, clean_dir):
    """The base tessellation filled in the tess hue with thin white edges.
    Cells are collected over the ``WIN_FRAC * L`` box and the view is
    ``TESS_VIEW_FRAC * win``, an interior inset."""
    faces = _face_polygons(net)
    center = np.array([L / 2.0, L / 2.0])
    win = WIN_FRAC * L              # collection box (cells gathered here)
    view = TESS_VIEW_FRAC * win     # displayed crop
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    polys = []
    for f in faces:
        v = f["verts"]
        shift = -L * np.round((v.mean(axis=0) - center) / L)
        vv = v + shift
        if (vv[:, 0].max() < center[0] - win
                or vv[:, 0].min() > center[0] + win
                or vv[:, 1].max() < center[1] - win
                or vv[:, 1].min() > center[1] + win):
            continue
        polys.append(vv)
    ax.add_collection(PatchCollection(
        [MplPolygon(p, closed=True) for p in polys],
        facecolor=_shade(color, FILL_SHADE), edgecolor="white",
        linewidths=1.7, linestyle="-", zorder=2))
    _scatter_generators(ax, net, center, view)
    _frame(ax, center, view)
    _save_clean_only(fig, clean_dir, f"anatomy_2d_{ntype}_tess")


def main():
    panels = {s.strip() for s in
              os.environ.get("RUN_PANELS", "").split(",") if s.strip()}
    if panels - {"tess", "tess_patch"}:
        raise SystemExit(
            f"RUN_PANELS: unknown panel(s) {sorted(panels)}; "
            "only 'tess' is available")
    only = os.environ.get("RUN_PROFILES", "").strip()
    selected = (PROFILES if not only
                else [p for p in PROFILES
                      if p.name in {s.strip() for s in only.split(",")}])
    if not selected:
        raise SystemExit(
            f"RUN_PROFILES={only!r} matches none of "
            f"{[p.name for p in PROFILES]}")
    for p in selected:
        os.makedirs(p.clean_dir, exist_ok=True)
        if p.ctype == "URL":
            n_side = int(round(N ** 0.5))
            pts = url_2d(n_side, L, a=p.a, seed=SEED)
            ptag = f"URL n_side={n_side} a={p.a:g}"
        else:
            pts = poisson_2d(N, L, seed=SEED)
            ptag = f"poisson N={N}"
        print(f"\n##### profile {p.name}: seed={SEED} L={L:g} "
              f"{ptag} -> {p.clean_dir} #####")
        for ntype in ("D", "V", "G", "C"):
            # Only edges and generators feed the figure: skip the
            # secondary cell-area passes (C centroid Voronoi, V circumcenter
            # Voronoi), which never touch the edge table.
            net = build_network(pts, L, ntype, c_area_mode="triangle",
                                compute_cell_areas=False)
            panel_tess_patch(ntype, net, TESS_COLORS[ntype], p.clean_dir)
        print(f"\n=== {p.name} done -> {p.clean_dir} ===")


if __name__ == "__main__":
    main()
