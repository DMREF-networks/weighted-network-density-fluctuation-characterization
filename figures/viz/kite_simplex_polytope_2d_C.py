"""2D simplex-polytope kite mesh -- Centroidal (C) panel only.

Variant of ``kite_simplex_polytope_2d.py`` narrowed
to the C tessellation, so one panel regenerates in one shot instead of the
D/V/G/C x circ/bary set of eight. Deliberate differences from the parent, and
nothing else:

  * C only -- no D/V/G loop, no all-placements loop.
  * ``bary`` is now the TRUE area COM (signed-area-weighted mean of the fan
    simplices' vertex means = the exact polygon centroid). The parent used
    ``P.mean(axis=0)``, the mean of the polytope's corners.
  * 15 cells in the cluster (parent: 8). Same style at the same sizes -- more
    of the same mesh, not a rescaled one.
  * Black dots on the primal vertices (the polytope corners = network nodes),
    alongside the white dual-vertex dots the parent already drew.
  * The (a)/(b)/(c) leader labels are byte-for-byte the parent's logic, now
    switchable: ``--no-labels`` drops them (and tags the filename).
  * Writes to its own output subdir under figures/viz/outputs/.

Run (from the release root):
    python figures/viz/kite_simplex_polytope_2d_C.py
    .venv-voro/Scripts/python.exe kite_simplex_polytope_2d_C.py --no-labels
Knobs (env): KMESH_SEED, KMESH_N, KMESH_L, KMESH_WIN, KMESH_CELLS, KMESH_PAD,
KMESH_FIGIN, KMESH_TRIM_M, KMESH_DPI.
Ink: KMESH_SCALE multiplies every line width and dot together (default 2);
KMESH_LW, KMESH_PDOT, KMESH_DDOT, KMESH_KITE_DIV_EDGE set what it scales.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon as MplPolygon

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))          # release root (wndfc/)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from wndfc.generators import poisson_2d
from wndfc.network import build_network
from anatomy_kite_2d import TESS_COLORS, _shade
import kite_simplex_polytope_2d as _parent
from kite_simplex_polytope_2d import (_edge_map, _gather_polys, _grow_cluster,
                                      _nearest)

SEED = int(os.environ.get("KMESH_SEED", "16"))
L = float(os.environ.get("KMESH_L", "10"))
N = int(os.environ.get("KMESH_N", "100"))
WIN = float(os.environ.get("KMESH_WIN", "0.40"))      # candidate gather / L
CELLS = int(os.environ.get("KMESH_CELLS", "15"))      # cells in the cluster
PAD = float(os.environ.get("KMESH_PAD", "0.70"))      # frame margin (whitespace)
FIGIN = float(os.environ.get("KMESH_FIGIN", "7.0"))   # fixed figure side (in)
DPI = float(os.environ.get("KMESH_DPI", "300"))       # 7 in x 300 -> 2100 px
TRIM_M = float(os.environ.get("KMESH_TRIM_M", "0.015"))    # trimmed margin

# One dial for every mark. Widths scale linearly; scatter sizes are AREAS, so
# they take SCALE**2 -- that way SCALE=2 doubles the apparent size of a dot and
# of a line alike, instead of doubling a dot's area and only its diameter's
# square root.
SCALE = float(os.environ.get("KMESH_SCALE", "2.0"))
LW_BASE = float(os.environ.get("KMESH_LW", "1.6"))
LW = LW_BASE * SCALE
# The gradient panels carry their structure in the ramp, so their linework
# drops back to the unscaled base and lets the ground do the work.
GRAD_LW = float(os.environ.get("KMESH_GRAD_LW", str(LW_BASE)))
DUAL_LS = "-"            # dual bonds are solid black -- the routed network and
#                          the kite's own apex-to-apex diagonal alike
# The dash pattern is in points, so it has to ride SCALE somehow, but neither
# extreme works: left unscaled against a doubled width each dash is barely
# longer than it is wide and reads as a row of blocks, while scaling it fully
# fits only ~2 dashes on a fan segment and they read as isolated strokes.
# sqrt(SCALE) splits it; KMESH_DASH_S overrides.
DASH_S = float(os.environ.get("KMESH_DASH_S", str(SCALE ** 0.5)))
PDOT = float(os.environ.get("KMESH_PDOT", "30")) * SCALE ** 2   # primal dots
DDOT = float(os.environ.get("KMESH_DDOT", "42")) * SCALE ** 2   # dual dots
DDOT_RIM = float(os.environ.get("KMESH_DDOT_RIM", "1.1")) * SCALE
KITE_COLOR = os.environ.get("KMESH_KITE_COLOR", "#6A1B9A")     # outline
# The kite's dual bisector reads in its own colour, so the outline and the
# cut through it can be named separately in the caption.
BISECT_COLOR = os.environ.get("KMESH_BISECT_COLOR", "#C62828")
SHOW_BISECT = os.environ.get("KMESH_BISECT", "0") != "0"       # hidden by default; KMESH_BISECT=1 draws the cut
# Casing width as a FRACTION of the line it cases, so it tracks whatever width
# the panel is drawn at instead of needing its own scaling.
KITE_DIV_EDGE = float(os.environ.get("KMESH_KITE_DIV_EDGE", "0.5"))
INK_DARK = "#1a1a1a"
# Where the hue peaks along a gradient panel's white -> hue -> black ramp.
GRAD_MID = float(os.environ.get("KMESH_GRAD_MID", "0.45"))
# Strips per simplex on a gradient panel -- see _piece_tri: Gouraud blends
# vertex colours, so the ramp needs intermediate nodes to be sampled at all.
GRAD_BANDS = int(os.environ.get("KMESH_GRAD_BANDS", "16"))
# --bleed: window / content bbox. Measured -- 0.45 leaves <=1% white on
# the worst edge here; the ring is convex enough not to need more.
FILL_F = float(os.environ.get("KMESH_FILL_F", "0.374"))
# Fan treatment on the flat panels: dark grey round dots at the line's own
# width. Dash values are multiplied by the linewidth by matplotlib, so these
# are ratios -- a near-zero "on" under a round cap renders as a circle of
# exactly one linewidth across.
FAN_GREY = os.environ.get("KMESH_FAN_GREY", "#555555")
DOT_ON = float(os.environ.get("KMESH_DOT_ON", "0.05"))
DOT_GAP = float(os.environ.get("KMESH_DOT_GAP", "2.0"))
SHADE_HI = float(os.environ.get("KMESH_SHADE_HI", "0.45"))     # light end
SHADE_LO = float(os.environ.get("KMESH_SHADE_LO", "-0.35"))    # deep end
MASS_BY = os.environ.get("KMESH_MASS_BY", "simplex")     # or "cell"
SIMPLEX_NORM = os.environ.get("KMESH_SIMPLEX_NORM", "cell")   # or "global"
OUTDIR = os.environ.get("VIZ_OUT", os.path.join(_HERE, "outputs", "kite_simplex_C"))
PAPER_DIR = os.path.join(OUTDIR, "paper")

_parent.L = L        # _gather_polys min-images against the parent module's L


def _ink(shade):
    """Per-panel line and dot treatment.

    The gradient panels (``wells``/``simplex``) get the base linewidth and
    dash, and black dual-vertex dots: their ramp runs to white at the primal
    boundary so the fill blends out at the cell edges, and to its deepest at
    the centroid so it bleeds into the dot sitting there. Everything else keeps
    the scaled ink and the white dots.
    """
    grad = shade in ("wells", "simplex")
    lw = GRAD_LW if grad else LW
    ds = 1.0 if grad else DASH_S
    # The flat panels take round dots in dark grey; everything else keeps the
    # dashed black fan.
    flat = shade == "flat"
    dash = (0, (DOT_ON, DOT_GAP)) if flat else (0, (7 * ds, 4 * ds))
    # On a gradient panel the two dot sets swap against the ground they sit on:
    # the dual dots go black so the dark well floor bleeds into them, and the
    # primal dots go white with a purple rim so they stay findable where the
    # ramp has washed out to white at the primal boundary.
    return dict(grad=grad, lw=lw, dash=dash,
                fan_color=FAN_GREY if flat else INK_DARK,
                fan_cap="round" if flat else "butt",
                dot_face=INK_DARK if grad else "white",
                dot_rim="none" if grad else INK_DARK,
                dot_rim_lw=0.0 if grad else DDOT_RIM,
                pdot_face="white" if grad else INK_DARK,
                pdot_rim=KITE_COLOR if grad else "none",
                pdot_rim_lw=lw * 0.7 if grad else 0.0,
                # The ramp washes out to white exactly where the primal edges
                # run, so on a gradient panel they need the kite divider's
                # casing to stay readable -- same white interior, same purple
                # boundary, same width.
                cell_fx=([pe.withStroke(linewidth=lw * (1.0 + KITE_DIV_EDGE),
                                        foreground=KITE_COLOR)]
                         if grad else []))


def _area_com(P):
    """True area COM of the polytope: the signed-area-weighted mean of its fan
    simplices' vertex means.

    Simplex ``(apex, a, b)`` contributes its vertex mean ``(apex + a + b)/3``
    weighted by its signed area; the apex term cancels against the closed
    boundary, so this is the exact polygon centroid however the fan is
    anchored -- unlike ``P.mean(axis=0)``, which averages only the corners.
    Signed (not absolute) areas keep it exact for non-convex faces too.
    """
    o = P.mean(axis=0)                                # local origin, small sums
    a = P - o
    b = np.roll(a, -1, axis=0)
    cr = a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]        # 2 x signed simplex area
    tot = cr.sum()
    if abs(tot) < 1e-14:
        return o
    return o + ((a + b) * cr[:, None]).sum(axis=0) / (3.0 * tot)


def _shrink_to_ink(fig, ax, lo, hi, lw=None, margin=TRIM_M, iters=8,
                   tol=1e-4):
    """Pull the axes window in to the drawn ink.

    ``bbox_inches="tight"`` cannot do this job: it crops to the *axes*
    rectangle, which ``subplots_adjust(0, 0, 1, 1)`` has already made the whole
    figure, so the dead ring inside the axes survives. Shrinking the data
    limits is what actually trims it.

    ``lo``/``hi`` are the polygon bbox. Every mesh artist -- fan, kite,
    diagonal, leader lines, both dot sets -- lies inside it, so the only extras
    are the dot radius (points) and the labels. They are taken analytically
    rather than from ``get_window_extent`` on the collections: once the window
    starts shrinking those return *clipped* extents, which feed back into the
    next pass and collapse the view onto whatever is still on screen.

    The figure is reshaped to the ink's aspect too, or ``aspect equal`` would
    letterbox a non-square window back into the square canvas -- which is the
    whitespace being removed. Iterated because dots and labels are sized in
    points: shrinking the window makes them relatively larger.
    """
    pad_pt = 0.5 * np.sqrt(max(PDOT, DDOT)) + (LW if lw is None else lw)
    prev = None
    for _ in range(iters):
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        x0, x1 = ax.get_xlim()
        p = pad_pt * (x1 - x0) / (fig.get_size_inches()[0] * 72.0)
        bx0, by0 = lo[0] - p, lo[1] - p
        bx1, by1 = hi[0] + p, hi[1] + p
        inv = ax.transData.inverted()
        for t in ax.texts:                    # labels only; nothing else can
            bb = t.get_window_extent(rend)    # fall outside the polygon bbox
            (tx0, ty0), (tx1, ty1) = inv.transform(
                [[bb.x0, bb.y0], [bb.x1, bb.y1]])
            bx0, by0 = min(bx0, tx0), min(by0, ty0)
            bx1, by1 = max(bx1, tx1), max(by1, ty1)
        cx, cy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
        hx = 0.5 * (bx1 - bx0) * (1.0 + margin)
        hy = 0.5 * (by1 - by0) * (1.0 + margin)
        new = (cx, cy, hx, hy)
        if prev is not None and max(abs(a - b)
                                    for a, b in zip(new, prev)) < tol:
            break
        prev = new
        ax.set_xlim(cx - hx, cx + hx)
        ax.set_ylim(cy - hy, cy + hy)
        big = max(hx, hy)
        fig.set_size_inches(FIGIN * hx / big, FIGIN * hy / big)


def _center(P, placement):
    """True area COM, or least-squares circumcentre of the polygon's vertices
    (exact for a triangle; best-fit otherwise, may sit off-centre)."""
    if placement == "bary":
        return _area_com(P)
    x, y = P[:, 0], P[:, 1]
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(P))])
    try:
        sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
        return np.array([sol[0], sol[1]])
    except np.linalg.LinAlgError:
        return P.mean(axis=0)


def _dual_segments(cl_emap, cof, mode, fringe=()):
    """Dual bonds over the drawn region, as plain 2-point segments.

    ``straight`` links the two incident polygon centroids directly. ``mid``
    runs every incident centroid to the edge barycentre instead, so an interior
    edge comes out as the bent centroid -> midpoint -> centroid path. Same bond
    set either way -- only the route between the centroids differs.

    ``fringe`` entries are the bonds that jut out through the drawn boundary;
    in ``mid`` the inner half is already in ``cl_emap``, so only the outer
    centroid -> midpoint leg is added here.
    """
    seg = []
    if mode == "straight":
        for lst in cl_emap.values():
            if len(lst) != 2:
                continue
            (i, _, _), (j, _, _) = lst
            seg.append([cof[i], cof[j]])
        for j, va, vb, i in fringe:
            seg.append([cof[i], cof[j]])
    else:
        for lst in cl_emap.values():
            for i, va, vb in lst:
                seg.append([cof[i], 0.5 * (va + vb)])
        for j, va, vb, i in fringe:
            seg.append([cof[j], 0.5 * (va + vb)])
    return seg


def _tri_area(c, a, b):
    return 0.5 * abs((a[0] - c[0]) * (b[1] - c[1])
                     - (b[0] - c[0]) * (a[1] - c[1]))


def _fan_areas(P, c):
    n = len(P)
    return np.array([_tri_area(c, P[k], P[(k + 1) % n]) for k in range(n)])


def _pieces(cluster, cands, cof, fringe=()):
    """Every simplex that gets drawn, as ``(apex, va, vb, area, owner)``.

    A core cell contributes its whole fan; a fringe entry contributes only the
    single wedge that the boundary edge cuts out of its polygon -- that wedge
    plus the core wedge on the near side is exactly the kite of that edge, so
    the drawn region ends on kite outlines.
    """
    out = []
    for i in cluster:
        P, c = cands[i], cof[i]
        n = len(P)
        for k in range(n):
            a, b = P[k], P[(k + 1) % n]
            out.append((c, a, b, _tri_area(c, a, b), i))
    for j, va, vb, _ in fringe:
        c = cof[j]
        out.append((c, va, vb, _tri_area(c, va, vb), j))
    return out


def _piece_tri(pieces, val_of_apex, bands=1):
    """The pieces as one Triangulation with PRIVATE nodes per simplex.

    Private nodes let each simplex carry its own apex value. For a constant
    apex value that is identical to sharing one apex node per cell (the ramp
    along a shared radial edge interpolates the same either way), so the same
    builder serves every shaded mode.

    ``bands`` splits each simplex into that many strips running from its primal
    edge up to its apex. Gouraud shading interpolates the vertex COLOURS, not
    the scalar, so at one band per simplex a ramp is only ever sampled at its
    two ends: a white-to-black ramp then renders as a straight greyscale blend
    and every hue between the ends is lost. Intermediate nodes put real samples
    on the ramp so the mesh actually follows the colormap.
    """
    xy, val, tris = [], [], []
    for (c, a, b, _, owner), v in zip(pieces, val_of_apex):
        for k in range(bands):
            t0, t1 = k / bands, (k + 1) / bands
            a0, b0 = a + (c - a) * t0, b + (c - b) * t0
            n = len(xy)
            if t1 >= 1.0:                     # top strip closes on the apex
                xy.extend([a0, b0, c])
                val.extend([v * t0, v * t0, float(v)])
                tris.append([n, n + 1, n + 2])
            else:
                a1, b1 = a + (c - a) * t1, b + (c - b) * t1
                xy.extend([a0, b0, b1, a1])
                val.extend([v * t0, v * t0, v * t1, v * t1])
                tris.append([n, n + 1, n + 2])
                tris.append([n, n + 2, n + 3])
    xy = np.asarray(xy, dtype=float)
    return (mtri.Triangulation(xy[:, 0], xy[:, 1], np.asarray(tris)),
            np.asarray(val))


def _paint(ax, color, shade, pieces, cands, cof):
    """Lay down the shaded ground and return the cell-outline facecolor.

    ``flat``  -- one hue, the parent's fill.
    ``wells`` -- Gouraud ramp per cell, light on the primal polygon boundary
                 deepening to the dual vertex, so each cell reads as a basin
                 whose floors the dual bonds run between.
    ``mass``  -- one flat hue per simplex from its area (``KMESH_MASS_BY=cell``
                 shades by whole-cell area instead), i.e. the mass distribution
                 the area COM is the balance point of.
    ``simplex`` -- ``wells`` quantised by the decomposition: every simplex gets
                 its own ramp to the centroid, reaching a depth set by its
                 share of the polygon's area.

    Area references are taken from each owner's WHOLE polygon, so a fringe
    wedge is scaled against the cell it was cut from rather than against
    itself.
    """
    if shade == "flat":
        # Painted per simplex, not per cell, so a fringe wedge (which has no
        # whole polygon behind it) still gets its ground. Edge matched to face
        # closes the antialiasing seams between neighbouring simplices.
        f = _shade(color, 0.0)
        ax.add_collection(PatchCollection(
            [MplPolygon(np.array([c, a, b]), closed=True)
             for c, a, b, _, _ in pieces],
            facecolor=f, edgecolor=f, linewidths=0.3, zorder=2))
        return "none"
    # Gradient panels run white -> hue -> black: white at the primal boundary
    # so the ground fades out and the cell edges dissolve into it rather than
    # ruling it off, black at the centroid so the well floor meets the dot
    # sitting on it. The hue has to be a third stop in the middle or a
    # white-to-black ramp would be pure greyscale and the tessellation would
    # lose its colour. mass is flat per simplex, so it keeps the two-stop ramp.
    if shade in ("wells", "simplex"):
        cmap = LinearSegmentedColormap.from_list(
            "kmesh", [(0.0, "white"), (GRAD_MID, _shade(color, 0.0)),
                      (1.0, INK_DARK)])
    else:
        cmap = LinearSegmentedColormap.from_list(
            "kmesh", [_shade(color, SHADE_HI), _shade(color, SHADE_LO)])
    owners = {p[4] for p in pieces}
    fan = {i: _fan_areas(cands[i], cof[i]) for i in owners}

    if shade == "mass":
        if MASS_BY == "cell":
            m = np.array([fan[p[4]].sum() for p in pieces])
        else:
            m = np.array([p[3] for p in pieces])
        span = m.max() - m.min()
        t = (m - m.min()) / span if span > 0 else np.zeros_like(m)
        tri, _ = _piece_tri(pieces, np.zeros(len(pieces)))
        ax.tripcolor(tri, facecolors=t, shading="flat", cmap=cmap, zorder=2)
        return "none"

    if shade == "wells":
        v = np.ones(len(pieces))
    else:                                     # simplex
        gmax = max((a.max() for a in fan.values() if a.size), default=1.0)
        v = []
        for p in pieces:
            ref = fan[p[4]].max() if SIMPLEX_NORM == "cell" else gmax
            v.append(p[3] / ref if ref > 0 else 1.0)
        v = np.asarray(v)
    tri, val = _piece_tri(pieces, v, bands=GRAD_BANDS)
    ax.tripcolor(tri, val, shading="gouraud", cmap=cmap, zorder=2,
                 vmin=0.0, vmax=1.0)
    return "none"



def _edge_key(a, b):
    return tuple(sorted((tuple(np.round(a, 6)), tuple(np.round(b, 6)))))


def _shared_edge(emap, i, j):
    """The primal edge between two polygons, or None if they do not share one."""
    for key, lst in emap.items():
        if {t[0] for t in lst} == {i, j}:
            return np.array(key[0]), np.array(key[1])
    return None


def _edges_at(cands, i, key_v):
    """The two edges of face ``i`` that meet the vertex with key ``key_v``."""
    P = cands[i]
    n = len(P)
    for k in range(n):
        if tuple(np.round(P[k], 6)) == key_v:
            return [(P[k], P[(k - 1) % n]), (P[k], P[(k + 1) % n])]
    return []


def _dual_face(emap, cands, key_v, start):
    """The cycle of faces meeting one primal vertex, in rotational order.

    That cycle IS the dual face at that vertex. Where the network is 3-regular
    (V and C) it is a triangle and the centre's two edge-neighbours happen to
    be adjacent to each other, but a Delaunay vertex carries 7-8 faces and a
    Gabriel vertex 4-6, so the ring has to be walked edge by edge rather than
    read off the centre's neighbours -- assuming the short version would invent
    bonds between faces that never touch. Returns None if the walk runs off the
    gathered window.
    """
    cyc, cur, prev = [start], start, None
    while True:
        es = [e for e in _edges_at(cands, cur, key_v) if _edge_key(*e) != prev]
        if not es:
            return None
        prev = _edge_key(*es[0])
        others = [t[0] for t in emap.get(prev, []) if t[0] != cur]
        if not others:
            return None
        nxt = others[0]
        if nxt == start:
            return cyc
        if nxt in cyc:
            return None
        cyc.append(nxt)
        cur = nxt


def _ring_structure(emap, cands, seed):
    """Everything the ring figure draws around one central polygon.

    Walks the dual face at each of the centre's corners, reduces them to a
    deduped bond list, and takes from that the wedges each bond needs (its
    kite's two halves) plus the primal edges that have to be drawn white
    because their polygons are only partly shown. The centre itself is drawn
    whole, so it contributes no wedge.

    Returns ``(faces, bonds, wedges, outer, missed)``.
    """
    P = cands[seed]
    faces, missed = [], 0
    for k in range(len(P)):
        cyc = _dual_face(emap, cands, tuple(np.round(P[k], 6)), seed)
        if cyc is not None and len(cyc) >= 3:
            faces.append(cyc)
        else:
            missed += 1

    bonds, seen_b = [], set()
    for cyc in faces:
        n = len(cyc)
        for k in range(n):
            i, j = cyc[k], cyc[(k + 1) % n]
            key = (min(i, j), max(i, j))
            if i == j or key in seen_b:
                continue
            e = _shared_edge(emap, i, j)
            if e is None:
                continue
            seen_b.add(key)
            bonds.append((i, j, e[0], e[1]))

    wedges, seen_w, outer = [], set(), []
    for i, j, a, b in bonds:
        ek = _edge_key(a, b)
        if seed not in (i, j):
            outer.append((a, b))          # the centre's own edges come from
        for f in (i, j):                  # its polygon patch already
            if f == seed or (f, ek) in seen_w:
                continue
            seen_w.add((f, ek))
            wedges.append((f, a, b, seed))
    return faces, bonds, wedges, outer, missed


def _ring_duals(bonds, cof, mode):
    """The dual bonds as segments: straight centroid to centroid, or bent
    through the primal edge barycentre."""
    seg = []
    for i, j, a, b in bonds:
        if mode == "straight":
            seg.append([cof[i], cof[j]])
        else:
            mid = 0.5 * (a + b)
            seg.append([cof[i], mid])
            seg.append([cof[j], mid])
    return seg


def panel(ntype, placement, pts, color, cells, labels, trim=True, dual="none",
          shade="flat", scope="ring", bleed=False, close_cells=False):
    net = (build_network(pts, L, "C", c_area_mode="triangle") if ntype == "C"
           else build_network(pts, L, ntype))
    center = np.array([L / 2.0, L / 2.0])
    cands = _gather_polys(net, center, WIN * L)
    if len(cands) < 2:
        print(f"  [skip] {ntype}/{placement}: too few faces")
        return
    cent = [p.mean(axis=0) for p in cands]

    # Face adjacency (share an edge).
    adj = {i: set() for i in range(len(cands))}
    for lst in _edge_map(cands, range(len(cands))).values():
        idxs = [t[0] for t in lst]
        for a in idxs:
            adj[a].update(j for j in idxs if j != a)
    seed = int(np.argmin([np.linalg.norm(c - center) for c in cent]))

    emap = _edge_map(cands, range(len(cands)))
    bonds = []
    if scope == "ring":
        # Just the one primal polygon. Its neighbours enter only as the wedge
        # each contributes to a shared kite, so the drawn region is cut at the
        # kites bounding the dual faces rather than at whole cells.
        cluster = {seed}
        faces, bonds, fringe, outer_edges, missed = _ring_structure(
            emap, cands, seed)
        if missed:
            print(f"  [warn] {missed} dual face(s) ran off the gathered "
                  f"window -- raise KMESH_WIN (now {WIN:g})")
    else:
        cluster = _grow_cluster(seed, adj, cent, center, cells)
        faces, fringe, outer_edges = [], [], []
        if len(cluster) < cells:
            print(f"  [warn] cluster grew to {len(cluster)}/{cells} cells -- "
                  f"raise KMESH_WIN (now {WIN:g}) for a bigger pool")
    cof = {i: _center(cands[i], placement)
           for i in set(cluster) | {f[0] for f in fringe}
           | {b for i, j, _, _ in bonds for b in (i, j)}}
    if close_cells and scope == "ring":
        # Finish every neighbour the ring touches: the whole polygon (white
        # boundary, full dotted fan, corner dots) instead of just the wedge
        # its kite cuts out. Only the PRIMAL side grows -- ``bonds`` is left
        # alone, so the dual network stays exactly the ring's.
        ring = sorted({f[0] for f in fringe} - {seed})
        cluster = [seed] + ring
        fringe = []
        print(f"  close-cells: {len(ring)} neighbour polygon(s) drawn whole")
    polys = [cands[i] for i in cluster]
    pieces = _pieces(cluster, cands, cof, fringe)
    if scope == "ring":
        sizes = "/".join(str(len(f)) for f in faces)
        print(f"  ring: 1 central {len(cands[seed])}-gon | dual faces [{sizes}]"
              f" | {len(bonds)} bonds, {len(fringe)} wedges")

    ink = _ink(shade)
    fig, ax = plt.subplots(figsize=(FIGIN, FIGIN))
    facecolor = _paint(ax, color, shade, pieces, cands, cof)
    ax.add_collection(PatchCollection(
        [MplPolygon(p, closed=True) for p in polys],
        facecolor=facecolor, edgecolor="white", linewidths=ink["lw"],
        zorder=2.5, path_effects=ink["cell_fx"]))
    # The outward primal edges are cell edges too, but their polygons are only
    # partly drawn, so they need laying in by hand to stay white.
    if outer_edges:
        ax.add_collection(LineCollection(
            [[a, b] for a, b in outer_edges], colors="white",
            linewidths=ink["lw"], zorder=2.6,
            path_effects=ink["cell_fx"]))
    # Fan (apex -> primal corners): dashed, the same hand as the kite's sides,
    # since they are the same kind of segment. A fringe wedge contributes only
    # its own two, which are what close the kite on the far side.
    radial = [[c, a] for c, a, b, _, _ in pieces] \
        + [[c, b] for c, a, b, _, _ in pieces]
    ax.add_collection(LineCollection(radial, colors=ink["fan_color"],
                                     linewidths=ink["lw"],
                                     linestyles=[ink["dash"]],
                                     capstyle=ink["fan_cap"], zorder=3))

    cl_emap = _edge_map(cands, cluster)
    if dual != "none":
        # Solid black: the dual polygon reads as the one continuous structure,
        # against everything primal-vertex-to-centroid being dashed.
        seg = (_ring_duals(bonds, cof, dual) if scope == "ring"
               else _dual_segments(cl_emap, cof, dual, fringe))
        ax.add_collection(LineCollection(
            seg, colors="black", linewidths=ink["lw"], linestyles=[DUAL_LS],
            zorder=3.5))

    # Kite: a cell edge with the two centres on opposite sides (a clean,
    # non-self-intersecting kite). Most central such edge.
    kite = None
    if scope == "ring":
        # Most central bond that touches the centre, so the highlight sits on
        # one of its own edges.
        best = None
        for i, j, va, vb in bonds:
            if seed not in (i, j):
                continue
            far = j if i == seed else i
            d = np.linalg.norm(0.5 * (va + vb) - center)
            if best is None or d < best[0]:
                best = (d, np.array([cof[seed], va, cof[far], vb]))
        kite = None if best is None else best[1]
    else:
        for key, lst in sorted(cl_emap.items(), key=lambda kv: np.linalg.norm(
                0.5 * (np.array(kv[0][0]) + np.array(kv[0][1])) - center)):
            if len(lst) != 2:
                continue
            (i, va, vb), (j, _, _) = lst
            c1, c2, mid, e = cof[i], cof[j], 0.5 * (va + vb), vb - va
            if np.cross(e, c1 - mid) * np.cross(e, c2 - mid) >= 0:
                continue
            kite = np.array([c1, va, c2, vb])
            break
        if kite is None:
            for key, lst in cl_emap.items():
                if len(lst) == 2:
                    (i, va, vb), (j, _, _) = lst
                    kite = np.array([cof[i], va, cof[j], vb])
                    break

    kite_tgt = kite.mean(axis=0) if kite is not None else None
    edge_mids = [0.5 * (np.array(k[0]) + np.array(k[1])) for k in cl_emap]
    edge_tgt = _nearest(edge_mids, [kite_tgt], center, 0.05 * L)
    ctr_tgt = _nearest(list(cof.values()), [kite_tgt, edge_tgt], center,
                       0.05 * L)

    if kite is not None:
        ax.add_patch(MplPolygon(kite, closed=True, fill=False,
                                edgecolor=KITE_COLOR, linewidth=ink["lw"],
                                zorder=4))
        # Dual divider c1-c2: the outline's own hand, so the centroid-to-
        # centroid cut reads as part of the highlight rather than as one more
        # black bond.
        if SHOW_BISECT:
            ax.plot([kite[0][0], kite[2][0]], [kite[0][1], kite[2][1]],
                    color=BISECT_COLOR, linewidth=ink["lw"], linestyle=DUAL_LS,
                    zorder=4.5)
        # Primal divider va-vb: still a white cell edge at the cell-edge
        # weight, only cased in purple -- it joins the highlight without
        # changing what it is.
        ax.plot([kite[1][0], kite[3][0]], [kite[1][1], kite[3][1]],
                color="white", linewidth=ink["lw"], linestyle=DUAL_LS,
                zorder=4.5, path_effects=[pe.withStroke(
                    linewidth=ink["lw"] * (1.0 + KITE_DIV_EDGE),
                    foreground=KITE_COLOR)])
    # Primal vertices (polytope corners = network nodes), taken off the drawn
    # simplices so the wedges' outer corners are dotted too, and deduped across
    # shared edges so overlapping markers do not stack.
    pverts = np.unique(np.round(
        np.array([p[1] for p in pieces] + [p[2] for p in pieces]), 6), axis=0)
    ax.scatter(pverts[:, 0], pverts[:, 1], s=PDOT,
               facecolors=ink["pdot_face"], edgecolors=ink["pdot_rim"],
               linewidths=ink["pdot_rim_lw"], zorder=5)
    centers = np.array(list(cof.values()))
    ax.scatter(centers[:, 0], centers[:, 1], s=DDOT,
               facecolors=ink["dot_face"], edgecolors=ink["dot_rim"],
               linewidths=ink["dot_rim_lw"], zorder=6)

    # Fixed frame around the cluster (so every panel is the same pixel size):
    # the cluster fills 1/(1+PAD) of the view, leaving a whitespace ring where
    # the big (a)(b)(c) labels sit at a constant radius.
    # Wedges reach past the core polygons on every side, so the bbox is taken
    # over all three corners of every drawn simplex, not just the whole cells.
    cv = np.vstack(polys + [np.array([p[j] for p in pieces])
                            for j in (0, 1, 2)])
    lo, hi = cv.min(axis=0), cv.max(axis=0)
    cc = 0.5 * (lo + hi)
    extent = float((hi - lo).max())
    half = 0.5 * extent * (1.0 + PAD)
    ax.set_xlim(cc[0] - half, cc[0] + half)
    ax.set_ylim(cc[1] - half, cc[1] + half)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if labels:
        for lbl, tgt in (("a", ctr_tgt), ("b", edge_tgt), ("c", kite_tgt)):
            if tgt is None:
                continue
            d = tgt - cc
            n = np.linalg.norm(d)
            d = d / n if n > 1e-9 else np.array([0.0, 1.0])
            ax.annotate(f"({lbl})", xy=tgt, xytext=cc + d * 0.78 * half,
                        fontsize=40, ha="center", va="center", color="#111111",
                        zorder=7, annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color="#111111",
                                        lw=LW * 0.625))

    # PAD above still sets the window the labels are placed against, so their
    # tuned radial positions are untouched; this only pulls the frame in to
    # the ink afterwards.
    focus = cof[seed]
    if bleed:
        # Crop INTO the ink instead of fitting round it: the frame becomes a
        # window ON the tessellation, so the mesh runs off all four sides.
        # FILL_F shrinks the square window inside the content bbox; a roughly
        # round patch needs <= 1/sqrt(2) for its corners to stay covered.
        # Centred on the featured polygon, not on the content bbox: the bbox
        # centre drifts with whichever way the patch happens to sprawl, so the
        # three figures would each sit on a different point and stop reading
        # as the same neighbourhood.
        half = FILL_F * 0.5 * float((hi - lo).max())
        ax.set_xlim(focus[0] - half, focus[0] + half)
        ax.set_ylim(focus[1] - half, focus[1] + half)
        print(f"  bleed window: {2 * half:.3f} data units across, "
              f"centred on the featured polygon")
    elif trim:
        _shrink_to_ink(fig, ax, lo, hi, lw=ink["lw"])
        gx, gy = ax.get_xlim(), ax.get_ylim()
        if not (gx[0] <= lo[0] and hi[0] <= gx[1]
                and gy[0] <= lo[1] and hi[1] <= gy[1]):
            raise RuntimeError(f"trim cropped the mesh: bbox {lo}..{hi} "
                               f"outside window {gx} x {gy}")

    stem = (f"kitemesh_2d_{ntype}_{placement}_clean"
            + ("" if scope == "ring" else "_cluster")
            + ("" if labels else "_nolabels")
            + ("" if dual == "none" else f"_dual{dual}")
            + ("" if shade == "flat" else f"_{shade}")
            + ("_bleed" if bleed else ""))
    out = os.path.join(OUTDIR, stem + ".png")
    fig.savefig(out, dpi=DPI, facecolor="white")
    try:
        fig.savefig(out[:-4] + ".eps", format="eps")
    except Exception as exc:
        print(f"  [eps skip] {out[:-4]}.eps: {exc}")
    plt.close(fig)
    print(f"  saved {out}  ({len(pverts)} primal verts)")
    for src in (out, out[:-4] + ".eps"):
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(PAPER_DIR,
                                              os.path.basename(src)))


def main(argv=None):
    p = argparse.ArgumentParser(
        description="2D simplex-polytope kite mesh, Centroidal (C) only.")
    p.add_argument("--ntype", default="C",
                   choices=("D", "V", "G", "C", "all"),
                   help="tessellation, or 'all' for D/V/G/C (default: C)")
    p.add_argument("--placement", choices=("bary", "circ"), default="bary",
                   help="dual-vertex placement (default: bary = true area COM)")
    p.add_argument("--cells", type=int, default=CELLS,
                   help=f"cells in the cluster (default: {CELLS})")
    p.add_argument("--labels", dest="labels", action="store_true",
                   help="draw the (a)/(b)/(c) leader labels (default)")
    p.add_argument("--no-labels", dest="labels", action="store_false",
                   help="omit them; output is tagged _nolabels")
    p.set_defaults(labels=True)
    p.add_argument("--trim", dest="trim", action="store_true",
                   help="crop the dead border so the ink reaches the edges "
                        "(default)")
    p.add_argument("--no-trim", dest="trim", action="store_false",
                   help="keep the parent's fixed square frame")
    p.set_defaults(trim=True)
    p.add_argument("--dual", choices=("none", "mid", "straight"),
                   default="none",
                   help="overlay the dual bonds: 'straight' joins the incident "
                        "polygon centroids directly, 'mid' routes each through "
                        "the primal edge barycentre (default: none)")
    p.add_argument("--shade", choices=("flat", "wells", "mass", "simplex"),
                   default="flat",
                   help="ground fill: 'wells' ramps each cell from its primal "
                        "boundary down to its dual vertex, 'mass' tints each "
                        "simplex by area, 'simplex' gives every simplex its "
                        "own ramp to the centroid scaled by its area share "
                        "(default: flat)")
    p.add_argument("--bleed", action="store_true",
                   help="crop into the ink so the network reaches every edge "
                        "of the image instead of being framed by it")
    p.add_argument("--scope", choices=("ring", "cluster"), default="ring",
                   help="'ring' = one central polygon + the cycle of "
                        "edge-neighbours, cut at the kites beyond; 'cluster' = "
                        "the grown --cells blob (default: ring)")
    p.add_argument("--close-cells", action="store_true",
                   help="ring scope: draw each neighbour polygon whole (white "
                        "boundary + full dotted fan) instead of just its kite "
                        "wedge; the dual bonds are NOT extended. No filename "
                        "tag.")
    args = p.parse_args(argv)

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(PAPER_DIR, exist_ok=True)
    pts = poisson_2d(N, L, seed=SEED)
    print(f"=== 2D simplex-polytope kite mesh: seed={SEED} N={N} "
          f"L={L:g} cells={args.cells} placement={args.placement} "
          f"labels={args.labels} trim={args.trim} dual={args.dual} "
          f"shade={args.shade} scope={args.scope} "
          f"-> {OUTDIR} (+ {PAPER_DIR}) ===")
    types = ("D", "V", "G", "C") if args.ntype == "all" else (args.ntype,)
    for nt in types:
        print(f"--- {nt} ---")
        try:
            panel(nt, args.placement, pts, TESS_COLORS[nt], args.cells,
                  args.labels, args.trim, args.dual, args.shade, args.scope,
                  args.bleed, args.close_cells)
        except Exception as exc:
            print(f"  [skip] {nt}: {exc}")
    print(f"=== done -> {OUTDIR} ===")


if __name__ == "__main__":
    main()
