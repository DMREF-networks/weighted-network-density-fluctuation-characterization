"""Midpoint-Voronoi weight cells (``midpt_sec_area``), 2D.

Companion to ``kite_simplex_polytope_2d_C.py``: same seed, same box, same ink,
so the two figures sit side by side in the paper. Where that one shows the kite
a bond gets its weight from, this shows the cell -- ``midpt_sec_area`` weights
each primal edge by the area of its MIDPOINT's cell in the secondary Voronoi
built on every edge midpoint of the network (``wndfc.network
.midpt_secondary_voronoi``).

The figure is four neighbouring midpoint cells with each cell's own primal edge
drawn through it, protruding a stub past the boundary before being cut -- the
stub is what says the edge continues while the cell does not.

All styling is imported from the kite script rather than restated, so a tweak
there follows here.

Run:
    .venv-voro/Scripts/python.exe midptsec_cells_2d.py
    .venv-voro/Scripts/python.exe midptsec_cells_2d.py --cells 6
Knobs (env): MSEC_CELLS (4), MSEC_PROT (stub length / sqrt(cell area)), plus
every KMESH_* knob from the kite script.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import Voronoi

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))          # release root (wndfc/)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from wndfc.generators import poisson_2d
from wndfc.network import build_network
from anatomy_kite_2d import TESS_COLORS, _shade
# One source of truth for the look -- see module docstring.
from kite_simplex_polytope_2d_C import (
    DDOT, DPI, DUAL_LS, FIGIN, INK_DARK, KITE_COLOR, L, N, PAPER_DIR, PDOT,
    SEED, WIN, _area_com, _ink, _shrink_to_ink)
from kite_simplex_polytope_2d import _edge_map, _gather_polys
from kite_simplex_polytope_2d_C import _edge_key

# Half-width of the drawn patch. 1.7 reproduces the bary schematic's region:
# that figure spans 3.07 x 3.14 data units about the box centre.
RAD = float(os.environ.get("MSEC_RAD", "3.25"))
NCELLS = int(os.environ.get("MSEC_CELLS", "1"))     # cells highlighted
# --bleed: window / content bbox. The face patch is raggeder than the
# kite ring, so it has to crop in further -- measured <=3% edge white.
FILL_F = float(os.environ.get("MSEC_FILL_F", "0.308"))
OUTDIR = os.environ.get("VIZ_OUT", os.path.join(_HERE, "outputs", "midptsec"))
PAPER_DIR = os.path.join(OUTDIR, "paper")


def _edges_local(net, center):
    """Every network edge as ``(v1, v2, midpoint)`` in one frame anchored on
    ``center``.

    ``edges_v1``/``edges_v2`` are already min-imaged against each other, so a
    single whole-box shift per edge brings the pair into the window without
    breaking either one.
    """
    v1 = np.asarray(net.edges_v1, dtype=float)
    v2 = np.asarray(net.edges_v2, dtype=float)
    mid = np.asarray(net.edges_midpoints, dtype=float)
    shift = -L * np.round((mid - center) / L)
    return v1 + shift, v2 + shift, mid + shift


def _ring_ground(net, center, rad):
    """The central primal polygon plus the faces sharing an edge with it.

    That is exactly the ground the chain of secondary cells wrapping the
    centre needs: each of those cells straddles one of the centre's edges, so
    it reaches into the neighbour on the far side and would otherwise sit half
    on bare page.
    """
    cands = _gather_polys(net, center, rad)
    cent = [P.mean(axis=0) for P in cands]
    seed = int(np.argmin([np.linalg.norm(c - center) for c in cent]))
    emap = _edge_map(cands, range(len(cands)))
    P = cands[seed]
    nbr = set()
    for k in range(len(P)):
        for t in emap.get(_edge_key(P[k], P[(k + 1) % len(P)]), []):
            if t[0] != seed:
                nbr.add(int(t[0]))
    return cands, seed, [cands[i] for i in [seed] + sorted(nbr)]


def _chain(gens, targets):
    """Index in ``gens`` nearest each target point -- how the chain's cells are
    picked out of the wider Voronoi sample."""
    out = []
    for t in targets:
        out.append(int(np.argmin(np.linalg.norm(gens - t, axis=1))))
    return out


def _ground(net, center, rad):
    """The primal faces of the patch -- the primary embedding, whose edges are
    the white ones. Whole faces, never clipped, so the figure ends on real
    cell boundaries the way the bary schematic does."""
    return [P for P in _gather_polys(net, center, rad)
            if np.linalg.norm(P.mean(axis=0) - center) <= rad]


def _walls(vor, show):
    """Finite Voronoi ridges with at least one generator in ``show``.

    Taken as ridges rather than as cell outlines so a shared wall is drawn
    once; doubling it would thicken every interior wall against the rim.
    """
    seg = []
    for (a, b), (p, q) in zip(vor.ridge_points, vor.ridge_vertices):
        if p < 0 or q < 0:
            continue
        if a in show or b in show:
            seg.append([vor.vertices[p], vor.vertices[q]])
    return seg


def _cell_polys(vor, idx):
    """Finite Voronoi cell polygons for the requested generator indices."""
    out = {}
    for i in idx:
        reg = vor.regions[vor.point_region[i]]
        if not reg or -1 in reg:
            continue
        out[i] = np.asarray([vor.vertices[k] for k in reg], dtype=float)
    return out


def _exit_t(poly, p, u):
    """Smallest t > 0 at which the ray ``p + t*u`` leaves ``poly``."""
    best = None
    n = len(poly)
    for k in range(n):
        a, b = poly[k], poly[(k + 1) % n]
        d = b - a
        den = u[0] * d[1] - u[1] * d[0]
        if abs(den) < 1e-14:
            continue
        w = a - p
        t = (w[0] * d[1] - w[1] * d[0]) / den
        s = (u[0] * w[1] - u[1] * w[0]) / -den
        if t > 1e-9 and -1e-9 <= s <= 1.0 + 1e-9:
            if best is None or t < best:
                best = t
    return best


def _continuations(net, v1, v2, node, at, stub):
    """The other edges leaving ``node``, cut to a stub.

    A midpoint cell reaches almost exactly to its edge's own endpoints -- the
    neighbouring midpoints sit half a length the other side of the shared
    vertex, so the bisector lands on the vertex itself (measured protrusions
    here run +0.14 to -0.01 against half-edges of ~0.43). So the edge cannot
    protrude out of its own cell; what protrudes is the network CARRYING ON
    through the vertex, which is what these stubs draw.
    """
    out = []
    for f in np.where((net.edges_left_cell == node)
                      | (net.edges_right_cell == node))[0]:
        near, far = ((v1[f], v2[f]) if net.edges_left_cell[f] == node
                     else (v2[f], v1[f]))
        d = far - near
        h = float(np.linalg.norm(d))
        if h < 1e-12:
            continue
        out.append([at, at + d / h * min(h, stub)])
    return out


def panel(ntype, pts, color, ncells, trim=True, bleed=False):
    net = (build_network(pts, L, "C", c_area_mode="triangle") if ntype == "C"
           else build_network(pts, L, ntype))
    center = np.array([L / 2.0, L / 2.0])
    v1, v2, mid = _edges_local(net, center)

    # Voronoi over a wider set than is drawn, so every shown cell is exact
    # rather than shaped by the edge of the sample.
    gw = max(WIN * L, RAD + 1.0)
    keep = np.where((np.abs(mid - center) <= gw).all(axis=1))[0]
    if keep.size < 12:
        print(f"  [skip] {ntype}: only {keep.size} midpoints in the window")
        return
    vor = Voronoi(mid[keep])

    cands, seed, ground = _ring_ground(net, center, RAD)
    Pc = cands[seed]
    n = len(Pc)
    # One dual chain: the midpoint cell of each edge of the central polygon.
    targets = np.array([0.5 * (Pc[k] + Pc[(k + 1) % n]) for k in range(n)])
    chain = _chain(mid[keep], targets)
    show = set(chain)
    home = chain[int(np.argmin(np.linalg.norm(targets - center, axis=1)))]
    polys = _cell_polys(vor, chain)

    ink = _ink("flat")
    fig, ax = plt.subplots(figsize=(FIGIN, FIGIN))
    ax.add_collection(PatchCollection(
        [MplPolygon(Q, closed=True) for Q in ground],
        facecolor=_shade(color, 0.0), edgecolor="white",
        linewidths=ink["lw"], zorder=2))
    ax.add_collection(LineCollection(_walls(vor, show), colors="black",
                                     linewidths=ink["lw"],
                                     linestyles=[DUAL_LS], zorder=3))
    if home in polys:
        ax.add_patch(MplPolygon(polys[home], closed=True, fill=False,
                                edgecolor=KITE_COLOR, linewidth=ink["lw"],
                                zorder=4))
        g = keep[home]
        ax.plot([v1[g][0], v2[g][0]], [v1[g][1], v2[g][1]],
                color=KITE_COLOR, linewidth=ink["lw"], linestyle=DUAL_LS,
                zorder=4.5)

    pv = np.unique(np.round(np.vstack(ground), 6), axis=0)
    ax.scatter(pv[:, 0], pv[:, 1], s=PDOT, facecolors=INK_DARK,
               edgecolors="none", zorder=5)
    gen = mid[keep[chain]]
    ax.scatter(gen[:, 0], gen[:, 1], s=DDOT, facecolors="white",
               edgecolors=INK_DARK, linewidths=ink["dot_rim_lw"], zorder=6)

    # Include the ground: the frame then contains whole primal cells rather
    # than clipping them, so the centroid cells the chain sits on are
    # readable as cells instead of as fragments.
    cv = np.vstack([polys[i] for i in polys] + [gen, Pc] + list(ground))
    lo, hi = cv.min(axis=0), cv.max(axis=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    focus = _area_com(Pc)
    if bleed:
        half = FILL_F * 0.5 * float((hi - lo).max())
        ax.set_xlim(focus[0] - half, focus[0] + half)
        ax.set_ylim(focus[1] - half, focus[1] + half)
        print(f"  bleed window: {2 * half:.3f} data units across")
    elif trim:
        _shrink_to_ink(fig, ax, lo, hi, lw=ink["lw"])

    stem = (f"midptsec_2d_{ntype}_clean_nolabels"
            + ("_bleed" if bleed else ""))
    out = os.path.join(OUTDIR, stem + ".png")
    fig.savefig(out, dpi=DPI, facecolor="white")
    try:
        fig.savefig(out[:-4] + ".eps", format="eps")
    except Exception as exc:
        print(f"  [eps skip] {out[:-4]}.eps: {exc}")
    plt.close(fig)
    print(f"  saved {out}  (central {n}-gon + {len(polys)}-cell dual chain)")
    for src in (out, out[:-4] + ".eps"):
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(PAPER_DIR,
                                              os.path.basename(src)))


def _poly_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Midpoint-Voronoi weight cells (midpt_sec_area), 2D.")
    p.add_argument("--ntype", default="C",
                   choices=("D", "V", "G", "C", "all"))
    p.add_argument("--cells", type=int, default=NCELLS)
    p.add_argument("--no-trim", dest="trim", action="store_false")
    p.add_argument("--bleed", action="store_true",
                   help="crop into the ink so the network reaches every edge "
                        "of the image instead of being framed by it")
    p.set_defaults(trim=True)
    args = p.parse_args(argv)

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(PAPER_DIR, exist_ok=True)
    pts = poisson_2d(N, L, seed=SEED)
    print(f"=== midpt_sec cells: seed={SEED} N={N} L={L:g} "
          f"cells={args.cells} -> {OUTDIR} (+ {PAPER_DIR}) ===")
    types = ("D", "V", "G", "C") if args.ntype == "all" else (args.ntype,)
    for nt in types:
        print(f"--- {nt} ---")
        try:
            panel(nt, pts, TESS_COLORS[nt], args.cells, args.trim,
                  args.bleed)
        except Exception as exc:
            print(f"  [skip] {nt}: {exc}")


if __name__ == "__main__":
    main()
