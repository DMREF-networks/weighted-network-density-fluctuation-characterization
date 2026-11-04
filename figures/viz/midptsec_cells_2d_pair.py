"""Midpoint-Voronoi weight cells, TWO adjacent primal cells (thinking aid).

Sibling of ``midptsec_cells_2d.py`` (Fig. 7, T scheme), which draws the central
polygon, its edge-neighbours as ground, and the chain of midpoint cells around
the central polygon's edges.  This one features a PAIR: the central polygon and
the neighbour across its most central edge -- the edge whose midpoint cell the
parent highlights -- with the ground grown to both polygons' neighbours and the
chain grown to the midpoint cells of every edge of both.  The highlighted cell
is the one on the shared edge, so it sits in both chains.

All styling is imported from the parent scripts; nothing there is modified.

Run (from the release root):
    python figures/viz/midptsec_cells_2d_pair.py
Knobs: MSEC_* and KMESH_* env knobs of the parents; ``--second`` as in the kite
pair script (``kite`` = across the most central edge, ``far``, or an integer
rank by edge-midpoint distance).
Outputs: <VIZ_OUT>/midptsec_pair_2d_<ntype>_clean_nolabels[...].png (+ .eps);
VIZ_OUT defaults to figures/viz/outputs/midptsec_pair.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import Voronoi

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from wndfc.generators import poisson_2d
from wndfc.network import build_network
from anatomy_kite_2d import TESS_COLORS, _shade
from kite_simplex_polytope_2d_C import (DDOT, DPI, DUAL_LS, FIGIN, INK_DARK, KITE_COLOR, L, N,
                                        PDOT, SEED, WIN, _area_com, _edge_key, _ink,
                                        _shrink_to_ink)
from kite_simplex_polytope_2d import _edge_map, _gather_polys
from midptsec_cells_2d import (FILL_F, RAD, _cell_polys, _chain, _edges_local, _walls)

OUTDIR = os.environ.get("VIZ_OUT", os.path.join(_HERE, "outputs", "midptsec_pair"))


def _neighbours(emap, cands, i):
    """Faces sharing an edge with face ``i``, keyed by the shared edge."""
    P = cands[i]
    out = {}
    for k in range(len(P)):
        ek = _edge_key(P[k], P[(k + 1) % len(P)])
        for t in emap.get(ek, []):
            if t[0] != i:
                out[ek] = int(t[0])
    return out


def pair_ground(net, center, rad, second="kite"):
    """The central polygon, its partner across one edge, and the union of both
    polygons' edge-neighbours as ground.  Returns
    ``(cands, seeds, ground, shared_edge)``."""
    cands = _gather_polys(net, center, rad)
    cent = [P.mean(axis=0) for P in cands]
    seed = int(np.argmin([np.linalg.norm(c - center) for c in cent]))
    emap = _edge_map(cands, range(len(cands)))
    nb0 = _neighbours(emap, cands, seed)
    ranked = sorted(nb0.items(), key=lambda kv: np.linalg.norm(
        0.5 * (np.array(kv[0][0]) + np.array(kv[0][1])) - center))
    if not ranked:
        raise RuntimeError("central polygon has no gathered neighbour")
    if second == "far":
        ek, partner = ranked[-1]
    else:
        k = 0 if second == "kite" else int(second)
        ek, partner = ranked[min(k, len(ranked) - 1)]
    seeds = [seed, partner]
    nbr = set(nb0.values()) | set(_neighbours(emap, cands, partner).values())
    nbr -= set(seeds)
    ground = [cands[i] for i in seeds + sorted(nbr)]
    return cands, seeds, ground, (np.array(ek[0]), np.array(ek[1]))


def panel(ntype, pts, color, trim=True, bleed=False, second="kite"):
    net = (build_network(pts, L, "C", c_area_mode="triangle") if ntype == "C"
           else build_network(pts, L, ntype))
    center = np.array([L / 2.0, L / 2.0])
    v1, v2, mid = _edges_local(net, center)

    gw = max(WIN * L, RAD + 1.0)
    keep = np.where((np.abs(mid - center) <= gw).all(axis=1))[0]
    if keep.size < 12:
        print(f"  [skip] {ntype}: only {keep.size} midpoints in the window")
        return
    vor = Voronoi(mid[keep])

    cands, seeds, ground, shared = pair_ground(net, center, RAD, second)
    # Two dual chains merged: the midpoint cell of every edge of BOTH polygons;
    # the shared edge's cell appears once.
    targets, seen = [], set()
    for s in seeds:
        P = cands[s]
        for k in range(len(P)):
            a, b = P[k], P[(k + 1) % len(P)]
            ek = _edge_key(a, b)
            if ek in seen:
                continue
            seen.add(ek)
            targets.append(0.5 * (a + b))
    targets = np.array(targets)
    chain = _chain(mid[keep], targets)
    show = set(chain)
    home = _chain(mid[keep], [0.5 * (shared[0] + shared[1])])[0]   # the shared edge's cell
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
                                edgecolor=KITE_COLOR, linewidth=ink["lw"], zorder=4))
        g = keep[home]
        ax.plot([v1[g][0], v2[g][0]], [v1[g][1], v2[g][1]],
                color=KITE_COLOR, linewidth=ink["lw"], linestyle=DUAL_LS, zorder=4.5)

    pv = np.unique(np.round(np.vstack(ground), 6), axis=0)
    ax.scatter(pv[:, 0], pv[:, 1], s=PDOT, facecolors=INK_DARK, edgecolors="none", zorder=5)
    gen = mid[keep[chain]]
    ax.scatter(gen[:, 0], gen[:, 1], s=DDOT, facecolors="white",
               edgecolors=INK_DARK, linewidths=ink["dot_rim_lw"], zorder=6)

    cv = np.vstack([polys[i] for i in polys] + [gen] + [cands[s] for s in seeds] + list(ground))
    lo, hi = cv.min(axis=0), cv.max(axis=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    focus = 0.5 * (_area_com(cands[seeds[0]]) + _area_com(cands[seeds[1]]))
    if bleed:
        half = FILL_F * 0.5 * float((hi - lo).max())
        ax.set_xlim(focus[0] - half, focus[0] + half)
        ax.set_ylim(focus[1] - half, focus[1] + half)
        print(f"  bleed window: {2 * half:.3f} data units across")
    elif trim:
        _shrink_to_ink(fig, ax, lo, hi, lw=ink["lw"])

    stem = (f"midptsec_pair_2d_{ntype}_clean_nolabels"
            + ("" if second == "kite" else f"_second{second}")
            + ("_bleed" if bleed else ""))
    out = os.path.join(OUTDIR, stem + ".png")
    fig.savefig(out, dpi=DPI, facecolor="white")
    try:
        fig.savefig(out[:-4] + ".eps", format="eps")
    except Exception as exc:
        print(f"  [eps skip] {out[:-4]}.eps: {exc}")
    plt.close(fig)
    print(f"  saved {out}  (pair {len(cands[seeds[0]])}-gon + {len(cands[seeds[1]])}-gon, "
          f"{len(polys)}-cell dual chain, {len(ground)} ground polygons)")


def main(argv=None):
    p = argparse.ArgumentParser(description="Midpoint-Voronoi weight cells, two adjacent primal cells.")
    p.add_argument("--ntype", default="C", choices=("D", "V", "G", "C", "all"))
    p.add_argument("--no-trim", dest="trim", action="store_false")
    p.add_argument("--bleed", action="store_true")
    p.add_argument("--second", default="kite",
                   help="partner cell: 'kite' (across the most central edge), 'far', or an integer rank")
    p.set_defaults(trim=True)
    args = p.parse_args(argv)

    os.makedirs(OUTDIR, exist_ok=True)
    pts = poisson_2d(N, L, seed=SEED)
    print(f"=== midpt_sec cells, PAIR: seed={SEED} N={N} L={L:g} second={args.second} -> {OUTDIR} ===")
    types = ("D", "V", "G", "C") if args.ntype == "all" else (args.ntype,)
    for nt in types:
        print(f"--- {nt} ---")
        try:
            panel(nt, pts, TESS_COLORS[nt], args.trim, args.bleed, args.second)
        except Exception as exc:
            print(f"  [skip] {nt}: {exc}")


if __name__ == "__main__":
    main()
