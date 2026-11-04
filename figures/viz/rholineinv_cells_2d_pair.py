"""Vertex-Voronoi weight cells (``rho_line_inv``), TWO adjacent primal cells
(thinking aid).

Sibling of ``rholineinv_cells_2d.py`` (Fig. 7, V scheme), which draws the central
polygon, its edge-neighbours as ground, the chain of vertex cells at the central
polygon's corners, and highlights the two corners of the most central edge with
their stars.  This one features a PAIR: the central polygon and the neighbour
across that same edge, with the ground grown to both polygons' neighbours and
the chain grown to the vertex cells of every corner of both.  The highlighted
corners are the shared edge's two ends, so they sit in both chains.

All styling is imported from the parent scripts; nothing there is modified.

Run (from the release root):
    python figures/viz/rholineinv_cells_2d_pair.py
Knobs: RLI_* / MSEC_* / KMESH_* env knobs of the parents; ``--second`` as in the
kite pair script.
Outputs: <VIZ_OUT>/rholineinv_pair_2d_<ntype>_clean_nolabels[...].png (+ .eps);
VIZ_OUT defaults to figures/viz/outputs/rholineinv_pair.
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
from kite_simplex_polytope_2d_C import (BISECT_COLOR, DPI, DUAL_LS, FIGIN, INK_DARK, KITE_COLOR,
                                        L, N, PDOT, SEED, WIN, _area_com, _ink, _shrink_to_ink)
from midptsec_cells_2d import RAD, _cell_polys, _chain, _walls
from rholineinv_cells_2d import FILL_F
from midptsec_cells_2d_pair import pair_ground

OUTDIR = os.environ.get("VIZ_OUT", os.path.join(_HERE, "outputs", "rholineinv_pair"))


def panel(ntype, pts, color, trim=True, bleed=False, second="kite"):
    net = (build_network(pts, L, ntype, c_area_mode="sec_vor") if ntype == "C"
           else build_network(pts, L, ntype))
    center = np.array([L / 2.0, L / 2.0])
    P = np.asarray(net.points, dtype=float)
    Pl = P - L * np.round((P - center) / L)

    gw = max(WIN * L, RAD + 1.0)
    keep = np.where((np.abs(Pl - center) <= gw).all(axis=1))[0]
    if keep.size < 8:
        print(f"  [skip] {ntype}: only {keep.size} vertices in the window")
        return
    vor = Voronoi(Pl[keep])

    cands, seeds, ground, shared = pair_ground(net, center, RAD, second)
    # Two dual chains merged: the vertex cell of every corner of BOTH polygons.
    corners = np.unique(np.round(np.vstack([cands[s] for s in seeds]), 6), axis=0)
    chain = _chain(Pl[keep], corners)
    show = set(chain)
    polys = _cell_polys(vor, chain)
    # The highlighted bond is the shared edge; both its ends get the A/L*
    # treatment, as in the parent.
    homes = _chain(Pl[keep], [shared[0], shared[1]])
    nodes = [int(keep[h]) for h in homes]

    ink = _ink("flat")
    fig, ax = plt.subplots(figsize=(FIGIN, FIGIN))
    ax.add_collection(PatchCollection(
        [MplPolygon(Q, closed=True) for Q in ground],
        facecolor=_shade(color, 0.0), edgecolor="white",
        linewidths=ink["lw"], zorder=2))
    ax.add_collection(LineCollection(_walls(vor, show), colors="black",
                                     linewidths=ink["lw"],
                                     linestyles=[DUAL_LS], zorder=3))
    for h in homes:
        if h in polys:
            ax.add_patch(MplPolygon(polys[h], closed=True, fill=False,
                                    edgecolor=KITE_COLOR, linewidth=ink["lw"], zorder=4))
    star, fulls = [], []
    for node in nodes:
        full = 0.0
        for f in np.where((net.edges_left_cell == node)
                          | (net.edges_right_cell == node))[0]:
            other = (net.edges_right_cell[f] if net.edges_left_cell[f] == node
                     else net.edges_left_cell[f])
            d = Pl[other] - Pl[node]
            d -= L * np.round(d / L)
            full += float(np.linalg.norm(d))
            star.append([Pl[node], Pl[node] + d])
        fulls.append(full)
    ax.add_collection(LineCollection(star, colors=BISECT_COLOR,
                                     linewidths=ink["lw"],
                                     linestyles=[DUAL_LS], zorder=4.5))

    A = np.asarray(net.cell_areas, dtype=float)
    S = np.asarray(net.cell_star_lengths, dtype=float)
    for node, full in zip(nodes, fulls):
        if abs(full - S[node]) > 1e-6:
            print(f"  [warn] star at {node} drawn {full:.5f} vs stored L* {S[node]:.5f}")
        print(f"  vertex {node}: A = {A[node]:.4f}, L* = {S[node]:.4f}  ->  A/L* = {A[node] / S[node]:.5f}")
    print(f"  w_e = mean of the two = {0.5 * sum(A[n] / S[n] for n in nodes):.5f}")

    pv = np.unique(np.round(np.vstack(ground), 6), axis=0)
    ax.scatter(pv[:, 0], pv[:, 1], s=PDOT, facecolors=INK_DARK, edgecolors="none", zorder=5)

    cv = np.vstack([polys[i] for i in polys] + [cands[s] for s in seeds]
                   + [np.asarray(q) for q in star] + list(ground))
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

    stem = (f"rholineinv_pair_2d_{ntype}_clean_nolabels"
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
          f"{len(polys)}-cell chain, {len(ground)} ground polygons)")


def main(argv=None):
    p = argparse.ArgumentParser(description="Vertex-Voronoi weight cells, two adjacent primal cells.")
    p.add_argument("--ntype", default="C", choices=("D", "V", "G", "C", "all"))
    p.add_argument("--no-trim", dest="trim", action="store_false")
    p.add_argument("--bleed", action="store_true")
    p.add_argument("--second", default="kite",
                   help="partner cell: 'kite' (across the most central edge), 'far', or an integer rank")
    p.set_defaults(trim=True)
    args = p.parse_args(argv)

    os.makedirs(OUTDIR, exist_ok=True)
    pts = poisson_2d(N, L, seed=SEED)
    print(f"=== rho_line_inv cells, PAIR: seed={SEED} N={N} L={L:g} second={args.second} -> {OUTDIR} ===")
    types = ("D", "V", "G", "C") if args.ntype == "all" else (args.ntype,)
    for nt in types:
        print(f"--- {nt} ---")
        try:
            panel(nt, pts, TESS_COLORS[nt], args.trim, args.bleed, args.second)
        except Exception as exc:
            print(f"  [skip] {nt}: {exc}")


if __name__ == "__main__":
    main()
