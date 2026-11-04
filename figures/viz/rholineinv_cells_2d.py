"""Vertex-Voronoi weight cells (``rho_line_inv``), 2D.

Companion to ``kite_simplex_polytope_2d_C.py`` and ``midptsec_cells_2d.py``:
same seed, same box, same ink, so all three sit together in the paper.

``rho_line_inv`` weights a bond by ``w_e = (A_L/L*_L + A_R/L*_R)/2`` -- for each
of the bond's two END VERTICES, the area of that vertex's Voronoi cell divided
by its star length (the total length of the edges meeting there). So the figure
is the two vertex cells the bond runs between, with every edge of both stars
protruding out of the cells and cut there: the cells give the ``A``, the
protruding stars give the ``L*``.

The vertex cells are exactly what ``build_network(..., c_area_mode='sec_vor')``
stores as ``cell_areas``, and the stars are ``cell_star_lengths``; both are
read back here as a check rather than recomputed.

Run:
    .venv-voro/Scripts/python.exe rholineinv_cells_2d.py
Knobs (env): RLI_PROT (stub length / sqrt(cell area)), plus every KMESH_* knob
from the kite script.
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
from kite_simplex_polytope_2d_C import (
    DDOT, DPI, DUAL_LS, FIGIN, INK_DARK, KITE_COLOR, L, N, PAPER_DIR, PDOT,
    SEED, WIN, _area_com, _ink, _shrink_to_ink)
from kite_simplex_polytope_2d_C import BISECT_COLOR
from midptsec_cells_2d import (RAD, _cell_polys, _chain, _poly_area,
                               _ring_ground, _walls)

PROT = float(os.environ.get("RLI_PROT", "0.55"))   # stub / sqrt(cell area)
# --bleed: window / content bbox; see midptsec_cells_2d for the measure.
FILL_F = float(os.environ.get("RLI_FILL_F", "0.308"))
OUTDIR = os.environ.get("VIZ_OUT", os.path.join(_HERE, "outputs", "rholineinv"))
PAPER_DIR = os.path.join(OUTDIR, "paper")


def _star(net, nodes_xy, node, at, stub, skip=None):
    """The edges leaving ``node``, cut to a stub -- the ``L*`` of the weight.

    Each far end is min-imaged against the node's own drawn position, so a
    star that straddles the periodic seam still radiates correctly. ``skip``
    drops the bond shared with the other focal vertex: it belongs to both
    stars, so drawing it here would stub it twice and strand a tip dot in the
    middle of the bond that is already drawn whole as the highlight. Its
    length still counts toward the returned ``L*``.
    """
    out, full = [], 0.0
    for f in np.where((net.edges_left_cell == node)
                      | (net.edges_right_cell == node))[0]:
        other = (net.edges_right_cell[f] if net.edges_left_cell[f] == node
                 else net.edges_left_cell[f])
        d = nodes_xy[other] - nodes_xy[node]
        d -= L * np.round(d / L)
        h = float(np.linalg.norm(d))
        if h < 1e-12:
            continue
        full += h
        if other == skip:
            continue
        out.append([at, at + d / h * min(h, stub)])
    return out, full


def panel(ntype, pts, color, trim=True, bleed=False):
    # sec_vor is what makes cell_areas the VERTEX Voronoi areas the weight
    # actually uses; the kite figure's 'triangle' mode would be the wrong A.
    net = build_network(pts, L, ntype, c_area_mode="sec_vor")         if ntype == "C" else build_network(pts, L, ntype)
    center = np.array([L / 2.0, L / 2.0])
    P = np.asarray(net.points, dtype=float)
    Pl = P - L * np.round((P - center) / L)          # one frame, box-centred

    gw = max(WIN * L, RAD + 1.0)
    keep = np.where((np.abs(Pl - center) <= gw).all(axis=1))[0]
    if keep.size < 8:
        print(f"  [skip] {ntype}: only {keep.size} vertices in the window")
        return
    vor = Voronoi(Pl[keep])

    cands, seed, ground = _ring_ground(net, center, RAD)
    Pc = cands[seed]
    # One dual chain: the vertex cell of each corner of the central polygon.
    chain = _chain(Pl[keep], Pc)
    show = set(chain)
    polys = _cell_polys(vor, chain)
    # w_e averages A/L* over BOTH ends of a bond, so both ends get made
    # explicit: one edge of the central polygon, its two corners' cells in
    # purple and both their stars in red. Showing one end only would draw half
    # the formula.
    nP = len(Pc)
    cand = [(k, (k + 1) % nP) for k in range(nP)
            if chain[k] in polys and chain[(k + 1) % nP] in polys]
    ka, kb = min(cand, key=lambda kk: np.linalg.norm(
        0.5 * (Pc[kk[0]] + Pc[kk[1]]) - center))
    homes = [chain[ka], chain[kb]]
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
        ax.add_patch(MplPolygon(polys[h], closed=True, fill=False,
                                edgecolor=KITE_COLOR, linewidth=ink["lw"],
                                zorder=4))
    # The L* of the ratio: every primal edge meeting that corner, in red, so
    # the two ingredients of A/L* are both explicit on the one highlighted
    # cell -- the purple outline is the A, the red star is the L*.
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
            print(f"  [warn] star at {node} drawn {full:.5f} vs stored "
                  f"L* {S[node]:.5f}")
    w = 0.5 * sum(A[n] / S[n] for n in nodes)
    for node in nodes:
        print(f"  vertex {node}: A = {A[node]:.4f}, L* = {S[node]:.4f}"
              f"  ->  A/L* = {A[node] / S[node]:.5f}")
    print(f"  w_e = mean of the two = {w:.5f}")

    pv = np.unique(np.round(np.vstack(ground), 6), axis=0)
    ax.scatter(pv[:, 0], pv[:, 1], s=PDOT, facecolors=INK_DARK,
               edgecolors="none", zorder=5)

    # Include the ground: whole primal cells in frame, not clipped ones.
    cv = np.vstack([polys[i] for i in polys] + [Pc]
                   + [np.asarray(q) for q in star] + list(ground))
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

    stem = (f"rholineinv_2d_{ntype}_clean_nolabels"
            + ("_bleed" if bleed else ""))
    out = os.path.join(OUTDIR, stem + ".png")
    fig.savefig(out, dpi=DPI, facecolor="white")
    try:
        fig.savefig(out[:-4] + ".eps", format="eps")
    except Exception as exc:
        print(f"  [eps skip] {out[:-4]}.eps: {exc}")
    plt.close(fig)
    print(f"  saved {out}  (central {len(Pc)}-gon + {len(polys)}-cell chain)")
    for src in (out, out[:-4] + ".eps"):
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(PAPER_DIR,
                                              os.path.basename(src)))


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Vertex-Voronoi weight cells (rho_line_inv), 2D.")
    p.add_argument("--ntype", default="C",
                   choices=("D", "V", "G", "C", "all"))
    p.add_argument("--no-trim", dest="trim", action="store_false")
    p.add_argument("--bleed", action="store_true",
                   help="crop into the ink so the network reaches every edge "
                        "of the image instead of being framed by it")
    p.set_defaults(trim=True)
    args = p.parse_args(argv)

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(PAPER_DIR, exist_ok=True)
    pts = poisson_2d(N, L, seed=SEED)
    print(f"=== rho_line_inv cells: seed={SEED} N={N} L={L:g} "
          f"-> {OUTDIR} (+ {PAPER_DIR}) ===")
    types = ("D", "V", "G", "C") if args.ntype == "all" else (args.ntype,)
    for nt in types:
        print(f"--- {nt} ---")
        try:
            panel(nt, pts, TESS_COLORS[nt], args.trim, args.bleed)
        except Exception as exc:
            print(f"  [skip] {nt}: {exc}")


if __name__ == "__main__":
    main()
