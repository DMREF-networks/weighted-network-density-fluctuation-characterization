"""2D kite mesh, TWO adjacent primal cells with their dual rings (thinking aid).

Sibling of ``kite_simplex_polytope_2d_C.py`` (Fig. 6), which draws one central
polygon and the ring of dual faces at its corners.  This one features a PAIR:
the central polygon plus the neighbour across its most central edge -- the
same edge the parent highlights as the kite -- and builds the ring of each,
so the two rings' overlap (the shared kite, the dual faces at the two shared
corners, the neighbours both cells touch) is visible in one frame.

Everything visual is imported from the parent: ink, dot sizes, colours, the
ring walk, the flat/gradient grounds, the trim/bleed framing.  Only the
selection of a second seed and the union of the two ring structures is new.
The parent is not modified.

Run (from the release root; the Fig. 6 invocation plus nothing):
    python figures/viz/kite_simplex_polytope_2d_C_pair.py --no-labels --dual mid --close-cells
Knobs: every KMESH_* env knob of the parent; ``--second`` picks the partner
(default ``kite`` = across the most central edge; ``far`` = across the edge
farthest from the box centre; an integer = that neighbour by rank of edge
midpoint distance, 0 = kite).
Outputs: <VIZ_OUT>/kitemesh_pair_2d_<ntype>_<placement>_clean[...].png (+ .eps);
VIZ_OUT defaults to figures/viz/outputs/kite_simplex_C_pair.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon as MplPolygon

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from wndfc.generators import poisson_2d
from wndfc.network import build_network
from anatomy_kite_2d import TESS_COLORS
from kite_simplex_polytope_2d import _edge_map, _gather_polys, _nearest
from kite_simplex_polytope_2d_C import (
    BISECT_COLOR, DDOT, DPI, DUAL_LS, FIGIN, FILL_F, KITE_COLOR, KITE_DIV_EDGE,
    L, LW, N, PAD, PDOT, SEED, SHOW_BISECT, WIN,
    _center, _ink, _paint, _pieces, _ring_duals, _ring_structure, _shrink_to_ink)

OUTDIR = os.environ.get("VIZ_OUT", os.path.join(_HERE, "outputs", "kite_simplex_C_pair"))


def _pick_second(bonds, seed, cof, center, how):
    """The partner cell: a face sharing an edge with ``seed``, chosen by the
    distance of the shared edge's midpoint from the box centre."""
    ranked = []
    for i, j, va, vb in bonds:
        if seed not in (i, j):
            continue
        far = j if i == seed else i
        ranked.append((float(np.linalg.norm(0.5 * (va + vb) - center)), far))
    if not ranked:
        return None
    ranked.sort()
    if how == "far":
        return ranked[-1][1]
    k = 0 if how == "kite" else int(how)
    return ranked[min(k, len(ranked) - 1)][1]


def _union_rings(emap, cands, seeds):
    """Ring structures of every seed, merged: deduped bonds, wedges of faces
    that are not themselves seeds, and the primal edges of bonds touching no
    seed (these are drawn white by hand, as in the parent)."""
    bonds, seen_b = [], set()
    wedges, seen_w = [], set()
    faces_all, missed = [], 0
    for s in seeds:
        faces, b_s, w_s, _outer, m = _ring_structure(emap, cands, s)
        faces_all.append(faces)
        missed += m
        for i, j, a, b in b_s:
            key = (min(i, j), max(i, j))
            if key in seen_b:
                continue
            seen_b.add(key)
            bonds.append((i, j, a, b))
    from kite_simplex_polytope_2d_C import _edge_key
    for i, j, a, b in bonds:
        ek = _edge_key(a, b)
        for f in (i, j):
            if f in seeds or (f, ek) in seen_w:
                continue
            seen_w.add((f, ek))
            wedges.append((f, a, b, i if f == j else j))
    outer = [(a, b) for i, j, a, b in bonds if i not in seeds and j not in seeds]
    return faces_all, bonds, wedges, outer, missed


def panel(ntype, placement, pts, color, labels, trim=True, dual="none",
          shade="flat", bleed=False, close_cells=False, second="kite"):
    net = (build_network(pts, L, "C", c_area_mode="triangle") if ntype == "C"
           else build_network(pts, L, ntype))
    center = np.array([L / 2.0, L / 2.0])
    cands = _gather_polys(net, center, WIN * L)
    if len(cands) < 2:
        print(f"  [skip] {ntype}/{placement}: too few faces")
        return
    cent = [p.mean(axis=0) for p in cands]
    seed = int(np.argmin([np.linalg.norm(c - center) for c in cent]))
    emap = _edge_map(cands, range(len(cands)))

    # Partner: across one of the centre's own edges (ring bonds touching it).
    _, b0, _, _, _ = _ring_structure(emap, cands, seed)
    cof0 = {seed: _center(cands[seed], placement)}
    partner = _pick_second(b0, seed, cof0, center, second)
    if partner is None:
        print(f"  [skip] {ntype}: no neighbour found for the pair")
        return
    seeds = [seed, partner]

    faces_all, bonds, fringe, outer_edges, missed = _union_rings(emap, cands, seeds)
    if missed:
        print(f"  [warn] {missed} dual face(s) ran off the gathered window -- "
              f"raise KMESH_WIN (now {WIN:g})")
    cluster = list(seeds)
    cof = {i: _center(cands[i], placement)
           for i in set(cluster) | {f[0] for f in fringe}
           | {b for i, j, _, _ in bonds for b in (i, j)}}
    if close_cells:
        ring = sorted({f[0] for f in fringe} - set(seeds))
        cluster = list(seeds) + ring
        fringe = []
        print(f"  close-cells: {len(ring)} neighbour polygon(s) drawn whole")
    polys = [cands[i] for i in cluster]
    pieces = _pieces(cluster, cands, cof, fringe)
    for s, faces in zip(seeds, faces_all):
        sizes = "/".join(str(len(f)) for f in faces)
        print(f"  ring of {len(cands[s])}-gon #{s}: dual faces [{sizes}]")
    print(f"  pair: {len(bonds)} bonds, {len(fringe)} wedges, {len(cluster)} whole polygons")

    ink = _ink(shade)
    fig, ax = plt.subplots(figsize=(FIGIN, FIGIN))
    facecolor = _paint(ax, color, shade, pieces, cands, cof)
    ax.add_collection(PatchCollection(
        [MplPolygon(p, closed=True) for p in polys],
        facecolor=facecolor, edgecolor="white", linewidths=ink["lw"],
        zorder=2.5, path_effects=ink["cell_fx"]))
    if outer_edges:
        ax.add_collection(LineCollection(
            [[a, b] for a, b in outer_edges], colors="white",
            linewidths=ink["lw"], zorder=2.6, path_effects=ink["cell_fx"]))
    radial = [[c, a] for c, a, b, _, _ in pieces] \
        + [[c, b] for c, a, b, _, _ in pieces]
    ax.add_collection(LineCollection(radial, colors=ink["fan_color"],
                                     linewidths=ink["lw"],
                                     linestyles=[ink["dash"]],
                                     capstyle=ink["fan_cap"], zorder=3))
    if dual != "none":
        ax.add_collection(LineCollection(
            _ring_duals(bonds, cof, dual), colors="black", linewidths=ink["lw"],
            linestyles=[DUAL_LS], zorder=3.5))

    # Kite: the one on the edge the two featured cells share.
    kite = None
    for i, j, va, vb in bonds:
        if {i, j} == set(seeds):
            kite = np.array([cof[seed], va, cof[partner], vb])
            break
    kite_tgt = kite.mean(axis=0) if kite is not None else None
    cl_emap = _edge_map(cands, cluster)
    edge_mids = [0.5 * (np.array(k[0]) + np.array(k[1])) for k in cl_emap]
    edge_tgt = _nearest(edge_mids, [kite_tgt], center, 0.05 * L)
    ctr_tgt = _nearest(list(cof.values()), [kite_tgt, edge_tgt], center, 0.05 * L)
    if kite is not None:
        ax.add_patch(MplPolygon(kite, closed=True, fill=False,
                                edgecolor=KITE_COLOR, linewidth=ink["lw"], zorder=4))
        if SHOW_BISECT:
            ax.plot([kite[0][0], kite[2][0]], [kite[0][1], kite[2][1]],
                    color=BISECT_COLOR, linewidth=ink["lw"], linestyle=DUAL_LS, zorder=4.5)
        ax.plot([kite[1][0], kite[3][0]], [kite[1][1], kite[3][1]],
                color="white", linewidth=ink["lw"], linestyle=DUAL_LS, zorder=4.5,
                path_effects=[pe.withStroke(linewidth=ink["lw"] * (1.0 + KITE_DIV_EDGE),
                                            foreground=KITE_COLOR)])
    pverts = np.unique(np.round(
        np.array([p[1] for p in pieces] + [p[2] for p in pieces]), 6), axis=0)
    ax.scatter(pverts[:, 0], pverts[:, 1], s=PDOT,
               facecolors=ink["pdot_face"], edgecolors=ink["pdot_rim"],
               linewidths=ink["pdot_rim_lw"], zorder=5)
    centers = np.array(list(cof.values()))
    ax.scatter(centers[:, 0], centers[:, 1], s=DDOT,
               facecolors=ink["dot_face"], edgecolors=ink["dot_rim"],
               linewidths=ink["dot_rim_lw"], zorder=6)

    cv = np.vstack(polys + [np.array([p[j] for p in pieces]) for j in (0, 1, 2)])
    lo, hi = cv.min(axis=0), cv.max(axis=0)
    cc = 0.5 * (lo + hi)
    half = 0.5 * float((hi - lo).max()) * (1.0 + PAD)
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
                        arrowprops=dict(arrowstyle="-", color="#111111", lw=LW * 0.625))

    focus = 0.5 * (cof[seed] + cof[partner])          # between the two featured cells
    if bleed:
        half = FILL_F * 0.5 * float((hi - lo).max())
        ax.set_xlim(focus[0] - half, focus[0] + half)
        ax.set_ylim(focus[1] - half, focus[1] + half)
        print(f"  bleed window: {2 * half:.3f} data units across, centred between the pair")
    elif trim:
        _shrink_to_ink(fig, ax, lo, hi, lw=ink["lw"])
        gx, gy = ax.get_xlim(), ax.get_ylim()
        if not (gx[0] <= lo[0] and hi[0] <= gx[1] and gy[0] <= lo[1] and hi[1] <= gy[1]):
            raise RuntimeError(f"trim cropped the mesh: bbox {lo}..{hi} outside window {gx} x {gy}")

    stem = (f"kitemesh_pair_2d_{ntype}_{placement}_clean"
            + ("" if labels else "_nolabels")
            + ("" if dual == "none" else f"_dual{dual}")
            + ("" if shade == "flat" else f"_{shade}")
            + ("" if second == "kite" else f"_second{second}")
            + ("_closed" if close_cells else "")      # unlike the parent: keep both variants
            + ("_bleed" if bleed else ""))
    out = os.path.join(OUTDIR, stem + ".png")
    fig.savefig(out, dpi=DPI, facecolor="white")
    try:
        fig.savefig(out[:-4] + ".eps", format="eps")
    except Exception as exc:
        print(f"  [eps skip] {out[:-4]}.eps: {exc}")
    plt.close(fig)
    print(f"  saved {out}  ({len(pverts)} primal verts)")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="2D kite mesh: two adjacent primal cells and their dual rings.")
    p.add_argument("--ntype", default="C", choices=("D", "V", "G", "C", "all"))
    p.add_argument("--placement", choices=("bary", "circ"), default="bary")
    p.add_argument("--labels", dest="labels", action="store_true")
    p.add_argument("--no-labels", dest="labels", action="store_false")
    p.set_defaults(labels=True)
    p.add_argument("--trim", dest="trim", action="store_true")
    p.add_argument("--no-trim", dest="trim", action="store_false")
    p.set_defaults(trim=True)
    p.add_argument("--dual", choices=("none", "mid", "straight"), default="none")
    p.add_argument("--shade", choices=("flat", "wells", "mass", "simplex"), default="flat")
    p.add_argument("--bleed", action="store_true")
    p.add_argument("--close-cells", action="store_true",
                   help="draw every neighbour either ring touches whole; dual bonds stay the rings'")
    p.add_argument("--second", default="kite",
                   help="partner cell: 'kite' (across the most central edge), 'far', or an "
                        "integer rank by edge-midpoint distance (default: kite)")
    args = p.parse_args(argv)

    os.makedirs(OUTDIR, exist_ok=True)
    pts = poisson_2d(N, L, seed=SEED)
    print(f"=== 2D kite mesh, PAIR: seed={SEED} N={N} L={L:g} placement={args.placement} "
          f"labels={args.labels} trim={args.trim} dual={args.dual} shade={args.shade} "
          f"second={args.second} -> {OUTDIR} ===")
    types = ("D", "V", "G", "C") if args.ntype == "all" else (args.ntype,)
    for nt in types:
        print(f"--- {nt} ---")
        try:
            panel(nt, args.placement, pts, TESS_COLORS[nt], args.labels, args.trim,
                  args.dual, args.shade, args.bleed, args.close_cells, args.second)
        except Exception as exc:
            print(f"  [skip] {nt}: {exc}")
    print(f"=== done -> {OUTDIR} ===")


if __name__ == "__main__":
    main()
