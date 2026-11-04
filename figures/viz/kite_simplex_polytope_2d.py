"""2D simplex-polytope kite mesh, one panel per tessellation x dual placement.

Same clean style as the 2D anatomy plots, cropped to a small connected cluster
of cells (faces sharing edges). Each panel is a "simplex-polytope mesh": the
network's planar faces are the POLYTOPES (solid, tess hue); each is
fan-triangulated from a central point into SIMPLICES (dotted radial edges
meeting in the middle); that centre is dotted. The centre is the dual vertex
under one of two placements:

    barycentric  : the face centroid (always interior).
    circumcentric: the least-squares circumcentre of the face's vertices
                   (a true circumcentre for triangles; for polygons a best-fit
                   centre that can land off-centre / outside -- that is fine
                   and is the point of the circumcentric panel).

One per tessellation (D / V / G / C) x {circumcentric, barycentric} = 8. The
only text on each image is three features marked (a), (b), (c) with leader
lines: (a) a centre/circumcentre, (b) a cell edge, (c) a kite -- two adjacent
fan-simplices across a shared cell edge, the quad [c1, va, c2, vb] -- drawn
with a black outline plus its centre-centre diagonal (longer dashes).

Clean borderless PNGs (+ EPS siblings) are written to figures/viz/outputs/
(override with VIZ_OUT).

Run (from the release root):
    python figures/viz/kite_simplex_polytope_2d.py
Knobs (env): KMESH_SEED, KMESH_N, KMESH_L, KMESH_WIN, KMESH_CELLS, KMESH_PAD.
"""
from __future__ import annotations

import heapq
import os
import shutil
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon as MplPolygon

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))          # release root (wndfc/)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from wndfc.generators import poisson_2d
from wndfc.network import build_network
from anatomy_kite_2d import _face_polygons, TESS_COLORS, _shade

SEED = int(os.environ.get("KMESH_SEED", "16"))
L = float(os.environ.get("KMESH_L", "10"))
N = int(os.environ.get("KMESH_N", "100"))
WIN = float(os.environ.get("KMESH_WIN", "0.25"))      # candidate gather / L
CELLS = int(os.environ.get("KMESH_CELLS", "8"))       # cells in the cluster
PAD = float(os.environ.get("KMESH_PAD", "0.70"))      # frame margin (whitespace)
FIGIN = float(os.environ.get("KMESH_FIGIN", "7.0"))   # fixed figure side (in)
OUTDIR = os.environ.get("VIZ_OUT", os.path.join(_HERE, "outputs", "kite_simplex"))
PAPER_DIR = os.path.join(OUTDIR, "paper")


def _center(P, placement):
    """Barycentre, or least-squares circumcentre of the polygon's vertices
    (exact for a triangle; best-fit otherwise, may sit off-centre)."""
    if placement == "bary":
        return P.mean(axis=0)
    x, y = P[:, 0], P[:, 1]
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(P))])
    try:
        sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
        return np.array([sol[0], sol[1]])
    except np.linalg.LinAlgError:
        return P.mean(axis=0)


def _gather_polys(net, center, win):
    """Network faces, min-image re-anchored to the window, that fall in it."""
    out = []
    for f in _face_polygons(net):
        v = f["verts"]
        vv = v - L * np.round((v.mean(axis=0) - center) / L)
        if (vv[:, 0].max() < center[0] - win or vv[:, 0].min() > center[0] + win
                or vv[:, 1].max() < center[1] - win
                or vv[:, 1].min() > center[1] + win):
            continue
        out.append(vv)
    return out


def _grow_cluster(seed, adj, cent, center, target):
    """A connected cluster of ``target`` faces grown nearest-the-centre-first
    from ``seed`` over the face-adjacency graph ``adj``."""
    cluster = {seed}
    heap = [(float(np.linalg.norm(cent[nb] - center)), nb) for nb in adj[seed]]
    heapq.heapify(heap)
    while heap and len(cluster) < target:
        _, u = heapq.heappop(heap)
        if u in cluster:
            continue
        cluster.add(u)
        for nb in adj[u]:
            if nb not in cluster:
                heapq.heappush(heap, (float(np.linalg.norm(cent[nb] - center)),
                                      nb))
    return cluster


def _edge_map(polys, indices):
    """sorted-rounded-endpoint key -> [(poly_index, va, vb), ...] over the
    given polygon indices."""
    em = {}
    for i in indices:
        p = polys[i]
        m = p.shape[0]
        for k in range(m):
            a, b = p[k], p[(k + 1) % m]
            key = tuple(sorted((tuple(np.round(a, 6)), tuple(np.round(b, 6)))))
            em.setdefault(key, []).append((i, a, b))
    return em


def _nearest(cands, avoid, center, rmin):
    avoid = [a for a in avoid if a is not None]
    best = None
    for pt in cands:
        if all(np.linalg.norm(pt - u) > rmin for u in avoid):
            d = np.linalg.norm(pt - center)
            if best is None or d < best[0]:
                best = (d, pt)
    return None if best is None else best[1]


def panel(ntype, placement, pts, color):
    c_area = "triangle" if ntype == "C" else "sec_vor"
    net = build_network(pts, L, ntype, c_area_mode=c_area)
    center = np.array([L / 2.0, L / 2.0])
    cands = _gather_polys(net, center, WIN * L)
    if len(cands) < 2:
        print(f"  [skip] {ntype}/{placement}: too few faces")
        return
    cent = [p.mean(axis=0) for p in cands]

    # Face adjacency (share an edge) -> grow a compact connected cluster.
    adj = {i: set() for i in range(len(cands))}
    for lst in _edge_map(cands, range(len(cands))).values():
        idxs = [t[0] for t in lst]
        for a in idxs:
            adj[a].update(j for j in idxs if j != a)
    seed = int(np.argmin([np.linalg.norm(c - center) for c in cent]))
    cluster = _grow_cluster(seed, adj, cent, center, CELLS)
    polys = [cands[i] for i in cluster]
    cof = {i: _center(cands[i], placement) for i in cluster}

    fig, ax = plt.subplots(figsize=(FIGIN, FIGIN))
    ax.add_collection(PatchCollection(
        [MplPolygon(p, closed=True) for p in polys],
        facecolor=_shade(color, 0.0), edgecolor="white", linewidths=1.6,
        zorder=2))
    radial = [[cof[i], v] for i in cluster for v in cands[i]]
    ax.add_collection(LineCollection(radial, colors="#1a1a1a", linewidths=0.7,
                                     linestyles=":", zorder=3))

    # Kite: a cell edge shared by two cluster cells, centres on opposite sides
    # (a clean, non-self-intersecting kite). Most central such edge.
    cl_emap = _edge_map(cands, cluster)
    kite = None
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
                                edgecolor="black", linewidth=1.8, zorder=4))
        # centre-centre diagonal (the two centres joined) -- longer dashes than
        # the dotted simplex edges, so the kite's construction reads clearly.
        ax.plot([kite[0][0], kite[2][0]], [kite[0][1], kite[2][1]],
                color="black", linewidth=1.2, linestyle=(0, (7, 4)), zorder=4)
    centers = np.array(list(cof.values()))
    ax.scatter(centers[:, 0], centers[:, 1], s=42, facecolors="white",
               edgecolors="#1a1a1a", linewidths=1.1, zorder=5)

    # Frame to the cluster's bounding box; the big (a)(b)(c) labels live in the
    # surrounding whitespace with leader lines.
    # Fixed frame around the cluster (so every panel is the same pixel size):
    # the cluster fills 1/(1+PAD) of the view, leaving a whitespace ring where
    # the big (a)(b)(c) labels sit at a constant radius.
    cv = np.vstack(polys)
    lo, hi = cv.min(axis=0), cv.max(axis=0)
    cc = 0.5 * (lo + hi)
    extent = float((hi - lo).max())
    half = 0.5 * extent * (1.0 + PAD)
    ax.set_xlim(cc[0] - half, cc[0] + half)
    ax.set_ylim(cc[1] - half, cc[1] + half)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    for lbl, tgt in (("a", ctr_tgt), ("b", edge_tgt), ("c", kite_tgt)):
        if tgt is None:
            continue
        d = tgt - cc
        n = np.linalg.norm(d)
        d = d / n if n > 1e-9 else np.array([0.0, 1.0])
        ax.annotate(f"({lbl})", xy=tgt, xytext=cc + d * 0.78 * half,
                    fontsize=40, ha="center", va="center", color="#111111",
                    zorder=6, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color="#111111", lw=1.0))

    out = os.path.join(OUTDIR, f"kitemesh_2d_{ntype}_{placement}_clean.png")
    save_kw = dict(dpi=175, facecolor="white")   # fixed size -> uniform panels
    fig.savefig(out, **save_kw)
    try:
        fig.savefig(out[:-4] + ".eps", format="eps")
    except Exception as exc:
        print(f"  [eps skip] {out[:-4]}.eps: {exc}")
    plt.close(fig)
    print(f"  saved {out}  ({len(polys)} cells)")
    for src in (out, out[:-4] + ".eps"):
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(PAPER_DIR,
                                              os.path.basename(src)))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(PAPER_DIR, exist_ok=True)
    pts = poisson_2d(N, L, seed=SEED)
    print(f"=== 2D simplex-polytope kite mesh: seed={SEED} N={N} L={L:g} "
          f"cells={CELLS} -> {OUTDIR} (+ {PAPER_DIR}) ===")
    for ntype in ("D", "V", "G", "C"):
        for placement in ("circ", "bary"):
            try:
                panel(ntype, placement, pts, TESS_COLORS[ntype])
            except Exception as exc:
                print(f"  [skip] {ntype}/{placement}: {exc}")
    print(f"=== done -> {OUTDIR} ===")


if __name__ == "__main__":
    main()
