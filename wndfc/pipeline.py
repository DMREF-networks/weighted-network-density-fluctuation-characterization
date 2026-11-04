"""Points -> network -> w_eff, with the tiling and structural checks.

Degenerate generator sets (lattices, jammed packings, cocircular / cospherical
tuples) break the tessellation layer's genericity assumption: voro++ resolves
a k-fold Voronoi vertex as one k-valent vertex, and a Delaunay derived from
that output is not a triangulation.  The V scheme is immune by construction
(its ancillary sites are the distinct Voronoi vertices, network.py); the C
tessellation and the secondary passes are not, and neither is the tiling
identity sum_e w_eff,e = V that the V/T/D schemes must satisfy.

A network that fails the structural check (build_diag["degenerate"]) or the
tiling sum rule raises TilingError; the points are never altered.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .pbc import cell_volume
from .weight_names import SCHEME, SKIP_CELL_AREAS_2D, NEED_CELL_VOLUMES_3D, resolve_weight

NETWORK_SOURCES = ("tessellation", "edges")

TILING_TOL = 1e-8                          # |sum w_eff / V - 1| accepted for V/T/D


def contact_gap_stats(points, L, edges, radii, tol: float = 1e-3) -> dict:
    """Relative gaps |r_ij| / (R_i + R_j) - 1 over the triangulation edges
    (diagnostic: contacts have gap ~ 0)."""
    from .pbc import min_image_delta, wrap_to_box

    pts = wrap_to_box(np.asarray(points, dtype=float), L)
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    r = np.asarray(radii, dtype=float)
    d = np.linalg.norm(min_image_delta(pts[e[:, 1]] - pts[e[:, 0]], L), axis=1)
    gap = d / (r[e[:, 0]] + r[e[:, 1]]) - 1.0
    return {"tol": float(tol), "n_edges": int(e.shape[0]),
            "n_gap_gt_1e-4": int((gap > 1e-4).sum()), "n_gap_gt_tol": int((gap > tol).sum()),
            "max_gap": float(gap.max()) if gap.size else 0.0,
            "min_gap": float(gap.min()) if gap.size else 0.0}


class TilingError(RuntimeError):
    """The tiling sum rule (or the structural check) failed."""


def _build_once(points, L, ntype, weight, dim, theta, c_area_mode,
                diag_out, network_source="tessellation", edges=None, radii=None):
    if dim == 2 and network_source != "tessellation":
        from .network import build_from_triangulation
        from .weights import compute_weights
        if edges is None:
            raise ValueError("network_source='edges' needs an edge list")
        net = build_from_triangulation(points, L, edges, ntype, radii=radii,
                                       compute_cell_areas=(weight not in SKIP_CELL_AREAS_2D))
        net.build_diag["primal_edges"] = np.asarray(edges, dtype=np.int64)
        w = compute_weights(net, weight, theta=theta, diag_out=diag_out)
        return net, w
    if dim == 3:
        from .network_3d import build_network_3d
        from .weights_3d import compute_weights_3d
        net = build_network_3d(points, L, ntype,
                               compute_cell_volumes=(weight in NEED_CELL_VOLUMES_3D))
        w = compute_weights_3d(net, weight, theta=theta, diag_out=diag_out)
    else:
        from .network import build_network
        from .weights import compute_weights
        net = build_network(points, L, ntype, c_area_mode=c_area_mode,
                            compute_cell_areas=(weight not in SKIP_CELL_AREAS_2D))
        w = compute_weights(net, weight, theta=theta, diag_out=diag_out)
    return net, w


def build_weighted_network(points, L: float, ntype: str, weight: str, *, dim: int = 2,
                           theta: float = 1.0,
                           c_area_mode: str = "sec_vor",
                           diag_out: Optional[dict] = None,
                           network_source: str = "tessellation",
                           edges: Optional[np.ndarray] = None,
                           radii: Optional[np.ndarray] = None):
    """(net, w_eff, info) for ``weight`` on the ``ntype`` tessellation of ``points``.

    Raises TilingError when the tiling check (V/T/D: |sum w_eff / V - 1| <= TILING_TOL)
    or the structural check fails.  ``info``: tiling_resid (None for the theta family),
    build_diag.  ``diag_out`` receives the weight diagnostics (dual_diamond_planar).
    ``L`` may be a ``pbc.Cell`` (2D general cell).  ``network_source``: the primal
    network -- ``tessellation`` (voro++ on the points; orthogonal cells only)
    or ``edges`` (a given periodic planar graph: a packing's contact list, a
    strut layout).
    """
    weight = resolve_weight(weight)
    if network_source not in NETWORK_SOURCES:
        raise ValueError(f"network_source={network_source!r}; expected one of {NETWORK_SOURCES}")
    V = cell_volume(L) if int(dim) == 2 else float(L) ** 3
    tiles = SCHEME[weight] != "theta"
    if diag_out is not None:
        diag_out.clear()
    net, w = _build_once(points, L, ntype, weight, dim, theta,
                         c_area_mode, diag_out, network_source, edges, radii)
    resid = abs(float(np.sum(w)) / V - 1.0) if tiles else None
    diag = dict(getattr(net, "build_diag", None) or {})
    degenerate = bool(diag.get("degenerate", False))
    if degenerate or not (resid is None or resid <= TILING_TOL):
        raise TilingError(
            f"{ntype}/{weight}: tiling or structural check failed (tiling resid={resid}, "
            f"degenerate={degenerate}); the generator set is degenerate")
    return net, w, {"tiling_resid": resid, "build_diag": diag}
