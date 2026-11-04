"""Per-edge kernel weights w_eff (2D).  See weight_names.py for the schemes.

Every scheme returns the KERNEL weight w_eff = w * ell that the spectral sum
multiplies the rod form factor by:

  length_power         w_eff = ell^theta
  rho_line_inv   (V)   w_eff = ell * sum_{v in e} A_v / ell*_v
                       (A_v = ancillary Voronoi cell area of endpoint v,
                        ell*_v = total length of the edges incident to v)
  midpt_sec_area (T)   w_eff = Voronoi cell area of the edge midpoint in the
                       tessellation of all edge midpoints
  dual_diamond_planar (D)  w_eff = signed kite area (1/2) ell d_e sin(theta_e)
                       of the barycentric planar dual (planar_dual.py)
  d2_ring        (D_2) w_eff = sum over vertices of moment-matched shares of the
                       vertex's barycentric D cell on its one-ring edges, starting
                       from the D kite halves (moment_weights.py); 2D, D network
                       of a triangulation, square box

The cell-measure schemes tile the box: sum_e w_eff,e = L^2.
"""
from __future__ import annotations

import numpy as np

from .network import Network2D
from .weight_names import WEIGHTS


def compute_weights(net: Network2D, weight: str = "length_power", *,
                    theta: float = 1.0, eps: float = 1e-30,
                    diag_out: dict | None = None) -> np.ndarray:
    """w_eff for ``weight`` (see module doc). ``diag_out`` receives the planar-dual
    diagnostics for ``dual_diamond_planar``."""
    if weight not in WEIGHTS:
        raise ValueError(f"unknown weight: {weight!r}; expected one of {WEIGHTS}")
    if net.n_edges == 0:
        return np.zeros(0, dtype=float)
    ell = net.edges_lengths

    if weight == "length_power":
        return np.power(ell, float(theta))

    if weight == "rho_line_inv":
        A_L = net.cell_areas[net.edges_left_cell]
        A_R = net.cell_areas[net.edges_right_cell]
        Lstar_L = net.cell_star_lengths[net.edges_left_cell]
        Lstar_R = net.cell_star_lengths[net.edges_right_cell]
        w = (A_L / np.maximum(Lstar_L, eps) + A_R / np.maximum(Lstar_R, eps))
        return w * ell

    if weight == "midpt_sec_area":
        from .network import build_network as _build_network
        from .network import _dedup_points_2d, voronoi_cells_general
        from .pbc import wrap_to_box
        cell = net.cell_like
        midpts = wrap_to_box(net.edges_midpoints, cell)
        uniq, inverse = _dedup_points_2d(midpts, cell)
        mult = np.bincount(inverse, minlength=uniq.shape[0])   # merged cells split by multiplicity
        if net.cell is not None:                               # general cell: voro++ shim
            areas, _ = voronoi_cells_general(uniq, net.cell)
        else:
            areas = _build_network(uniq, net.L, "D").cell_areas
        return (areas / mult)[inverse]

    if weight == "d2_ring":
        from .moment_weights import d2_ring
        return d2_ring(net, diag_out=diag_out)

    # dual_diamond_planar
    from .planar_dual import planar_face_dual
    pd = planar_face_dual(net)
    kite = 0.5 * ell * pd["edge_dual_len"] * pd["edge_sin_theta"]
    if diag_out is not None:
        diag_out["dual_diamond_planar"] = {
            "n_homology":           int(pd["n_homology"]),
            "n_bad_edges":          int(pd["n_bad_edges"]),
            "n_nonconvex_faces":    int(pd["n_nonconvex_faces"]),
            "n_small_faces":        int(pd["n_small_faces"]),
            "n_zero_kite_unmasked": int(pd["n_zero_kite_unmasked"]),
            "n_faces":              int(pd["n_faces"]),
            "kite_sum":             float(pd["kite_sum"]),
            "kite_resid":           float(pd["kite_resid"]),
        }
    return kite
