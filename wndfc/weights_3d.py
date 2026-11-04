"""Per-edge kernel weights w_eff (3D).  Same schemes as weights.py with cell
volumes in place of areas:

  length_power         w_eff = ell^theta
  rho_line_inv   (V)   w_eff = ell * sum_{v in e} V_v / ell*_v
  midpt_sec_area (T)   w_eff = Voronoi cell volume of the edge midpoint
  dual_diamond_planar (D)  w_eff = signed bipyramid volume of the barycentric
                       dual (volumetric_dual_3d.py; cell apex = volume centroid,
                       face anchor = area centroid).  Undefined for G.

The cell-measure schemes tile: sum_e w_eff,e = L^3.
"""
from __future__ import annotations

import numpy as np

from .network_3d import Network3D
from .weight_names import WEIGHTS


def compute_weights_3d(net: Network3D, weight: str = "length_power", *,
                       theta: float = 1.0, eps: float = 1e-30,
                       diag_out: dict | None = None) -> np.ndarray:
    if weight not in WEIGHTS:
        raise ValueError(f"unknown weight: {weight!r}; expected one of {WEIGHTS}")
    if weight == "d2_ring":
        raise ValueError(f"{weight} is defined in 2D only")
    if net.n_edges == 0:
        return np.zeros(0, dtype=float)
    ell = net.edges_lengths

    if weight == "length_power":
        return np.power(ell, float(theta))

    if weight == "rho_line_inv":
        V_L = net.cell_volumes[net.edges_left_cell]
        V_R = net.cell_volumes[net.edges_right_cell]
        Lstar_L = net.cell_star_lengths[net.edges_left_cell]
        Lstar_R = net.cell_star_lengths[net.edges_right_cell]
        w = (V_L / np.maximum(Lstar_L, eps) + V_R / np.maximum(Lstar_R, eps))
        return w * ell

    if weight == "midpt_sec_area":
        from .network import _dedup_points_2d
        from .network_3d import _cell_volumes_streaming
        from .pbc import wrap_to_box
        midpts = wrap_to_box(net.edges_midpoints, net.L)
        uniq, inverse = _dedup_points_2d(midpts, net.L)     # dimension-agnostic dedup
        mult = np.bincount(inverse, minlength=uniq.shape[0])
        return (_cell_volumes_streaming(uniq, net.L) / mult)[inverse]

    # dual_diamond_planar
    if net.ntype == "G":
        raise ValueError("dual_diamond_planar is undefined for the Gabriel network in 3D")
    from .volumetric_dual_3d import volumetric_face_dual
    vd = volumetric_face_dual(net, g_mode="merge", apex_mode="volume_com",
                              face_anchor="area_centroid")
    if diag_out is not None:
        dd = {
            "apex_mode":         str(vd.get("apex_mode", "volume_com")),
            "face_anchor":       str(vd.get("face_anchor", "area_centroid")),
            "n_tets":            int(vd["n_tets"]),
            "n_edges_unmatched": int(vd["n_edges_unmatched"]),
            "kite_sum":          float(vd["kite_sum"]),
            "kite_resid":        float(vd["kite_resid"]),
        }
        for k in ("n_subtets", "frac_neg_subtets", "edge_coherence_p01", "edge_coherence_p05",
                  "edge_coherence_p50", "n_deficit_subtets", "deficit_abs_sum",
                  "gauge_spread", "com_tiling_resid"):
            if k in vd:
                dd[k] = float(vd[k])
        for k in ("n_nonsimple_rings", "n_c3_cells"):
            if k in vd:
                dd[k] = int(vd[k])
        diag_out["dual_diamond_planar"] = dd
    return vd["per_edge_weight"]
