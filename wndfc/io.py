"""NPZ persistence: arrays + one embedded metadata JSON (schema in norms.py).

Arrays (always): k, chi_ell, chi_self, counts.
Optional: w (kernel weights), ell; edge table (edge_ell, edge_midpoints,
edge_ell_dual, cell_areas, edges_left_cell, edges_right_cell); sigma2_R,
sigma2_direct, sigma2_spectral; the real-space pair deposits h_ell_r_edges,
h_ell_self, h_ell_dist, h_ell_self_sq, h_ell_dist_sq, h_ell_self_mom,
h_ell_dist_mom (h_ell.PairDeposits; curves and diagnostics are read-time).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np



def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def save_result(path: str, *, chi, net, w_eff: np.ndarray, meta: Dict[str, Any],
                sigma2_R: Optional[np.ndarray] = None, sigma2_direct: Optional[np.ndarray] = None,
                sigma2_spectral: Optional[np.ndarray] = None, h_ell=None,
                save_weights: bool = False, save_edge_data: bool = False) -> str:
    """Write the NPZ. ``meta`` is the norms.build_metadata dict (+ any extras);
    ``h_ell`` an h_ell.PairDeposits."""
    meta = dict(meta)
    arrays: Dict[str, np.ndarray] = {
        "k": np.asarray(chi.k, dtype=np.float64),
        "chi_ell": np.asarray(chi.chi_ell, dtype=np.float64),
        "chi_self": np.asarray(chi.chi_self, dtype=np.float64),
        "counts": np.asarray(chi.counts, dtype=np.int64),
    }
    if save_weights:
        arrays["w"] = np.asarray(w_eff, dtype=np.float64)
        arrays["ell"] = np.asarray(net.edges_lengths, dtype=np.float64)
    if save_edge_data:
        arrays["edge_ell"] = np.asarray(net.edges_lengths, dtype=np.float64)
        arrays["edge_midpoints"] = np.asarray(net.edges_midpoints, dtype=np.float64)
        for src, key, dt in (("edges_dual_length", "edge_ell_dual", np.float64),
                             ("cell_areas", "cell_areas", np.float64),
                             ("edges_left_cell", "edges_left_cell", np.int64),
                             ("edges_right_cell", "edges_right_cell", np.int64)):
            val = getattr(net, src, None)
            if val is not None:
                arrays[key] = np.asarray(val, dtype=dt)
    if sigma2_R is not None:
        arrays["sigma2_R"] = np.asarray(sigma2_R, dtype=np.float64)
        if sigma2_direct is not None:
            arrays["sigma2_direct"] = np.asarray(sigma2_direct, dtype=np.float64)
        if sigma2_spectral is not None:
            arrays["sigma2_spectral"] = np.asarray(sigma2_spectral, dtype=np.float64)
    if h_ell is not None:
        for key, val in (("h_ell_r_edges", h_ell.r_edges), ("h_ell_self", h_ell.dep_self),
                         ("h_ell_dist", h_ell.dep_dist), ("h_ell_self_sq", h_ell.sq_self),
                         ("h_ell_dist_sq", h_ell.sq_dist), ("h_ell_self_mom", h_ell.mom_self),
                         ("h_ell_dist_mom", h_ell.mom_dist)):
            arrays[key] = np.asarray(val, dtype=np.float64)
        meta["h_ell"] = h_ell.meta()
    arrays["metadata_json"] = np.asarray(json.dumps(_jsonable(meta)))
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_result(path: str) -> Dict[str, Any]:
    """Arrays + ``metadata`` dict from an NPZ written by ``save_result``."""
    with np.load(path, allow_pickle=False) as data:
        out = {k: data[k].copy() for k in data.files if k != "metadata_json"}
        out["metadata"] = json.loads(str(data["metadata_json"]))
    return out
