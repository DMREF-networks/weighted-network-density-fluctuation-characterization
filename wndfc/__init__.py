"""wndfc (weighted network density fluctuation characterization) -- spectral
density and edge-length variance of weighted periodic tessellation networks
(2D and 3D).

Pipeline: points -> network (D/V/G/C tessellation, PBC) -> per-edge kernel
weights w_eff (theta family; V/T/D cell-measure schemes) -> chi(k) in the
bare 1/V gauge (with the per-mode self term) -> direct sigma^2(R), the
Parseval bridge, the real-space pair deposits (self / distinct channels of
h_w(r)), and the small-R analytic library.
"""
from __future__ import annotations

from .network import Network2D, build_network, build_from_graph, voronoi_cells_general
from .network_3d import Network3D, build_network_3d
from .generators import poisson_2d, poisson_3d, url_2d, url_3d, hip_2d, hip_m_for_target_n
from .loaders import load_points_2d, load_points_3d, load_packing, load_lattice, load_edges, load_radii
from .weight_names import WEIGHTS, SCHEME, WEIGHT_ALIASES, weight_tag, dir_tag, resolve_weight
from .weights import compute_weights
from .weights_3d import compute_weights_3d
from .spectral import ChiEll2D, chi_ell_modes, compute_chi_ell
from .spectral_3d import ChiEll3D, compute_chi_ell_3d
from .variance import Sigma2Result, compute_sigma2
from .variance_3d import Sigma2Result3D, compute_sigma2_3d
from .parseval import parseval_sigma2
from .h_ell import (PairDeposits, RealSpaceCurves, sample_pair_deposits, real_space_curves,
                    real_space_moments, real_space_diagnostics, deposits_from_npz, spectral_h_w,
                    hankel_inv_2d, contact_plateau)
from .norms import build_metadata, derived, axes, moment_sums, vertex_wedge_sum
from .pipeline import build_weighted_network, TilingError
from .io import save_result, load_result
from .small_r import (g_kernel, g_d_form_factor, H_infty, c4_exact, c4_r4_prediction,
                      unified_prediction, close_pair_distinct_sigma2, weighted_smallR_leading,
                      crossover_Rstar, porod_moment, f_w_self_baseline, vertex_angle_sum)
from .small_r_3d import (unified_prediction_3d, weighted_smallR_leading_3d, c5_exact,
                         c5_r5_prediction, close_pair_distinct_sigma2_3d)

__all__ = [
    "Network2D", "build_network", "Network3D", "build_network_3d",
    "poisson_2d", "poisson_3d", "url_2d", "url_3d", "hip_2d", "hip_m_for_target_n",
    "load_points_2d", "load_points_3d", "load_packing", "load_lattice", "load_edges", "load_radii",
    "build_from_graph", "voronoi_cells_general",
    "WEIGHTS", "SCHEME", "WEIGHT_ALIASES", "weight_tag", "dir_tag", "resolve_weight",
    "compute_weights", "compute_weights_3d",
    "ChiEll2D", "chi_ell_modes", "compute_chi_ell", "ChiEll3D", "compute_chi_ell_3d",
    "Sigma2Result", "compute_sigma2", "Sigma2Result3D", "compute_sigma2_3d",
    "parseval_sigma2",
    "PairDeposits", "RealSpaceCurves", "sample_pair_deposits", "real_space_curves", "real_space_moments",
    "real_space_diagnostics", "deposits_from_npz", "spectral_h_w", "hankel_inv_2d", "contact_plateau",
    "build_metadata", "derived", "axes", "moment_sums", "vertex_wedge_sum",
    "build_weighted_network", "TilingError", "run_pipeline", "run_pipeline_3d",
    "save_result", "load_result",
    "g_kernel", "g_d_form_factor", "H_infty", "c4_exact", "c4_r4_prediction",
    "unified_prediction", "close_pair_distinct_sigma2", "weighted_smallR_leading",
    "crossover_Rstar", "porod_moment", "f_w_self_baseline", "vertex_angle_sum",
    "unified_prediction_3d", "weighted_smallR_leading_3d", "c5_exact", "c5_r5_prediction",
    "close_pair_distinct_sigma2_3d",
]


def run_pipeline(points, L, ntype="D", weight="length_power", *, theta=1.0, kmax=200.0,
                 shell_mode="width", backend="numpy"):
    """points -> network -> w_eff -> chi (2D); returns (net, w_eff, chi)."""
    net, w, _ = build_weighted_network(points, L, ntype, weight, dim=2, theta=theta)
    return net, w, compute_chi_ell(net, w, kmax=kmax, shell_mode=shell_mode, backend=backend)


def run_pipeline_3d(points, L, ntype="D", weight="length_power", *, theta=1.0, kmax=20.0,
                    shell_mode="width", backend="numpy"):
    """points -> network -> w_eff -> chi (3D); returns (net, w_eff, chi)."""
    net, w, _ = build_weighted_network(points, L, ntype, weight, dim=3, theta=theta)
    return net, w, compute_chi_ell_3d(net, w, kmax=kmax, shell_mode=shell_mode, backend=backend)
