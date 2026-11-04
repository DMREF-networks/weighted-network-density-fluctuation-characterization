"""Weight tokens, scheme names, and output tags -- single source of truth.

Five edge-weight schemes (kernel weight w_eff = w * ell):

  length_power         theta family, w_eff = ell^theta (theta = 1: unweighted)
  rho_line_inv         V scheme: ancillary Voronoi at the network vertices,
                       cell content split among incident edges by length
  midpt_sec_area       T scheme: Voronoi cell content of the edge midpoint
  dual_diamond_planar  D scheme: signed kite (2D) / bipyramid (3D) of the
                       barycentric dual; undefined for G in 3D
  d2_ring              D_2 scheme: the D kite halves on each vertex's own edges,
                       corrected over its one-ring to match the barycentric D
                       cell through second order; 2D, D network of a
                       triangulation, square box

The cell-measure schemes tile: sum_e w_eff,e = V.

The paper's symbols are accepted as aliases of the tokens on the CLI and in
``build_weighted_network`` (case-insensitive; ``resolve_weight``):

  paper   alias        token                 stored ``scheme``
  zeta    theta/zeta   length_power          theta   (the exponent is --theta, alias --zeta)
  V       V / MCV      rho_line_inv          V
  T       T / MCT      midpt_sec_area        T
  D       D / MCD      dual_diamond_planar   D
  D_2     D2 / MCD2    d2_ring               D2
"""
from __future__ import annotations

WEIGHTS = ("length_power", "rho_line_inv", "midpt_sec_area", "dual_diamond_planar", "d2_ring")
SCHEME = {
    "length_power": "theta",
    "rho_line_inv": "V",
    "midpt_sec_area": "T",
    "dual_diamond_planar": "D",
    "d2_ring": "D2",
}
# paper symbols / short names -> token (looked up case-insensitively)
WEIGHT_ALIASES = {
    "theta": "length_power", "zeta": "length_power",
    "v": "rho_line_inv", "mcv": "rho_line_inv",
    "t": "midpt_sec_area", "mct": "midpt_sec_area",
    "d": "dual_diamond_planar", "mcd": "dual_diamond_planar",
    "d2": "d2_ring", "mcd2": "d2_ring",
}
WEIGHTS_2D = WEIGHTS
WEIGHTS_3D = WEIGHTS
# 2D: weights that never read the generator-cell areas (skip the secondary voro++ pass)
SKIP_CELL_AREAS_2D = frozenset({"length_power", "dual_diamond_planar", "d2_ring"})
# 3D: weights that need generator-cell volumes
NEED_CELL_VOLUMES_3D = frozenset({"rho_line_inv"})


def resolve_weight(name: str) -> str:
    """Canonical weight token for a token or one of its aliases; ValueError otherwise."""
    if name in WEIGHTS:
        return name
    key = str(name).strip().lower()
    if key in WEIGHTS:
        return key
    if key in WEIGHT_ALIASES:
        return WEIGHT_ALIASES[key]
    raise ValueError(f"unknown weight {name!r}; expected one of {WEIGHTS} or an alias "
                     f"{tuple(sorted(WEIGHT_ALIASES))}")


def weight_tag(weight: str, theta: float | None = None) -> str:
    """File/dir tag of a weight: scheme token plus the theta exponent for the theta family."""
    if weight == "length_power":
        t = 1.0 if theta is None else float(theta)
        s = f"{t:g}"
        return f"length_power_th{s}"
    return weight


def dir_tag(ntype: str, ctype: str, weight: str, theta: float | None = None) -> str:
    """<T>_<ctype>_<weight_tag>, e.g. D_poi_length_power_th1, V_url_rho_line_inv."""
    return f"{ntype}_{ctype}_{weight_tag(weight, theta)}"
