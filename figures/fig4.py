"""Fig. 4 -- real-space components of the edge autocovariance, 2D theta = 1,
Poisson and URL: (a) self channel, (b) distinct h_w, (c) contact-normalized
total (direct route).

    python figures/fig4.py --data <root> --out <dir> [--norm contact|master|rhow]

<root>/<ctype>/<T>/*.npz with the pair deposits (h_ell_r_edges, h_ell_self,
h_ell_dist, ...), k / chi_ell / chi_self and edge_ell (+ w); legacy NPZs with
h_ell_r_grid, h_ell_spectral, h_ell_direct, edge_ell are read as they are.

--norm contact (default): x = r lambda_ell^{1/2}, channels divided by
    lambda_ell <w^2 ell>.
--norm master: x = r / ell*, ell* = <w^2 ell^2>/<w^2 ell>; self channel drawn as
    ell* p_w(r) = 2 <w^2 (ell-r)_+>/<w^2 ell> (y(0) = 2, unit area) between the
    monodisperse and exponential reference curves; total carries the same factor.
    Output names get a ``_master`` suffix.
--norm rhow: x = r lambda_ell^{1/2}; every channel divided by rho_w^2 only
    (self f_w, distinct h_w, total f_w + h_w).  Output names get a ``_rhow`` suffix.
--total-spectral: the total panel (c, f) draws the sampled total as markers against
    the Hankel inversion of the whole bare spectrum as the line (default: sampled
    total as the line, no markers).  Output names get a ``_tspec`` suffix.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chi_real_panels import CHANNELS, NORMS, load_real, plot_real_panel  # noqa: E402

# Tessellation-legend placement per (channel, substrate) or (channel, substrate,
# norm); the more specific key wins, anything not listed uses the panel spec in
# CHANNELS.  Channels: chisnorm (self), chid (distinct), hnormd (total).  The value
# "off" suppresses that panel's legend (e.g. when one panel of the set carries it).
LEGEND_LOC = {
    ("chid", "poi"): "off",
    ("hnormd", "poi"): "off",
    ("chisnorm", "url"): "off",
    ("hnormd", "url"): "off",
    ("chid", "url"): "off",
}

# Contact-value legend of the distinct panel (the r -> 0 ticks, CONTACT_LEGEND in
# chi_real_panels), per substrate or (substrate, norm).  Independent of LEGEND_LOC
# so one distinct panel keys the ticks while the self panel carries the
# tessellation legend.  "off" suppresses it; a substrate not listed follows that
# panel's tessellation legend.
CONTACT_LEGEND_LOC = {
    "poi": "upper left",
    "url": "off",
}


def _legend_loc(chan, ctype, norm):
    return LEGEND_LOC.get((chan, ctype, norm)) or LEGEND_LOC.get((chan, ctype))


def _contact_legend_loc(ctype, norm):
    return CONTACT_LEGEND_LOC.get((ctype, norm)) or CONTACT_LEGEND_LOC.get(ctype)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--norm", choices=NORMS, default="contact")
    ap.add_argument("--total-spectral", action="store_true", default=False)
    ap.add_argument("--porod-moment", action="store_true", default=False,
                    help="normalize by the finite-k_max Porod moment <w^2 ell>_kmax (sum_w2l_kmax) "
                         "instead of the bare <w^2 ell>; needed for theta = 0 (suffix _porod)")
    a = ap.parse_args(argv)
    suffix = (("" if a.norm == "contact" else f"_{a.norm}") + ("_tspec" if a.total_spectral else "")
              + ("_porod" if a.porod_moment else ""))
    for ctype in ("poi", "url"):
        curves = []
        for t in "CDGV":                 # a missing tessellation or substrate is skipped, not fatal
            try:
                curves.append(load_real(os.path.join(a.data, ctype, t, "*.npz"), t, porod=a.porod_moment))
            except FileNotFoundError:
                print(f"{ctype}/{t}: no NPZs, skipped")
        if not curves:
            continue
        for chan in CHANNELS:
            loc = _legend_loc(chan, ctype, a.norm)
            out = plot_real_panel(curves, chan, os.path.join(a.out, f"{chan}_{ctype}{suffix}.png"),
                                  norm=a.norm, total_spectral=a.total_spectral,
                                  legend=(loc != "off"), legend_loc=(None if loc == "off" else loc),
                                  contact_legend_loc=_contact_legend_loc(ctype, a.norm))
            print(out, " ".join(f"{c.ntype}:n={c.meta['n_files']},rho_w={c.meta['rho_w_emp']:.4f},"
                                f"ell*={c.meta['ell_star']:.4f}" for c in curves),
                  f"[{curves[0].meta['m2_key']}]")


if __name__ == "__main__":
    main()
