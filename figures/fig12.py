"""Fig. 12 -- 2D chi(k) for the V / T / D weight schemes on Poisson and URL substrates.

    python figures/fig12.py --data <root> --out <dir>

<root> holds release-schema NPZs in subdirectories <T>_<ctype>_<weight>/ for
weight in {rho_line_inv, midpt_sec_area, dual_diamond_planar}, T in {C,D,G,V},
ctype in {poi, url}.  Writes vc_<ctype>_rho_line_inv.png, dual_2d_<ctype>_T.png,
dual_2d_<ctype>_D.png (+ .eps) into <out>.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chi_panels import PAPER_STYLE, curves_from_roots, guide, plot_chi_panel  # noqa: E402

SCHEMES = {  # scheme: (weight token, annotation, output stem)
    "V": ("rho_line_inv", r"$\mathcal{V}$", "vc_{c}_rho_line_inv"),
    "T": ("midpt_sec_area", r"$\mathcal{T}$", "dual_2d_{c}_T"),
    "D": ("dual_diamond_planar", r"$\mathcal{D}$", "dual_2d_{c}_D"),
}
# per panel: guides, legend layout, legend anchor
PANELS = {
    ("V", "poi"): ([guide(2, (0.001, 0.45), 0.57), guide(3, (0.001, 0.50), 0.35)], "rows", (0.99, 0.02)),
    ("V", "url"): ([guide(2, (0.001, 0.50), 0.53), guide(3, (0.001, 0.65), 0.30)], "refs_left", (0.99, 0.02)),
    ("T", "poi"): ([guide(3, (0.001, 0.53), 0.48), guide(4, (0.001, 0.62), 0.33)], "rows", (0.99, 0.04)),
    ("T", "url"): ([guide(3, (0.001, 0.50), 0.33), guide(4, (0.001, 0.55), 0.14)], "refs_left", (0.99, 0.01)),
    ("D", "poi"): ([guide(3, (0.001, 0.57), 0.50), guide(4, (0.001, 0.62), 0.32)], "refs_left", (0.99, 0.01)),
    ("D", "url"): ([guide(3, (0.001, 0.62), 0.44), guide(4, (0.001, 0.77), 0.28)], "refs_left", (0.99, 0.01)),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="root with <T>_<ctype>_<weight>/ subdirs (release NPZs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--schemes", nargs="*", default=list(SCHEMES))
    a = ap.parse_args(argv)
    for scheme in a.schemes:
        weight, annot, stem = SCHEMES[scheme]
        for ctype in ("poi", "url"):
            refs, layout, bbox = PANELS[(scheme, ctype)]
            roots = {nt: os.path.join(a.data, f"{nt}_{ctype}_{weight}", "*.npz") for nt in "CDGV"}
            curves = curves_from_roots(roots)
            out = plot_chi_panel(curves, os.path.join(a.out, stem.format(c=ctype) + ".png"), refs=refs,
                                 annot=annot, legend_layout=layout, legend_bbox=bbox, style=PAPER_STYLE)
            print(out, " ".join(f"{c.ntype}:n={c.meta['n_files']},rho_w={c.meta['rho_w_emp']:.4f}" for c in curves))


if __name__ == "__main__":
    main()
