"""Fig. 13 -- 3D chi(k) for the V / T / D weight schemes on Poisson and URL substrates.

    python figures/fig13.py --data <root> --out <dir> [--schemes T D V]

<root> holds release-schema NPZs in <T>_<ctype>_<weight>/ subdirectories.
V panels require vertex-Voronoi rho_line_inv data in 3D (the release definition);
they are skipped when absent.  D-scheme panels plot D, V, C (no Gabriel: the
barycentric dual is undefined for G in 3D).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chi_panels import PAPER_STYLE, curves_from_roots, guide, plot_chi_panel  # noqa: E402

XLIM = (10 ** -2.1, 10 ** -0.2)
YLIM = (10 ** -7.2, 10 ** 1.3)

SCHEMES = {  # scheme: (weight token, annotation, output stem, tessellations, line width)
    "V": ("rho_line_inv", r"$\mathcal{V}$", "3d_{c}_rli_DVGC", "CDGV", 2.8),
    "T": ("midpt_sec_area", r"$\mathcal{T}$", "3d_{c}_midpt_DVGC", "CDGV", 2.8),
    "D": ("dual_diamond_planar", r"$\mathcal{D}$", "dual_3d_{c}_fcom", "CDV", 3.0),
}
# per panel: guides, legend columns, legend layout, legend anchor (None -> the axes corner)
PANELS = {
    ("V", "poi"): ([guide(2, (0.001, 0.53), 0.68), guide(4, (0.001, 0.55), 0.40)], 2, "rows", None),
    ("V", "url"): ([guide(2, (0.001, 0.30), 0.41), guide(3, (0.001, 0.55), 0.24)], 2, "rows", None),
    ("T", "poi"): ([guide(3, (0.001, 0.45), 0.52), guide(4, (0.001, 0.45), 0.33)], 2, "rows", (0.99, 0.04)),
    ("T", "url"): ([guide(3, (0.001, 0.40), 0.35), guide(4, (0.001, 0.44), 0.17)], 1, "refs_left", (0.99, 0.04)),
    ("D", "poi"): ([guide(3, (0.001, 0.45), 0.50), guide(4, (0.001, 0.45), 0.34)], 1, "rows", (0.96, 0.04)),
    ("D", "url"): ([guide(3, (0.001, 0.50), 0.43), guide(4, (0.001, 0.53), 0.27)], 1, "rows", (0.96, 0.04)),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--schemes", nargs="*", default=["T", "D", "V"])
    a = ap.parse_args(argv)
    for scheme in a.schemes:
        weight, annot, stem, ntypes, lw = SCHEMES[scheme]
        for ctype in ("poi", "url"):
            refs, ncol, layout, bbox = PANELS[(scheme, ctype)]
            roots = {nt: os.path.join(a.data, f"{nt}_{ctype}_{weight}", "*.npz") for nt in ntypes}
            if not all(os.path.isdir(os.path.dirname(p)) for p in roots.values()):
                print(f"skip {scheme}/{ctype}: data absent")
                continue
            curves = curves_from_roots(roots, ntypes=tuple(ntypes))
            out = plot_chi_panel(curves, os.path.join(a.out, stem.format(c=ctype) + ".png"), dim=3,
                                 refs=refs, annot=annot, xlim=XLIM, ylim=YLIM, lw=lw,
                                 legend_ncol=ncol, legend_layout=layout, legend_bbox=bbox, style=PAPER_STYLE)
            print(out, " ".join(f"{c.ntype}:n={c.meta['n_files']},rho_w={c.meta['rho_w_emp']:.4f}" for c in curves))


if __name__ == "__main__":
    main()
