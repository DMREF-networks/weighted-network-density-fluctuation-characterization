"""Fig. 14 boiled-down glass row + Fig. 15 contact panel: the 2D ideal-glass
CONTACT network only (ntype D on the authors' contact list == the radical
Delaunay), measured (a) under the three cell-measure schemes and (b) at
theta = 1.

    python figures/fig14_contact.py --data <root> --out <dir> [--ntype V]
    python figures/fig14_contact.py --data ../outputs/glass_contact --out plots/contact1

<root>/<ntype>_packing_<weight>/*.npz (the glass_contact.sl layout); writes
contact_schemes.png (one curve per cell-measure scheme; the power kite P joins
when its directory exists), contact_th1.png (the single theta = 1 curve for the
Fig. 15 panel) and contact_th0.png (theta = 0: equal weight per contact,
w = 1/ell).  --ntype V reads the power-Voronoi runs and writes power_*.png.
A panel whose data directory is absent is skipped with a notice.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chi_panels                                                        # noqa: E402
from chi_panels import DASH_LONG, DASH_SHORT, RELEASE_STYLE, Curve, RefLine, plot_chi_panel, scaled_axes  # noqa: E402

# scheme -> weight dir token (all on ntype D: the contact network is the measured network)
WEIGHT = {"V": "rho_line_inv", "T": "midpt_sec_area", "D": "dual_diamond_planar", "P": "power_kite"}
LABEL = {"V": r"$\mathcal{V}$", "T": r"$\mathcal{T}$", "D": r"$\mathcal{D}$", "P": r"$\mathcal{P}$"}
# placeholder set, deliberately distinct from the tessellation colors (final set TBD)
SCHEME_COLORS = {"Vs": "#2A9D8F", "Ts": "#E76F51", "Ds": "#264653", "Ps": "#E9C46A"}
chi_panels.TESS_COLORS.update(SCHEME_COLORS)     # extra keys only; the C/D/G/V entries are untouched

DASH_DOT = (0, (0.8, 1.4))
SCHEMES_STYLE = dict(RELEASE_STYLE, legend_fs=11, ystride=2)   # the paper's schemes panel: 11 pt legend and annotation
XLIM = (10 ** -2, 10 ** -0.03)
YLIM = (10 ** -7.5, 10 ** 0.6)


NAME = {"D": ("contact", r"Contact network"), "V": ("power", r"Power Voronoi")}   # stem, annotation


def _curve(data: str, weight_tag: str, label: str, color_key: str, ntype: str = "D") -> Curve:
    """Each packing carries its own sheared cell, hence its own k grid: scale every
    realization by its own metadata (norms.axes), then average on the first file's
    scaled grid over the shared range (linear interpolation in log x)."""
    files = sorted(glob.glob(os.path.join(data, f"{ntype}_packing_{weight_tag}", "*.npz")))
    if not files:
        raise FileNotFoundError(os.path.join(data, f"{ntype}_packing_{weight_tag}", "*.npz"))
    xs, ys, rho_w = [], [], []
    for f in files:
        with np.load(f, allow_pickle=False) as d:
            meta = json.loads(str(d["metadata_json"]))
            x, y = scaled_axes(d["k"], d["chi_ell"], meta)
            m = (x > 0) & np.isfinite(y)
            xs.append(x[m]); ys.append(y[m]); rho_w.append(meta.get("rho_w_emp"))
    lo = max(x[0] for x in xs)
    hi = min(x[-1] for x in xs)
    xg = xs[0][(xs[0] >= lo) & (xs[0] <= hi)]
    yg = np.mean([np.interp(np.log(xg), np.log(x), y) for x, y in zip(xs, ys)], axis=0)
    return Curve(label=label, ntype=color_key, x=xg, y=yg,
                 meta={"n_files": len(files), "rho_w_emp": float(np.mean([r for r in rho_w if r is not None]))})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ntype", default="D", choices=("D", "V", "C", "G"),
                    help="network derived from the contact list (D = the contacts, V = the power Voronoi)")
    a = ap.parse_args(argv)
    nt = a.ntype
    stem, annot = NAME.get(nt, (nt.lower(), f"{nt} network"))

    # (a) the cell-measure schemes on the network
    # RefLine(2, (0.001, 0.53), 0.63, DASH_SHORT, r"$k^{2}$"),
    #  3: (0, (4.0, 1.4, 1.0, 1.4)),        # dash-dotted
    #     4: (0, (5.0, 2.0)),
    try:
        curves = [_curve(a.data, WEIGHT[s], LABEL[s], s + "s", nt) for s in ("V", "T", "D")]
        if glob.glob(os.path.join(a.data, f"{nt}_packing_power_kite", "*.npz")):   # P: optional fourth curve
            curves.append(_curve(a.data, WEIGHT["P"], LABEL["P"], "Ps", nt))
        refs = [RefLine(3, (0.001, 0.53), 0.63, (0, (4.0, 1.4, 1.0, 1.4)), r"$k^{3}$"),
        RefLine(4, (0.001, 0.53), 0.29, (0, (5.0, 2.0)), r"$k^{4}$")]
        out = plot_chi_panel(curves, os.path.join(a.out, f"{stem}_schemes.png"), refs=refs,
                             annot=annot, annot_loc=(0.1, 0.90), xlim=XLIM, ylim=YLIM,
                             legend_loc="lower right", legend_bbox=None, style=SCHEMES_STYLE)
        print(out, " ".join(f"{c.label}:n={c.meta['n_files']},rho_w={c.meta['rho_w_emp']:.4f}" for c in curves))
    except FileNotFoundError as e:
        print(f"skipped {stem}_schemes: no data ({e})")

    # (b, c) theta = 1 (Fig. 15 first panel) and theta = 0 (equal weight per edge)
    # refs = [RefLine(1, (0.001, 0.53), 0.55, DASH_SHORT, r"$k^{1}$"),
    #     RefLine(2, (0.001, 0.56), 0.35, DASH_LONG, r"$k^{2}$")]
    for th_tag, th_label in (("th1", r"$\theta=1$"), ("th0", r"$\theta=0$")):
        try:
            c = _curve(a.data, f"length_power_{th_tag}", annot, "Ds", nt)
            refs = []
            out = plot_chi_panel([c], os.path.join(a.out, f"{stem}_{th_tag}.png"), refs=refs,
                                 annot=None, annot_loc=(0.80, 0.10),
                                 xlim=XLIM, ylim=(10 ** -4.5, 10 ** 0.9),
                                 legend_loc="lower left", legend_bbox=None, legend_ncol=1)
            print(out, f"{th_tag}:n={c.meta['n_files']},rho_w={c.meta['rho_w_emp']:.4f}")
        except FileNotFoundError as e:
            print(f"skipped {stem}_{th_tag}: no data ({e})")


if __name__ == "__main__":
    main()
