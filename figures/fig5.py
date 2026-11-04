"""Fig. 5 -- large-wavenumber decay of chi(k) (Porod regime), 2D and 3D.

    python figures/fig5.py --data <root> --out <dir>

2D panel: unweighted (theta = 1) C/D/G/V on Poisson, kmax = 600.
3D panel: V-scheme (rho_line_inv) C/D/G/V on Poisson (D at N = 6/side, kmax = 300).
On these axes x*y -> 1/pi at large k in 2D and 3D.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chi_panels import curves_from_roots, plot_chi_panel  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    c2 = curves_from_roots({nt: os.path.join(a.data, f"{nt}_poi_length_power_th1p0", "*.npz") for nt in "CDGV"})
    print(plot_chi_panel(c2, os.path.join(a.out, "2d_k600.png"), dim=2, xlim=(0.01, 70.0), ylim=(1e-3, 100.0),
                         legend_loc="upper right", legend_bbox=None))
    c3 = curves_from_roots({nt: os.path.join(a.data, f"{nt}_poi_rho_line_inv", "*.npz") for nt in "CDGV"})
    print(plot_chi_panel(c3, os.path.join(a.out, "3d_hk.png"), dim=3, xlim=(0.4, 8.0), ylim=(1e-2, 10.0),
                         legend_loc="lower right", legend_bbox=(0.99, 0.04)))


if __name__ == "__main__":
    main()
