"""Figs. 2 and 10 -- scaled edge-length variance sigma^2(R): direct window sampling
(markers) vs the Parseval bridge (lines), theta = 1 (Fig. 2) and theta = 2 (Fig. 10),
Poisson and URL substrates, per tessellation.

    python figures/fig2.py --data <root> --out <dir>

<root>/<ctype>_th<theta>/<T>.npz, each carrying sigma2_R / sigma2_spectral and
sigma2_direct (own R grid in sigma2_R_direct when it differs).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_legend import STYLE as LEGEND_STYLE  # noqa: E402
from sigma2_panels import load_sigma2, plot_sigma2_panel  # noqa: E402

# the Direct | Spectral table in the paper's geometry
TABLE = dict(LEGEND_STYLE, line_span_frac=0.75, colsep_em=0.3, cellpad_em=0.05, headerpad_em=0.05,
             rowsep_em=0.257, borderpad=0.2, borderaxespad=1.15)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    for ctype in ("poi", "URL"):
        for th in (1, 2):
            files = sorted(glob.glob(os.path.join(a.data, f"{ctype}_th{th}", "*.npz")))
            if not files:
                continue
            curves = [load_sigma2(f) for f in files]
            out = plot_sigma2_panel(curves, os.path.join(a.out, f"{ctype}_{th}.png"),legend=(ctype == "poi" and th == 1), legend_style=TABLE)
            print(out, " ".join(f"{c.ntype}:N={c.meta['N_ell']}" for c in curves))


if __name__ == "__main__":
    main()
