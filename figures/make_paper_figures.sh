#!/usr/bin/env bash
# Render the paper's figures from the published data deposit.
#   DATA=/path/to/unzipped/deposit bash figures/make_paper_figures.sh            # all
#   DATA=... bash figures/make_paper_figures.sh fig09 fig10                        # some
# Outputs land in $OUT (default plots/paper/<figNN_tag>/, png + eps).
set -u
cd "$(dirname "$0")/.."
: "${DATA:?set DATA to the directory holding the unzipped fig*_ datasets}"
PY="${PY:-python}"
OUT="${OUT:-plots/paper}"
ONLY=" $* "
run() { local tag=$1; shift; [ "$ONLY" = "  " ] || [[ "$ONLY" == *" $tag "* ]] || return 0
        echo; echo "## $tag: $*"; "$@" || echo "!! $tag failed"; }

run fig01 env VIZ_OUT=$OUT/fig01_networks "$PY" figures/viz/anatomy_kite_2d.py
run fig02 "$PY" figures/fig4.py --data $DATA/fig02_realspace --out $OUT/fig02_realspace --norm selfnorm --total-spectral
run fig03 "$PY" figures/fig_highk_2d3d.py --data $DATA/fig03_highk_2d3d --out $OUT/fig03_highk_2d3d
run fig04 "$PY" figures/fig2.py --data $DATA/fig04_bridge --out $OUT/fig04_bridge
run fig05 "$PY" figures/fig3_163.py --data $DATA/fig05_smallr --out $OUT/fig05_smallr
run fig06 env VIZ_OUT=$OUT/fig06_kite "$PY" figures/viz/kite_simplex_polytope_2d_C.py --no-labels --dual mid --close-cells
run fig07 env VIZ_OUT=$OUT/fig07_schemes "$PY" figures/viz/midptsec_cells_2d.py
run fig07 env VIZ_OUT=$OUT/fig07_schemes "$PY" figures/viz/rholineinv_cells_2d.py
run fig08 "$PY" figures/fig8.py --data $DATA/fig08_theta --out $OUT/fig08_theta
run fig09 "$PY" figures/fig12.py --data $DATA/fig09_schemes2d --out $OUT/fig09_schemes2d
run fig10 "$PY" figures/fig13.py --data $DATA/fig10_schemes3d --out $OUT/fig10_schemes3d
run fig11 "$PY" figures/fig9.py --data $DATA/fig11_decomp --out $OUT/fig11_decomp --tess C --stems th0_poi th2_poi th0_url dual_bary_poi
run fig12 "$PY" figures/fig14_contact.py --data $DATA/fig12_13_contact --out $OUT/fig12_13_contact
run fig12 "$PY" figures/fig_loaded_th1.py --data $DATA/fig12_loaded_th1 --out $OUT/fig12_loaded_th1 --systems stealthy hip
run fig13 "$PY" figures/fig14.py --data $DATA/fig13_loaded_schemes --out $OUT/fig13_loaded_schemes --systems stealthy hip
run fig14 "$PY" figures/fig_d2_compare.py --data $DATA/fig14_d2 --out $OUT/fig14_d2
