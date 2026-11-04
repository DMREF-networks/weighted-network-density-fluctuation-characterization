# Schematic figures (Figs. 1, 6, 7)

Self-contained renders on a fixed seed (16), N = 100, L = 10; no data files.

    # Fig. 1: base-tessellation windows, Poisson + URL (a = 1), C/D/G/V
    RUN_PROFILES=clean,cleanurl RUN_PANELS=tess CLEAN_WIN_FRAC=0.40 CLEAN_VIEW_FACTOR=0.78 \
      CLEAN_LW_SCALE=2.5 CLEAN_BLOCK_DOT=80 python figures/viz/anatomy_kite_2d.py
    # (those values are the script's defaults, so a bare
    #  `python figures/viz/anatomy_kite_2d.py` renders the same 8 panels)
    # Fig. 6: kite partition of a centroidal cell with the barycentric dual
    python figures/viz/kite_simplex_polytope_2d_C.py --no-labels --dual mid --close-cells
    # Fig. 7: midpoint-sector (T) and vertex-Voronoi (V) weight schematics
    python figures/viz/midptsec_cells_2d.py
    python figures/viz/rholineinv_cells_2d.py

Outputs land under `figures/viz/outputs/` (override with `VIZ_OUT`).
