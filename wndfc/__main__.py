"""Command line: points -> network -> w_eff -> chi(k) [+ sigma^2(R), h_w(r)] -> NPZ.

    python -m wndfc --poisson 10000 -L 100 --ntype D --weight length_power --theta 1 \\
        --kmax 30 --with-direct-sigma2 --save-h-ell --out run.npz

Gauge: bare 1/V (see spectral.py); metadata schema: norms.py.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from .generators import hip_2d, hip_m_for_target_n, poisson_2d, poisson_3d, url_2d, url_3d
from .io import save_result
from .loaders import load_edges, load_lattice, load_packing, load_points_2d, load_points_3d
from .pbc import Cell, cell_volume, nominal_side, safe_radius
from .norms import build_metadata
from .pipeline import TilingError, build_weighted_network
from .spectral import compute_chi_ell
from .spectral_3d import compute_chi_ell_3d
from .weight_names import SCHEME, WEIGHTS, dir_tag, resolve_weight

CODE_VERSION = "release"


def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="wndfc",
                                description="Weighted-network spectral density chi(k), 2D or 3D, periodic.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--load-file", type=str, default=None, help="point file (whitespace/CSV or .npy)")
    src.add_argument("--load-packing", type=str, default=None, metavar="DIR",
                     help="disk-packing directory (positions.dat, radii.dat, latticeVectors.dat; "
                          "the authors' <name>edges.npy next to the library is picked up)")
    src.add_argument("--poisson", type=int, default=None, metavar="N", help="N uniform points in [0, L)^d")
    src.add_argument("--url", type=int, default=None, metavar="N_SIDE",
                     help="N_SIDE^d lattice with uniform shifts in [-a/2, a/2)^d (a from --url-a)")
    src.add_argument("--hip", type=int, default=None, metavar="N_TARGET",
                     help="2D hyperplane-intersection process with E[N] >= N_TARGET (M from --hip-m)")
    p.add_argument("--url-a", type=float, default=0.0, help="URL displacement amplitude (default 0)")
    p.add_argument("--hip-m", type=int, default=None, metavar="M", help="HIP line count (overrides --hip)")
    p.add_argument("--dim", type=int, default=2, choices=[2, 3])
    p.add_argument("-L", "--box-size", type=float, default=None, help="periodic box side")
    p.add_argument("--lattice-file", type=str, default=None,
                   help="2 x 2 lattice matrix (rows = lattice vectors) for a general periodic cell (2D)")
    p.add_argument("--edges-file", type=str, default=None,
                   help="edge list (i j per line, or .npy) = the primal triangulation; implies --network edges")
    p.add_argument("--network", type=str, default=None, choices=["tessellation", "edges"],
                   help="primal network source (default: edges when an edge list is given, else the voro++ "
                        "tessellation of the points)")
    p.add_argument("--contact-tol", type=float, default=1e-3,
                   help="diagnostic only: relative gap |r_ij|/(R_i+R_j) - 1 above which a triangulation "
                        "edge is reported as a non-contact")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (generation and window sampling)")
    # tessellation
    p.add_argument("--ntype", type=str, default="D", choices=["D", "V", "G", "C"])
    p.add_argument("--c-area-mode", type=str, default="sec_vor", choices=["sec_vor", "triangle"],
                   help="C: per-vertex cell area from a secondary Voronoi (sec_vor) or the Delaunay simplex")
    # weights
    p.add_argument("--weight", type=resolve_weight, default="length_power", metavar="W",
                   help=f"one of {' '.join(WEIGHTS)}, or the paper's symbol V T D D2 (MCV MCT MCD MCD2); "
                        "theta/zeta = length_power")
    p.add_argument("--theta", "--zeta", dest="theta", type=float, default=1.0,
                   help="length_power exponent: w_eff = ell^theta (the paper's zeta)")
    # spectral
    p.add_argument("--kmax", type=float, default=None, help="default 200 (2D) / 20 (3D)")
    p.add_argument("--bin-spacing", type=float, default=None, help="radial bin width (default 2 pi / L)")
    p.add_argument("--shell-mode", type=str, default="width", choices=["width", "exact"])
    p.add_argument("--chunk-size", "--cpu-chunk-size", dest="chunk_size", type=int, default=None,
                   help="modes per batch (numpy: default 1024 in 2D / 512 in 3D; cupy: default and "
                        "ceiling set by free device memory; numba: unused)")
    p.add_argument("--chi-backend", type=str, default="numpy", choices=["numpy", "numba", "cupy"],
                   help="numpy: vectorized reference; numba: parallel CPU; cupy: optional CUDA GPU")
    # direct variance
    p.add_argument("--with-direct-sigma2", action="store_true", default=False)
    p.add_argument("--sigma2-n-windows", type=int, default=10000)
    p.add_argument("--sigma2-n-R", type=int, default=50)
    p.add_argument("--sigma2-r-min", type=float, default=None, help="default L/200")
    p.add_argument("--sigma2-r-max", type=float, default=None, help="default L/4 (<= L/2)")
    # real-space pair deposits (self + distinct channels of h_w(r)), 2D
    p.add_argument("--save-h-ell", action="store_true", default=False,
                   help="sampled pair deposits on [0, r_max] (h_ell.sample_pair_deposits); curves, "
                        "moments and the sum rule are read-time arithmetic on the stored deposits")
    p.add_argument("--h-ell-r-max", type=float, default=None, help="default L/4")
    p.add_argument("--h-ell-n-bins", type=int, default=200)
    p.add_argument("--h-ell-n-samples", type=int, default=256,
                   help="independent uniform (s, t) draws per edge pair (self and distinct alike)")
    # output
    p.add_argument("--out", type=str, default=None, help="NPZ path")
    p.add_argument("--out-root", type=str, default=None,
                   help="write <root>/<T>_<ctype>_<weight>/<same>_seed<S>.npz")
    p.add_argument("--save-weights", action="store_true", default=False)
    p.add_argument("--save-edge-data", action="store_true", default=False)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.chi_backend == "cupy":
        from .spectral_gpu import _import_cupy
        try:
            _import_cupy()
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    kmax = args.kmax if args.kmax is not None else (20.0 if args.dim == 3 else 200.0)

    # ---- points ---------------------------------------------------------
    edges = radii = None
    phi = None
    if args.load_packing is not None:
        if args.dim != 2:
            print("ERROR: --load-packing is 2D only", file=sys.stderr)
            return 2
        pk = load_packing(args.load_packing, edges=args.edges_file)
        points, L, radii, edges, phi = pk["points"], pk["cell"], pk["radii"], pk["edges"], pk["phi"]
        gen, source = "packing", os.path.abspath(args.load_packing)
    elif args.load_file is not None:
        loader = load_points_3d if args.dim == 3 else load_points_2d
        points, detected_L = loader(args.load_file, box_size=args.box_size)
        L = float(args.box_size) if args.box_size is not None else detected_L
        gen, source = "load", os.path.abspath(args.load_file)
        if args.lattice_file is not None:
            if args.dim != 2:
                print("ERROR: --lattice-file is 2D only", file=sys.stderr)
                return 2
            L = load_lattice(args.lattice_file)
        if args.edges_file is not None:
            edges = load_edges(args.edges_file)
    else:
        if args.box_size is None:
            print("ERROR: --box-size is required with a generator", file=sys.stderr)
            return 2
        L = float(args.box_size)
        if args.poisson is not None:
            points = (poisson_3d if args.dim == 3 else poisson_2d)(args.poisson, L, seed=args.seed)
            gen, source = "poi", f"poisson(N={args.poisson},seed={args.seed},dim={args.dim})"
        elif args.url is not None:
            points = (url_3d if args.dim == 3 else url_2d)(args.url, L, a=args.url_a, seed=args.seed)
            gen, source = "url", f"url(n_side={args.url},a={args.url_a},seed={args.seed},dim={args.dim})"
        else:
            if args.dim != 2:
                print("ERROR: --hip is 2D only", file=sys.stderr)
                return 2
            M = args.hip_m if args.hip_m is not None else hip_m_for_target_n(args.hip)
            points = hip_2d(M, L, seed=args.seed)
            gen, source = "hip", f"hip(M={M},L={L:.0f},seed={args.seed},dim=2)"
    if isinstance(L, Cell) and L.is_square:
        L = L.L
    general = isinstance(L, Cell)
    network_source = args.network
    if network_source is None:
        network_source = "edges" if edges is not None else "tessellation"
    if network_source == "edges" and edges is None:
        print("ERROR: --network edges needs --edges-file", file=sys.stderr)
        return 2
    if general and network_source == "tessellation":
        print("ERROR: a general (sheared) cell needs an edge list (--edges-file); voro++ tessellates "
              "orthogonal boxes only", file=sys.stderr)
        return 2
    cell_txt = f"cell=general V={L.V:.6g}" if general else f"L={L}"
    print(f"[wndfc] dim={args.dim} points={points.shape[0]} {cell_txt} ntype={args.ntype} "
          f"weight={args.weight} kmax={kmax} network={network_source}")

    # ---- network + weights (tiling and structural checks) -----------------
    t0 = time.time()
    weight_diag: dict = {}
    try:
        net, w, binfo = build_weighted_network(
            points, L, args.ntype, args.weight, dim=args.dim, theta=args.theta,
            c_area_mode=args.c_area_mode,
            diag_out=weight_diag, network_source=network_source, edges=edges, radii=radii)
    except TilingError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    V = cell_volume(L) if args.dim == 2 else L ** 3
    side = nominal_side(L) if args.dim == 2 else L
    r_safe = safe_radius(L) if args.dim == 2 else 0.5 * L
    print(f"  network: n_edges={net.n_edges} <ell>={net.mean_edge_length:.4f}  "
          f"({time.time() - t0:.2f}s)")
    print(f"  weights: {args.weight} sum(w_eff)/V={float(np.sum(w)) / V:.6f}")

    # ---- chi(k) ---------------------------------------------------------
    t2 = time.time()
    kw = {"chunk_size": args.chunk_size, "backend": args.chi_backend}
    if args.dim == 3:
        chi = compute_chi_ell_3d(net, w, kmax=kmax, bin_spacing=args.bin_spacing,
                                 shell_mode=args.shell_mode, **kw)
    else:
        chi = compute_chi_ell(net, w, kmax=kmax, bin_spacing=args.bin_spacing,
                              shell_mode=args.shell_mode, **kw)
    print(f"  chi: {chi.n_bins} shells ({args.shell_mode})  ({time.time() - t2:.2f}s)")

    # ---- direct sigma^2 + bridge ----------------------------------------
    sigma2_R = sigma2_direct = sigma2_spectral = None
    sigma2_meta = None
    if args.with_direct_sigma2:
        from .parseval import parseval_sigma2
        if args.dim == 3:
            from .variance_3d import compute_sigma2_3d as _compute_sigma2
        else:
            from .variance import compute_sigma2 as _compute_sigma2
        r_min = args.sigma2_r_min if args.sigma2_r_min is not None else side / 200.0
        r_max = args.sigma2_r_max if args.sigma2_r_max is not None else min(side / 4.0, r_safe)
        if r_max > r_safe + 1e-12:
            print(f"  ERROR: sigma2-r-max={r_max} exceeds the PBC-safe radius {r_safe}", file=sys.stderr)
            return 2
        sigma2_R = np.geomspace(r_min, r_max, args.sigma2_n_R)
        w_eff_arr = np.asarray(w, dtype=float)
        # the window observable is sum_e w_e * (clipped length): per-length marks w = w_eff / ell
        w_user = w_eff_arr / np.maximum(np.asarray(net.edges_lengths, dtype=float), 1e-30)
        t3 = time.time()
        sigma2_spectral = parseval_sigma2(chi, sigma2_R, args.dim)
        direct_res = _compute_sigma2(net, w_user, sigma2_R, n_windows=args.sigma2_n_windows, seed=args.seed)
        sigma2_direct = direct_res.sigma2
        mask = sigma2_direct > 0
        med_ratio = float(np.median(sigma2_spectral[mask] / sigma2_direct[mask])) if mask.any() else float("nan")
        print(f"  sigma2: n_windows={direct_res.n_windows} bridge/direct median={med_ratio:.4f}  "
              f"({time.time() - t3:.2f}s)")
        sigma2_meta = {"n_windows": int(direct_res.n_windows), "sigma2_seed": int(args.seed),
                       "r_min": float(r_min), "r_max": float(r_max), "n_R": int(args.sigma2_n_R),
                       "ratio_spec_over_direct_median": med_ratio}

    # ---- real-space pair deposits (2D) ----------------------------------
    h_ell_result = None
    if args.save_h_ell:
        if args.dim == 3:
            print("  WARN: h_ell is 2D only; skipped", file=sys.stderr)
        else:
            from .h_ell import real_space_diagnostics, sample_pair_deposits
            r_max_hell = args.h_ell_r_max if args.h_ell_r_max is not None else min(side / 4.0, r_safe)
            if r_max_hell > r_safe + 1e-12:
                print(f"  ERROR: h-ell-r-max={r_max_hell} exceeds the PBC-safe radius {r_safe}", file=sys.stderr)
                return 2
            t_h = time.time()
            w_arr = np.asarray(w, dtype=float)
            h_ell_result = sample_pair_deposits(net, w_arr, r_max=r_max_hell, n_bins=args.h_ell_n_bins,
                                                n_samples=args.h_ell_n_samples, seed=args.seed)
            dg = real_space_diagnostics(h_ell_result, V, float(np.sum(w_arr)) / V, float(np.sum(w_arr ** 2)))
            if h_ell_result.cutoff_capped:
                print(f"  WARN: h_ell pair cutoff capped at the PBC-safe radius {r_safe:.4g} "
                      f"(r_max + ell_max = {r_max_hell + float(np.max(net.edges_lengths)):.4g}); "
                      f"deposits near r_max are incomplete -- lower --h-ell-r-max", file=sys.stderr)
            print(f"  h_ell: n_bins={h_ell_result.n_bins} n_samples={h_ell_result.n_samples} "
                  f"pairs={h_ell_result.n_pairs_dist} M_0={dg['M_0']:+.4g}+-{dg['M_0_sd']:.2g} "
                  f"vs {dg['sum_rule_rhs']:+.4g} (tiling schemes close) h_w0={dg['h_w0']:+.3f}+-{dg['h_w0_sd']:.3f}  "
                  f"({time.time() - t_h:.2f}s)")

    # ---- metadata + output ----------------------------------------------
    scheme = SCHEME[args.weight]
    meta = build_metadata(net=net, w_eff=w, dim=args.dim, gen=gen, tess=args.ntype, scheme=scheme,
                          theta=(args.theta if args.weight == "length_power" else None),
                          seed=args.seed, ens=None, k_max=kmax, n_bins=chi.n_bins,
                          shell_mode=args.shell_mode, code_version=CODE_VERSION)
    meta["source"] = source
    meta["chi_backend"] = args.chi_backend
    if args.dim == 2:
        meta["c_area_mode"] = args.c_area_mode
        meta["network_source"] = network_source
        if phi is not None:
            meta["phi"] = float(phi)
        if network_source != "tessellation" and radii is not None:
            from .pipeline import contact_gap_stats
            prim = edges if edges is not None else binfo["build_diag"].get("primal_edges")
            if prim is not None:
                meta["contact_stats"] = contact_gap_stats(points, L, prim, radii, args.contact_tol)
    if "dual_diamond_planar" in weight_diag:
        meta["dual_diamond_planar_diag"] = weight_diag["dual_diamond_planar"]
    if "d2_ring" in weight_diag:
        meta["d2_ring_diag"] = weight_diag["d2_ring"]
    if getattr(net, "build_diag", None):
        bd = net.build_diag
        meta["c_build_diag" if bd.get("kind") == "C_symdel" else "v_build_diag"] = bd
    if sigma2_meta is not None:
        meta["sigma2"] = sigma2_meta

    if args.out is not None:
        out = args.out
    elif args.out_root is not None:
        tag = dir_tag(args.ntype, gen, args.weight, args.theta)
        out = os.path.join(args.out_root, tag, f"{tag}_seed{args.seed}.npz")
    else:
        out = f"chi_{args.ntype}_{args.weight}_seed{args.seed}.npz"
    save_result(out, chi=chi, net=net, w_eff=w, meta=meta, sigma2_R=sigma2_R,
                sigma2_direct=sigma2_direct, sigma2_spectral=sigma2_spectral,
                h_ell=h_ell_result, save_weights=args.save_weights,
                save_edge_data=args.save_edge_data)
    print(f"  saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
