# weighted-network-density-fluctuation-characterization

## About

This code generates network structures by computing the spatial tessellation of a point pattern or by loading a network configuration. It then applies a prescribed weighting scheme to the edges of the structure.

The code can be used to compute the orientationally averaged weighted network spectral density, the weighted edge length variance, or the weighted network autocovariance.

The structures are assumed to be in a periodic square or cubic box.

Companion code for the paper cited [below](#citation). Import and command name: `wndfc`.

## Data

The ensemble results used in the paper are available at [Google Drive](https://drive.google.com/drive/folders/1K3lpf-nONYxQ4OK8QV42Lvp_dYe_VJV8). <!-- TODO: move to Zenodo and replace with DOI -->

## Setup

**Local install:**
Requires Python 3.9–3.10 (the ceiling comes from `pyvoro==1.3.2`, whose sdist doesn't build on 3.11+; `wndfc` itself is fine on newer Pythons) and a C++ compiler.
```sh
git clone https://github.com/DMREF-networks/weighted-network-density-fluctuation-characterization.git
cd weighted-network-density-fluctuation-characterization
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[test,figures]"                # extras: pytest + matplotlib for figures/
python -m pytest tests -q                       # verifies the build (~1 min)
```

**Docker:**
Easiest route if you just want to run the code or reproduce the paper's results — pinned Python 3.10 and locked dependencies, no local build. Use the local install if you're developing.
```sh
docker build -t wndfc .                         # ~5 min; builds pyvoro
docker run --rm wndfc                           # no command → runs the test suite
# mount /out so results survive --rm
docker run --rm -v "$PWD/out:/out" wndfc wndfc \
    --poisson 1600 -L 40 --ntype D --weight length_power --theta 2 \
    --kmax 30 --out /out/run.npz
```

### Performance extras

**Cython streaming extension:** Auto-enabled when built; speeds up large $N$. Set `WNDFC_NO_STREAM=1` to disable.
```sh
pip install -e ".[streaming]"
python wndfc/streaming/setup.py build_ext --inplace
```
**CUDA GPU spectra (2D/3D):** The spectral sum is $O(\text{edges} \times \text{wavevectors})$ and dominates runtime at large $N$ or high `kmax`. The CuPy backend moves that sum onto the GPU.
```sh
pip install -e ".[gpu]"
```
Select with `--chi-backend cupy` or `backend="cupy"` (only the spectral sum runs on the GPU). The `[gpu]` extra targets CUDA 12.x; otherwise install a CuPy build matching your CUDA version. Float64 throughout, so consumer cards are slow.

## Usage

```sh
# Delaunay network on 1600 uniform i.i.d. points in [0, 40)^2, apex-based (𝒟) weights;
# computes \tilde{chi}(k) to k = 30 plus the direct weighted edge-length variance -> run.npz
wndfc --poisson 1600 -L 40 --ntype D --weight dual_diamond_planar \
    --kmax 30 --with-direct-sigma2 --out run.npz
```

```python
import numpy as np
import wndfc

# 1. Pipeline: points -> network -> kernel weights (w_eff) -> spectral density chi(k)
pts = wndfc.poisson_2d(1600, 40.0, seed=1)          # uniform i.i.d. on [0, 40)^2
net, w_eff, chi = wndfc.run_pipeline(pts, 40.0, ntype="D",
                                     weight="dual_diamond_planar", kmax=30.0)

# 2. Variance: direct windows vs. Parseval
R = np.geomspace(0.5, 8.0, 40)
w = w_eff / net.edges_lengths                       # per-length mark, not w_eff
direct = wndfc.compute_sigma2(net, w, R, seed=1)
spectral = wndfc.parseval_sigma2(chi, R, dim=2)

# 3. Save, load, rescaled plot axes
meta = wndfc.build_metadata(net=net, w_eff=w_eff, dim=2, gen="poi", tess="D", scheme="D",
                            theta=None, seed=1, ens=1, k_max=30.0, n_bins=chi.n_bins,
                            shell_mode="width", code_version="local")
wndfc.save_result("run.npz", chi=chi, net=net, w_eff=w_eff, meta=meta)
res = wndfc.load_result("run.npz")
x, y = wndfc.axes(res["metadata"], res["k"], res["chi_ell"])
```

---

## Python API Reference

All symbols below are exported at top level (`import wndfc`) except `wndfc.pbc.*`.

- **`w_eff` (kernel weight, `w * ell`):** returned by `compute_weights*`, `build_weighted_network`, `run_pipeline*`; passed to `compute_chi_ell*`, `chi_ell_modes`, `sample_pair_deposits`, `build_metadata`, `moment_sums`, `vertex_wedge_sum`, `save_result`.
- **`w` (per-length mark, `w_eff / ell`):** passed to `compute_sigma2*`, `contact_plateau`, and all `small_r` / `small_r_3d` functions (`None` = unweighted `w = 1`).
- **`L` (box/cell):** square/cubic side float `L`, or `pbc.Cell(A)` for general 2D periodic cells with row vectors `A` `(2, 2)`.

### Pipelines ([`pipeline.py`](wndfc/pipeline.py))

| Function | Returns | Description |
|---|---|---|
| `run_pipeline(points, L, ntype="D", weight="length_power", *, theta=1.0, kmax=200.0, shell_mode="width", backend="numpy")` / `run_pipeline_3d(..., kmax=20.0)` | `(net, w_eff, chi)` | End-to-end 2D / 3D pipeline (`Network2D`/`3D`, `(E,)` `w_eff`, `ChiEll2D`/`3D`). |
| `build_weighted_network(points, L, ntype, weight, *, dim=2, theta=1.0, c_area_mode="sec_vor", network_source="tessellation", edges=None, radii=None)` | `(net, w_eff, info)` | Checked network + weights. `info` keys: `tiling_resid`, `build_diag`. Raises `TilingError`. |

### Inputs ([`generators.py`](wndfc/generators.py), [`loaders.py`](wndfc/loaders.py))

| Function | Returns | Description |
|---|---|---|
| `poisson_2d(N, L, *, seed=None)` / `poisson_3d` | `ndarray (N, d)` | Uniform points in `[0, L)^d`. |
| `url_2d(n_side, L, *, a=0.0, seed=None)` / `url_3d` | `ndarray (N, d)` | `n_side^d` lattice with uniform shifts in `[-a/2, a/2)^d`. |
| `hip_2d(M, L, *, seed=None)` | `ndarray (N, 2)` | In-box intersections of `M` random lines. |
| `load_points_2d(path, *, box_size=None, delimiter=None)` / `load_points_3d` | `(points, L)` | Coordinates from text/CSV/`.npy`; infers `L = ceil(max)` if omitted. |
| `load_lattice(path)` | `Cell` | 2D periodic `Cell` from a `(2, 2)` lattice matrix file. |
| `load_radii(path)` | `ndarray (N,)` | Per-point radii for power/radical constructions. |
| `load_edges(path)` | `ndarray (E, 2)` | 0-based integer edge index pairs. |
| `load_packing(directory, *, edges=None, edges_search=True)` | `dict` | Keys: `points`, `cell`, `radii`, `edges`, `phi`, `name` (`radii`/`edges`/`phi` may be `None`). |

### Networks & Periodic Cells ([`network.py`](wndfc/network.py), [`network_3d.py`](wndfc/network_3d.py), [`pbc.py`](wndfc/pbc.py))

| Function / Class | Returns / Attributes | Description |
|---|---|---|
| `build_network(points, L, ntype, c_area_mode="sec_vor", compute_cell_areas=True)` / `build_network_3d(points, L, ntype, *, compute_cell_volumes=True)` | `Network2D` / `Network3D` | Periodic `D`, `V`, `G`, or `C` network by [voro++](https://github.com/chr1shr/voro) (Rycroft 2009)[^1] via [pyvoro](https://github.com/joe-jordan/pyvoro) or `wndfc`'s streaming interface. |
| `build_from_graph(points, cell, edges, ntype, *, radii=None)` | `Network2D` | `D`, `C`, `V`, or `G` network from a planar graph in any 2D `Cell` (`V` requires triangulation). |
| `voronoi_cells_general(points, cell, radii=None)` | `(areas, pairs)` | Voronoi/power cell areas `(N,)` and adjacent pairs `(E, 2)` in a general 2D `Cell`. |
| `Network2D` / `Network3D` | Dataclass | **Fields:** `points`, `generators`, `edges_v1`, `edges_v2` (minimum-imaged), `edges_lengths`, `edges_midpoints`, `edges_left_cell`, `edges_right_cell`, `cell_areas` / `cell_volumes`, `build_diag`.<br>**Properties:** `V`, `n_edges`, `n_cells`, `mean_edge_length`, `mean_edge_length_sq`, `rho_line`, `rho_b`, `coordination`, `edge_moment(theta)`. |
| `pbc.Cell(A)` / `pbc.Cell.square(L)` | `Cell` | 2D cell from `(2, 2)` row vectors. Members: `V`, `inradius`, `reciprocal()`, `to_fractional(x)`, `to_cartesian(f)`. |
| `pbc.min_image_delta(dx, L)` / `pbc.min_image_distance(a, b, L)` | `ndarray` | Minimum-image displacements / periodic distances (`L` = float or `Cell`). |
| `pbc.wrap_to_box(p, L)` / `pbc.unwrap_relative(ref, target, L)` | `ndarray` | Wraps into primary cell / unwraps `target` relative to `ref`. |
| `pbc.periodic_pairs(points, L, r)` | `ndarray (P, 2)` | Point index pairs within periodic distance `r`. |
| `pbc.cell_volume(L)` / `pbc.nominal_side(L)` / `pbc.safe_radius(L)` | `float` | Cell measure `V`, nominal side `V^(1/d)`, and maximum unambiguous PBC radius. |

### Weights ([`weights.py`](wndfc/weights.py), [`weights_3d.py`](wndfc/weights_3d.py))

| Function / Constant | Returns | Description |
|---|---|---|
| `compute_weights(net, weight="length_power", *, theta=1.0, diag_out=None)` / `compute_weights_3d(net, weight="length_power", *, theta=1.0, diag_out=None)` | `ndarray (E,)` | Kernel weights `w_eff` (`diag_out` dict populated in-place if passed). |
| `WEIGHTS` / `SCHEME` | `tuple` / `dict` | Valid weight tokens / mapping from token to scheme code (`"theta"`, `"V"`, `"T"`, `"D"`, `"D2"`). |

| Token (`weight` / `--weight`) | Code | Support |
|---|---|---|
| `length_power` | `theta` | 2D & 3D, all networks (`w_eff = ell^theta`) |
| `rho_line_inv` | `V` | 2D & 3D, all networks |
| `midpt_sec_area` | `T` | 2D & 3D, all networks |
| `dual_diamond_planar` | `D` | 2D all networks; 3D `D`, `V`, `C` |
| `d2_ring` | `D2` | 2D `D` triangulation (square box) |

### Spectra & Variance ([`spectral.py`](wndfc/spectral.py), [`spectral_3d.py`](wndfc/spectral_3d.py), [`variance.py`](wndfc/variance.py), [`parseval.py`](wndfc/parseval.py))

| Function / Class | Returns / Attributes | Description |
|---|---|---|
| `compute_chi_ell(net, w_eff, *, kmax=200.0, bin_spacing=None, chunk_size=None, shell_mode="width", backend="numpy")` / `compute_chi_ell_3d(..., kmax=20.0)` | `ChiEll2D` / `ChiEll3D` | Shell-averaged spectral density for `\|k\| <= kmax` (`backend`: `"numpy"`, `"numba"`, `"cupy"`; `chunk_size`: modes per batch, numpy default 1024 / 512 in 3D, cupy default set by device memory). |
| `chi_ell_modes(net, w_eff, kvecs, *, chunk_size=1024)` | `ndarray (K,)` | Unnormalized complex amplitudes `A(k)` at wavevectors `kvecs` `(K, 2)`. |
| `compute_sigma2(net, w, R_grid=None, *, n_windows=10000, seed=None)` / `compute_sigma2_3d` | `Sigma2Result` / `Sigma2Result3D` | Window variance across random disks/spheres (`R_grid` defaults to 50 radii in `[L/200, L/4]`). |
| `parseval_sigma2(chi, R_grid, dim, values=None)` | `ndarray` | Spectral Parseval variance at `R_grid` (`values` selects total, `chi.chi_self`, or `chi.chi_distinct`). |
| `ChiEll2D` / `ChiEll3D` | Dataclass | **Fields:** `k`, `chi_ell`, `chi_self`, `counts`. **Properties:** `n_bins`, `chi_distinct`. |
| `Sigma2Result` / `Sigma2Result3D` | Dataclass | **Fields:** `R`, `sigma2`, `n_windows`. |

### Real-Space Correlations (2D; [`h_ell.py`](wndfc/h_ell.py))

| Function / Class | Returns / Attributes | Description |
|---|---|---|
| `sample_pair_deposits(net, w_eff=None, *, r_max=None, n_bins=200, n_samples=256, seed=0)` | `PairDeposits` | Monte-Carlo arclength-pair distance histogram split into self and distinct channels. |
| `real_space_curves(dep, V, rho_w)` | `RealSpaceCurves` | Normalized correlation curves. **Fields:** `r`, `f_self`, `h_dist`, `h_total`, `sd_self`, `sd_dist`. |
| `real_space_moments(dep, V, rho_w)` | `dict` | Bin-free radial moments `M_0`–`M_3` of `h_w(r)`. |
| `real_space_diagnostics(dep, V, rho_w, sum_w2l2, vertex_wedge_sum=None)` | `dict` | Moments, sum-rule/mass residuals, and contact plateau comparison. |
| `contact_plateau(net, w)` | `float` | Analytic distinct small-`r` contact limit `h_w(0+)`. |
| `spectral_h_w(k, chi, chi_self, rho_w, r, taper="hann")` | `ndarray` | `h_w(r)` via 2D Hankel inversion of `chi - chi_self`. |
| `hankel_inv_2d(k, f_k, r)` | `ndarray` | 2D inverse Hankel transform at `r`. |
| `PairDeposits` | Dataclass | **Fields:** `r_edges`, `dep_self`, `dep_dist`, `sq_self`, `sq_dist`, `mom_self`, `mom_dist`, sampling metadata (`cutoff_capped`). |

### Small-$R$ Analytic Predictions ([`small_r.py`](wndfc/small_r.py), [`small_r_3d.py`](wndfc/small_r_3d.py))

| Function | Returns | Description |
|---|---|---|
| `unified_prediction(net, R_grid, weights=None)` / `unified_prediction_3d` | `ndarray` | Exact all-`R` self variance (2D / 3D). |
| `weighted_smallR_leading(net, weights, R_grid)` / `weighted_smallR_leading_3d` | `ndarray` | Leading small-`R` variance (`O(R^3)` in 2D, `O(R^4)` in 3D). |
| `c4_exact(net, *, weights=None)` / `c5_exact` | `dict` | Exact 2D `C4` / 3D `C5` coefficients, leading/baseline coefficients, validity radius `R0`, and diagnostics. |
| `c4_r4_prediction(net, R_grid, c4=None, weights=None)` / `c5_r5_prediction(..., c5=None, ...)` | `ndarray` | Small-`R` polynomial variance expansion valid for `R <= R0` (2D / 3D). |
| `close_pair_distinct_sigma2(net, R_grid, *, weights=None, n_gauss=24)` / `close_pair_distinct_sigma2_3d` | `ndarray` | Close-pair distinct variance via Gauss–Legendre quadrature (2D / 3D). |
| `crossover_Rstar(net)` | `float` | 2D self-term crossover radius. |
| `g_kernel(alpha, order=0)` | `ndarray` | Single-edge window overlap kernel `G(alpha)` or derivative. |
| `g_d_form_factor(x, dim=2)` | `ndarray` | Orientation-averaged squared rod form factor. |
| `porod_moment(net, weights, kmax, *, dim=2)` | `float` | Finite-`kmax` Porod moment normalizer. |

### Diagnostics ([`small_r.py`](wndfc/small_r.py))

Used for the paper's analysis and figures; not needed for a measurement.

| Function | Returns | Description |
|---|---|---|
| `vertex_angle_sum(net, w)` | `tuple` | `(angle_sum, n_pairs, n_collinear, n_clamped, r0_adjacent)`; the contact plateau `chi_d(0) = angle_sum / (pi V) - rho_w^2`. |
| `H_infty(B, dim=2)` | `ndarray` | Vertex-adjacent pair kernel for direction cosine `B`. |
| `f_w_self_baseline(net, r_grid, weights=None)` | `ndarray` | Closed-form 2D self correlation `f_w(r)`. |

### Metadata, I/O ([`norms.py`](wndfc/norms.py), [`io.py`](wndfc/io.py))

| Function / Class | Returns | Description |
|---|---|---|
| `build_metadata(*, net, w_eff, dim, gen, tess, scheme, theta, seed, ens, k_max, n_bins, shell_mode, code_version, with_wedge=True, extra=None)` | `dict` | Run metadata dictionary. |
| `moment_sums(edge_lengths, w_eff, k_max, dim)` | `dict` | `sum_wl`, `sum_w2l`, `sum_w2l2`, `sum_w2l3`, `sum_w2l_kmax`. |
| `vertex_wedge_sum(net, w_eff)` | `float` | Weighted vertex wedge sum scalar. |
| `derived(meta)` | `dict` | Read-time scalars: `lambda_ell`, `rho_w`, `mean_wl`, `mean_w2l`, `mean_w2l2`, `mean_w2l3`, `mean_w2l_kmax`, `C_d`, `K_d`, `chi_self_0`, `porod_tail`. |
| `axes(meta, k, chi)` | `(ndarray, ndarray)` | Dimensionless rescaled `(x, y)` plot coordinates. |
| `save_result(path, *, chi, net, w_eff, meta, sigma2_R=None, sigma2_direct=None, sigma2_spectral=None, h_ell=None, save_weights=False, save_edge_data=False)` | `Path` | Writes run `.npz`. |
| `load_result(path)` | `dict` | Arrays + decoded `"metadata"` dict. |

---

## CLI Reference

```sh
wndfc SOURCE [OPTIONS]
python -m wndfc SOURCE [OPTIONS]
```
Exit codes: `0` success, `2` invalid arguments, `3` tiling/structural check failed.

### Source (pick one)

| Flag | Input |
|---|---|
| `--poisson N` | `N` uniform points in `[0, L)^d`. |
| `--url N_SIDE` | `N_SIDE^d` perturbed lattice (`--url-a FLOAT` shift amplitude, default `0`). |
| `--hip N_TARGET` | 2D hyperplane intersections with `E[N] >= N_TARGET` (`--hip-m M` sets the exact line count; `wndfc.hip_m_for_target_n` picks `M` for a target `N`). |
| `--load-file FILE` | Point coordinates from text/CSV or `.npy`. |
| `--load-packing DIR` | 2D packing directory (`positions.dat`, `latticeVectors.dat`/`boxSize.dat`, optional radii/scalars/edges). |

### Options

| Flag | Default | Description |
|---|---|---|
| `--dim {2,3}` | `2` | Spatial dimension. |
| `-L, --box-size FLOAT` | — | Box side `L` (required unless lattice/packing file is given). |
| `--lattice-file FILE` | — | `(2, 2)` lattice matrix file (2D general cell). |
| `--edges-file FILE` | — | 0-based `i j` primal edge list. |
| `--network {tessellation,edges}` | auto | Primal construction (`edges` are 2D only). |
| `--ntype {D,V,G,C}` | `D` | Delaunay (`D`), Voronoi (`V`), Gabriel (`G`), or Centroidal (`C`). |
| `--c-area-mode {sec_vor,triangle}` | `sec_vor` | 2D `C`-network cell area mode. |
| `--contact-tol FLOAT` | `0.001` | Relative contact-gap diagnostic threshold. |
| `--seed INT` | `0` | RNG seed for generators and window sampling. |
| `--weight TOKEN` | `length_power` | Weight token (see table above). |
| `--theta FLOAT` | `1.0` | Exponent for `length_power`. |
| `--kmax FLOAT` | `200` (2D) / `20` (3D) | Wavenumber cutoff `\|k\| <= kmax`. |
| `--bin-spacing FLOAT` | `2*pi / L` | Radial shell width. |
| `--shell-mode {width,exact}` | `width` | Fixed-width bins or exact reciprocal lattice shells. |
| `--chi-backend {numpy,numba,cupy}` | `numpy` | Spectral summation backend; CuPy is optional. |
| `--chunk-size INT` | numpy: `1024` (2D) / `512` (3D); cupy: device memory | Wavevector batch size (cupy: also capped by device memory). |
| `--with-direct-sigma2` | `False` | Compute window-sampled and Parseval `sigma2(R)`. |
| `--sigma2-n-windows INT` | `10000` | Random windows per radius. |
| `--sigma2-n-R INT` | `50` | Log-spaced radii count. |
| `--sigma2-r-min`, `--sigma2-r-max` | `L/200`, `L/4` | Radius range (capped at PBC safe radius). |
| `--save-h-ell` | `False` | Sample 2D pair deposits on `[0, r_max]`. |
| `--h-ell-r-max FLOAT` | `L/4` | Max pair distance (capped at PBC safe radius). |
| `--h-ell-n-bins`, `--h-ell-n-samples` | `200`, `256` | Distance bins and arclength draws per pair. |
| `--out PATH` | `chi_<ntype>_<weight>_seed<S>.npz` | Output result `.npz` path (overrides `--out-root`). |
| `--out-root DIR` | — | Write to `DIR/<tag>/<tag>_seed<S>.npz`, `<tag>` = `<ntype>_<gen>_<weight>[_th<theta>]` (`wndfc.dir_tag`; the layout `figures/` and the data deposit use). |
| `--save-weights`, `--save-edge-data` | `False` | Include `w`/`ell` or full edge/cell arrays in result `.npz`. |

---

## Output File Schemas

### Result NPZ (`save_result` / `--out`)

| Keys | Condition | Description |
|---|---|---|
| `k`, `chi_ell`, `chi_self`, `counts` | always | `(B,)` shell wavenumbers, total/self spectra, and mode counts. |
| `metadata_json` | always | JSON string (decoded into `"metadata"` by `load_result`). |
| `w`, `ell` | `--save-weights` | `(E,)` kernel weights `w_eff` and edge lengths. |
| `edge_ell`, `edge_midpoints`, `edge_ell_dual`, `cell_areas`, `edges_left_cell`, `edges_right_cell` | `--save-edge-data` | Available network edge and cell arrays. |
| `sigma2_R`, `sigma2_direct`, `sigma2_spectral` | `--with-direct-sigma2` | `(R,)` window radii, direct window variance, and Parseval variance. |
| `h_ell_r_edges`, `h_ell_self`, `h_ell_dist`, `h_ell_self_sq`, `h_ell_dist_sq`, `h_ell_self_mom`, `h_ell_dist_mom` | `--save-h-ell` | Bin edges, self/distinct deposit sums, squared sums, and radial moments (`n = 0..3`). |

**`metadata` keys:**
- **Identity & grid:** `gauge`, `d`, `L`, `V`, `N_ell`, `gen`, `tess`, `scheme`, `theta`, `seed`, `ens`, `code_version`, `cell`, `dk`, `k_max`, `n_bins`, `shell_mode`
- **Sums & diagnostics:** `sum_wl`, `sum_w2l`, `sum_w2l2`, `sum_w2l3`, `sum_w2l_kmax`, `vertex_wedge_sum`, `rho_w_emp`, `source`, `chi_backend`, `network_source`, `c_area_mode`, `phi`, `contact_stats`, `sigma2`, `h_ell`

## Citation

If you use this code, please cite:

> Copeland and Maher, "Quantification of density fluctuations in weighted networks across length scales and generation of hyperuniform network structures," in preparation (2026).

```bibtex
@unpublished{copeland2026weighted,
  title  = {Quantification of density fluctuations in weighted networks across length scales and generation of hyperuniform network structures},
  author = {Copeland, Leon and Maher, Charles},
  year   = {2026},
  note   = {In preparation}
}
```

[^1]: C. H. Rycroft, "VORO++: A three-dimensional Voronoi cell library in C++," Chaos 19, 041111 (2009). doi:10.1063/1.3215722
