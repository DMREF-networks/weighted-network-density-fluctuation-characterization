"""Real-space channels of the radial edge autocovariance, one panel per channel
with the four tessellations overlaid (lines = the analytic / spectral checks,
markers = the sampled pair deposits).

Stored per NPZ (release): the sampled pair deposits h_ell_self / h_ell_dist on
h_ell_r_edges (h_ell.PairDeposits) plus k, chi_ell, chi_self and the edge table
(edge_ell, optional w).  Read-time (h_ell.real_space_curves): self f_w = g_self,
distinct h_w = g_dist - 1, total.  Markers are the deposits; the self line is
the closed form lambda_ell <w^2 (ell - r)_+> / (pi r) (small_r.self_pair_moment,
needs the edge table and the weights) and the distinct line the Hankel
inversion of the stored distinct spectrum (h_ell.spectral_h_w).
Legacy NPZs (h_ell_r_grid + h_ell_direct / h_ell_spectral curves) are read as
they are: markers = the stored direct curve, lines = the stored spectral curve,
self markers = small_r.self_pair_direct on the edge table.

Five normalizations, selected per panel by ``norm``:

contact  (divisor lambda_ell <w^2 ell>; x = r lambda_ell^{1/2})
  self   H_s = pi r chi_s / (lam m2)            = <w^2 (ell-r)_+> / <w^2 ell>
  dist   h_w                                    (the stored array)
  total  H = H_s + pi r rho_w^2 h_w / (lam m2)  (sampled, drawn as the line; with
         ``total_spectral`` as markers against the Hankel inversion of the WHOLE
         bare spectrum / rho_w^2 = f_w + h_w, the same check as the dist line)
A legacy direct curve is undefined below r_min = 2 r_max / n_bins and is not
drawn there (the last grid point takes the last computed bin); the deposits are
defined on every bin.  The dist panel carries a tick at its left edge at the
analytic contact plateau h_w(0) = vertex_wedge_sum / (pi V rho_w^2) - 1 when
the metadata stores vertex_wedge_sum.

master   (the self channel's intrinsic scale ell* = <w^2 ell^2> / <w^2 ell>;
          x = r / ell* = r <w^2 ell> / <w^2 ell^2>)
  self   2 pi r chi_s / (lam <w^2 ell>) = ell* p_w(r) = 2 <w^2 (ell-r)_+> / <w^2 ell> = 2 H_s
         with p_w(r) = <w^2 2(ell-r)_+> / <w^2 ell^2> the unit-mass pair-length
         density, so every curve has y(0) = 2 and unit area in x, bracketed by
         the monodisperse 2(1-x)_+ and the exponential 2 exp(-2x).
  dist   h_w                                    (unchanged)
  total  2 H                                    (same factor as self, so the
                                                 channels still add)
<w^2 ell^2> comes from the stored sum when present, else from the edge table.
<w^2 ell> is the bare sum_w2l, or with ``porod`` the finite-k_max Porod moment
sum_w2l_kmax (needed at theta = 0, where the bare moment is shortest-edge noise).

rhow     (divisor rho_w^2 on every channel; x = r lambda_ell^{1/2})
  self   chi_s / rho_w^2 = f_w = lam <w^2 (ell-r)_+> / (pi r rho_w^2)
  dist   chi_d / rho_w^2 = h_w                  (unchanged)
  total  chi / rho_w^2   = f_w + h_w
The contact arrays are rescaled by lam <w^2 ell> / (pi r rho_w^2), with the
scalar prefactor taken as its ensemble mean (``rhow_fac``).

selfnorm (one divisor lambda_ell^{3/2} <w^2 ell> on every channel -- the self
          term's small-r scale; x = r lambda_ell^{1/2} = r~)
  self   chi_s / (lam^{3/2} m2) = H_s / (pi r~)                 -> 1/(pi r~) at small r~
  dist   chi_d / (lam^{3/2} m2) = H_d / (pi r~) = lam^{1/2} <w ell>^2 / m2 * h_w
                                                (not expected to collapse across systems)
  total  (H_s + H_d) / (pi r~)                  (= self + dist, the same factor)
The contact arrays are rescaled by 1/(pi r~), no per-channel constant; the
distinct channel is taken from its per-file contact form H_d, so the plotted
total is exactly self + dist.  The total uses the closed-form self in both
routes (direct: + sampled H_d; spectral: + Hankel distinct, as rhow), and the
dist panel's contact tick sits at rho_w^2 h_w(0) / (lam^{3/2} m2), per file.

contact_pir (contact, with the distinct channel in the self/total units;
          x = r lambda_ell^{1/2})
  self   H_s                                    (as contact)
  dist   H_d = pi r chi_d / (lam m2) = pi r rho_w^2 h_w / (lam m2)   -> 0 at r = 0
  total  H = H_s + H_d                          (as contact, = self + dist)
The distinct markers are the per-file contact form H_d, the line the Hankel
distinct times r / rhow_fac.  No contact tick (pi r times a finite plateau).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import matplotlib
import matplotlib.patheffects as pe
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

from chi_panels import LABEL_FS, LEGEND_FS, TESS_COLORS, TICK_FS, _RC, check_edge_count  # noqa: E402
from paired_legend import paired_table_legend  # noqa: E402
from sigma2_panels import MARK_OFFSET, TESS_MARKERS  # noqa: E402

_REL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
import sys  # noqa: E402
sys.path.insert(0, _REL)
from wndfc.h_ell import deposits_from_npz, real_space_curves, spectral_h_w  # noqa: E402
from wndfc.small_r import self_pair_direct, self_pair_moment  # noqa: E402

NORMS = ("contact", "master", "rhow", "selfnorm", "contact_pir")
XLABEL = {"contact": r"$r\,\lambda_\ell^{1/2}$",
          "master": r"$r\,\langle w^2\ell\rangle/\langle w^2\ell^2\rangle$",   # = r / ell*
          "rhow": r"$r\,\lambda_\ell^{1/2}$",
          "selfnorm": r"$r\,\lambda_\ell^{1/2}$",
          "contact_pir": r"$r\,\lambda_\ell^{1/2}$"}

SELF_N_PAIRS = 2048      # sampled same-edge pairs per edge (self_pair_direct)
SELF_SEED = 0

# Paired-legend column headers per channel: (markers, line).  Markers are always the
# sampled deposits; the line is that channel's check (closed-form self, inverted
# distinct spectrum, inverted whole spectrum for the total with ``total_spectral``).
LEGEND_COLS = {"chisnorm": ("Direct", "Spectral"),
               "chid": ("Direct", "Spectral"),
               "hnormd": ("Direct", "Spectral")}

# Separate legend for the analytic contact value of the distinct channel (the
# ticks at the left edge of the chid panel): one white, black-bordered sample in
# the ticks' line style.  ``loc=None`` switches it off.
CONTACT_LEGEND = dict(label=r"$\chi^{(d)}_{_{\ell,w}}(0)/\rho_w^2$", loc="upper left")
# Per-norm override of that label (norms not listed use CONTACT_LEGEND["label"]).
CONTACT_LEGEND_LABEL = {"selfnorm": r"$\chi^{(d)}_{_{\ell,w}}(0)/(\lambda_\ell^{3/2}\langle w^2\ell\rangle)$"}


def _self_direct_cached(npz_path: str, r: np.ndarray, ell: np.ndarray,
                        w_pl: Optional[np.ndarray]) -> np.ndarray:
    """self_pair_direct for one NPZ, memoized in ``.cache/`` beside it.

    The entry is keyed on the NPZ size and mtime, the r grid and the sampling
    parameters, so a changed input or setting is recomputed, never reused.
    """
    st = os.stat(npz_path)
    key = "%d:%d:%d:%d:%s" % (st.st_size, st.st_mtime_ns, SELF_N_PAIRS, SELF_SEED,
                              hashlib.sha1(np.ascontiguousarray(r, dtype=np.float64).tobytes()).hexdigest()[:16])
    cdir = os.path.join(os.path.dirname(os.path.abspath(npz_path)), ".cache")
    cpath = os.path.join(cdir, os.path.splitext(os.path.basename(npz_path))[0] + ".selfdir.npz")
    if os.path.exists(cpath):
        with np.load(cpath, allow_pickle=False) as c:
            if str(c["key"]) == key:
                return c["moment"]
    m = self_pair_direct(r, ell, w_pl, n_pairs=SELF_N_PAIRS, seed=SELF_SEED)
    os.makedirs(cdir, exist_ok=True)
    np.savez(cpath, key=np.array(key), moment=m)
    return m


@dataclass
class RealCurves:
    ntype: str
    r: np.ndarray                       # r grid (bin centres)
    x: np.ndarray                       # r lambda_ell^{1/2}
    H_self: Optional[np.ndarray]        # closed form, drawn as the line (None without weights)
    H_self_dir: Optional[np.ndarray]    # sampled self deposits, drawn as markers (legacy: None without weights)
    h_spec: Optional[np.ndarray]        # spectral check, drawn as the line
    h_dir: Optional[np.ndarray]         # sampled distinct deposits (legacy: the GL curve), markers
    H_dist_dir: Optional[np.ndarray]   # pi r rho_w^2 h_w / (lam m2), per file, from the markers
    H_tot_spec: Optional[np.ndarray]   # pi r rho_w^2 (f_w + h_w) / (lam m2) from the inverted total chi
    meta: dict = field(default_factory=dict)


def _per_length_w(d, meta, ell):
    """Per-length marks from the stored kernel weights; ones for theta = 1;
    None when the scheme needs weights that were not stored."""
    if "w" in d.files:
        return d["w"] / ell
    if meta.get("scheme") == "theta" and float(meta.get("theta") or 0.0) == 1.0:
        return np.ones_like(ell)
    return None


def _read_one(f: str):
    """(r, meta, ell, w_pl, self_dep, self_closed, h_dist_markers, h_spec_line, h_tot_spec)
    for one NPZ.  h_tot_spec = Hankel inversion of the whole bare spectrum / rho_w^2
    = f_w + h_w (the total correlation without the background), band-limited."""
    with np.load(f, allow_pickle=False) as d:
        meta = json.loads(str(d["metadata_json"]))
        check_edge_count(meta, f)
        ell = d["edge_ell"] if "edge_ell" in d.files else None       # closed-form line + legacy only
        w_pl = None if ell is None else _per_length_w(d, meta, ell)
        lam = float(meta["N_ell"]) / float(meta["V"])
        rho_w = float(meta["rho_w_emp"])
        if "h_ell_self" in d.files:
            data = {k: d[k] for k in d.files if k.startswith("h_ell_")}
            data["metadata"] = meta
            dep = deposits_from_npz(data)
            c = real_space_curves(dep, float(meta["V"]), rho_w)
            r = c.r
            # sampled self as the moment <w^2 (ell - r)_+>:  f_w pi r rho_w^2 / lam
            self_dep = c.f_self * np.pi * r * rho_w ** 2 / lam
            h_dir = c.h_dist
            h_spec = spectral_h_w(d["k"], d["chi_ell"], d["chi_self"], rho_w, r) if "chi_self" in d.files else None
        else:
            r = d["h_ell_r_grid"]
            # the legacy sampled self needs the marks: unweighted markers against
            # a weighted normalization would be wrong, so none without weights
            self_dep = None if w_pl is None else _self_direct_cached(f, r, ell, w_pl)
            h_spec = d["h_ell_spectral"].copy() if "h_ell_spectral" in d.files else None
            h_dir = None
            if "h_ell_direct" in d.files:
                # a legacy direct curve is defined for 2 dr <= r < r_grid.max()
                h_dir = d["h_ell_direct"].astype(float)
                h_dir[(r < 2.0 * (r[1] - r[0])) | (r >= r[-1])] = np.nan
        self_closed = None if (w_pl is None or ell is None) else self_pair_moment(r, ell, w_pl)
        h_tot_spec = spectral_h_w(d["k"], d["chi_ell"], 0.0, rho_w, r)   # chi_self = 0: the whole spectrum
    return r, meta, ell, w_pl, self_dep, self_closed, h_dir, h_spec, h_tot_spec


def load_real(pattern: str, ntype: str, porod: bool = False) -> RealCurves:
    """Ensemble mean over release NPZs sharing an r grid.

    ``porod``: divide by the finite-k_max Porod moment <w^2 ell>_kmax (metadata
    ``sum_w2l_kmax``, the choice norms.axes makes) instead of the bare <w^2 ell>;
    the two agree for theta >= 1, and the bare moment is shortest-edge noise for
    theta = 0 on V/G/C.  Falls back to the bare moment where the key is absent.
    """
    Hs, Hsd, hs, hd, Hd, Ht, ells, h0s, r0, metas = [], [], [], [], [], [], [], [], None, []
    facs, h0sn = [], []
    for f in sorted(glob.glob(pattern)):
        r, meta, ell, w_pl, self_dep, self_closed, h_dir, h_spec, h_tot_spec = _read_one(f)
        if r0 is None:
            r0 = r.copy()
        elif not np.allclose(r, r0):
            raise ValueError(f"r grid mismatch: {f}")
        lam = float(meta["N_ell"]) / float(meta["V"])
        m2_key = "sum_w2l_kmax" if porod and meta.get("sum_w2l_kmax") is not None else "sum_w2l"
        m2 = float(meta[m2_key]) / float(meta["N_ell"])
        rho_w = float(meta["rho_w_emp"])
        # ell* = <w^2 ell^2> / <w^2 ell>; the second moment from the stored
        # sum when present, else from the edge table (per-length w).
        if meta.get("sum_w2l2") is not None:
            m2l2 = float(meta["sum_w2l2"]) / float(meta["N_ell"])
        elif ell is not None:
            w2 = np.ones_like(ell) if w_pl is None else w_pl ** 2
            m2l2 = float(np.mean(w2 * ell ** 2))
        else:
            raise ValueError(f"{f}: neither sum_w2l2 nor edge_ell -- cannot form ell*")
        ells.append(m2l2 / m2)
        facs.append(lam * m2 / (np.pi * rho_w ** 2))   # contact array * this / r = channel / rho_w^2
        Hs.append(None if self_closed is None else self_closed / m2)
        Hsd.append(None if self_dep is None else self_dep / m2)
        hs.append(h_spec)
        hd.append(h_dir)
        Hd.append(np.pi * r * rho_w ** 2 * h_dir / (lam * m2) if h_dir is not None else None)
        Ht.append(np.pi * r * rho_w ** 2 * h_tot_spec / (lam * m2))
        # Analytic contact plateau of the distinct channel from the vertex
        # table: h_w(0) = vertex_wedge_sum / (pi V rho_w^2) - 1, with
        # vertex_wedge_sum = sum_v sum_{a<b} w_a w_b (pi - gamma_ab)/sin gamma_ab.
        vws = meta.get("vertex_wedge_sum")
        h0s.append(None if vws is None else float(vws) / (np.pi * float(meta["V"]) * rho_w ** 2) - 1.0)
        # the same plateau in selfnorm units, per file as the distinct markers
        # (rho_w^2 h_w(0) / (lam m2); the ensemble lam^{1/2} is applied below)
        h0sn.append(None if h0s[-1] is None else rho_w ** 2 * h0s[-1] / (lam * m2))
        metas.append(meta)
    if not metas:
        raise FileNotFoundError(pattern)
    lam = float(np.mean([m["N_ell"] / m["V"] for m in metas]))
    mean = lambda L: (np.mean(L, axis=0) if all(v is not None for v in L) else None)  # noqa: E731
    meta = dict(metas[0]); meta["n_files"] = len(metas)
    for key in ("N_ell", "sum_wl", "sum_w2l", "rho_w_emp"):
        meta[key] = float(np.mean([m[key] for m in metas]))
    meta["ell_star"] = float(np.mean(ells))
    meta["rhow_fac"] = float(np.mean(facs))
    meta["m2_key"] = m2_key                        # which <w^2 ell> normalized the channels
    meta["h_w0"] = float(np.mean(h0s)) if all(v is not None for v in h0s) else None
    meta["h_w0_selfnorm"] = (float(np.mean(h0sn)) / np.sqrt(lam)
                             if all(v is not None for v in h0sn) else None)
    return RealCurves(ntype, r0, r0 * np.sqrt(lam), mean(Hs), mean(Hsd),
                      mean(hs), mean(hd), mean(Hd), mean(Ht), meta)


# Per channel and normalization: y label, axis windows, horizontal guides
# (y0, linestyle) and reference curves (f(x), linestyle) drawn in black.
# ``self_vline`` (self panel, rhow): linestyle of a per-tessellation vertical
# eye guide at r = lambda_ell <w^2 ell> / rho_w^2 (x = r lambda_ell^{1/2}), no legend entry.
CHANNELS = {
    "chisnorm": {
        "contact": dict(ylabel=r"$\chi^{(s)}_{_{\ell,w}}(r)\,\pi\,r\,/(\lambda_\ell\langle w^2\ell\rangle)$",
                        xlim=(10 ** -1.7, 10 ** 1.4), ylim=(0, 10), legend_loc="center right",
                        guides=(), curves=()),
        "master": dict(ylabel=r"$2\pi r\,\chi^{(s)}_{_{\ell,w}}(r)/(\lambda_\ell\langle w^2\ell\rangle)$",
                       xlim=(10 ** -1.7, 10 ** 1.4), ylim=(0, 2.1), legend_loc="center right",
                       guides=(),
                       curves=((lambda x: 2.0 * np.clip(1.0 - x, 0.0, None), "--"),
                               (lambda x: 2.0 * np.exp(-2.0 * x), ":")),
                       annot=(r"$\begin{aligned}\text{dashed:\,}& 2(1-x)_{+} \\ \text{dotted:\,}& 2\mathrm{e}^{-2x}\end{aligned}$")),
        "rhow": dict(ylabel=r"$\chi^{(s)}_{_{\ell,w}}(r)/\rho_w^2$",
                     xlim=(10 ** -1.7, 20), ylim=(-0.7, 9.5), legend_loc="center right",
                     guides=((0.0, "--"),), curves=(), self_vline=":"), # (0.0, "--"),
        "selfnorm": dict(ylabel=r"$\chi^{(s)}_{_{\ell,w}}(r)/(\lambda_\ell^{3/2}\langle w^2\ell\rangle)$",
                         xlim=(10 ** -1.7, 20), ylim=None, legend_loc="center right",
                         guides=((0.0, "--"),),
                         curves=()),
                         #  curves=((lambda x: 1.0 / (np.pi * x), "--"),),          # small-r limit 1/(pi r~)
                         #  annot=r"dashed: $1/(\pi\,r\lambda_\ell^{1/2})$"),
        "contact_pir": dict(ylabel=r"$\chi^{(s)}_{_{\ell,w}}(r)\,\pi\,r\,/(\lambda_\ell\langle w^2\ell\rangle)$",
                            xlim=(10 ** -1.8, 10 ** 1.4), ylim=(0, 1.5), legend_loc="upper right",
                            guides=(), curves=()),
    },
    "chid": {
        "contact": dict(ylabel=r"$\chi^{(d)}_{_{\ell,w}}(r)/\rho_w^2$",
                        xlim=(10 ** -1, 15), ylim=(-0.4, .2), legend_loc="lower right",
                        guides=((0.0, "--"),), curves=()),
        "master": dict(ylabel=r"$\chi^{(d)}_{_{\ell,w}}(r)/\rho_w^2$",
                       xlim=(10 ** -1, 10 ** 1.8), ylim=(-0.4, .4), legend_loc="upper right",
                       guides=((0.0, "--"),), curves=()),
        "rhow": dict(ylabel=r"$\chi^{(d)}_{_{\ell,w}}(r)/\rho_w^2$",
                     xlim=(10 ** -1.7, 20), ylim=(-0.5, .18), legend_loc="lower right",
                     guides=((0.0, "--"),), curves=()),
        "selfnorm": dict(ylabel=r"$\chi^{(d)}_{_{\ell,w}}(r)/(\lambda_\ell^{3/2}\langle w^2\ell\rangle)$",
                         xlim=(10 ** -1.7, 20), ylim=(-0.47, 0.2), legend_loc="lower right",
                         guides=((0.0, "--"),), curves=()),
        "contact_pir": dict(ylabel=r"$\pi r\,\chi^{(d)}_{_{\ell,w}}(r)/(\lambda_\ell\langle w^2\ell\rangle)$",
                            xlim=(10 ** -1.8, 15), ylim=None, legend_loc="upper right",
                            guides=((0.0, "--"),), curves=()),
    },
    "hnormd": {
        "contact": dict(ylabel=r"$\pi r\,\chi_{_{\ell,w}}(r)/(\lambda_\ell\langle w^2\ell\rangle)$",
                        xlim=(10 ** -1.2, 15.0), ylim=None, legend_loc="upper right",
                        guides=((0.0, "--"),), curves=()),
        "master": dict(ylabel=r"$2\pi r\,\chi_{_{\ell,w}}(r)/(\lambda_\ell\langle w^2\ell\rangle)$",
                       xlim=(10 ** -1.4, 15), ylim=None, legend_loc="upper right",
                       guides=(), curves=()), # , (2.0, ":")
        "rhow": dict(ylabel=r"$\chi_{_{\ell,w}}(r)/\rho_w^2$",
                     xlim=(10 ** -1.7, 20), ylim=(-0.7, 9.5), legend_loc="upper right",
                     guides=((0.0, "--"),), curves=()),
        "selfnorm": dict(ylabel=r"$\chi_{_{\ell,w}}(r)/(\lambda_\ell^{3/2}\langle w^2\ell\rangle)$",
                         xlim=(10 ** -1.7, 20), ylim=None, legend_loc="upper right",
                         guides=((0.0, "--"),), curves=()),
        "contact_pir": dict(ylabel=r"$\pi r\,\chi_{_{\ell,w}}(r)/(\lambda_\ell\langle w^2\ell\rangle)$",
                            xlim=(10 ** -1.8, 15.0), ylim=None, legend_loc="upper right",
                            guides=((0.0, "--"),), curves=()),
    },
}


def channel_x(c: RealCurves, norm: str) -> np.ndarray:
    if norm in ("contact", "rhow", "selfnorm", "contact_pir"):
        return c.x
    if norm == "master":
        return c.r / c.meta["ell_star"]
    raise KeyError(norm)


def channel_xy(c: RealCurves, chan: str, norm: str = "contact", total_spectral: bool = False):
    """(y_line, y_markers) for a channel; None where a route is absent.
    ``total_spectral``: the total panel draws the sampled total as markers against
    the inverted whole spectrum as the line (else the sampled total as the line)."""
    s = 2.0 if norm == "master" else 1.0
    if norm == "rhow":         # contact array -> channel / rho_w^2 (chid is already h_w)
        s = c.meta["rhow_fac"] / c.r
    if norm == "selfnorm":     # contact array -> channel / (lam^{3/2} m2), every channel
        s = 1.0 / (np.pi * c.x)
    if chan == "chisnorm":     # line = closed form, markers = sampled self deposits
        # return (None if c.H_self is None else s * c.H_self), (None if c.H_self_dir is None else s * c.H_self_dir)
        # -> chi_s/rho_w^2
        m2  = c.meta["sum_w2l"] / c.meta["N_ell"]
        lam = c.meta["N_ell"] / c.meta["V"]
        # s   = s / (c.r * c.meta["rho_w_emp"])   # -> chi_s/rho_w^2
        return (None if c.H_self is None else s * c.H_self), (None if c.H_self_dir is None else s * c.H_self_dir)
    if chan == "chid":         # line = spectral check, markers = sampled distinct deposits
        if norm == "contact_pir":
            # contact form H_d = pi r rho_w^2 h_w / (lam m2), the self/total units
            return ((None if c.h_spec is None else c.h_spec * c.r / c.meta["rhow_fac"]), c.H_dist_dir)
        if norm == "selfnorm":
            # contact form H_d (markers per file; spectral line via
            # h_spec * r / rhow_fac = pi r rho_w^2 h_spec / (lam m2)), same s as self and total
            return ((None if c.h_spec is None else s * c.h_spec * c.r / c.meta["rhow_fac"]),
                    (None if c.H_dist_dir is None else s * c.H_dist_dir))
        return c.h_spec, c.h_dir
    if chan == "hnormd":
        ok = c.H_dist_dir is not None and c.H_self_dir is not None
        total = (s * (c.H_self_dir + c.H_dist_dir)) if ok else None
        if norm == "selfnorm" and c.H_self is not None and c.H_dist_dir is not None:
            # direct total = closed-form self + sampled distinct (self as in the spectral line)
            total = s * (c.H_self + c.H_dist_dir)
        if total_spectral:
            if norm in ("rhow", "selfnorm") and c.H_self is not None and c.h_spec is not None:
                # LEON CHECK LATER: closed-form self (exact, no k_max) + Hankel distinct
                # instead of the whole-spectrum inversion, whose Hann band limit
                # overshoots the 1/r self term by ~1.2x at the first bin (k_max r = 3.75).
                # h_spec * r / rhow_fac = pi r rho_w^2 h_spec / (lam m2) = H_dist_spec.
                return s * (c.H_self + c.h_spec * c.r / c.meta["rhow_fac"]), total
            return (None if c.H_tot_spec is None else s * c.H_tot_spec), total
        return total, None
    raise KeyError(chan)


def plot_real_panel(curves, chan: str, out_png: str, *, norm: str = "contact",
                    lw: float = 2.8, legend_scale: float = 0.7, legend: bool = True,
                    legend_loc: Optional[str] = None, contact_legend_loc: Optional[str] = None,
                    total_spectral: bool = False, figsize=(3.375, 2.8), dpi: int = 600) -> str:
    if norm not in NORMS:
        raise KeyError(norm)
    spec = CHANNELS[chan][norm]
    loc = legend_loc or spec["legend_loc"]      # per-render override of the panel spec
    # Contact-value legend of the distinct panel: by default it follows ``legend`` at
    # CONTACT_LEGEND["loc"]; an explicit loc places it regardless of ``legend``,
    # "off" suppresses it.
    if contact_legend_loc is None:
        contact_loc = CONTACT_LEGEND["loc"] if legend else None
    else:
        contact_loc = None if contact_legend_loc == "off" else contact_legend_loc
    paired = chan != "hnormd" or total_spectral      # markers + line -> the two-column legend
    with matplotlib.rc_context(_RC):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xscale("log")
        # ax.set_yscale("log")
        present = []
        contact_tick = None
        xmin, xmax = np.inf, 0.0
        for c in sorted(curves, key=lambda c: c.ntype):
            col = TESS_COLORS[c.ntype]
            x = channel_x(c, norm)
            xmin, xmax = min(xmin, x[x > 0].min()), max(xmax, x.max())
            y_s, y_d = channel_xy(c, chan, norm, total_spectral)
            if y_s is not None:
                ax.plot(x, y_s, "-", color=col, lw=lw, zorder=2, label=c.ntype)
            if chan == "chisnorm" and norm == "rhow" and spec.get("self_vline"):
                # vertical eye guide at r = lambda_ell <w^2 ell> / rho_w^2 = pi * rhow_fac,
                # rhow_fac = lambda_ell <w^2 ell> / (pi rho_w^2)
                # placed on the x = r lambda_ell^{1/2} axis; same style as the guides
                # lam = c.meta["N_ell"] / c.meta["V"]
                # ax.axvline(c.meta["rhow_fac"], color="k", lw=.8,
                #            ls=spec["self_vline"], zorder=1)
                pass
            if y_d is not None:
                ax.plot(x, y_d, ls="none", marker=TESS_MARKERS[c.ntype], markersize=4, markerfacecolor=col,
                        markeredgecolor="black", markeredgewidth=.8, markevery=(MARK_OFFSET[c.ntype], 0.1), zorder=11)
            h0_key = "h_w0_selfnorm" if norm == "selfnorm" else "h_w0"
            if chan == "chid" and norm != "contact_pir" and c.meta.get(h0_key) is not None:
                # analytic contact plateau h_w(0): short tick at the left edge
                x0 = spec["xlim"][0]
                pass
                # ax.plot([x0, x0 * 10 ** 1], [c.meta["h_w0"]] * 2, color=col, lw=1,
                #         solid_capstyle="butt", zorder=6, ls="--")
                # ax.plot([x0, x0 * 10 ** .7], [c.meta["h_w0"]] * 2, color=col, lw=2, ls="--",
                #     solid_capstyle="projecting", zorder=6,
                #     path_effects=[pe.Stroke(linewidth=2.7, foreground="black"), pe.Normal()])
                contact_tick, = ax.plot([x0, x0 * 10 ** .7], [c.meta[h0_key]] * 2, color=col, lw=2,
                    ls=(0, (2.5, 3.0)), dash_capstyle="projecting", zorder=6,
                    path_effects=[pe.Stroke(linewidth=3.2, foreground="black"), pe.Normal()])
            present.append(c.ntype)
        for y0, ls in spec["guides"]:            # guides and reference curves sit behind the data
            # pass
            ax.axhline(y0, color="k", lw=.8, ls=ls, zorder=1)
        if spec["curves"]:
            xg = np.logspace(np.log10(xmin), np.log10(xmax), 400)
            for fn, ls in spec["curves"]:
                ax.plot(xg, fn(xg), color="k", lw=.8, ls=ls, zorder=0)
        if spec.get("annot"):    # reference-curve key, top right, legend font size
            ax.text(0.97, 0.97, spec["annot"], transform=ax.transAxes, ha="right", va="top",
                    fontsize=LEGEND_FS, zorder=600)
        ax.set_xlim(*spec["xlim"])
        if spec["ylim"]:
            ax.set_ylim(*spec["ylim"])
        ax.set_xlabel(XLABEL[norm], fontsize=LABEL_FS); ax.set_ylabel(spec["ylabel"], fontsize=LABEL_FS)
        ax.tick_params(which="both", labelsize=TICK_FS, direction="in", top=True, right=True)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=999))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10)), numticks=999))
        ax.xaxis.set_minor_formatter(NullFormatter())
        if not legend:           # bare panel (set carries its legend elsewhere)
            pass
        elif not paired:         # one line per tess, no table
            ax.legend(loc=loc, ncol=2, fontsize=LEGEND_FS, frameon=False,
                      columnspacing=1.0)
        else:
            rows = [(nt, TESS_COLORS[nt], TESS_MARKERS[nt]) for nt in present]
            mark_hdr, line_hdr = LEGEND_COLS[chan]
            paired_table_legend(ax, rows, [(mark_hdr, "marker"), (line_hdr, "line")],
                                loc=loc, fontsize=LEGEND_FS,
                                scale=legend_scale, frameon=False,
                                marker_edgecolor="black", marker_edgewidth=0.8)
        if contact_loc and contact_tick is not None:
            sample = Line2D([], [])
            sample.update_from(contact_tick)     # the ticks' dash pattern, width and border
            sample.set_color("white")            # neutral fill inside the ticks' black border
            ax.legend([sample], [CONTACT_LEGEND_LABEL.get(norm, CONTACT_LEGEND["label"])], loc=contact_loc,
                      fontsize=LEGEND_FS * legend_scale, frameon=False, handlelength=4.0)
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        fig.savefig(os.path.splitext(out_png)[0] + ".eps", bbox_inches="tight")
        plt.close(fig)
    return out_png
