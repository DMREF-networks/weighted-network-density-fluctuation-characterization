"""Optional CuPy spectral summation, shared by 2D and 3D."""
from __future__ import annotations

from .spectral import _bin_accumulate, _bin_finalize


def _import_cupy():
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "backend='cupy' requires CuPy; install the gpu extra for CUDA 12.x "
            "or a CuPy build matching your CUDA installation") from exc
    try:
        cp.cuda.Device().mem_info
    except Exception as exc:
        raise RuntimeError("backend='cupy' requires a working CUDA device: " + str(exc)) from exc
    return cp


def _dot(kv, x):
    """(n_k, E) products k . x_e, elementwise: no cuBLAS workspace, so the only
    allocation failure is CuPy's OutOfMemoryError."""
    out = kv[:, 0:1] * x[:, 0]
    for j in range(1, kv.shape[1]):
        out += kv[:, j:j + 1] * x[:, j]
    return out


def _mode_intensities_cupy(cp, midpoints, half, weights, kvecs):
    """Unnormalized total and self intensities on the device; float64/complex128 throughout."""
    kv = cp.asarray(kvecs, dtype=cp.float64)
    amp = weights * cp.sinc(_dot(kv, half) / cp.pi)
    sums = cp.sum(amp * cp.exp(1j * _dot(kv, midpoints)), axis=1)
    return sums.real ** 2 + sums.imag ** 2, cp.sum(amp * amp, axis=1)


def _compute_chi_cupy(net, weights, tiles, dk, shell_mode, chunk_size):
    """Keep edges resident; accumulate mode tiles into the shared bare-1/V bins."""
    cp = _import_cupy()
    midpoints = cp.asarray(net.edges_midpoints, dtype=cp.float64)
    half = cp.asarray(0.5 * (net.edges_v2 - net.edges_v1), dtype=cp.float64)
    w = cp.asarray(weights, dtype=cp.float64)
    free_bytes, _ = cp.cuda.Device().mem_info
    free_bytes += cp.get_default_memory_pool().free_bytes()
    # Reserve half the available memory; allow 80 bytes per (mode, edge) for intermediates.
    max_batch = max(1, int(0.5 * free_bytes / (80 * net.n_edges)))
    batch = max_batch if chunk_size is None else min(max(1, int(chunk_size)), max_batch)
    acc: dict = {}
    for kv, km in tiles:
        total = cp.empty(km.size)
        self_ = cp.empty(km.size)
        start = 0
        while start < km.size:
            stop = min(start + batch, km.size)
            try:
                total[start:stop], self_[start:stop] = _mode_intensities_cupy(
                    cp, midpoints, half, w, kv[start:stop])
            except cp.cuda.memory.OutOfMemoryError:
                if batch == 1:
                    raise
                batch = max(1, batch // 2)
                continue
            start = stop
        _bin_accumulate(acc, km, cp.asnumpy(total), cp.asnumpy(self_), dk, shell_mode)
    return _bin_finalize(acc, net.V)
