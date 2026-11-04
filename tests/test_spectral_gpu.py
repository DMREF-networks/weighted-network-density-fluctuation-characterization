"""Optional GPU parity and lightweight backend dispatch checks."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.environ.get("WNDFC_TEST_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import wndfc                                             # noqa: E402
from wndfc import spectral_gpu                           # noqa: E402
from wndfc.__main__ import main                            # noqa: E402
from wndfc.pbc import Cell                                 # noqa: E402
from wndfc.spectral import _mode_intensities                # noqa: E402


def _network(dim, general=False):
    rng = np.random.default_rng(13)
    cell = Cell(np.array([[8.0, 1.0], [0.0, 7.0]])) if general else None
    v1 = rng.uniform(size=(9, dim))
    v1 = v1 @ cell.A if general else 8.0 * v1
    v2 = v1 + rng.uniform(-0.8, 0.8, size=v1.shape)
    kw = dict(L=float("nan") if general else 8.0, ntype="D", points=v1,
              edges_v1=v1, edges_v2=v2, edges_lengths=np.linalg.norm(v2 - v1, axis=1),
              edges_left_cell=np.arange(9), edges_right_cell=np.roll(np.arange(9), 1),
              cell_star_lengths=np.ones(9))
    if dim == 2:
        return wndfc.Network2D(**kw, cell_areas=np.ones(9), cell=cell)
    return wndfc.Network3D(**kw, cell_volumes=np.ones(9))


def _compute(dim):
    return wndfc.compute_chi_ell if dim == 2 else wndfc.compute_chi_ell_3d


def _same(cpu, gpu):
    np.testing.assert_array_equal(cpu.counts, gpu.counts)
    np.testing.assert_allclose(cpu.k, gpu.k, rtol=0, atol=1e-14)
    for field in ("chi_ell", "chi_self", "chi_distinct"):
        np.testing.assert_allclose(getattr(cpu, field), getattr(gpu, field), rtol=1e-10, atol=1e-13)


@pytest.fixture
def cuda():
    pytest.importorskip("cupy")
    try:
        return spectral_gpu._import_cupy()
    except RuntimeError as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize("dim,general", [(2, False), (2, True), (3, False)])
@pytest.mark.parametrize("shell_mode", ["width", "exact"])
@pytest.mark.parametrize("signed", [False, True])
def test_gpu_matches_cpu(cuda, dim, general, shell_mode, signed):
    net = _network(dim, general)
    weights = np.linspace(-1.5 if signed else 0.3, 2.7, net.n_edges)
    compute = _compute(dim)
    cpu = compute(net, weights, kmax=2.2, shell_mode=shell_mode)
    for batch in (1, 7):
        gpu = compute(net, weights, kmax=2.2, shell_mode=shell_mode, backend="cupy",
                      chunk_size=batch, rows_per_tile=1)
        _same(cpu, gpu)


@pytest.mark.parametrize("dim", [2, 3])
def test_gpu_cancelling_weights(cuda, dim):
    net = _network(dim)
    net.edges_v1[:] = net.edges_v1[0]
    net.edges_v2[:] = net.edges_v2[0]
    weights = np.array([2.0, -2.0] + [0.0] * 7)
    gpu = _compute(dim)(net, weights, kmax=1.6, backend="cupy", chunk_size=3)
    np.testing.assert_allclose(gpu.chi_ell, 0.0, rtol=0, atol=1e-25)
    assert np.all(gpu.chi_self > 0)
    _same(_compute(dim)(net, weights, kmax=1.6), gpu)


@pytest.mark.parametrize("dim", [2, 3])
def test_cupy_batching_and_dispatch(monkeypatch, dim):
    """Exercise memory limits and retry dispatch without a CUDA device."""
    net = _network(dim)
    weights = np.linspace(-1.5, 2.7, net.n_edges)
    uploads, batches = [], []

    class OutOfMemoryError(MemoryError):
        pass

    def asarray(a, dtype):
        uploads.append(np.shape(a))
        return np.asarray(a, dtype=dtype)

    cp = SimpleNamespace(
        asarray=asarray, float64=np.float64, empty=np.empty, asnumpy=np.asarray,
        get_default_memory_pool=lambda: SimpleNamespace(free_bytes=lambda: 80 * net.n_edges * 2),
        cuda=SimpleNamespace(Device=lambda: SimpleNamespace(mem_info=(80 * net.n_edges * 2, 0)),
                             memory=SimpleNamespace(OutOfMemoryError=OutOfMemoryError)))

    def intensities(cp, mid, half, w, kv):
        batches.append(len(kv))
        if len(batches) == 1:
            raise OutOfMemoryError()
        return _mode_intensities(mid, half, w, kv, len(kv))

    monkeypatch.setattr(spectral_gpu, "_import_cupy", lambda: cp)
    monkeypatch.setattr(spectral_gpu, "_mode_intensities_cupy", intensities)
    compute = _compute(dim)
    cpu = compute(net, weights, kmax=2.2)
    result = compute(net, weights, kmax=2.2, backend="cupy", chunk_size=20, rows_per_tile=1)
    _same(cpu, result)
    assert uploads == [(9, dim), (9, dim), (9,)]
    assert batches[0] == 2 and max(batches[1:]) == 1


@pytest.mark.parametrize("dim", [2, 3])
def test_missing_cupy_is_optional(monkeypatch, dim):
    monkeypatch.setitem(sys.modules, "cupy", None)
    net = _network(dim)
    compute = _compute(dim)
    assert compute(net, net.edges_lengths, kmax=1.6).n_bins > 0
    with pytest.raises(RuntimeError, match="requires CuPy"):
        compute(net, net.edges_lengths, kmax=1.6, backend="cupy")


def test_cli_missing_cupy(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "cupy", None)
    assert main(["--poisson", "9", "-L", "8", "--chi-backend", "cupy"]) == 2
    assert "requires CuPy" in capsys.readouterr().err


def test_unavailable_cuda(monkeypatch):
    def device():
        raise RuntimeError("no CUDA driver")

    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=SimpleNamespace(Device=device)))
    with pytest.raises(RuntimeError, match="requires a working CUDA device"):
        spectral_gpu._import_cupy()
