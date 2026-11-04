from __future__ import annotations
import os
import numpy as np

# WNDFC_NO_STREAM=1 forces every streaming entry point off so all
# tessellations fall through to the pyvoro path. Diagnostic toggle for
# A/B-testing streaming vs pyvoro on the same input (covers 2D D/G/V/C
# and 3D, since both AVAILABLE flags are gated here).
_NO_STREAM = os.environ.get("WNDFC_NO_STREAM", "").strip() in ("1", "true", "TRUE", "yes")

try:
    from ._streaming import stream_volumes_periodic as _impl3d
    AVAILABLE = True and not _NO_STREAM
except ImportError as _exc:
    _import_error_3d = _exc
    AVAILABLE = False

try:
    from ._streaming_2d import stream_network_periodic_2d as _impl2d
    AVAILABLE_2D = True and not _NO_STREAM
except ImportError as _exc:
    _import_error_2d = _exc
    AVAILABLE_2D = False

if _NO_STREAM:
    print("[wndfc.streaming] WNDFC_NO_STREAM=1 -> streaming "
          "disabled (pyvoro fallback for all tess).")


def stream_volumes_periodic(points, L, n_blocks=0):
    if not AVAILABLE:
        raise RuntimeError(
            "wndfc.streaming._streaming not built. Run "
            "`python wndfc/streaming/setup.py build_ext --inplace` "
            "first. Original ImportError: " + repr(_import_error_3d))
    pts = np.ascontiguousarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    return _impl3d(pts, float(L), int(n_blocks))


def stream_network_periodic_2d(points, L, n_blocks=0, initial_capacity=0):
    """Stream the 2D periodic Voronoi tessellation directly from voro++.

    Returns ``(areas, edge_cell, edge_neighbor, edge_v1, edge_v2, n_done)``;
    each unique Voronoi edge appears twice (once per incident cell). See
    ``_streaming_2d.pyx`` for full output semantics.
    """
    if not AVAILABLE_2D:
        raise RuntimeError(
            "wndfc.streaming._streaming_2d not built. Run "
            "`python wndfc/streaming/setup.py build_ext --inplace` "
            "with 2D voro++ source available (wndfc/c_voro/2d/src/ "
            "or VOROPP_2D_SRC). Original ImportError: "
            + repr(_import_error_2d))
    pts = np.ascontiguousarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be (N, 2)")
    return _impl2d(pts, float(L), int(n_blocks), int(initial_capacity))


__all__ = [
    "stream_volumes_periodic", "AVAILABLE",
    "stream_network_periodic_2d", "AVAILABLE_2D",
]
