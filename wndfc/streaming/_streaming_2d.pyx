# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
cimport cython


cdef extern from "streaming_helper_2d.hpp" namespace "wndfc_streaming_2d":
    int compute_network_periodic_2d(
        const double* pts, int N, double L, int n_blocks,
        double* out_areas,
        int*    out_edge_cell,
        int*    out_edge_neighbor,
        double* out_edge_v1x,
        double* out_edge_v1y,
        double* out_edge_v2x,
        double* out_edge_v2y,
        int     cap,
        int*    out_n_edges) except +


@cython.boundscheck(False)
@cython.wraparound(False)
def stream_network_periodic_2d(double[:, ::1] points, double L,
                               int n_blocks=0, int initial_capacity=0):
    """Stream a 2D periodic Voronoi tessellation via voro++ 2D.

    Returns
    -------
    areas : (N,) float64
        Per-cell Voronoi area, indexed by generator id.
    edge_cell : (E,) int32
    edge_neighbor : (E,) int32
        Cell ids on either side of each cell-edge record. Each unique
        Voronoi edge appears twice (once per incident cell). Caller
        dedupes by keeping ``edge_cell < edge_neighbor``.
    edge_v1, edge_v2 : (E, 2) float64
        Voronoi vertex coordinates of the edge as seen from cell
        ``edge_cell``. v1 -> v2 follows the cell's own boundary
        traversal order (CCW under voro++ convention); ``v2 - v1`` is
        the absolute displacement (no min-image needed: voro++
        unwraps for the cell's local frame).
    """
    cdef int N = points.shape[0]
    if points.shape[1] != 2:
        raise ValueError("points must be (N, 2)")
    if L <= 0:
        raise ValueError("L must be positive")

    cdef int cap
    if initial_capacity > 0:
        cap = initial_capacity
    else:
        cap = max(64, 14 * N)

    cdef np.ndarray[double, ndim=1] areas = np.full(N, np.nan, dtype=np.float64)
    cdef np.ndarray[int, ndim=1] e_cell
    cdef np.ndarray[int, ndim=1] e_nbr
    cdef np.ndarray[double, ndim=1] v1x
    cdef np.ndarray[double, ndim=1] v1y
    cdef np.ndarray[double, ndim=1] v2x
    cdef np.ndarray[double, ndim=1] v2y

    cdef int n_done = -1
    cdef int n_edges = 0
    cdef int attempt = 0

    while True:
        e_cell = np.empty(cap, dtype=np.int32)
        e_nbr  = np.empty(cap, dtype=np.int32)
        v1x    = np.empty(cap, dtype=np.float64)
        v1y    = np.empty(cap, dtype=np.float64)
        v2x    = np.empty(cap, dtype=np.float64)
        v2y    = np.empty(cap, dtype=np.float64)

        n_done = compute_network_periodic_2d(
            &points[0, 0], N, L, n_blocks,
            &areas[0],
            <int*>     e_cell.data,
            <int*>     e_nbr.data,
            <double*>  v1x.data,
            <double*>  v1y.data,
            <double*>  v2x.data,
            <double*>  v2y.data,
            cap, &n_edges)

        if n_done >= 0:
            break
        attempt += 1
        if attempt > 5:
            raise RuntimeError(
                f"wndfc 2D streaming exceeded capacity 5 times "
                f"(last cap={cap}, n_edges so far={n_edges}); aborting.")
        cap = max(cap * 2, n_edges * 2 + 1024)
        # reset areas for next attempt
        areas[:] = np.nan

    edge_v1 = np.empty((n_edges, 2), dtype=np.float64)
    edge_v2 = np.empty((n_edges, 2), dtype=np.float64)
    edge_v1[:, 0] = v1x[:n_edges]
    edge_v1[:, 1] = v1y[:n_edges]
    edge_v2[:, 0] = v2x[:n_edges]
    edge_v2[:, 1] = v2y[:n_edges]

    return (areas,
            np.ascontiguousarray(e_cell[:n_edges]),
            np.ascontiguousarray(e_nbr[:n_edges]),
            edge_v1, edge_v2, n_done)
