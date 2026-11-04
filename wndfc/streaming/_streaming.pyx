# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
cimport cython
from libcpp cimport bool as cbool


cdef extern from "streaming_helper.hpp" namespace "wndfc_streaming":
    int compute_volumes_periodic(const double* pts, int N, double L,
                                 double* out_volumes, int n_blocks) except +


@cython.boundscheck(False)
@cython.wraparound(False)
def stream_volumes_periodic(double[:, ::1] points, double L,
                            int n_blocks=0):
    cdef int N = points.shape[0]
    if points.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    if L <= 0:
        raise ValueError("L must be positive")

    if n_blocks <= 0:
        nb = max(1, int(round((N / 3.0) ** (1.0 / 3.0))))
    else:
        nb = n_blocks

    cdef np.ndarray[double, ndim=1] volumes = np.full(
        N, np.nan, dtype=np.float64)
    cdef double[::1] vols_view = volumes

    cdef int n_done = compute_volumes_periodic(
        &points[0, 0], N, L, &vols_view[0], nb)

    return volumes, n_done
