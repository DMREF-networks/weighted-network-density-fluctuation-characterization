#pragma once
#include "voro++.hh"
#include <cmath>   // std::fmod

namespace wndfc_streaming {

inline int compute_volumes_periodic(const double* pts, int N, double L,
                                    double* out_volumes, int n_blocks) {
    using namespace voro;

    container_periodic con(L, 0.0, L, 0.0, 0.0, L,
                           n_blocks, n_blocks, n_blocks, 8);

    // Defensive domain clamp: voro++ requires inputs in [0, L)^3. np.mod
    // can return *exactly* L for tiny-negative inputs (FP rounding); a
    // point at x == L lands in block nx (out of bounds). Clamp here so
    // the helper is robust regardless of caller hygiene.
    for (int i = 0; i < N; ++i) {
        double x = pts[3*i + 0];
        double y = pts[3*i + 1];
        double z = pts[3*i + 2];
        if (!(x >= 0.0)) x = 0.0; else if (x >= L) x = std::fmod(x, L);
        if (!(y >= 0.0)) y = 0.0; else if (y >= L) y = std::fmod(y, L);
        if (!(z >= 0.0)) z = 0.0; else if (z >= L) z = std::fmod(z, L);
        if (x >= L) x = 0.0;
        if (y >= L) y = 0.0;
        if (z >= L) z = 0.0;
        con.put(i, x, y, z);
    }

    voronoicell_neighbor c;
    c_loop_all_periodic vl(con);
    int n_done = 0;

    if (vl.start()) {
        do {
            if (con.compute_cell(c, vl)) {
                int pid = vl.pid();
                if (pid >= 0 && pid < N) {
                    out_volumes[pid] = c.volume();
                    ++n_done;
                }
            }
        } while (vl.inc());
    }
    return n_done;
}

}
