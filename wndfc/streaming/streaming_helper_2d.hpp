#pragma once
// 2D periodic Voronoi streaming helper. Uses voro++ 2D (separate source tree
// in wndfc/c_voro/2d/src/). Emits per-cell areas and a flat list of
// per-cell-edge records (cell_id, neighbor_id, vertex pair) so the Python
// side can dedupe to unique (left, right) generator pairs and reproduce
// pyvoro-equivalent network geometry without materializing Python lists/dicts.
//
// Output buffers are caller-allocated with a fixed capacity. If the actual
// edge count exceeds capacity, the function returns -1 and the caller retries
// with a larger buffer. For Poisson-like generators in 2D the average cell
// has ~6 edges, so each cell contributes ~6 cell-edge records (each unique
// edge appears twice, once per incident cell). Capacity = 14 * N is safe.

#include "voro++_2d.hh"
#include <cmath>   // std::fmod

namespace wndfc_streaming_2d {

// Returns the number of cells processed on success, or -1 on capacity
// overflow. *out_n_edges is set to the total cell-edge records written
// (including both sides of each unique edge).
inline int compute_network_periodic_2d(
        const double* pts, int N, double L, int n_blocks,
        double* out_areas,
        int*    out_edge_cell,
        int*    out_edge_neighbor,
        double* out_edge_v1x,
        double* out_edge_v1y,
        double* out_edge_v2x,
        double* out_edge_v2y,
        int     cap,
        int*    out_n_edges) {

    using namespace voro;

    int nb = n_blocks > 0 ? n_blocks : (int) (sqrt((double) N / 3.0) + 1);
    if (nb < 1) nb = 1;

    container_2d con(0.0, L, 0.0, L,
                     nb, nb,
                     /*xperiodic=*/true, /*yperiodic=*/true,
                     /*init_mem=*/8);

    // Defensive domain clamp: voro++ requires inputs in the half-open box
    // [0, L). np.mod(p, L) can return *exactly* L for tiny-negative inputs
    // (FP rounding), and unwrapped callers can drift slightly. A point at
    // x == L lands in block nx (out of bounds) and voro++ silently
    // misbehaves. Clamp into [0, L) here so the helper is robust regardless
    // of caller hygiene. Same fix mirrored in compute_volumes_periodic (3D).
    for (int i = 0; i < N; ++i) {
        double x = pts[2*i + 0];
        double y = pts[2*i + 1];
        if (!(x >= 0.0)) x = 0.0;       // also catches NaN
        else if (x >= L)  x = std::fmod(x, L);
        if (!(y >= 0.0)) y = 0.0;
        else if (y >= L)  y = std::fmod(y, L);
        // fmod can still return L for x slightly above a multiple of L on
        // some libc implementations; final hard clamp.
        if (x >= L) x = 0.0;
        if (y >= L) y = 0.0;
        con.put(i, x, y);
    }

    voronoicell_neighbor_2d c;
    c_loop_all_2d vl(con);

    std::vector<int> nbrs;
    std::vector<double> verts;

    int n_done = 0;
    int n_edges = 0;

    if (vl.start()) {
        do {
            if (con.compute_cell(c, vl)) {
                int pid = vl.pid();
                if (pid < 0 || pid >= N) continue;
                double gx, gy;
                vl.pos(gx, gy);
                out_areas[pid] = c.area();

                c.neighbors(nbrs);
                c.vertices(gx, gy, verts);
                int p = (int) nbrs.size();
                // verts/nbrs are in voro++ 2D STORAGE order, which is NOT
                // the polygon ring order (vertices() dumps the raw pts
                // array; the connectivity lives in the ed table). The ring
                // successor of vertex k is ed[2*k], and ne[k] (== nbrs[k])
                // is the neighbor across the edge (k -> ed[2*k]) -- the
                // same traversal voro++'s own perimeter()/edge_lengths()
                // use. The previous k -> (k+1)%p pairing produced scrambled
                // vertex pairs for any cell whose cut history left storage
                // order non-cyclic (~93% of Poisson cells; found 2026-06-12
                // via the moments fan cross-check).
                {
                    int k = 0, steps = 0;
                    do {
                        int l = c.ed[2*k];
                        if (l < 0 || l >= p) break;   // corrupt ring guard
                        // Self-periodic neighbor: at very small L / very
                        // few generators, a cell can border its own
                        // periodic image. Both reports of that edge satisfy
                        // (e_cell == e_nbr), so the standard
                        // `keep = e_cell < e_nbr` dedup drops them. Skip
                        // here (but keep walking) to make the omission
                        // explicit.
                        if (nbrs[k] != pid) {
                            if (n_edges >= cap) {
                                *out_n_edges = n_edges;
                                return -1;
                            }
                            out_edge_cell[n_edges]     = pid;
                            out_edge_neighbor[n_edges] = nbrs[k];
                            out_edge_v1x[n_edges]      = verts[2*k];
                            out_edge_v1y[n_edges]      = verts[2*k + 1];
                            out_edge_v2x[n_edges]      = verts[2*l];
                            out_edge_v2y[n_edges]      = verts[2*l + 1];
                            ++n_edges;
                        }
                        k = l;
                    } while (k != 0 && ++steps <= p);
                }
                ++n_done;
            }
        } while (vl.inc());
    }

    *out_n_edges = n_edges;
    return n_done;
}

}  // namespace wndfc_streaming_2d
