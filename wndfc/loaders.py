from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np


def load_points_2d(path: str,
                   *,
                   box_size: Optional[float] = None,
                   delimiter: Optional[str] = None) -> Tuple[np.ndarray, float]:
    """Load a 2D point pattern from a text file (whitespace/comma) or .npy.

    Tries the following column layouts in order:
      ``(x, y)``, ``(id, x, y)``, ``(x, y, z)`` (drops z).

    The detected box side ``L`` is the supplied ``box_size`` if non-None,
    else ``ceil(max(coord)) + 1e-12`` (rounds up to nominal integer L).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    arr = np.load(path) if path.endswith(".npy") else \
        np.loadtxt(path, delimiter=delimiter)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    n_cols = arr.shape[1]
    if n_cols == 2:
        pts = arr[:, :2]
    elif n_cols == 3:
        sample_int = np.allclose(arr[:, 0], arr[:, 0].astype(int))
        if sample_int and arr[:, 0].min() >= 0:
            pts = arr[:, 1:3]
        else:
            pts = arr[:, :2]
    elif n_cols >= 4:
        pts = arr[:, 1:3]
    else:
        raise ValueError(f"unsupported column count: {n_cols}")

    pts = np.ascontiguousarray(pts, dtype=float)
    if box_size is not None:
        L = float(box_size)
    else:
        L = float(np.ceil(np.max(pts)))
        if L <= 0:
            L = float(np.max(pts)) + 1e-12
    return pts, L


def load_points_3d(path: str,
                   *,
                   box_size: Optional[float] = None,
                   delimiter: Optional[str] = None) -> Tuple[np.ndarray, float]:
    """Load a 3D point pattern from a text file (whitespace/comma) or .npy.

    Tries the following column layouts in order:
      ``(x, y, z)``, ``(id, x, y, z)``, ``5+ cols`` -> takes columns 1..3.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    arr = np.load(path) if path.endswith(".npy") else \
        np.loadtxt(path, delimiter=delimiter)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    n_cols = arr.shape[1]
    if n_cols == 3:
        pts = arr[:, :3]
    elif n_cols == 4:
        sample_int = np.allclose(arr[:, 0], arr[:, 0].astype(int))
        if sample_int and arr[:, 0].min() >= 0:
            pts = arr[:, 1:4]
        else:
            pts = arr[:, :3]
    elif n_cols >= 5:
        pts = arr[:, 1:4]
    else:
        raise ValueError(f"unsupported column count: {n_cols}")

    pts = np.ascontiguousarray(pts, dtype=float)
    if box_size is not None:
        L = float(box_size)
    else:
        L = float(np.ceil(np.max(pts)))
        if L <= 0:
            L = float(np.max(pts)) + 1e-12
    return pts, L


# --- General cells, radii, edge lists, packing directories ------------------

def load_lattice(path: str):
    """2 x 2 lattice matrix (rows = lattice vectors) -> ``pbc.Cell``."""
    from .pbc import Cell

    A = np.loadtxt(path, dtype=float)
    if A.shape != (2, 2):
        raise ValueError(f"lattice file must hold a 2 x 2 matrix; got {A.shape}")
    return Cell(A)


def load_radii(path: str) -> np.ndarray:
    r = np.load(path) if path.endswith(".npy") else np.loadtxt(path, dtype=float)
    return np.ascontiguousarray(np.asarray(r, dtype=float).ravel())


def load_edges(path: str) -> np.ndarray:
    """(E, 2) int edge list: ``.npy`` or text with ``i j`` per line (0-based).

    A third column, if present, is ignored (weights / flags of other tools).
    """
    e = np.load(path) if path.endswith(".npy") else np.loadtxt(path)
    e = np.asarray(e)
    if e.ndim == 1:
        e = e.reshape(-1, 2)
    if e.shape[1] < 2:
        raise ValueError("edge list needs two index columns")
    e = e[:, :2]
    if not np.allclose(e, np.round(e)):
        raise ValueError("edge list indices are not integers")
    return np.ascontiguousarray(np.round(e).astype(np.int64))


def load_packing(directory: str, *, edges: Optional[str] = None,
                 edges_search: bool = True) -> dict:
    """A disk-packing directory in the idealPackingLibrary layout:

        positions.dat   N x 2 centres        radii.dat    N radii
        latticeVectors.dat  2 x 2 (rows = lattice vectors)   scalars.dat  key value

    ``edges``: an explicit edge-list file; with ``edges_search`` the loader also
    looks for ``<name>edges.npy`` / ``<name>edges.txt`` up to four directory
    levels above the packing (the ``Network_Scattering`` layout).  Returns a dict
    with ``points``, ``cell``, ``radii``, ``edges`` (or None), ``phi`` (or None)
    and ``name``.
    """
    from .pbc import Cell

    d = os.path.abspath(directory)
    name = os.path.basename(d.rstrip(os.sep))
    pts = np.loadtxt(os.path.join(d, "positions.dat"), dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"positions.dat must be (N, 2); got {pts.shape}")
    lat = os.path.join(d, "latticeVectors.dat")
    if os.path.exists(lat):
        cell = load_lattice(lat)
    else:
        side = float(np.loadtxt(os.path.join(d, "boxSize.dat")).ravel()[0])
        cell = Cell.square(side)
    radii = None
    if os.path.exists(os.path.join(d, "radii.dat")):
        radii = load_radii(os.path.join(d, "radii.dat"))
    phi = None
    sc = os.path.join(d, "scalars.dat")
    if os.path.exists(sc):
        for line in open(sc, encoding="utf-8"):
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "phi":
                phi = float(parts[1])
    edge_arr = None
    if edges is not None:
        edge_arr = load_edges(edges)
    elif edges_search:
        for up in (1, 2, 3, 4):
            base = d
            for _ in range(up):
                base = os.path.dirname(base)
            for ext in (".npy", ".txt"):
                cand = os.path.join(base, name + "edges" + ext)
                if os.path.exists(cand):
                    edge_arr = load_edges(cand)
                    break
            if edge_arr is not None:
                break
    return {"points": np.ascontiguousarray(pts), "cell": cell, "radii": radii,
            "edges": edge_arr, "phi": phi, "name": name}
