"""Periodic-cell geometry: square boxes and general (sheared) 2D cells.

A ``Cell`` is the periodic parallelogram spanned by the rows of ``A`` (the
lattice vectors); positions x are ``f A`` with fractional coordinates f.  The
square box of side L is ``A = L I``.  Every helper below accepts either a float
side ``L`` (square box) or a ``Cell``; for square cells the original float code
path runs unchanged, so square-box results are identical to the letter.

General-cell conventions (row vectors throughout):
  fractional     f = x A^{-1}
  minimum image  delta -> (delta A^{-1} - round(delta A^{-1})) A
  wrap           x -> (x A^{-1} mod 1) A
  volume         V = |det A|
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass(frozen=True)
class Cell:
    """Periodic 2D cell spanned by the rows of ``A``."""

    A: np.ndarray

    def __post_init__(self):
        A = np.array(self.A, dtype=float)
        if A.shape != (2, 2):
            raise ValueError(f"cell matrix must be (2, 2); got {A.shape}")
        if abs(np.linalg.det(A)) <= 0.0:
            raise ValueError("cell matrix is singular")
        object.__setattr__(self, "A", A)

    @staticmethod
    def square(L: float) -> "Cell":
        return Cell(np.array([[float(L), 0.0], [0.0, float(L)]]))

    @property
    def is_square(self) -> bool:
        A = self.A
        return bool(A[0, 1] == 0.0 and A[1, 0] == 0.0 and A[0, 0] == A[1, 1] and A[0, 0] > 0.0)

    @property
    def L(self) -> float:
        """Side of a square cell (NaN for a general cell -- never use it there)."""
        return float(self.A[0, 0]) if self.is_square else float("nan")

    @property
    def Ainv(self) -> np.ndarray:
        return np.linalg.inv(self.A)

    @property
    def V(self) -> float:
        return float(abs(np.linalg.det(self.A)))

    @property
    def inradius(self) -> float:
        """Half the smallest distance between opposite sides: the largest disk
        radius for which the minimum image is unambiguous."""
        lengths = np.linalg.norm(self.A, axis=1)
        return float(0.5 * self.V / lengths.max())

    @property
    def sigma_min(self) -> float:
        """Smallest singular value of A: |delta x| >= sigma_min |delta f|."""
        return float(np.linalg.svd(self.A, compute_uv=False).min())

    def reciprocal(self) -> np.ndarray:
        """Rows b_i with a_i . b_j = 2 pi delta_ij (k = n @ reciprocal())."""
        return 2.0 * np.pi * np.linalg.inv(self.A).T

    def to_fractional(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) @ self.Ainv

    def to_cartesian(self, f: np.ndarray) -> np.ndarray:
        return np.asarray(f, dtype=float) @ self.A

    def as_meta(self) -> list:
        return self.A.tolist()


CellLike = Union[float, int, Cell]


def as_cell(cell: CellLike) -> Cell:
    return cell if isinstance(cell, Cell) else Cell.square(float(cell))


def _side(cell: CellLike):
    """(float side, None) for a square box / square Cell, (None, Cell) otherwise."""
    if isinstance(cell, Cell):
        return (cell.L, None) if cell.is_square else (None, cell)
    return (float(cell), None)


def min_image_delta(delta: np.ndarray, L: CellLike) -> np.ndarray:
    """Minimum-image of a displacement vector under the periodic cell."""
    delta = np.asarray(delta, dtype=float)
    side, cell = _side(L)
    if cell is None:
        return delta - side * np.round(delta / side)
    f = delta @ cell.Ainv
    return (f - np.round(f)) @ cell.A


def min_image_distance(a: np.ndarray, b: np.ndarray, L: CellLike) -> np.ndarray:
    """Minimum-image Euclidean distance(s) between point(s) ``a`` and ``b``."""
    return np.linalg.norm(min_image_delta(np.asarray(b) - np.asarray(a), L), axis=-1)


def wrap_to_box(p: np.ndarray, L: CellLike) -> np.ndarray:
    """Wrap point coordinates into the canonical cell.

    Square box: ``np.mod(p, L)`` is mathematically in [0, L) but for tiny-negative
    inputs (e.g. p = -1e-17) returns *exactly* L due to FP rounding. voro++
    requires inputs in the strict half-open interval [0, L) -- a point at
    x == L lands in block nx (out of bounds) and is silently misplaced.
    Snap any boundary-equal coords back to 0 to make the wrap rigorous.
    General cell: the same in fractional coordinates.
    """
    side, cell = _side(L)
    if cell is None:
        q = np.mod(np.asarray(p, dtype=float), side)
        q[q == side] = 0.0
        return q
    f = np.mod(np.asarray(p, dtype=float) @ cell.Ainv, 1.0)
    f[f == 1.0] = 0.0
    return f @ cell.A


def wrap_fractional(p: np.ndarray, cell: Cell) -> np.ndarray:
    """Fractional coordinates in [0, 1)^2 (for periodic KD-trees)."""
    f = np.mod(np.asarray(p, dtype=float) @ cell.Ainv, 1.0)
    f[f == 1.0] = 0.0
    return f


def unwrap_relative(reference: np.ndarray, target: np.ndarray, L: CellLike) -> np.ndarray:
    """Return ``target`` shifted by a lattice vector so its displacement from
    ``reference`` is the minimum-image displacement."""
    reference = np.asarray(reference, dtype=float)
    target = np.asarray(target, dtype=float)
    return reference + min_image_delta(target - reference, L)


def image_shift(delta: np.ndarray, L: CellLike) -> np.ndarray:
    """Integer lattice coefficients n such that ``delta - n A`` is the minimum
    image (square box: ``round(delta / L)`` per axis)."""
    delta = np.asarray(delta, dtype=float)
    side, cell = _side(L)
    if cell is None:
        return np.round(delta / side)
    return np.round(delta @ cell.Ainv)


def shift_vectors(n: np.ndarray, L: CellLike) -> np.ndarray:
    """Cartesian displacement of integer lattice coefficients ``n``."""
    n = np.asarray(n, dtype=float)
    side, cell = _side(L)
    if cell is None:
        return n * side
    return n @ cell.A


def periodic_pairs(points: np.ndarray, L: CellLike, r: float) -> np.ndarray:
    """(n_pairs, 2) index pairs with minimum-image distance <= r.

    Square box: ``cKDTree(points, boxsize=L).query_pairs(r)`` (points must be in
    [0, L)).  General cell: a periodic tree on the fractional coordinates with
    the conservative radius ``r / sigma_min``, then an exact Cartesian filter.
    """
    from scipy.spatial import cKDTree

    side, cell = _side(L)
    if cell is None:
        return cKDTree(points, boxsize=side).query_pairs(float(r), output_type="ndarray")
    f = wrap_fractional(points, cell)
    cand = cKDTree(f, boxsize=1.0).query_pairs(float(r) / cell.sigma_min, output_type="ndarray")
    if cand.shape[0] == 0:
        return cand
    df = f[cand[:, 1]] - f[cand[:, 0]]
    df -= np.round(df)
    d = np.linalg.norm(df @ cell.A, axis=1)
    return cand[d <= float(r)]


def sample_uniform(rng: np.random.Generator, n: int, L: CellLike) -> np.ndarray:
    """``n`` points uniform in the cell (square box: ``rng.uniform(0, L)``)."""
    side, cell = _side(L)
    if cell is None:
        return rng.uniform(0.0, side, size=(int(n), 2))
    return rng.uniform(0.0, 1.0, size=(int(n), 2)) @ cell.A


def cell_volume(L: CellLike) -> float:
    side, cell = _side(L)
    return side * side if cell is None else cell.V


def nominal_side(L: CellLike) -> float:
    """The square side itself (exact) or sqrt(V) for a general cell -- for default
    grids and cutoffs that scale with the box size."""
    side, cell = _side(L)
    return side if cell is None else float(np.sqrt(cell.V))


def safe_radius(L: CellLike) -> float:
    """Largest radius with an unambiguous minimum image: L/2, or the inradius."""
    side, cell = _side(L)
    return 0.5 * side if cell is None else cell.inradius
