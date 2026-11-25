"""Physical nearest images and reciprocal shells for square and hexagonal cells."""

import itertools
import numpy as np
from scipy.spatial import Voronoi


def cell_kind(lattice):
    gram = lattice.T @ lattice
    if not np.isclose(gram[0, 0], gram[1, 1]):
        raise ValueError(
            "Spatial estimators support equal-sided square and hexagonal cells"
        )
    cosine = gram[0, 1] / gram[0, 0]
    if np.isclose(cosine, 0):
        return "square"
    if np.isclose(cosine, 0.5):
        return "hexagonal"
    raise ValueError("Use square or 60-degree hexagonal primitive vectors")


def translations(lattice):
    cell_kind(lattice)
    return np.array(list(itertools.product([-1, 0, 1], repeat=2))) @ lattice.T


def cell_polygon(lattice):
    shifts = translations(lattice)
    voronoi = Voronoi(shifts)
    origin = np.flatnonzero(np.linalg.norm(shifts, axis=1) == 0)[0]
    region = voronoi.regions[voronoi.point_region[origin]]
    polygon = voronoi.vertices[region]
    return polygon[np.argsort(np.arctan2(polygon[:, 1], polygon[:, 0]))]


def pixel_areas(xedges, yedges, lattice):
    """Area of each histogram pixel inside the physical nearest-image cell."""
    normals = translations(lattice)
    normals = normals[np.linalg.norm(normals, axis=1) > 0]
    areas = np.zeros((len(xedges) - 1, len(yedges) - 1))
    for i, (x0, x1) in enumerate(zip(xedges[:-1], xedges[1:])):
        for j, (y0, y1) in enumerate(zip(yedges[:-1], yedges[1:])):
            polygon = [np.array(p) for p in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]
            for normal in normals:
                bound = np.dot(normal, normal) / 2
                clipped = []
                for previous, current in zip(polygon[-1:] + polygon[:-1], polygon):
                    before, after = previous @ normal - bound, current @ normal - bound
                    if (before <= 0) != (after <= 0):
                        clipped.append(
                            previous + before / (before - after) * (current - previous)
                        )
                    if after <= 0:
                        clipped.append(current)
                polygon = clipped
                if not polygon:
                    break
            if polygon:
                p = np.array(polygon)
                areas[i, j] = (
                    abs(
                        np.dot(p[:, 0], np.roll(p[:, 1], -1))
                        - np.dot(p[:, 1], np.roll(p[:, 0], -1))
                    )
                    / 2
                )
    return areas


def nearest_image(displacements, lattice):
    shape = displacements.shape
    flat = np.asarray(displacements, dtype=float).reshape(-1, 2)
    inverse = np.linalg.inv(lattice)
    shifts = translations(lattice)
    result = np.empty_like(flat)
    for start in range(0, len(flat), 4096):
        frac = flat[start : start + 4096] @ inverse.T
        wrapped = (frac - np.rint(frac)) @ lattice.T
        candidates = wrapped[:, None, :] - shifts
        nearest = np.square(candidates).sum(axis=-1).argmin(axis=1)
        result[start : start + len(wrapped)] = candidates[
            np.arange(len(wrapped)), nearest
        ]
    return result.reshape(shape)


def pair_displacements(positions, lattice):
    n = positions.shape[-1] // 2
    if n < 2:
        raise ValueError("Pair correlations require at least two particles")
    x = np.asarray(positions, dtype=float).reshape(*positions.shape[:-1], n, 2)
    i, j = np.triu_indices(n, 1)
    return nearest_image(x[..., i, :] - x[..., j, :], lattice)


def reciprocal_vectors(lattice, n_particles, kmax=None):
    kind = cell_kind(lattice)
    density = n_particles / abs(np.linalg.det(lattice))
    if kmax is None:
        spacing = np.sqrt(2 / (np.sqrt(3) * density))
        kmax = 1.5 * (
            4 * np.pi / (np.sqrt(3) * spacing)
            if kind == "hexagonal"
            else 2 * np.pi * np.sqrt(density)
        )
    if not np.isfinite(kmax) or kmax <= 0:
        raise ValueError("--kmax must be positive and finite")
    reciprocal = 2 * np.pi * np.linalg.inv(lattice).T
    bound = int(np.ceil(kmax / np.linalg.svd(reciprocal, compute_uv=False).min()))
    indices = np.array(list(itertools.product(range(-bound, bound + 1), repeat=2)))
    vectors = indices @ reciprocal.T
    norm = np.linalg.norm(vectors, axis=1)
    keep = (norm > 0) & (norm <= kmax * (1 + 1e-12))
    if not keep.any():
        raise ValueError("--kmax is below the first nonzero reciprocal vector")
    return indices[keep], vectors[keep]


def structure_values(positions, vectors):
    """S(k) per time and chain, excluding no particles from the density sum."""
    n = positions.shape[-1] // 2
    flat = positions.reshape(-1, n, 2)
    chunks = []
    for start in range(0, len(flat), 32):
        phase = np.einsum("wjd,kd->wjk", flat[start : start + 32], vectors)
        density = np.exp(1j * phase).sum(axis=1)
        chunks.append(np.square(np.abs(density)) / n)
    return np.concatenate(chunks).reshape(*positions.shape[:2], len(vectors))


def shell_values(positions, lattice, n_particles, kmax=None):
    _, vectors = reciprocal_vectors(lattice, n_particles, kmax)
    squared = np.square(vectors).sum(axis=1)
    shells, inverse = np.unique(np.round(squared, 10), return_inverse=True)
    values = structure_values(positions, vectors)
    averaged = np.stack(
        [values[..., inverse == i].mean(axis=-1) for i in range(len(shells))], axis=-1
    )
    radii = np.array(
        [np.sqrt(squared[inverse == i].mean()) for i in range(len(shells))]
    )
    return radii, np.bincount(inverse), averaged
