"""Two-dimensional disk and torus geometry; lattice vectors are columns."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import jax.numpy as jnp


def make_lattice(lattice_type: str, length: float, ndim: int = 2) -> np.ndarray:
    """Build a square cell or a triangular cell with a hexagonal Brillouin zone."""
    if ndim != 2 or not np.isfinite(length) or length <= 0:
        raise ValueError(
            "Geometry requires ndim=2 and a positive finite lattice length"
        )
    if lattice_type == "square":
        return length * np.eye(2)
    if lattice_type == "hexagonal":
        return length * np.array([[1.0, 0.5], [0.0, np.sqrt(3.0) / 2.0]])
    raise ValueError(f"Unknown lattice_type: {lattice_type}")


@dataclass(frozen=True, eq=False)
class Geometry:
    """Physical domain captured by the model and Hamiltonian.

    center is the disk center or the origin of periodic coordinates. A torus
    has no wall. Its lattice contains two independent primitive vectors.
    """

    kind: str = "disk"
    lattice: Optional[np.ndarray] = None
    radius: float = 10.0
    center: tuple = (0.0, 0.0)
    ndim: int = 2

    def __post_init__(self):
        if self.kind not in ("disk", "periodic") or self.ndim != 2:
            raise ValueError("Use disk or periodic geometry in two dimensions")
        if np.shape(self.center) != (2,) or not np.isfinite(self.center).all():
            raise ValueError("center must contain two finite coordinates")
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("disk radius must be positive and finite")
        if (self.kind == "periodic") != (self.lattice is not None):
            raise ValueError(
                "Periodic geometry requires a lattice; disk geometry requires no lattice"
            )
        if self.lattice is not None:
            lattice = np.array(self.lattice, dtype=float, copy=True)
            if (
                lattice.shape != (2, 2)
                or not np.isfinite(lattice).all()
                or abs(np.linalg.det(lattice)) < 1e-12
            ):
                raise ValueError("lattice must be a finite nonsingular 2 by 2 matrix")
            lattice.setflags(write=False)
            object.__setattr__(self, "lattice", lattice)

    @classmethod
    def from_config(cls, cfg):
        return cls(
            kind=cfg.geometry,
            lattice=cfg.lattice,
            radius=cfg.disk_radius,
            center=tuple(cfg.center),
            ndim=cfg.ndim,
        )

    def wrap(self, positions):
        """Wrap flat or (..., 2) positions to the primitive cell."""
        if self.lattice is None:
            return positions
        shape = positions.shape
        pos = jnp.reshape(positions, (-1, 2)) - jnp.asarray(self.center)
        frac = jnp.einsum("ij,nj->ni", jnp.linalg.inv(jnp.asarray(self.lattice)), pos)
        wrapped = jnp.einsum("ij,nj->ni", jnp.asarray(self.lattice), frac % 1.0)
        return (wrapped + jnp.asarray(self.center)).reshape(shape)
