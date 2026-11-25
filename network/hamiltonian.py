# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kinetic energy, dipolar and Gaussian pair interactions, and disk confinement."""

import itertools
from typing import Callable, Optional, Union
import chex
from network import networks
from network.geometry import Geometry
from network.pbc.feature_layer import periodic_norm
from network.utils import utils
import folx
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

Array = Union[jnp.ndarray, np.ndarray]
KineticEnergy = Callable
LocalEnergy = Callable


@chex.dataclass
class LocalEnergyAux:
    kinetic: jax.Array
    potential: jax.Array


def local_kinetic_energy(
    f: networks.NetworkLike,
    use_scan: bool = False,
    complex_output: bool = False,
    laplacian_method: str = "default",
) -> KineticEnergy:
    r"""Creates a function to for the local kinetic energy, -1/2 \nabla^2 ln|f|.

    Args:
      f: Callable which evaluates the wavefunction as a
        (sign or phase, log magnitude) tuple.
      use_scan: Whether to use a `lax.scan` for computing the laplacian.
      complex_output: If true, the output of f is complex-valued.
      laplacian_method: Laplacian calculation method. One of:
        'default': take jvp(grad), looping over inputs
        'folx': use Microsoft's implementation of forward laplacian

    Returns:
      Callable which evaluates the local kinetic energy,
      -1/2f \nabla^2 f = -1/2 (\nabla^2 log|f| + (\nabla log|f|)^2).
    """

    phase_f = utils.select_output(f, 0)
    logabs_f = utils.select_output(f, 1)

    if laplacian_method == "default":

        def _lapl_over_f(params, data):
            n = data.positions.shape[0]
            eye = jnp.eye(n)
            grad_f = jax.grad(logabs_f, argnums=1)

            def grad_f_closure(x):
                return grad_f(params, x, data.interaction_strength)

            primal, dgrad_f = jax.linearize(grad_f_closure, data.positions)

            if complex_output:
                grad_phase = jax.grad(phase_f, argnums=1)

                def grad_phase_closure(x):
                    return grad_phase(params, x, data.interaction_strength)

                phase_primal, dgrad_phase = jax.linearize(
                    grad_phase_closure, data.positions
                )
                hessian_diagonal = lambda i: (
                    dgrad_f(eye[i])[i] + 1.0j * dgrad_phase(eye[i])[i]
                )
            else:
                hessian_diagonal = lambda i: dgrad_f(eye[i])[i]

            if use_scan:
                _, diagonal = lax.scan(
                    lambda i, _: (i + 1, hessian_diagonal(i)), 0, None, length=n
                )
                result = -0.5 * jnp.sum(diagonal)
            else:
                result = -0.5 * lax.fori_loop(
                    0, n, lambda i, val: val + hessian_diagonal(i), 0.0
                )
            result -= 0.5 * jnp.sum(primal**2)
            if complex_output:
                result += 0.5 * jnp.sum(phase_primal**2)
                result -= 1.0j * jnp.sum(primal * phase_primal)
            return result

    elif laplacian_method == "folx":

        def _lapl_over_f(params, data):
            f_closure = lambda x: f(params, x, data.interaction_strength)
            f_wrapped = folx.forward_laplacian(f_closure, sparsity_threshold=6)
            output = f_wrapped(data.positions)
            result = (
                -(output[1].laplacian + jnp.sum(output[1].jacobian.dense_array ** 2))
                / 2
            )
            if complex_output:
                result -= 0.5j * output[0].laplacian
                result += 0.5 * jnp.sum(output[0].jacobian.dense_array ** 2)
                result -= 1.0j * jnp.sum(
                    output[0].jacobian.dense_array * output[1].jacobian.dense_array
                )
            return result
    else:
        raise NotImplementedError(
            f"Laplacian method {laplacian_method} not implemented."
        )

    return _lapl_over_f


def periodic_pair_distances(pairs: Array, lattice: Array) -> jnp.ndarray:
    """Computes periodic particle-pair distances using minimum image convention.

    Args:
      pairs: Shape (n_particles, n_particles, ndim). Particle-particle displacement vectors.
      lattice: Shape (ndim, ndim). Matrix of lattice vectors (columns are vectors).

    Returns:
      distances: Shape (n_particles, n_particles, 1). Periodic distances.
    """
    reciprocal_vecs = jnp.linalg.inv(lattice)
    lattice_metric = lattice.T @ lattice
    # Convert to fractional coordinates
    scaled_ee = jnp.einsum("il,jkl->jki", reciprocal_vecs, pairs)
    # Compute periodic norm
    n = pairs.shape[0]
    # Add identity to diagonal to avoid zero-norm issues, then mask out
    scaled_ee_safe = scaled_ee + jnp.eye(n)[..., None]
    distances = periodic_norm(lattice_metric, scaled_ee_safe) * (1.0 - jnp.eye(n))
    return distances[..., None]


def pair_potential(
    pairs: Array,
    distances: Array,
    interaction_strength: float,
    interaction_small_length_cutoff: float = 0.1,
    lattice: Optional[Array] = None,
    interaction_truncation_limit: int = 5,
    interaction_cutoff_radius: Optional[float] = None,
    interaction_type: str = "inverse_cubic",
    short_range_width: float = 0.1,
) -> jnp.ndarray:
    """Returns the particle-pair repulsion potential.

    Args:
      pairs: Shape (n_particles, n_particles, ndim). Particle-particle displacements.
      distances: Shape (neletrons, n_particles, :). distances[i,j,0] gives the distance
        between particles i and j under minimum-image convention.
      interaction_strength: Lambda multiplying the selected pair potential.
      interaction_small_length_cutoff: Softening length r0 that regularises the
        1/r^3 singularity at short range.
      lattice: Shape (ndim, ndim). Lattice vectors for periodic cell.
      interaction_truncation_limit: Number of unit-cell images included in each
        positive/negative lattice-vector direction for the real-space summation.
      interaction_cutoff_radius: Optional spherical cutoff R_c. When set, pair
        (and image) contributions at distance > R_c are dropped from the explicit
        sum; the neglected long-range part is meant to be supplied separately as
        an isotropic tail correction (see ``pair_tail_coefficient``).
        When None (default) the explicit sum is unchanged.
      interaction_type: 'inverse_cubic' or normalized 2D 'short_range' Gaussian.
      short_range_width: Gaussian standard deviation (positive).
    """
    if interaction_type not in ("inverse_cubic", "short_range"):
        raise ValueError(f"Unknown interaction_type: {interaction_type}")
    if short_range_width <= 0:
        raise ValueError("short_range_width must be positive")
    r0 = interaction_small_length_cutoff

    def pair_potential(r):
        if interaction_type == "short_range":
            return jnp.exp(-(r**2) / (2 * short_range_width**2)) / (
                2 * jnp.pi * short_range_width**2
            )
        return 1.0 / (r + r0) ** 3

    n = pairs.shape[0]
    if lattice is None:
        r_pairs = distances[jnp.triu_indices_from(distances[..., 0], 1)]
        pair_energy = pair_potential(r_pairs)
        if interaction_cutoff_radius is not None:
            pair_energy = jnp.where(
                r_pairs <= interaction_cutoff_radius, pair_energy, 0.0
            )
        return interaction_strength * jnp.sum(pair_energy)

    dim = pairs.shape[-1]
    ordinals = jnp.arange(
        -interaction_truncation_limit, interaction_truncation_limit + 1
    )
    image_indices = jnp.array(list(itertools.product(ordinals, repeat=dim)))
    # R = A n with lattice vectors as columns in A.
    image_shifts = jnp.einsum("ij,nj->ni", lattice, image_indices)

    idx_i, idx_j = jnp.triu_indices(n, k=1)
    pair_displacements = pairs[idx_i, idx_j]
    # Wrap pair displacements to the minimum-image cell so the lattice sum is
    # correct even when MCMC walkers have drifted many unit cells away from [0,L).
    reciprocal = jnp.linalg.inv(lattice)
    frac = jnp.einsum("ij,pj->pi", reciprocal, pair_displacements)
    pair_displacements = jnp.einsum("ij,pj->pi", lattice, frac - jnp.round(frac))
    all_displacements = pair_displacements[:, None, :] + image_shifts[None, :, :]
    distances = jnp.linalg.norm(all_displacements, axis=-1)
    pair_energy = pair_potential(distances)
    if interaction_cutoff_radius is not None:
        pair_energy = jnp.where(
            distances <= interaction_cutoff_radius, pair_energy, 0.0
        )
    return interaction_strength * jnp.sum(pair_energy)


def pair_tail_coefficient(
    lattice: Array,
    n_particles: int,
    interaction_small_length_cutoff: float,
    interaction_cutoff_radius: float,
    g2_r: Optional[Array] = None,
    g2_values: Optional[Array] = None,
) -> float:
    """Isotropic large-r tail of the 1/(r+r0)^3 pair energy, per unit strength.

    Returns a scalar ``tail_coeff`` such that the long-range correction to the
    particle-pair potential energy of one configuration is

        E_tail = interaction_strength * tail_coeff,

    with

        E_tail / k = 0.5 * N * n * \\int_{R_c}^{\\infty}
                          [1 / (r + r0)^3] g2(r) (2 pi r) dr ,

    where ``N`` is the particle number, ``n = N / A`` the 2D number density,
    ``R_c`` the spherical cutoff applied to the explicit pair sum, and ``g2(r)``
    the radial pair-correlation function. ``g2(r)`` defaults to 1 (uniform fluid,
    the standard mean-field tail). A tabulated ``g2`` may be supplied via
    ``(g2_r, g2_values)``; it is integrated numerically over its range (which
    should start at ``R_c``) with ``g2 = 1`` assumed beyond the last tabulated
    point.

    This is computed once at construction time in NumPy (it does not depend on
    the instantaneous walker positions) and added as a constant inside the jitted
    local-energy evaluation.

    Args:
      lattice: Shape (ndim, ndim). Real-space lattice vectors (as columns).
      n_particles: Number of particles N.
      interaction_small_length_cutoff: Softening length r0.
      interaction_cutoff_radius: Spherical cutoff R_c (lower limit of the tail).
      g2_r: Optional radii at which g2 is tabulated.
      g2_values: Optional g2 values matching ``g2_r``.

    Returns:
      Python float ``tail_coeff``.
    """
    import numpy as np  # local import; runs at construction, not under jit.

    lattice = np.asarray(lattice, dtype=np.float64)
    area = np.abs(np.linalg.det(lattice))
    n_density = n_particles / area
    r0 = float(interaction_small_length_cutoff)
    r_cut = float(interaction_cutoff_radius)

    def analytic_uniform(a):
        # \int_a^\infty (2 pi r) / (r + r0)^3 dr
        #   = 2 pi [ 1 / (a + r0) - r0 / (2 (a + r0)^2) ]
        return 2.0 * np.pi * (1.0 / (a + r0) - r0 / (2.0 * (a + r0) ** 2))

    if g2_r is None or g2_values is None:
        integral = analytic_uniform(r_cut)
    else:
        g2_r = np.asarray(g2_r, dtype=np.float64)
        g2_values = np.asarray(g2_values, dtype=np.float64)
        mask = g2_r >= r_cut
        rr = g2_r[mask]
        gg = g2_values[mask]
        integrand = (1.0 / (rr + r0) ** 3) * gg * (2.0 * np.pi * rr)
        trapezoid = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2.x rename.
        integral = float(trapezoid(integrand, rr))
        # Add the analytic remainder beyond the tabulated range assuming g2 -> 1.
        integral += analytic_uniform(rr[-1])

    return float(0.5 * n_particles * n_density * integral)


def make_ewald_1r3_potential(
    lattice: Array,
    n_particles: int,
    interaction_small_length_cutoff: float = 0.1,
    eta: Optional[float] = None,
    eta_scale: float = 1.0,
    real_shells: int = 3,
    recip_shells: int = 8,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Builds a jittable 2D Ewald evaluator for the 1/(r + a)^3 pair potential.

    Returns the *exact* periodic potential energy (per unit interaction strength)
    of one configuration under an arbitrary 2D Bravais lattice, with no cutoff or
    tail-correction error. This is the Ewald alternative to the minimum-image sum
    (`pair_potential`) and the analytic tail correction; unlike those
    it also includes the particle self-image (Madelung) interaction, which is the
    physically correct energy of the infinite periodic crystal.

    The softened dipole potential is split as

        1/(r + a)^3 = [1/(r + a)^3 - 1/r^3] + 1/r^3 ,

    where the bracket is short-ranged (~ -3a/r^4) and summed directly in real
    space over lattice images, and the bare 1/r^3 part is evaluated by the
    generalized 2D dipolar Ewald sum (cf. Astrakharchik-reproducing/ewald.py,
    extended here to a non-orthogonal lattice and to a > 0):

        phi_short(r) = [erfc(eta r) + (2/sqrt(pi)) eta r exp(-(eta r)^2)] / r^3
        I(G)         = 2 eta exp(-G^2/4eta^2) - sqrt(pi) G erfc(G/(2 eta))
        U = 0.5 sum_{i,j,R != self} [phi_short + (1/(r+a)^3 - 1/r^3)]
            + (sqrt(pi)/A) sum_{G != 0} I(G) |rho(G)|^2
            + 2 sqrt(pi) eta N^2 / A - 2 N eta^3 / (3 sqrt(pi)) .

    The result is independent of the Ewald parameter eta (verified numerically);
    a = 0 recovers the triangular-lattice Madelung constant 4.446 n^{3/2}.

    Args:
      lattice: Shape (2, 2). Real-space primitive vectors as *columns*.
      n_particles: Number of particles N.
      interaction_small_length_cutoff: Softening length a in 1/(r + a)^3.
      eta: Ewald splitting parameter. If None, uses eta_scale * sqrt(pi)/sqrt(A).
      eta_scale: Multiplier on the default eta (energy is eta-independent; this
        only trades real- vs reciprocal-space convergence).
      real_shells: Half-width (in lattice vectors) of the real-space image block.
      recip_shells: Half-width of the reciprocal-space G block.

    Returns:
      Callable f(positions) -> scalar, where positions has shape (N, 2), giving
      the periodic 1/(r + a)^3 energy per unit interaction strength.
    """
    A = np.asarray(lattice, dtype=np.float64)
    if A.shape != (2, 2):
        raise ValueError(
            "Ewald summation is implemented for 2D lattices only "
            f"(got lattice shape {A.shape})."
        )
    a = float(interaction_small_length_cutoff)
    area = float(np.abs(np.linalg.det(A)))
    if eta is None:
        eta = eta_scale * np.sqrt(np.pi) / np.sqrt(area)
    eta = float(eta)
    A_inv = np.linalg.inv(A)
    B = 2.0 * np.pi * A_inv.T  # reciprocal primitive vectors as columns.

    # Real-space image shifts R = A n (includes the origin).
    rr = np.arange(-real_shells, real_shells + 1)
    ns = np.array(list(itertools.product(rr, repeat=2)))
    Rsh = (A @ ns.T).T
    Rzero = np.linalg.norm(Rsh, axis=1) < 1e-9

    # Reciprocal vectors G = B m (excludes the origin).
    rg = np.arange(-recip_shells, recip_shells + 1)
    ms = np.array([m for m in itertools.product(rg, repeat=2) if m != (0, 0)])
    G = (B @ ms.T).T
    Gn = np.linalg.norm(G, axis=1)
    from scipy.special import erfc as _erfc_np  # host-side kernel precompute.

    IG = 2.0 * eta * np.exp(-(Gn**2) / (4.0 * eta**2)) - np.sqrt(np.pi) * Gn * _erfc_np(
        Gn / (2.0 * eta)
    )
    self_const = 2.0 * np.sqrt(
        np.pi
    ) * eta * n_particles**2 / area - 2.0 * n_particles * eta**3 / (
        3.0 * np.sqrt(np.pi)
    )

    # Move precomputed constants onto the device.
    A_j = jnp.asarray(A)
    A_inv_j = jnp.asarray(A_inv)
    Rsh_j = jnp.asarray(Rsh)
    self_pair = jnp.asarray(Rzero)  # (nR,) True where R == 0
    G_j = jnp.asarray(G)
    IG_j = jnp.asarray(IG)
    inv_sqrt_pi = float(1.0 / np.sqrt(np.pi))
    sqrt_pi_over_area = float(np.sqrt(np.pi) / area)
    n_elec = int(n_particles)

    def ewald_energy(positions: jnp.ndarray) -> jnp.ndarray:
        """positions: (N, 2) -> scalar periodic 1/(r+a)^3 energy per unit strength."""
        pos = positions.reshape(n_elec, 2)
        # Minimum-image pair displacements under the (possibly non-orthogonal) cell.
        d = pos[:, None, :] - pos[None, :, :]  # (N, N, 2)
        frac = jnp.einsum("ij,nmj->nmi", A_inv_j, d)
        d = jnp.einsum("ij,nmj->nmi", A_j, frac - jnp.round(frac))

        dd = d[:, :, None, :] + Rsh_j[None, None, :, :]  # (N, N, nR, 2)
        eye = jnp.eye(n_elec, dtype=bool)
        self_mask = eye[:, :, None] & self_pair[None, None, :]
        # Guard the (i=j, R=0) self entries before taking the norm: at those points
        # dd = 0, and both norm(0) and 1/r have NaN gradients. Replacing the squared
        # distance by 1 there keeps the value and gradient finite (the term is masked
        # to 0 below regardless).
        r2 = jnp.sum(dd**2, axis=-1)  # (N, N, nR)
        r_safe = jnp.sqrt(jnp.where(self_mask, 1.0, r2))
        z = eta * r_safe
        # Real-space near-field of the 1/(r+a)^3 energy. Naively this is
        #   phi_short(r) + [1/(r+a)^3 - 1/r^3]
        # with phi_short = [erfc(z) + (2/sqrt(pi)) z e^{-z^2}] / r^3, z = eta r. For
        # close particles (small r) the two 1/r^3 pieces are individually huge and
        # cancel, destroying float32 precision (bosons *can* approach closely, so this
        # blows training up). Regroup exactly as
        #   near = 1/(r+a)^3 + [phi_short - 1/r^3] = 1/(r+a)^3 + eta^3 * s(eta r),
        #   s(z) = ((2/sqrt(pi)) z e^{-z^2} - erf(z)) / z^3   (smooth, -> -4/(3 sqrt(pi))),
        # where 1/(r+a)^3 is bounded by 1/a^3 and s(z) is evaluated by a small-z
        # Taylor series (its leading O(z) terms cancel analytically), so no large
        # cancellation ever occurs.
        z2 = z * z
        c = 2.0 * inv_sqrt_pi
        # Closed form of s(z), accurate for z >= 0.5 (guard small z to keep the
        # unused branch's gradient finite).
        z_closed = jnp.where(z < 0.5, 1.0, z)
        s_closed = (
            c * z_closed * jnp.exp(-(z_closed**2)) - jax.scipy.special.erf(z_closed)
        ) / (z_closed**3)
        # Taylor series of s(z) for small z: s = c*(-2/3 + 2/5 z^2 - 1/7 z^4 + 1/27 z^6 - ...)
        s_series = c * (
            -2.0 / 3.0 + (2.0 / 5.0) * z2 - (1.0 / 7.0) * z2**2 + (1.0 / 27.0) * z2**3
        )
        s = jnp.where(z < 0.5, s_series, s_closed)
        near = 1.0 / (r_safe + a) ** 3 + (eta**3) * s
        u_real = 0.5 * jnp.sum(jnp.where(self_mask, 0.0, near))

        phase = pos @ G_j.T  # (N, nG)
        rho = jnp.sum(jnp.exp(1j * phase), axis=0)  # (nG,)
        u_rec = sqrt_pi_over_area * jnp.sum(IG_j * (rho.real**2 + rho.imag**2))

        return u_real + u_rec + self_const

    return ewald_energy


def disk_wall(radius, disk_radius=10.0, barrier_sharpness=1.0):
    """Smooth confining wall used by the disk example, in simulation units."""
    rim_width = 0.1 / barrier_sharpness
    height = 10 * barrier_sharpness
    return jnp.sum(height * (1 + jnp.tanh((radius - disk_radius) / rim_width)) / 2)


def local_energy(
    f,
    n_particles,
    geometry=None,
    *,
    interaction_small_length_cutoff=0.1,
    interaction_truncation_limit=5,
    barrier_sharpness=1.0,
    use_scan=False,
    laplacian_method="default",
    interaction_cutoff_radius=None,
    interaction_tail_correction=False,
    interaction_tail_g2_r=None,
    interaction_tail_g2_values=None,
    interaction_ewald=False,
    interaction_ewald_eta=None,
    interaction_ewald_eta_scale=1.0,
    interaction_ewald_real_shells=3,
    interaction_ewald_recip_shells=8,
    interaction_type="inverse_cubic",
    short_range_width=0.1,
):
    """Build H psi / psi for one bosonic walker in the configured domain."""
    geometry = geometry or Geometry()
    lattice = geometry.lattice
    if interaction_type not in ("inverse_cubic", "short_range"):
        raise ValueError(f"Unknown interaction_type: {interaction_type}")
    if short_range_width <= 0 or barrier_sharpness <= 0:
        raise ValueError("Interaction width and barrier sharpness must be positive")
    if (interaction_ewald or interaction_tail_correction) and (
        lattice is None or interaction_type != "inverse_cubic"
    ):
        raise ValueError(
            "Ewald and tail correction require periodic inverse_cubic physics"
        )
    if interaction_ewald and interaction_tail_correction:
        raise ValueError("Ewald and tail correction are mutually exclusive")
    tail_coeff = 0.0
    if interaction_tail_correction:
        if interaction_cutoff_radius is None:
            raise ValueError("Tail correction requires interaction_cutoff_radius")
        tail_coeff = pair_tail_coefficient(
            lattice,
            n_particles,
            interaction_small_length_cutoff,
            interaction_cutoff_radius,
            interaction_tail_g2_r,
            interaction_tail_g2_values,
        )
    ewald = None
    if interaction_ewald:
        ewald = make_ewald_1r3_potential(
            lattice,
            n_particles,
            interaction_small_length_cutoff,
            eta=interaction_ewald_eta,
            eta_scale=interaction_ewald_eta_scale,
            real_shells=interaction_ewald_real_shells,
            recip_shells=interaction_ewald_recip_shells,
        )
    kinetic_energy = local_kinetic_energy(
        f, use_scan=use_scan, laplacian_method=laplacian_method
    )

    def evaluate(params, key, data):
        del key
        _, pairs, radius, distances = networks.construct_input_features(
            data.positions, geometry.center, geometry.ndim
        )
        if ewald is not None:
            potential = data.interaction_strength * ewald(data.positions)
        else:
            if lattice is not None:
                distances = periodic_pair_distances(pairs, lattice)
            potential = pair_potential(
                pairs,
                distances,
                data.interaction_strength,
                interaction_small_length_cutoff=interaction_small_length_cutoff,
                lattice=lattice,
                interaction_truncation_limit=interaction_truncation_limit,
                interaction_cutoff_radius=interaction_cutoff_radius,
                interaction_type=interaction_type,
                short_range_width=short_range_width,
            )
            if interaction_tail_correction:
                potential += data.interaction_strength * tail_coeff
        if geometry.kind == "disk":
            potential += disk_wall(
                radius[..., 0, 0], geometry.radius, barrier_sharpness
            )
        kinetic = kinetic_energy(params, data)
        return potential + kinetic, LocalEnergyAux(kinetic=kinetic, potential=potential)

    return evaluate
