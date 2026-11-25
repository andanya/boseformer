"""Analytic regression test for the smooth periodic dipolar Jastrow."""

from network import smooth_jastrow
import jax
from jax.experimental import enable_x64
import jax.numpy as jnp
import numpy as np
from scipy.special import k0e, k1e


def _analytic_pair_kinetic(displacement, lattice, interaction_strength):
    """Returns -∇²ψ/ψ for the two-body factor as a function of separation."""
    metric = lattice.T @ lattice
    phase_jacobian = 2 * np.pi * np.linalg.inv(lattice)
    phase = phase_jacobian @ displacement
    sine = np.sin(phase)
    cosine = np.cos(phase)
    one_minus_cosine = 1 - cosine
    q = one_minus_cosine @ metric @ one_minus_cosine + sine @ metric @ sine

    # Differentiate q in phase coordinates, then transform to Cartesian space.
    grad_q = (
        2
        * phase_jacobian.T
        @ (sine * (metric @ one_minus_cosine) + cosine * (metric @ sine))
    )
    phase_hessian = 2 * (
        np.diag(cosine * (metric @ one_minus_cosine) - sine * (metric @ sine))
        + np.outer(sine, sine) * metric
        + np.outer(cosine, cosine) * metric
    )
    laplacian_q = np.trace(phase_jacobian.T @ phase_hessian @ phase_jacobian)

    radius = np.sqrt(q) / (2 * np.pi)
    grad_radius = grad_q / (4 * np.pi * np.sqrt(q))
    laplacian_radius = laplacian_q / (4 * np.pi * np.sqrt(q)) - (grad_q @ grad_q) / (
        8 * np.pi * q**1.5
    )

    saturation_length = np.min(np.linalg.norm(lattice, axis=0))
    tanh = np.tanh(radius / saturation_length)
    slope = 1 - tanh**2
    rho = saturation_length * tanh + 1e-12
    grad_rho = slope * grad_radius
    laplacian_rho = slope * laplacian_radius - 2 * tanh * slope / saturation_length * (
        grad_radius @ grad_radius
    )

    arg = 2 * np.sqrt(interaction_strength / rho)
    grad_arg = -arg / (2 * rho) * grad_rho
    laplacian_arg = -arg / (2 * rho) * laplacian_rho + 3 * arg / (4 * rho**2) * (
        grad_rho @ grad_rho
    )
    # K0' = -K1 and K0'' = K0 - K0'/arg. Scaling cancels in K1/K0.
    k0_log_slope = -k1e(arg) / k0e(arg)
    return -(
        (1 - k0_log_slope / arg) * (grad_arg @ grad_arg) + k0_log_slope * laplacian_arg
    )


def test_smooth_contact_matches_analytic():
    # Place one pair in the hexagonal cell for N=37 at density 1/pi.
    side = np.sqrt(2 * np.pi * 37 / np.sqrt(3))
    lattice = side * np.array([[1, 0.5], [0, np.sqrt(3) / 2]])
    angles = np.deg2rad([0, 30, 60])
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    radii = np.array([0.1, 0.03, 0.01, 0.003, 0.001])
    displacements = (directions[:, None, :] * radii[None, :, None]).reshape(-1, 2)
    interaction_strength = 1.0
    expected = np.array(
        [
            _analytic_pair_kinetic(d, lattice, interaction_strength)
            for d in displacements
        ]
    )

    with jax.default_device(jax.devices("cpu")[0]), enable_x64():
        pair_jastrow = smooth_jastrow.make_smooth_periodic_log_jastrow(lattice)

        def log_pair(displacement):
            positions = jnp.concatenate((displacement, jnp.zeros_like(displacement)))
            return pair_jastrow(positions, interaction_strength)

        gradient = jax.grad(log_pair)
        hessian = jax.jacfwd(gradient)

        def kinetic(displacement):
            # Both particles give the same contribution and each carries -1/2.
            grad = gradient(displacement)
            return -(jnp.trace(hessian(displacement)) + grad @ grad)

        actual = jax.jit(jax.vmap(kinetic))(jnp.asarray(displacements))

    actual = np.asarray(actual)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=0, equal_nan=False)

    # Check the smaller residual after cancellation of the leading divergence.
    # The production Bessel approximation contributes up to about 2 energy units
    # of absolute error at the shortest separation in this check.
    separations = np.linalg.norm(displacements, axis=1)
    bare_potential = interaction_strength / separations**3
    np.testing.assert_allclose(
        actual + bare_potential,
        expected + bare_potential,
        rtol=0.1,
        atol=1e-4,
        equal_nan=False,
    )
