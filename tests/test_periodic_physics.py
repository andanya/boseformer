"""Numerical checks for the bare hexagonal Boseformer model."""

import os

from ferminet import base_2DEG_config
from ferminet import envelopes
from ferminet import hamiltonian
from ferminet import psiformer
from ferminet.pbc import feature_layer
import jax
from jax.experimental import enable_x64
import jax.numpy as jnp
import numpy as np
import pytest


N_PARTICLES = 25


def _paper_lattice():
  side = np.sqrt(2 * np.pi * N_PARTICLES / np.sqrt(3))
  return base_2DEG_config.make_lattice('hexagonal', side)


def _bare_two_particle_image_sum(lattice, displacement, shells):
  """Direct square-shell lattice sum including both particles' self images."""
  axis = np.arange(-shells, shells + 1)
  indices = np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1)
  shifts = indices.reshape(-1, 2) @ lattice.T
  self_images = shifts[np.any(shifts != 0, axis=1)]
  pair_radii = np.linalg.norm(shifts + displacement, axis=-1)
  self_radii = np.linalg.norm(self_images, axis=-1)
  return np.sum(pair_radii**-3) + np.sum(self_radii**-3)


def test_bare_ewald_matches_direct_periodic_image_sum():
  lattice = _paper_lattice()
  displacement = np.array([1.7, 2.1])
  positions = np.stack((np.zeros(2), displacement))
  # Extrapolate the leading 1 / shell width error for a 2D sum of 1/r³.
  coarse = _bare_two_particle_image_sum(lattice, displacement, 64)
  fine = _bare_two_particle_image_sum(lattice, displacement, 128)
  expected = fine + (fine - coarse) * (64.5 / 64)

  with enable_x64():
    potential = jax.jit(hamiltonian.make_ewald_1r3_potential(
        lattice, 2, interaction_small_length_cutoff=0.0,
        real_shells=4, recip_shells=12))
    energy = potential(positions)
    wrapped = positions.copy()
    wrapped[1] += lattice[:, 0] - lattice[:, 1]
    np.testing.assert_allclose(potential(wrapped), energy, atol=1e-10)
    np.testing.assert_allclose(energy, expected, rtol=2e-5, atol=1e-8)


@pytest.mark.parametrize(
    'backend',
    [
        'cpu',
        pytest.param('gpu', marks=pytest.mark.skipif(
            os.environ.get('BOSENET_RUN_GPU_TESTS') != '1',
            reason='Set BOSENET_RUN_GPU_TESTS=1 to run the GPU smoke test.')),
    ],
)
def test_paper_wavefunction_symmetries_and_conditioning(backend):
  """Check the full paper-sized forward pass without a training checkpoint."""
  if backend == 'gpu':
    assert jax.default_backend() in ('cuda', 'gpu'), 'Select a JAX CUDA device.'
  lattice = _paper_lattice()
  cfg = base_2DEG_config.default()
  nspins = (N_PARTICLES, 0)
  with jax.default_device(jax.devices(backend)[0]):
    atoms = jnp.zeros((1, 2))
    charges = jnp.ones((1,))
    spins = jnp.ones((N_PARTICLES,))
    features = feature_layer.make_pbc_feature_layer(
        natoms=1, nspins=nspins, ndim=2, lattice=lattice, include_r_ae=False)
    network = psiformer.make_fermi_net(
        nspins, charges, ndim=2, determinants=cfg.network.determinants,
        envelope=envelopes.make_null_envelope(), feature_layer=features,
        jastrow='none', predict_logits=True, smooth_periodic_jastrow=True,
        smooth_periodic_jastrow_lattice=lattice, **cfg.network.psiformer)
    params = network.init(jax.random.PRNGKey(23))
    rng = np.random.default_rng(23)
    positions = rng.uniform(size=(N_PARTICLES, 2)) @ lattice.T
    permuted = positions[rng.permutation(N_PARTICLES)]
    wrapped = positions.copy()
    wrapped[0] += lattice[:, 0]
    apply = jax.jit(network.apply)

    def evaluate(pos, coupling):
      return apply(params, jnp.asarray(pos).reshape(-1), spins, atoms,
                   charges, jnp.asarray(coupling))

    sign, log_amplitude = evaluate(positions, 25.0)
    assert float(sign) == 1.0
    assert np.isfinite(log_amplitude)
    for equivalent in (permuted, wrapped):
      other_sign, other_log_amplitude = evaluate(equivalent, 25.0)
      np.testing.assert_array_equal(other_sign, sign)
      np.testing.assert_allclose(other_log_amplitude, log_amplitude,
                                 rtol=0, atol=3e-4)

    # Check lambda conditioning in the learned orbitals separately from the
    # fixed Jastrow, which also depends on lambda.
    orbitals = jax.jit(network.orbitals)
    flat_positions = jnp.asarray(positions).reshape(-1)
    arguments = (params, flat_positions, spins, atoms, charges)
    low = orbitals(*arguments, jnp.asarray(25.0))
    high = orbitals(*arguments, jnp.asarray(35.0))
    differences = jax.tree_util.tree_map(
        lambda x, y: np.max(np.abs(x - y)), low, high)
    assert max(jax.tree_util.tree_leaves(differences)) > 1e-4
