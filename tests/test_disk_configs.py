"""Regression coverage for configurable geometry, interactions, and models."""

from ferminet import base_2DEG_config, checkpoint, hamiltonian, networks, psiformer
from ferminet import train2DEG
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from test_paper_workflow import _script_config, _short_run, _check_run, _csv_rows


def test_disk_wall_and_gaussian_local_energy():
  """Check the restored disk Hamiltonian against independent analytic values."""
  atoms = jnp.zeros((1, 2))
  charges = jnp.ones(1)
  positions = jnp.array([0.3, 0., 1.3, 0.])
  data = networks.FermiNetData(
      positions=positions, spins=jnp.ones(2), atoms=atoms, charges=charges,
      interaction_strength=jnp.asarray(0.7))
  # log psi = -a sum(r_i^2): T = a*d - 2*a^2*sum(r_i^2).
  a = 0.2
  def wavefunction(params, pos, spins, atoms, charges, coupling):
    del params, spins, atoms, charges, coupling
    return jnp.asarray(1.), -a * jnp.sum(pos**2)

  local_energy = hamiltonian.local_energy(
      wavefunction, charges, (2, 0), geometry='disk', disk_radius=1.,
      interaction_type='short_range', short_range_width=0.5)
  total, parts = jax.jit(local_energy)(None, jax.random.PRNGKey(0), data)
  wall = np.sum(10 * (1 + np.tanh((np.array([0.3, 1.3]) - 1) / 0.1)) / 2)
  pair = 0.7 * np.exp(-1 / (2 * 0.5**2)) / (2 * np.pi * 0.5**2)
  kinetic = a * 4 - 2 * a**2 * np.sum(np.array(positions)**2)
  np.testing.assert_allclose(parts.potential, wall + pair, rtol=1e-6)
  np.testing.assert_allclose(parts.kinetic, kinetic, rtol=1e-6)
  np.testing.assert_allclose(total, wall + pair + kinetic, rtol=1e-6)

  # The same radii in a periodic cell must never introduce a wall.
  radii = jnp.array([[[9.]], [[10.]], [[12.]]])
  assert hamiltonian.potential_electron_nuclear(charges, radii) == 0
  np.testing.assert_allclose(
      hamiltonian.potential_electron_nuclear(charges, radii, geometry='disk'),
      15., atol=1e-5)


@pytest.mark.parametrize('interaction', ['short_range', 'inverse_cubic'])
def test_pair_interaction_selection(interaction):
  pos = jnp.array([0., 0., 0.1, 0.])
  _, ee, _, r_ee = networks.construct_input_features(pos, jnp.zeros((1, 2)), ndim=2)
  actual = hamiltonian.potential_electron_electron(
      ee, r_ee, 0.1, interaction_type=interaction)
  expected = (0.1 * np.exp(-0.5) / (2 * np.pi * 0.1**2)
              if interaction == 'short_range' else 0.1 / 0.2**3)
  np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_invalid_physics_configs_fail_before_training():
  cfg = _script_config('dipole-pbc-hex-N25-pbc.py')
  assert train2DEG.validate_boson_config(cfg) == 'periodic'
  cfg.geometry = 'disk'
  with pytest.raises(ValueError, match='lattice'):
    train2DEG.train(cfg)
  cfg.geometry = 'periodic'
  cfg.interaction_type = 'short_range'
  with pytest.raises(ValueError, match='Ewald'):
    train2DEG.train(cfg)
  cfg.interaction_ewald = False
  with pytest.raises(ValueError, match='Jastrow'):
    train2DEG.train(cfg)
  cfg.network.smooth_periodic_jastrow = False
  assert train2DEG.validate_boson_config(cfg) == 'periodic'
  cfg.network.lambda_conditioning.mode = 'none'
  with pytest.raises(ValueError, match='single interaction strength'):
    train2DEG.train(cfg)


def test_unconditioned_model_has_no_lambda_parameters():
  cfg = base_2DEG_config.default()
  assert cfg.network.lambda_conditioning.mode == 'additive'
  model = psiformer.make_fermi_net(
      (2, 0), jnp.ones(1), ndim=2, determinants=2, num_layers=1,
      num_heads=2, heads_dim=4, mlp_hidden_dims=(8,), predict_logits=True,
      lambda_conditioning_mode='none', use_layer_norm=True)
  params = model.init(jax.random.PRNGKey(13))
  paths = [str(path) for path, _ in jax.tree_util.tree_flatten_with_path(params)[0]]
  assert not any('lambda_embed' in path or 'cross_attn' in path for path in paths)
  args = (params, jnp.array([0.1, 0.2, 0.8, 0.5]), jnp.ones(2),
          jnp.zeros((1, 2)), jnp.ones(1))
  first = model.apply(*args, jnp.asarray(0.1))
  second = model.apply(*args, jnp.asarray(10.))
  jax.tree_util.tree_map(np.testing.assert_array_equal, first, second)


@pytest.mark.parametrize('mode', ['additive', 'none'])
def test_disk_training_and_checkpoint_inference(tmp_path, mode):
  cfg = _script_config('bosef-10e-rep01.py')
  assert cfg.system.pyscf_mol.nelec == (10, 0)
  assert train2DEG.validate_boson_config(cfg) == 'disk'
  cfg.network.lambda_conditioning.mode = mode
  couplings = [0.0, 0.05, 0.1] if mode == 'additive' else [0.1]
  cfg.interaction_strength_training_set = couplings if mode == 'additive' else None
  cfg.log.tripwires = True
  train_path = tmp_path / 'train'
  _short_run(cfg, train_path, 2)
  trained = _check_run(train_path, [0, 1], couplings, 4)

  cfg.optim.optimizer = 'none'
  cfg.interaction_strength_training_set = None
  cfg.interaction_strength = 0.08 if mode == 'additive' else 0.1
  infer_path = tmp_path / 'infer'
  _short_run(cfg, infer_path, 1, restore=train_path)
  rows = _csv_rows(infer_path, 'train_stats.csv', [0],
                   ['energy', 'kinetic', 'potential', 'pmove'])
  row = rows[0]
  np.testing.assert_allclose(float(row['energy']),
      float(row['kinetic']) + float(row['potential']), rtol=5e-5, atol=1e-4)
  _, data, restored, optimizer, *_ = checkpoint.restore(
      checkpoint.find_last_checkpoint(str(infer_path)))
  np.testing.assert_allclose(data.interaction_strength, cfg.interaction_strength)
  assert optimizer is None
  jax.tree_util.tree_map(np.testing.assert_array_equal, trained, restored)
