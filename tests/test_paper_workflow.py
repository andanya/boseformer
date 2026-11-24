"""BoseNet regression tests for the N=25 bare dipolar paper runs."""

import csv
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

from ferminet import base_2DEG_config
from ferminet import checkpoint
from ferminet import train2DEG
from ferminet.utils import system
import jax
import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
STEM = 'dipole-pbc-hex-N25-pbc'
BASE_COUPLINGS = [0.1, 0.5, 5.0, 20.0, 50.0, 80.0]
FINE_TUNE_COUPLINGS = [20.0, 25.0, 30.0, 35.0, 40.0]
GPU_ONLY = pytest.mark.skipif(
    os.environ.get('BOSENET_RUN_GPU_TESTS') != '1',
    reason='Set BOSENET_RUN_GPU_TESTS=1 to run the GPU smoke test.',
)


def _script_config(filename, *args):
  """Load a paper entry point while intercepting its final training call."""
  script = SCRIPTS / filename
  with (
      patch.object(sys, 'argv', [str(script), *args]),
      patch.object(train2DEG, 'train') as train,
  ):
    runpy.run_path(str(script), run_name='__main__')
  train.assert_called_once()
  return train.call_args.args[0]


@pytest.mark.parametrize(
    ('suffix', 'couplings', 'training_steps'),
    [
        ('', BASE_COUPLINGS, 50_000),
        ('-ft2040', FINE_TUNE_COUPLINGS, 70_000),
        ('-int30', None, 50_000),
    ],
    ids=['base', 'fine-tune', 'single-coupling'],
)
def test_paper_entry_points(suffix, couplings, training_steps):
  train = _script_config(f'{STEM}{suffix}.py')
  infer = _script_config(f'infer-{STEM}{suffix}.py', '30.0', '7')

  for cfg in (train, infer):
    lattice = cfg.network.make_feature_layer_kwargs.lattice
    particles = cfg.system.electrons or cfg.system.pyscf_mol.nelec
    assert tuple(particles) == (25, 0)
    assert cfg.system.ndim == 2
    np.testing.assert_allclose(np.linalg.det(lattice), 25 * np.pi)
    assert cfg.interaction_ewald
    assert cfg.interaction_small_length_cutoff == 0
    assert (cfg.interaction_ewald_real_shells,
            cfg.interaction_ewald_recip_shells) == (4, 12)
    assert cfg.network.smooth_periodic_jastrow
    assert cfg.network.network_type == 'psiformer'
    assert cfg.network.predict_logits and cfg.network.boson_head == 'product'
    assert cfg.network.determinants == 16
    layers = cfg.network.psiformer
    assert (layers.num_layers, layers.num_heads, layers.heads_dim) == (4, 4, 64)
    assert tuple(layers.mlp_hidden_dims) == (256,)
    assert cfg.pretrain.method is None

  assert train.optim.optimizer == 'kfac'
  assert train.optim.iterations == training_steps
  assert train.interaction_strength_training_set == couplings
  assert train.optim.per_lambda_clip == (couplings is not None)
  if suffix == '-ft2040':
    assert train.log.restore_path == _script_config(f'{STEM}.py').log.save_path
  if suffix == '-int30':
    assert train.interaction_strength == 30.0

  assert infer.log.restore_path == train.log.save_path
  assert infer.optim.optimizer == 'none'
  assert infer.optim.iterations == 7
  assert infer.interaction_strength == 30.0
  assert not infer.interaction_strength_training_set
  assert infer.observables.nk0


def _short_run(cfg, directory, steps, *, restore=None, small=True):
  """Keep the paper physics while running a few steps in a temp directory."""
  cfg.log.save_path = str(directory)
  cfg.log.restore_path = str(restore) if restore is not None else ''
  cfg.log.save_frequency = 0
  cfg.log.tripwire_frequency = 1
  cfg.optim.iterations = steps
  cfg.batch_size = 12 * jax.device_count()
  cfg.mcmc.steps = 2
  cfg.mcmc.burn_in = 1
  cfg.mcmc.init_width = 1.0 if small else 3.0
  cfg.debug.deterministic = True
  cfg.debug.check_nan = True
  if small:
    # The current 2D API represents the periodic background with a dummy atom.
    cfg.system.pyscf_mol = None
    cfg.system.molecule = [system.Atom('H', (0.0, 0.0))]
    cfg.system.electrons = (4, 0)
    side = np.sqrt(2 * np.pi * 4 / np.sqrt(3))
    cfg.network.make_feature_layer_kwargs.lattice = (
        base_2DEG_config.make_lattice('hexagonal', side))
    cfg.network.determinants = 2
    cfg.network.psiformer.num_layers = 1
    cfg.network.psiformer.num_heads = 2
    cfg.network.psiformer.heads_dim = 4
    cfg.network.psiformer.mlp_hidden_dims = (8,)
  train2DEG.train(cfg)


def _csv_rows(directory, name, steps, numeric_columns):
  with (directory / name).open(newline='') as stream:
    rows = list(csv.DictReader(stream))
  assert [int(row['step']) for row in rows] == steps
  for column in numeric_columns:
    assert np.isfinite([float(row[column]) for row in rows]).all(), column
  return rows


def _check_run(directory, steps, couplings, n_particles, *, inference=False):
  columns = ['energy', 'ewmean', 'ewvar', 'pmove']
  if inference:
    columns += ['kinetic', 'potential']
  stats = _csv_rows(directory, 'train_stats.csv', steps, columns)
  assert all(0 <= float(row['pmove']) <= 1 for row in stats)
  tripwires = _csv_rows(
      directory, 'tripwires.csv', steps, ['rmin', 'frac_close', 'max_abs_el'])
  for row in tripwires:
    by_coupling = dict(item.split(':') for item in
                       row['per_lambda_energies'].split(';'))
    assert set(map(float, by_coupling)) == set(couplings)
    assert np.isfinite(list(map(float, by_coupling.values()))).all()

  previous, current = [
      np.load(directory / f'walkers_{name}.host0.npy', allow_pickle=False)
      for name in ('prev', 'last')
  ]
  for positions in (previous, current):
    assert positions.shape == (12 * jax.local_device_count(), 2 * n_particles)
    assert np.isfinite(positions).all()
  assert not np.array_equal(previous, current)

  filename = checkpoint.find_last_checkpoint(str(directory))
  assert filename is not None, f'No checkpoint in {directory}'
  next_step, walkers, params, opt_state, *_ = checkpoint.restore(filename)
  assert next_step == steps[-1] + 1
  expected = np.resize(couplings, 12).astype(np.float32)
  for device_couplings in walkers.interaction_strength:
    np.testing.assert_array_equal(device_couplings, expected)
  assert jax.tree_util.tree_leaves(params)
  assert all(np.isfinite(leaf).all()
             for leaf in jax.tree_util.tree_leaves(params))
  assert (opt_state is None) == inference

  if inference:
    energy = [float(row['energy']) for row in stats]
    components = [
        float(row['kinetic']) + float(row['potential']) for row in stats
    ]
    np.testing.assert_allclose(energy, components, rtol=5e-5, atol=1e-4)
    nk0 = _csv_rows(directory, 'nk0_samples.csv', steps,
                    ['ratio_mean', 'ratio_sq_mean'])
    assert all(int(row['n_particles']) == n_particles for row in nk0)
    assert all(float(row['ratio_mean']) > 0 for row in nk0)
  return params


def test_small_periodic_training_fine_tuning_and_inference(tmp_path):
  """Exercise KFAC, fine-tuning, and frozen sampling on small inputs."""
  base_path = tmp_path / 'base'
  base = _script_config(f'{STEM}.py')
  _short_run(base, base_path, 2)
  base_params = _check_run(base_path, [0, 1], BASE_COUPLINGS, 4)

  fine_path = tmp_path / 'fine-tune'
  fine = _script_config(f'{STEM}-ft2040.py')
  _short_run(fine, fine_path, 3, restore=base_path)
  fine_params = _check_run(fine_path, [2], FINE_TUNE_COUPLINGS, 4)
  changes = jax.tree_util.tree_map(
      lambda old, new: np.any(old != new), base_params, fine_params)
  assert any(jax.tree_util.tree_leaves(changes))

  infer_path = tmp_path / 'inference'
  infer = _script_config(f'infer-{STEM}-ft2040.py', '26.0')
  _short_run(infer, infer_path, 2, restore=fine_path)
  infer_params = _check_run(infer_path, [0, 1], [26.0], 4, inference=True)
  jax.tree_util.tree_map(
      np.testing.assert_array_equal, fine_params, infer_params)


def test_small_single_coupling_training(tmp_path):
  """Train the fixed-coupling reference without multi-coupling batches."""
  reference = _script_config(f'{STEM}-int30.py')
  _short_run(reference, tmp_path, 1)
  _check_run(tmp_path, [0], [30.0], 4)


@GPU_ONLY
def test_gpu_paper_size_training_and_frozen_inference(tmp_path):
  """Run the N=25 architecture with a small walker batch on NVIDIA CUDA."""
  assert jax.default_backend() in ('cuda', 'gpu'), 'Select a JAX CUDA device.'
  base_path = tmp_path / 'base'
  base = _script_config(f'{STEM}.py')
  _short_run(base, base_path, 2, small=False)
  base_params = _check_run(base_path, [0, 1], BASE_COUPLINGS, 25)

  infer_path = tmp_path / 'inference'
  infer = _script_config(f'infer-{STEM}.py', '26.0')
  _short_run(infer, infer_path, 1, restore=base_path, small=False)
  infer_params = _check_run(infer_path, [0], [26.0], 25, inference=True)
  jax.tree_util.tree_map(
      np.testing.assert_array_equal, base_params, infer_params)
