"""25-particle HEX gen model with the smooth-periodic K_0 two-body Jastrow.

Interaction: bare 1/r^3 dipole (interaction_small_length_cutoff = 0.0), evaluated by
exact 2D Ewald summation (includes the self-image / Madelung term; absolute
energies are NOT comparable to the truncated-sum gen-HEX runs).

New features enabled (all default-off elsewhere, so existing runs are unchanged):
  * cfg.network.smooth_periodic_jastrow -- fixed (non-trainable) two-body factor
      log f_2 = log K_0(2 sqrt(lambda / rho(r~))), r~ the smooth periodic
      distance. Imposes the exact 2D-dipole short-range cusp so E_L is finite at
      contact and the "contact pocket" instability is removed by construction;
      smooth everywhere (no min-image / matching seam), so the variational bound
      holds. This is FIX A from FABLE-FEEDBACK.md.
  * cfg.optim.per_lambda_clip -- clip local energies within each lambda group
      (the global clip window is dominated by the inter-lambda spread and never
      fires in multi-lambda training).
  * cfg.log.tripwires -- log per-walker min pair distance, max|E_L|, and per-
      lambda mean energies to tripwires.csv; warn on approach to contact.
"""
from pyscf import gto

from ferminet import base_2DEG_config
from ferminet import train2DEG
from ferminet.utils import system

from ferminet.pbc import envelopes

import numpy as np


N_PARTICLES = 25
# Hex primitive cell area = L^2 * sqrt(3)/2; match density n = 1/pi:
#   L^2 * sqrt(3)/2 = N*pi  ->  L = sqrt(2*N*pi/sqrt(3))
TORUS_LENGTH = np.sqrt(2 * N_PARTICLES * np.pi / np.sqrt(3))

mol = gto.Mole()
mol.build(
    atom = 'H  0 0 0',
    charge = 1 - N_PARTICLES,
    spin = N_PARTICLES,
    basis = 'sto-3g', unit='bohr')

cfg = base_2DEG_config.default()
cfg.system.pyscf_mol = mol

# Pretraining is not currently implemented for systems in PBC
cfg.pretrain.method = None

cfg.lattice_type = 'hexagonal'
lattice = base_2DEG_config.make_lattice(
    cfg.lattice_type, TORUS_LENGTH, ndim=2)

cfg.network.make_feature_layer_fn = (
    "ferminet.pbc.feature_layer.make_pbc_feature_layer")
cfg.network.make_feature_layer_kwargs = {
    "lattice": lattice,
    "include_r_ae": False
}

cfg.network.make_envelope_fn = (
    "ferminet.envelopes.make_null_envelope")
cfg.network.make_envelope_kwargs = {}
cfg.network.full_det = True
cfg.network.predict_logits = True

# Set training parameters
cfg.optim.iterations = 50000
cfg.network.network_type = 'psiformer'
cfg.log.save_path = './dipole-25e-pbc-gen-HEX-smoothperiodic-bare'

# Multi-lambda generative training
cfg.interaction_strength_training_set = [0.1, 0.5, 5.0, 20.0, 50.0, 80.0]
cfg.interaction_strength = 1.0  # default for inference
cfg.interaction_small_length_cutoff = 0.0  # bare 1/r^3 dipole

# === Exact Ewald summation for the interaction ===
cfg.interaction_ewald = True
cfg.interaction_ewald_real_shells = 4
cfg.interaction_ewald_recip_shells = 12

# === FIX A: smooth-periodic K_0 two-body Jastrow (uses the PBC lattice) ===
cfg.network.smooth_periodic_jastrow = True
cfg.network.smooth_periodic_jastrow_rmatch_frac = 1.0

# === Per-lambda clipping + tripwire diagnostics ===
cfg.optim.per_lambda_clip = True
cfg.log.tripwires = True
cfg.log.tripwire_frequency = 100

# === Conservative optimizer (matches WORKING) ===
cfg.optim.lr.rate = 0.02
cfg.optim.lr.delay = 10000.0
cfg.optim.lr.decay = 1.5
cfg.optim.clip_local_energy = 3.0
cfg.optim.kfac.damping = 0.01
cfg.optim.kfac.norm_constraint = 0.003

train2DEG.train(cfg)
