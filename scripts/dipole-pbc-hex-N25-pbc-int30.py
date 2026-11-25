"""25-particle SINGLE-lambda=30 reference (bare 1/r^3) with the smooth-periodic K_0 two-body Jastrow.

Interaction: bare 1/r^3 dipole (interaction_small_length_cutoff = 0.0), evaluated by
exact 2D Ewald summation (includes the self-image / Madelung term; absolute
energies are NOT comparable to the truncated-sum gen-HEX runs).

Enabled options:
  * cfg.network.smooth_periodic_jastrow -- fixed (non-trainable) two-body factor
      log f_2 = log K_0(2 sqrt(lambda / rho(r~))), r~ the smooth periodic
      distance. Imposes the exact 2D-dipole short-range cusp so E_L is finite at
      contact and the "contact pocket" instability is removed by construction;
      smooth everywhere (no min-image / matching seam), so the variational bound
      holds.
  * cfg.optim.per_lambda_clip -- clip local energies within each lambda group
      (the global clip window is dominated by the inter-lambda spread and never
      fires in multi-lambda training).
  * cfg.log.tripwires -- log per-walker min pair distance, max|E_L|, and per-
      lambda mean energies to tripwires.csv; warn on approach to contact.
"""

from network import config
from network import train

import numpy as np

N_PARTICLES = 25
# Hex primitive cell area = L^2 * sqrt(3)/2; match density n = 1/pi:
#   L^2 * sqrt(3)/2 = N*pi  ->  L = sqrt(2*N*pi/sqrt(3))
TORUS_LENGTH = np.sqrt(2 * N_PARTICLES * np.pi / np.sqrt(3))

cfg = config.default()
cfg.n_particles = N_PARTICLES


cfg.geometry = 'periodic'
cfg.lattice = config.make_lattice(
    'hexagonal', TORUS_LENGTH, ndim=2)

cfg.network.predict_logits = True

# Set training parameters
cfg.optim.iterations = 50000
cfg.log.save_path = './dipole-25e-pbc-gen-HEX-smoothperiodic-bare-int30'

# Multi-lambda generative training
cfg.interaction_strength_training_set = None  # single-lambda reference (no gen conditioning)
cfg.interaction_strength = 30.0  # dedicated reference at lambda=30
cfg.interaction_small_length_cutoff = 0.0

# === Exact Ewald summation for the interaction ===
cfg.interaction_ewald = True
cfg.interaction_ewald_real_shells = 4
cfg.interaction_ewald_recip_shells = 12

# === Smooth-periodic K_0 two-body Jastrow (uses the PBC lattice) ===
cfg.network.smooth_periodic_jastrow = True
cfg.network.smooth_periodic_jastrow_rmatch_frac = 1.0

# === Per-lambda clipping + tripwire diagnostics ===
cfg.optim.per_lambda_clip = False  # single lambda -> global clip
cfg.log.tripwires = True
cfg.log.tripwire_frequency = 100

# === Optimizer settings ===
cfg.optim.lr.rate = 0.02
cfg.optim.lr.delay = 10000.0
cfg.optim.lr.decay = 1.5
cfg.optim.clip_local_energy = 3.0
cfg.optim.kfac.damping = 0.01
cfg.optim.kfac.norm_constraint = 0.003

train.train(cfg)
