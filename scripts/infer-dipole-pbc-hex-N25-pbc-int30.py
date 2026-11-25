"""Inference for the dipole-pbc-hex-N25-pbc-int30 model (smooth-periodic K_0 Jastrow).

Runs VMC inference (no optimization) at a single interaction_strength. Mirrors
the training config in dipole-pbc-hex-N25-pbc-int30.py EXACTLY -- in particular the exact 2D Ewald
summation and the fixed smooth-periodic K_0 two-body Jastrow, which is PART of
the wavefunction and must be enabled here to reproduce the trained psi.

Also records the condensate fraction n(k=0) (observables.nk0) and tripwire
diagnostics per lambda.

Usage:
    uv run --locked python scripts/infer-dipole-pbc-hex-N25-pbc-int30.py <interaction_strength> [iterations]
"""

import sys

from network import config
from network import train

import numpy as np

if len(sys.argv) < 2:
    print(
        "Usage: uv run --locked python scripts/infer-dipole-pbc-hex-N25-pbc-int30.py <interaction_strength> [iterations]"
    )
    sys.exit(1)

lam = float(sys.argv[1])
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
print(f"Running inference at interaction_strength = {lam} ({iters} iterations)")

N_PARTICLES = 25
# Hex primitive cell area = L^2 * sqrt(3)/2; match density n = 1/pi:
#   L^2 * sqrt(3)/2 = N*pi  ->  L = sqrt(2*N*pi/sqrt(3))
TORUS_LENGTH = np.sqrt(2 * N_PARTICLES * np.pi / np.sqrt(3))

cfg = config.default()
cfg.n_particles = N_PARTICLES

cfg.geometry = "periodic"
cfg.lattice = config.make_lattice("hexagonal", TORUS_LENGTH, ndim=2)

cfg.network.predict_logits = True

# --- Inference settings ---
cfg.optim.optimizer = "none"
cfg.optim.iterations = iters

cfg.interaction_strength = lam
cfg.interaction_small_length_cutoff = 0.0

# === Exact Ewald summation (must match training) ===
cfg.interaction_ewald = True
cfg.interaction_ewald_real_shells = 4
cfg.interaction_ewald_recip_shells = 12

# === Smooth-periodic K_0 Jastrow (part of the wavefunction; must match) ===
cfg.network.smooth_periodic_jastrow = True
cfg.network.smooth_periodic_jastrow_rmatch_frac = 1.0

# === Observables / diagnostics ===
cfg.observables.nk0 = True  # condensate fraction n(k=0) order parameter
cfg.log.tripwires = True
cfg.log.tripwire_frequency = 100

# Restore from trained checkpoint, save to lambda-specific directory
cfg.log.restore_path = "./dipole-25e-pbc-gen-HEX-smoothperiodic-bare-int30"
cfg.log.save_path = f"./dipole-25e-pbc-gen-HEX-smoothperiodic-bare-int30/lambda_{lam}"

train.train(cfg)
