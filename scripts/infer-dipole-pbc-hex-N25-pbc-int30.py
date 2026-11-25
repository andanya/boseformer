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
from pyscf import gto

from ferminet import base_2DEG_config
from ferminet import train2DEG
from ferminet.utils import system

from ferminet.pbc import envelopes

import numpy as np
import os


if len(sys.argv) < 2:
    print("Usage: uv run --locked python scripts/infer-dipole-pbc-hex-N25-pbc-int30.py <interaction_strength> [iterations]")
    sys.exit(1)

lam = float(sys.argv[1])
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
print(f"Running inference at interaction_strength = {lam} ({iters} iterations)")

N_PARTICLES = 25
# Hex primitive cell area = L^2 * sqrt(3)/2; match density n = 1/pi:
#   L^2 * sqrt(3)/2 = N*pi  ->  L = sqrt(2*N*pi/sqrt(3))
TORUS_LENGTH = np.sqrt(2 * N_PARTICLES * np.pi / np.sqrt(3))

mol = gto.Mole()
mol.build(
    atom='H  0 0 0',
    charge=1 - N_PARTICLES,
    spin=N_PARTICLES,
    basis='sto-3g', unit='bohr')

cfg = base_2DEG_config.default()
cfg.system.pyscf_mol = mol
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
cfg.network.network_type = 'psiformer'

# --- Inference settings ---
cfg.optim.optimizer = 'none'
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
cfg.observables.nk0 = True          # condensate fraction n(k=0) order parameter
cfg.log.tripwires = True
cfg.log.tripwire_frequency = 100

# Restore from trained checkpoint, save to lambda-specific directory
cfg.log.restore_path = './dipole-25e-pbc-gen-HEX-smoothperiodic-bare-int30'
cfg.log.save_path = f'./dipole-25e-pbc-gen-HEX-smoothperiodic-bare-int30/lambda_{lam}'

train2DEG.train(cfg)
