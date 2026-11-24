import os
from pyscf import gto

from ferminet import base_2DEG_config
from ferminet import train2DEG
from ferminet.utils import system

# Parametrize number of particles
N_PARTICLES = 10

mol = gto.Mole()
mol.build(
    atom = 'H  0 0 0',
    charge = 1 - N_PARTICLES,
    spin = N_PARTICLES,
    basis = 'sto-3g', unit='bohr')

cfg = base_2DEG_config.default()
cfg.system.pyscf_mol = mol

# Set training parameters
# cfg.batch_size = 256
# cfg.pretrain.iterations = 100
cfg.optim.iterations = 10000
cfg.network.network_type = 'psiformer'
# cfg.network.complex = True
cfg.log.save_path = './' + os.path.basename(__file__).removesuffix('.py')

# cfg.update(system.pyscf_mol_to_internal_representation(cfg.system.pyscf_mol))
# for atom in cfg.system.molecule:
#     atom.coords = atom.coords[:2]

cfg.geometry = 'disk'
cfg.interaction_type = 'short_range'
cfg.short_range_width = 0.1
cfg.interaction_strength = 0.1
# Conditional by default. For an unconditioned single-lambda model, set
# cfg.network.lambda_conditioning.mode = 'none' and training_set = None.
cfg.interaction_strength_training_set = [0.0, 0.05, 0.1]
cfg.network.boson_head = 'sum'
cfg.network.predict_logits = True
cfg.pretrain.method = None
cfg.disk_radius = 10.

train2DEG.train(cfg)
