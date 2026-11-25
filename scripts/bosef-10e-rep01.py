import os

from network import config
from network import train

# Parametrize number of particles
N_PARTICLES = 10

cfg = config.default()
cfg.n_particles = N_PARTICLES

# Set training parameters
# cfg.batch_size = 256
cfg.optim.iterations = 10000
cfg.log.save_path = './' + os.path.basename(__file__).removesuffix('.py')

cfg.geometry = 'disk'
cfg.interaction_type = 'short_range'
cfg.short_range_width = 0.1
cfg.interaction_strength = 0.1
# Conditional by default. For an unconditioned single-lambda model, set
# cfg.network.lambda_conditioning.mode = 'none' and training_set = None.
cfg.interaction_strength_training_set = [0.0, 0.05, 0.1]
cfg.network.boson_head = 'sum'
cfg.network.predict_logits = True
cfg.disk_radius = 10.

train.train(cfg)
