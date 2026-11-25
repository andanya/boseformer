"""Defaults for bosonic VMC on a disk or periodic cell.

Set n_particles and geometry explicitly in run scripts. For a torus set
geometry='periodic' and lattice=make_lattice('hexagonal' or 'square', length).
The conditional model is the default; use network.lambda_conditioning.mode
='none' with a single interaction strength for an unconditioned model.
"""

import ml_collections
from network.geometry import make_lattice as make_lattice


def default() -> ml_collections.ConfigDict:
    """Return independent configuration values for a new run."""
    return ml_collections.ConfigDict(
        {
            "barrier_sharpness": 1.0,
            "batch_size": 4096,
            "debug": {"check_nan": False, "deterministic": False},
            "disk_radius": 10.0,
            "geometry": "disk",
            "interaction_cutoff_radius": None,
            "interaction_ewald": False,
            "interaction_ewald_eta": None,
            "interaction_ewald_eta_scale": 1.0,
            "interaction_ewald_real_shells": 3,
            "interaction_ewald_recip_shells": 8,
            "interaction_small_length_cutoff": 0.1,
            "interaction_strength": 0.0,
            "interaction_strength_training_set": None,
            "interaction_tail_correction": False,
            "interaction_tail_g2_path": None,
            "interaction_truncation_limit": 5,
            "interaction_type": "inverse_cubic",
            "log": {
                "features": False,
                "local_energies": False,
                "restore_path": "",
                "save_frequency": 10.0,
                "save_path": "",
                "stats_frequency": 1,
                "tripwire_frequency": 100,
                "tripwires": False,
                "walkers": False,
            },
            "mcmc": {
                "adapt_frequency": 100,
                "blocks": 1,
                "burn_in": 300,
                "init_width": 1.0,
                "move_width": 0.02,
                "steps": 30,
            },
            "network": {
                "bias_orbitals": False,
                "boson_head": "product",
                "lambda_conditioning": {"log_scale_input": False, "mode": "additive"},
                "predict_logits": False,
                "psiformer": {
                    "heads_dim": 64,
                    "mlp_hidden_dims": (256,),
                    "num_heads": 4,
                    "num_layers": 4,
                    "use_layer_norm": True,
                },
                "rescale_inputs": False,
                "smooth_periodic_jastrow": False,
                "smooth_periodic_jastrow_rmatch_frac": 1.0,
                "products": 16,
            },
            "observables": {"nk0": False},
            "optim": {
                "adam": {"b1": 0.9, "b2": 0.999, "eps": 1e-08, "eps_root": 0.0},
                "center_at_clip": True,
                "clip_local_energy": 5.0,
                "clip_median": False,
                "iterations": 50000,
                "kfac": {
                    "cov_ema_decay": 0.95,
                    "cov_update_every": 1,
                    "damping": 0.001,
                    "invert_every": 1,
                    "l2_reg": 0.0,
                    "mean_center": True,
                    "min_damping": 0.0001,
                    "momentum": 0.0,
                    "momentum_type": "regular",
                    "norm_constraint": 0.001,
                    "register_only_generic": False,
                },
                "laplacian": "default",
                "lr": {"decay": 1.5, "delay": 5000.0, "rate": 0.05},
                "max_vmap_batch_size": 0,
                "objective": "vmc",
                "optimizer": "kfac",
                "per_lambda_clip": False,
                "reset_if_nan": False,
                "vmc_weight": 0.0,
            },
            "short_range_width": 0.1,
            "n_particles": 25,
            "ndim": 2,
            "center": (0.0, 0.0),
            "lattice": None,
        }
    )


def resolve(cfg):
    return cfg.copy_and_resolve_references()
