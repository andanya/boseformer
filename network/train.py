# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core training loop for neural QMC in JAX."""

import functools
import os
import time
from typing import Optional, Tuple, Union
from absl import logging
import chex
from network.geometry import Geometry
from network import checkpoint
from network import constants
from network import curvature_tags_and_blocks
from network import hamiltonian
from network import loss as qmc_loss_functions
from network import mcmc
from network import networks
from network import psiformer
from network.utils import statistics
from network.utils import writers
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import kfac_jax
import ml_collections
import numpy as np
import optax
from typing_extensions import Protocol

OptimizerState = Union[optax.OptState, kfac_jax.Optimizer.State]
OptUpdateResults = Tuple[
    networks.ParamTree,
    Optional[OptimizerState],
    jnp.ndarray,
    Optional[qmc_loss_functions.AuxiliaryLossData],
]


class OptUpdate(Protocol):
    def __call__(
        self,
        params: networks.ParamTree,
        data: networks.WalkerData,
        opt_state: optax.OptState,
        key: chex.PRNGKey,
    ) -> OptUpdateResults:
        """Evaluates the loss and gradients and updates the parameters accordingly.

        Args:
          params: network parameters.
          data: particle positions and coupling.
          opt_state: optimizer internal state.
          key: RNG state.

        Returns:
          Tuple of (params, opt_state, loss, aux_data), where params and opt_state
          are the updated parameters and optimizer state, loss is the evaluated loss
          and aux_data auxiliary data (see AuxiliaryLossData docstring).
        """


StepResults = Tuple[
    networks.WalkerData,
    networks.ParamTree,
    Optional[optax.OptState],
    jnp.ndarray,
    qmc_loss_functions.AuxiliaryLossData,
    jnp.ndarray,
]


class Step(Protocol):
    def __call__(
        self,
        data: networks.WalkerData,
        params: networks.ParamTree,
        state: OptimizerState,
        key: chex.PRNGKey,
        mcmc_width: jnp.ndarray,
    ) -> StepResults:
        """Performs one set of MCMC moves and an optimization step.

        Args:
          data: batch of MCMC configurations and coupling.
          params: network parameters.
          state: optimizer internal state.
          key: JAX RNG state.
          mcmc_width: width of MCMC move proposal. See mcmc.make_mcmc_step.

        Returns:
          Tuple of (data, params, state, loss, aux_data, pmove).
            data: Updated MCMC configurations drawn from the network given the
              *input* network parameters.
            params: updated network parameters after the gradient update.
            state: updated optimization state.
            loss: energy of system based on input network parameters averaged over
              the entire set of MCMC configurations.
            aux_data: AuxiliaryLossData object also returned from evaluating the
              loss of the system.
            pmove: probability that a proposed MCMC move was accepted.
        """


def null_update(
    params: networks.ParamTree,
    data: networks.WalkerData,
    opt_state: Optional[optax.OptState],
    key: chex.PRNGKey,
) -> OptUpdateResults:
    """Performs an identity operation with an OptUpdate interface."""
    del data, key
    return (params, opt_state, jnp.zeros(1), None)


def make_opt_update_step(
    evaluate_loss: qmc_loss_functions.LossFn, optimizer: optax.GradientTransformation
) -> OptUpdate:
    """Returns an OptUpdate function for performing a parameter update."""
    loss_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)

    def opt_update(
        params: networks.ParamTree,
        data: networks.WalkerData,
        opt_state: Optional[optax.OptState],
        key: chex.PRNGKey,
    ) -> OptUpdateResults:
        """Evaluates the loss and gradients and updates the parameters using optax."""
        (loss, aux_data), grad = loss_and_grad(params, key, data)
        grad = constants.pmean(grad)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state, loss, aux_data)

    return opt_update


def make_loss_step(evaluate_loss: qmc_loss_functions.LossFn) -> OptUpdate:
    """Returns an OptUpdate function for evaluating the loss."""

    def loss_eval(
        params: networks.ParamTree,
        data: networks.WalkerData,
        opt_state: Optional[optax.OptState],
        key: chex.PRNGKey,
    ) -> OptUpdateResults:
        """Evaluates just the loss and gradients with an OptUpdate interface."""
        loss, aux_data = evaluate_loss(params, key, data)
        return (params, opt_state, loss, aux_data)

    return loss_eval


def make_training_step(
    mcmc_step, optimizer_step: OptUpdate, reset_if_nan: bool = False
) -> Step:
    """Factory to create traning step for non-KFAC optimizers.

    Args:
      mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
        for creating the callable.
      optimizer_step: OptUpdate callable which evaluates the forward and backward
        passes and updates the parameters and optimizer state, as required.
      reset_if_nan: If true, reset the params and opt state to the state at the
        previous step when the loss is NaN

    Returns:
      step, a callable which performs a set of MCMC steps and then an optimization
      update. See the Step protocol for details.
    """

    @functools.partial(constants.pmap, donate_argnums=(0, 1, 2))
    def step(
        data: networks.WalkerData,
        params: networks.ParamTree,
        state: Optional[optax.OptState],
        key: chex.PRNGKey,
        mcmc_width: jnp.ndarray,
    ) -> StepResults:
        """A full update iteration (except for KFAC): MCMC steps + optimization."""
        mcmc_key, loss_key = jax.random.split(key, num=2)
        data, pmove = mcmc_step(params, data, mcmc_key, mcmc_width)
        new_params, new_state, loss, aux_data = optimizer_step(
            params, data, state, loss_key
        )
        if reset_if_nan:
            new_params = jax.lax.cond(
                jnp.isnan(loss), lambda: params, lambda: new_params
            )
            new_state = jax.lax.cond(jnp.isnan(loss), lambda: state, lambda: new_state)
        return (data, new_params, new_state, loss, aux_data, pmove)

    return step


def make_kfac_training_step(
    mcmc_step, damping: float, optimizer: kfac_jax.Optimizer, reset_if_nan: bool = False
) -> Step:
    """Factory to create traning step for KFAC optimizers.

    Args:
      mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
        for creating the callable.
      damping: value of damping to use for each KFAC update step.
      optimizer: KFAC optimizer instance.
      reset_if_nan: If true, reset the params and opt state to the state at the
        previous step when the loss is NaN

    Returns:
      step, a callable which performs a set of MCMC steps and then an optimization
      update. See the Step protocol for details.
    """
    mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)
    shared_mom = kfac_jax.utils.replicate_all_local_devices(jnp.zeros([]))
    shared_damping = kfac_jax.utils.replicate_all_local_devices(jnp.asarray(damping))
    copy_tree = constants.pmap(
        functools.partial(jax.tree_util.tree_map, lambda x: (1.0 * x).astype(x.dtype))
    )

    def step(
        data: networks.WalkerData,
        params: networks.ParamTree,
        state: kfac_jax.Optimizer.State,
        key: chex.PRNGKey,
        mcmc_width: jnp.ndarray,
    ) -> StepResults:
        """A full update iteration for KFAC: MCMC steps + optimization."""
        mcmc_keys, loss_keys = kfac_jax.utils.p_split(key)
        data, pmove = mcmc_step(params, data, mcmc_keys, mcmc_width)
        if reset_if_nan:
            old_params = copy_tree(params)
            old_state = copy_tree(state)
        new_params, new_state, stats = optimizer.step(
            params=params,
            state=state,
            rng=loss_keys,
            batch=data,
            momentum=shared_mom,
            damping=shared_damping,
        )
        if reset_if_nan and jnp.isnan(stats["loss"]):
            new_params = old_params
            new_state = old_state
        return (data, new_params, new_state, stats["loss"], stats["aux"], pmove)

    return step


def _unshard_positions_to_host(pos_sharded) -> np.ndarray:
    """(n_local_devices, device_batch, ncoords) -> (batch_total, ncoords) on host."""
    arr = np.asarray(jax.device_get(pos_sharded))
    return arr.reshape(-1, arr.shape[-1])


def _add_suffix_before_ext(path: str, suffix: str) -> str:
    """Insert suffix (e.g., '.host0') before the file extension."""
    base, ext = os.path.splitext(path)
    return f"{base}{suffix}{ext}"


def _atomic_save_npy(path: str, arr: np.ndarray):
    """Atomically save a .npy file, without NumPy changing the filename."""
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.save(f, arr)
        os.replace(tmp, path)
    except OSError as e:
        logging.warning("Failed to save %s: %s. Skipping this save.", path, e)


def train(cfg: ml_collections.ConfigDict, writer_manager=None):
    """Runs training loop for QMC.

    Args:
      cfg: ConfigDict containing the system and training parameters to run on. See
        config.default for more details.
      writer_manager: context manager with a write method for logging output. If
        None, a default writer (network.utils.writers.Writer) is used.

    Raises:
      ValueError: if an illegal or unsupported value in cfg is detected.
    """
    geometry = validate_boson_config(cfg)
    num_devices = jax.local_device_count()
    num_hosts = jax.device_count() // num_devices
    logging.info(
        "Starting QMC with %i XLA devices per host across %i hosts.",
        num_devices,
        num_hosts,
    )
    if cfg.batch_size % (num_devices * num_hosts) != 0:
        raise ValueError(
            f"Batch size must be divisible by number of devices, got batch size {cfg.batch_size} for {num_devices * num_hosts} devices."
        )
    host_batch_size = cfg.batch_size // num_hosts
    total_host_batch_size = host_batch_size
    device_batch_size = host_batch_size // num_devices
    data_shape = (num_devices, device_batch_size)
    if cfg.debug.deterministic:
        seed = 23
    else:
        seed = jnp.asarray([1000000.0 * time.time()])
        seed = int(multihost_utils.broadcast_one_to_all(seed)[0])
    key = jax.random.PRNGKey(seed)
    network = psiformer.make_network(
        cfg.n_particles,
        geometry=geometry,
        products=cfg.network.products,
        bias_orbitals=cfg.network.bias_orbitals,
        rescale_inputs=cfg.network.rescale_inputs,
        predict_logits=cfg.network.predict_logits,
        boson_head=cfg.network.boson_head,
        lambda_conditioning_mode=cfg.network.lambda_conditioning.mode,
        lambda_log_scale_input=cfg.network.lambda_conditioning.log_scale_input,
        smooth_periodic_jastrow=cfg.network.smooth_periodic_jastrow,
        smooth_periodic_jastrow_rmatch_frac=cfg.network.smooth_periodic_jastrow_rmatch_frac,
        **cfg.network.psiformer,
    )
    key, subkey = jax.random.split(key)
    params = network.init(subkey)
    params = kfac_jax.utils.replicate_all_local_devices(params)
    signed_network = network.apply
    logabs_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]
    batch_network = jax.vmap(logabs_network, in_axes=(None, 0, 0), out_axes=0)

    ckpt_save_path = checkpoint.create_save_path(cfg.log.save_path)
    ckpt_restore_path = checkpoint.get_restore_path(cfg.log.restore_path)
    walkers_last_path = os.path.join(ckpt_save_path, "walkers_last.npy")
    walkers_prev_path = os.path.join(ckpt_save_path, "walkers_prev.npy")
    walkers_every = int(cfg.log.get("walkers_every", 1))
    density_bins = int(cfg.log.get("density_bins", 128))
    gather_hosts = bool(cfg.log.get("walkers_gather_hosts", False))
    save_density = bool(cfg.log.get("save_density_from_walkers", False))
    ckpt_restore_filename = checkpoint.find_last_checkpoint(
        ckpt_save_path
    ) or checkpoint.find_last_checkpoint(ckpt_restore_path)
    if ckpt_restore_filename:
        t_init, data, params, opt_state_ckpt, mcmc_width_ckpt = checkpoint.restore(
            ckpt_restore_filename, host_batch_size
        )
        training_set = cfg.get("interaction_strength_training_set", None)
        if training_set:
            training_set = list(training_set)
            num_lambdas = len(training_set)
            walker_lambdas = jnp.array(
                [training_set[i % num_lambdas] for i in range(device_batch_size)],
                dtype=jnp.float32,
            )
        else:
            walker_lambdas = jnp.full(
                (device_batch_size,), cfg.interaction_strength, dtype=jnp.float32
            )
        batch_interaction_strength = kfac_jax.utils.replicate_all_local_devices(
            walker_lambdas
        )
        data = networks.WalkerData(
            positions=data.positions, interaction_strength=batch_interaction_strength
        )
    else:
        logging.info("No checkpoint found. Training new model.")
        key, subkey = jax.random.split(key)
        subkey = jax.random.fold_in(subkey, jax.process_index())
        pos = init_particles(
            subkey,
            cfg.n_particles,
            geometry,
            batch_size=total_host_batch_size,
            init_width=cfg.mcmc.init_width,
        )
        pos = jnp.reshape(pos, data_shape + (-1,))
        pos = kfac_jax.utils.broadcast_all_local_devices(pos)
        training_set = cfg.get("interaction_strength_training_set", None)
        if training_set:
            training_set = list(training_set)
            num_lambdas = len(training_set)
            walker_lambdas = jnp.array(
                [training_set[i % num_lambdas] for i in range(device_batch_size)],
                dtype=jnp.float32,
            )
        else:
            walker_lambdas = jnp.full(
                (device_batch_size,), cfg.interaction_strength, dtype=jnp.float32
            )
        walker_lambdas = jnp.reshape(walker_lambdas, (device_batch_size,))
        batch_interaction_strength = kfac_jax.utils.replicate_all_local_devices(
            walker_lambdas
        )
        data = networks.WalkerData(
            positions=pos, interaction_strength=batch_interaction_strength
        )
        t_init = 0
        opt_state_ckpt = None
        mcmc_width_ckpt = None
    train_schema = ["step", "energy", "ewmean", "ewvar", "pmove"]
    is_inference_run = cfg.optim.optimizer == "none"
    if is_inference_run:
        train_schema += ["kinetic", "potential"]
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    pbc_lattice_for_mcmc = geometry.lattice
    mcmc_step = mcmc.make_mcmc_step(
        batch_network,
        device_batch_size,
        steps=cfg.mcmc.steps,
        ndim=cfg.ndim,
        blocks=cfg.mcmc.blocks,
        lattice=pbc_lattice_for_mcmc,
    )
    compute_nk0 = cfg.observables.get("nk0", False)
    nk0_csv_file = None
    p_nk0_step = None
    nk0_n_particles = cfg.n_particles
    if compute_nk0:
        if pbc_lattice_for_mcmc is None:
            raise ValueError(
                "cfg.observables.nk0 requires periodic boundary conditions: cfg.lattice is None."
            )
        nk0_lattice = jnp.asarray(pbc_lattice_for_mcmc)
        nk0_ndim = cfg.ndim

        def _nk0_one_walker(params, positions, lam, key):
            u = jax.random.uniform(key, shape=(nk0_ndim,))
            s_disp = nk0_lattice @ u
            new_positions = positions.at[:nk0_ndim].add(s_disp)
            sign_old, log_old = signed_network(params, positions, lam)
            sign_new, log_new = signed_network(params, new_positions, lam)
            return sign_old * sign_new * jnp.exp(log_new - log_old)

        def _nk0_one_device(params, data, key):
            n_local = data.positions.shape[0]
            keys = jax.random.split(key, n_local)
            ratios = jax.vmap(_nk0_one_walker, in_axes=(None, 0, 0, 0))(
                params, data.positions, data.interaction_strength, keys
            )
            mean = jnp.mean(ratios)
            sq_mean = jnp.mean(ratios**2)
            mean = constants.pmean(mean)
            sq_mean = constants.pmean(sq_mean)
            return jnp.stack([mean, sq_mean])

        p_nk0_step = constants.pmap(_nk0_one_device)
        nk0_csv_path = os.path.join(ckpt_save_path, "nk0_samples.csv")
        if jax.process_index() == 0:
            need_header = (
                not os.path.exists(nk0_csv_path) or os.path.getsize(nk0_csv_path) == 0
            )
            nk0_csv_file = open(nk0_csv_path, "a", buffering=1)
            if need_header:
                nk0_csv_file.write("step,ratio_mean,ratio_sq_mean,n_particles\n")
    tripwires_on = bool(cfg.log.get("tripwires", False))
    tripwire_frequency = int(cfg.log.get("tripwire_frequency", 100))
    tripwire_csv_file = None
    tripwire_lattice = pbc_lattice_for_mcmc
    tripwire_ndim = cfg.ndim
    if tripwires_on and jax.process_index() == 0:
        tripwire_csv_path = os.path.join(ckpt_save_path, "tripwires.csv")
        need_header = (
            not os.path.exists(tripwire_csv_path)
            or os.path.getsize(tripwire_csv_path) == 0
        )
        tripwire_csv_file = open(tripwire_csv_path, "a", buffering=1)
        if need_header:
            tripwire_csv_file.write(
                "step,rmin,frac_close,max_abs_el,per_lambda_energies\n"
            )

    def _min_pair_distance(positions_host: np.ndarray) -> tuple:
        """(rmin, frac of walkers with a pair closer than 0.5) via min-image."""
        n_coords = positions_host.shape[-1]
        n_particles = n_coords // tripwire_ndim
        r = positions_host.reshape(-1, n_particles, tripwire_ndim)
        d = r[:, :, None, :] - r[:, None, :, :]
        if tripwire_lattice is not None:
            lat = np.asarray(tripwire_lattice)
            inv = np.linalg.inv(lat)
            frac = np.einsum("ij,wabj->wabi", inv, d)
            d = np.einsum("ij,wabj->wabi", lat, frac - np.round(frac))
        dist = np.linalg.norm(d, axis=-1)
        iu = np.triu_indices(n_particles, 1)
        pair = dist[:, iu[0], iu[1]]
        per_walker_min = pair.min(axis=1)
        return (float(per_walker_min.min()), float((per_walker_min < 0.5).mean()))

    laplacian_method = cfg.optim.get("laplacian", "default")
    interaction_tail_correction = cfg.get("interaction_tail_correction", False)
    interaction_cutoff_radius = cfg.get("interaction_cutoff_radius", None)
    interaction_tail_g2_r = None
    interaction_tail_g2_values = None
    if interaction_tail_correction:
        g2_path = cfg.get("interaction_tail_g2_path", None)
        if g2_path:
            g2_data = np.load(g2_path)
            interaction_tail_g2_r = g2_data["r"]
            interaction_tail_g2_values = g2_data["g2"]
            logging.info("Loaded tabulated g2(r) for tail correction from %s", g2_path)
        logging.info(
            "Interaction tail correction ON: cutoff R_c=%s, g2=%s",
            interaction_cutoff_radius,
            "tabulated" if interaction_tail_g2_r is not None else "uniform (=1)",
        )
    if cfg.get("interaction_ewald", False):
        logging.info(
            "Interaction Ewald summation ON (exact periodic 1/(r+a)^3): eta_scale=%s real_shells=%s recip_shells=%s",
            cfg.get("interaction_ewald_eta_scale", 1.0),
            cfg.get("interaction_ewald_real_shells", 3),
            cfg.get("interaction_ewald_recip_shells", 8),
        )
    local_energy_fn = hamiltonian.local_energy(
        f=signed_network,
        use_scan=False,
        laplacian_method=laplacian_method,
        interaction_small_length_cutoff=cfg.interaction_small_length_cutoff,
        interaction_truncation_limit=cfg.interaction_truncation_limit,
        barrier_sharpness=cfg.barrier_sharpness,
        geometry=geometry,
        interaction_type=cfg.get("interaction_type", "inverse_cubic"),
        short_range_width=cfg.get("short_range_width", 0.1),
        interaction_cutoff_radius=interaction_cutoff_radius,
        interaction_tail_correction=interaction_tail_correction,
        interaction_tail_g2_r=interaction_tail_g2_r,
        interaction_tail_g2_values=interaction_tail_g2_values,
        interaction_ewald=cfg.get("interaction_ewald", False),
        interaction_ewald_eta=cfg.get("interaction_ewald_eta", None),
        interaction_ewald_eta_scale=cfg.get("interaction_ewald_eta_scale", 1.0),
        interaction_ewald_real_shells=cfg.get("interaction_ewald_real_shells", 3),
        interaction_ewald_recip_shells=cfg.get("interaction_ewald_recip_shells", 8),
        n_particles=cfg.n_particles,
    )
    local_energy = local_energy_fn
    if cfg.optim.objective == "vmc":
        evaluate_loss = qmc_loss_functions.make_loss(
            logabs_network,
            local_energy,
            clip_local_energy=cfg.optim.clip_local_energy,
            clip_from_median=cfg.optim.clip_median,
            center_at_clipped_energy=cfg.optim.center_at_clip,
            max_vmap_batch_size=cfg.optim.get("max_vmap_batch_size", 0),
            per_lambda_clip=cfg.optim.get("per_lambda_clip", False),
        )
    elif cfg.optim.objective == "wqmc":
        evaluate_loss = qmc_loss_functions.make_wqmc_loss(
            logabs_network,
            local_energy,
            clip_local_energy=cfg.optim.clip_local_energy,
            clip_from_median=cfg.optim.clip_median,
            center_at_clipped_energy=cfg.optim.center_at_clip,
            max_vmap_batch_size=cfg.optim.get("max_vmap_batch_size", 0),
            vmc_weight=cfg.optim.get("vmc_weight", 1.0),
        )
    else:
        raise ValueError(f"Not a recognized objective: {cfg.optim.objective}")

    def learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
        return cfg.optim.lr.rate * jnp.power(
            1.0 / (1.0 + t_ / cfg.optim.lr.delay), cfg.optim.lr.decay
        )

    if cfg.optim.optimizer == "none":
        optimizer = None
    elif cfg.optim.optimizer == "adam":
        optimizer = optax.chain(
            optax.scale_by_adam(**cfg.optim.adam),
            optax.scale_by_schedule(learning_rate_schedule),
            optax.scale(-1.0),
        )
    elif cfg.optim.optimizer == "lamb":
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(eps=1e-07),
            optax.scale_by_trust_ratio(),
            optax.scale_by_schedule(learning_rate_schedule),
            optax.scale(-1),
        )
    elif cfg.optim.optimizer == "kfac":
        val_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)
        optimizer = kfac_jax.Optimizer(
            val_and_grad,
            l2_reg=cfg.optim.kfac.l2_reg,
            norm_constraint=cfg.optim.kfac.norm_constraint,
            value_func_has_aux=True,
            value_func_has_rng=True,
            learning_rate_schedule=learning_rate_schedule,
            curvature_ema=cfg.optim.kfac.cov_ema_decay,
            inverse_update_period=cfg.optim.kfac.invert_every,
            min_damping=cfg.optim.kfac.min_damping,
            num_burnin_steps=0,
            register_only_generic=cfg.optim.kfac.register_only_generic,
            estimation_mode="fisher_exact",
            multi_device=True,
            pmap_axis_name=constants.PMAP_AXIS_NAME,
            auto_register_kwargs=dict(
                graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS
            ),
        )
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        opt_state = optimizer.init(params, subkeys, data)
        opt_state = opt_state_ckpt or opt_state
    else:
        raise ValueError(f"Not a recognized optimizer: {cfg.optim.optimizer}")
    if not optimizer:
        opt_state = None
        step = make_training_step(
            mcmc_step=mcmc_step, optimizer_step=make_loss_step(evaluate_loss)
        )
    elif isinstance(optimizer, optax.GradientTransformation):
        opt_state = jax.pmap(optimizer.init)(params)
        opt_state = opt_state_ckpt or opt_state
        step = make_training_step(
            mcmc_step=mcmc_step,
            optimizer_step=make_opt_update_step(evaluate_loss, optimizer),
            reset_if_nan=cfg.optim.reset_if_nan,
        )
    elif isinstance(optimizer, kfac_jax.Optimizer):
        step = make_kfac_training_step(
            mcmc_step=mcmc_step,
            damping=cfg.optim.kfac.damping,
            optimizer=optimizer,
            reset_if_nan=cfg.optim.reset_if_nan,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    if mcmc_width_ckpt is not None:
        mcmc_width = kfac_jax.utils.replicate_all_local_devices(mcmc_width_ckpt[0])
    else:
        mcmc_width = kfac_jax.utils.replicate_all_local_devices(
            jnp.asarray(cfg.mcmc.move_width)
        )
    pmoves = np.zeros(cfg.mcmc.adapt_frequency)
    if t_init == 0:
        logging.info("Burning in MCMC chain for %d steps", cfg.mcmc.burn_in)
        burn_in_step = make_training_step(
            mcmc_step=mcmc_step, optimizer_step=null_update
        )
        for t in range(cfg.mcmc.burn_in):
            sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
            data, params, *_ = burn_in_step(
                data, params, state=None, key=subkeys, mcmc_width=mcmc_width
            )
        logging.info("Completed burn-in MCMC steps")
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        ptotal_energy = constants.pmap(evaluate_loss)
        initial_energy, _ = ptotal_energy(params, subkeys, data)
        logging.info("Initial energy: %03.4f", initial_energy[0])
    time_of_last_ckpt = time.time()
    weighted_stats = None
    if cfg.optim.optimizer == "none" and opt_state_ckpt is not None:
        logging.info("No optimizer provided. Assuming inference run.")
        logging.info("Setting initial iteration to 0.")
        t_init = 0
    if writer_manager is None:
        writer_manager = writers.Writer(
            name="train_stats",
            schema=train_schema,
            directory=ckpt_save_path,
            iteration_key=None,
            log=False,
        )
    with writer_manager as writer:
        num_resets = 0
        for t in range(t_init, cfg.optim.iterations):
            save_walkers_now = t % walkers_every == 0
            if save_walkers_now:
                prev_pos_host = _unshard_positions_to_host(data.positions)
                if gather_hosts:
                    prev_pos_host = np.concatenate(
                        jax.experimental.multihost_utils.process_allgather(
                            prev_pos_host
                        ),
                        axis=0,
                    )
            sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
            data, params, opt_state, loss, aux_data, pmove = step(
                data, params, opt_state, subkeys, mcmc_width
            )
            if save_walkers_now:
                cur_pos_host = _unshard_positions_to_host(data.positions)
                if gather_hosts:
                    cur_pos_host = np.concatenate(
                        jax.experimental.multihost_utils.process_allgather(
                            cur_pos_host
                        ),
                        axis=0,
                    )
                if gather_hosts:
                    if jax.process_index() == 0:
                        _atomic_save_npy(walkers_prev_path, prev_pos_host)
                        _atomic_save_npy(walkers_last_path, cur_pos_host)
                else:
                    suffix = f".host{jax.process_index()}"
                    prev_path = _add_suffix_before_ext(walkers_prev_path, suffix)
                    last_path = _add_suffix_before_ext(walkers_last_path, suffix)
                    _atomic_save_npy(prev_path, prev_pos_host)
                    _atomic_save_npy(last_path, cur_pos_host)
                if save_density and cfg.ndim == 2:
                    n_particles = cur_pos_host.shape[-1] // 2
                    R = cur_pos_host.reshape(-1, n_particles, 2)
                    xs, ys = (R[..., 0].ravel(), R[..., 1].ravel())
                    H, xedges, yedges = np.histogram2d(xs, ys, bins=density_bins)
                    out = os.path.join(
                        ckpt_save_path,
                        "density_last.npy"
                        if gather_hosts
                        else f"density_last.host{jax.process_index()}.npy",
                    )
                    _atomic_save_npy(out, H)
            if compute_nk0:
                sharded_key, nk0_subkeys = kfac_jax.utils.p_split(sharded_key)
                nk0_stats = p_nk0_step(params, data, nk0_subkeys)
                if nk0_csv_file is not None:
                    ratio_mean = float(nk0_stats[0, 0])
                    ratio_sq_mean = float(nk0_stats[0, 1])
                    nk0_csv_file.write(
                        f"{t},{ratio_mean:.10g},{ratio_sq_mean:.10g},{nk0_n_particles}\n"
                    )
            if tripwires_on and t % tripwire_frequency == 0:
                try:
                    el_host = np.asarray(jax.device_get(aux_data.local_energy)).reshape(
                        -1
                    )
                    lam_host = np.asarray(
                        jax.device_get(data.interaction_strength)
                    ).reshape(-1)
                    max_abs_el = (
                        float(np.max(np.abs(el_host))) if el_host.size else float("nan")
                    )
                    pos_host = _unshard_positions_to_host(data.positions)
                    rmin, frac_close = _min_pair_distance(pos_host)
                    uniq = np.unique(lam_host)
                    per_lam = {
                        float(u): float(np.mean(el_host[lam_host == u])) for u in uniq
                    }
                    hot_lambda = float(uniq.max()) if uniq.size else 0.0
                    if rmin < 0.5 and hot_lambda > 5.0:
                        logging.warning(
                            "TRIPWIRE step %d: rmin=%.3f (%.1f%% walkers < 0.5) at lambda up to %.1f; max|E_L|=%.3g. Possible contact pocket.",
                            t,
                            rmin,
                            100 * frac_close,
                            hot_lambda,
                            max_abs_el,
                        )
                    if tripwire_csv_file is not None:
                        per_lam_str = ";".join(
                            (f"{k:g}:{v:.6g}" for k, v in sorted(per_lam.items()))
                        )
                        tripwire_csv_file.write(
                            f'{t},{rmin:.6g},{frac_close:.6g},{max_abs_el:.6g},"{per_lam_str}"\n'
                        )
                except Exception as e:
                    logging.warning("Tripwire computation failed at step %d: %s", t, e)
            loss = loss[0]
            weighted_stats = statistics.exponentialy_weighted_stats(
                alpha=0.1, observation=loss, previous_stats=weighted_stats
            )
            pmove = pmove[0]
            mcmc_width, pmoves = mcmc.update_mcmc_width(
                t, mcmc_width, cfg.mcmc.adapt_frequency, pmove, pmoves
            )
            if cfg.debug.check_nan:
                tree = {"params": params, "loss": loss}
                if cfg.optim.optimizer != "none":
                    tree["optim"] = opt_state
                try:
                    chex.assert_tree_all_finite(tree)
                    num_resets = 0
                except AssertionError as e:
                    if cfg.optim.reset_if_nan:
                        num_resets += 1
                        if num_resets > 100:
                            raise e
                    else:
                        raise e
            if t % cfg.log.stats_frequency == 0:
                logging_str = "Step %05d: %03.4f, exp. variance=%03.4f, pmove=%0.2f"
                logging_args = (t, loss, weighted_stats.variance, pmove)
                writer_kwargs = {
                    "step": t,
                    "energy": np.asarray(loss),
                    "ewmean": np.asarray(weighted_stats.mean),
                    "ewvar": np.asarray(weighted_stats.variance),
                    "pmove": np.asarray(pmove),
                }
                if is_inference_run and aux_data.kinetic_energy is not None:
                    writer_kwargs["kinetic"] = np.asarray(aux_data.kinetic_energy[0])
                    writer_kwargs["potential"] = np.asarray(
                        aux_data.potential_energy[0]
                    )
                logging.info(logging_str, *logging_args)
                writer.write(t, **writer_kwargs)
            if time.time() - time_of_last_ckpt > cfg.log.save_frequency * 60:
                checkpoint.save(ckpt_save_path, t, data, params, opt_state, mcmc_width)
                time_of_last_ckpt = time.time()
        if compute_nk0 and nk0_csv_file is not None:
            nk0_csv_file.close()
        if tripwire_csv_file is not None:
            tripwire_csv_file.close()


def init_particles(key, n_particles, geometry, batch_size, init_width):
    """Initialize Gaussian walkers about the coordinate origin/disk center."""
    key, subkey = jax.random.split(key)
    center = jnp.tile(jnp.asarray(geometry.center), n_particles)
    return (
        center
        + jax.random.normal(subkey, shape=(batch_size, n_particles * geometry.ndim))
        * init_width
    )


def validate_boson_config(cfg):
    """Validate physics before compiling or creating output files."""
    geometry = Geometry.from_config(cfg)
    if not isinstance(cfg.n_particles, int) or cfg.n_particles < 1:
        raise ValueError("n_particles must be a positive integer")
    if cfg.interaction_type not in ("inverse_cubic", "short_range"):
        raise ValueError("interaction_type must be inverse_cubic or short_range")
    if cfg.interaction_ewald or cfg.interaction_tail_correction:
        if geometry.kind != "periodic" or cfg.interaction_type != "inverse_cubic":
            raise ValueError(
                "Ewald and tail correction require periodic inverse_cubic physics"
            )
    if cfg.interaction_ewald and cfg.interaction_tail_correction:
        raise ValueError("Ewald and tail correction are mutually exclusive")
    if cfg.network.smooth_periodic_jastrow:
        if geometry.kind != "periodic" or cfg.interaction_type != "inverse_cubic":
            raise ValueError(
                "Smooth periodic Jastrow requires periodic inverse_cubic physics"
            )
    mode = cfg.network.lambda_conditioning.mode
    if mode not in ("additive", "film", "none"):
        raise ValueError("lambda conditioning mode must be additive, film, or none")
    training_set = cfg.interaction_strength_training_set
    if mode == "none" and training_set and len(set(training_set)) > 1:
        raise ValueError("Unconditioned models require a single interaction strength")
    if cfg.observables.nk0 and geometry.kind != "periodic":
        raise ValueError("nk0 requires a periodic lattice")
    return geometry
