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

"""Metropolis-Hastings Monte Carlo.

NOTE: these functions operate on batches of MCMC configurations and should not
be vmapped.
"""

import chex
from network import constants
from network import networks
import jax
from jax import lax
from jax import numpy as jnp
import numpy as np


def mh_accept(x1, x2, lp_1, lp_2, ratio, key, num_accepts):
    """Given state, proposal, and probabilities, execute MH accept/reject step."""
    key, subkey = jax.random.split(key)
    rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
    cond = ratio > rnd
    x_new = jnp.where(cond[..., None], x2, x1)
    lp_new = jnp.where(cond, lp_2, lp_1)
    num_accepts += jnp.sum(cond)
    return (x_new, key, lp_new, num_accepts)


def mh_update(
    params: networks.ParamTree,
    f: networks.LogNetworkLike,
    data: networks.WalkerData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
    stddev=0.02,
    ndim=2,
    blocks=1,
    i=0,
):
    """Performs one Metropolis-Hastings step using an all-particle move."""
    del i, blocks
    key, subkey = jax.random.split(key)
    x1 = data.positions
    x2 = x1 + stddev * jax.random.normal(subkey, shape=x1.shape)
    lp_2 = 2.0 * f(params, x2, data.interaction_strength)
    ratio = lp_2 - lp_1
    x_new, key, lp_new, num_accepts = mh_accept(
        x1, x2, lp_1, lp_2, ratio, key, num_accepts
    )
    new_data = networks.WalkerData(**dict(data) | {"positions": x_new})
    return (new_data, key, lp_new, num_accepts)


def mh_block_update(
    params: networks.ParamTree,
    f: networks.LogNetworkLike,
    data: networks.WalkerData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
    stddev=0.02,
    ndim=2,
    blocks=1,
    i=0,
):
    """Performs one Metropolis-Hastings step for a block of particles."""
    key, subkey = jax.random.split(key)
    batch_size = data.positions.shape[0]
    n_particles = data.positions.shape[1] // ndim
    pad = (blocks - n_particles % blocks) % blocks
    x1 = jnp.reshape(
        jnp.pad(data.positions, ((0, 0), (0, pad * ndim))),
        [batch_size, blocks, -1, ndim],
    )
    ii = i % blocks
    x2 = x1.at[:, ii].add(stddev * jax.random.normal(subkey, shape=x1[:, ii].shape))
    x2 = jnp.reshape(x2, [batch_size, -1])
    if pad > 0:
        x2 = x2[..., : -pad * ndim]
    lp_2 = 2.0 * f(params, x2, data.interaction_strength)
    ratio = lp_2 - lp_1
    x1 = jnp.reshape(x1, [batch_size, -1])
    if pad > 0:
        x1 = x1[..., : -pad * ndim]
    x_new, key, lp_new, num_accepts = mh_accept(
        x1, x2, lp_1, lp_2, ratio, key, num_accepts
    )
    new_data = networks.WalkerData(**dict(data) | {"positions": x_new})
    return (new_data, key, lp_new, num_accepts)


def make_mcmc_step(
    batch_network, batch_per_device, steps=10, ndim=2, blocks=1, lattice=None
):
    """Creates the MCMC step function."""
    inner_fun = mh_block_update if blocks > 1 else mh_update
    if lattice is not None:
        lattice = jnp.asarray(lattice)
        inv_lattice = jnp.linalg.inv(lattice)

    def mcmc_step(params, data, key, width):
        """Performs a set of MCMC steps.

        Args:
          params: parameters to pass to the network.
          data: (batched) MCMC configurations to pass to the network.
          key: RNG state.
          width: standard deviation to use in the move proposal.

        Returns:
          (data, pmove), where data is the updated MCMC configurations, key the
          updated RNG state and pmove the average probability a move was accepted.
        """
        pos = data.positions

        def step_fn(i, x):
            return inner_fun(
                params, batch_network, *x, stddev=width, ndim=ndim, blocks=blocks, i=i
            )

        nsteps = steps * blocks
        logprob = 2.0 * batch_network(params, pos, data.interaction_strength)
        new_data, key, _, num_accepts = lax.fori_loop(
            0, nsteps, step_fn, (data, key, logprob, 0.0)
        )
        if lattice is not None:
            wrapped_pos = new_data.positions
            lat_ndim = lattice.shape[0]
            batch_size = wrapped_pos.shape[0]
            pos_3d = jnp.reshape(wrapped_pos, [batch_size, -1, lat_ndim])
            frac = jnp.einsum("ij,bej->bei", inv_lattice, pos_3d)
            frac = frac % 1.0
            pos_3d = jnp.einsum("ij,bej->bei", lattice, frac)
            wrapped_pos = jnp.reshape(pos_3d, [batch_size, -1])
            new_data = networks.WalkerData(
                **dict(new_data) | {"positions": wrapped_pos}
            )
        pmove = jnp.sum(num_accepts) / (nsteps * batch_per_device)
        pmove = constants.pmean(pmove)
        return (new_data, pmove)

    return mcmc_step


def update_mcmc_width(
    t: int,
    width: jnp.ndarray,
    adapt_frequency: int,
    pmove: jnp.ndarray,
    pmoves: np.ndarray,
    pmove_max: float = 0.55,
    pmove_min: float = 0.5,
) -> tuple[jnp.ndarray, np.ndarray]:
    """Updates the width in MCMC steps.

    Args:
      t: Current step.
      width: Current MCMC width.
      adapt_frequency: The number of iterations after which the update is applied.
      pmove: Acceptance ratio in the last step.
      pmoves: Acceptance ratio over the last N steps, where N is the number of
        steps between MCMC width updates.
      pmove_max: The upper threshold for the range of allowed pmove values
      pmove_min: The lower threshold for the range of allowed pmove values

    Returns:
      width: Updated MCMC width.
      pmoves: Updated `pmoves`.
    """
    t_since_mcmc_update = t % adapt_frequency
    pmoves[t_since_mcmc_update] = pmove.reshape(-1)[0].item()
    if t > 0 and t_since_mcmc_update == 0:
        if np.mean(pmoves) > pmove_max:
            width *= 1.1
        elif np.mean(pmoves) < pmove_min:
            width /= 1.1
    return (width, pmoves)
