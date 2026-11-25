# Copyright 2022 DeepMind Technologies Limited.
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
# limitations under the License

"""Smooth periodic features in square or hexagonal lattice coordinates."""

from typing import Tuple
import chex
from network import networks
import jax.numpy as jnp


def periodic_norm(metric: jnp.ndarray, scaled_r: jnp.ndarray) -> jnp.ndarray:
    """Returns the periodic norm of a set of vectors.

    Args:
      metric: metric tensor in fractional coordinate system, A.T A, where A is the
        lattice vectors.
      scaled_r: vectors in fractional coordinates of the lattice cell, with
        trailing dimension ndim, to compute the periodic norm of.
    """
    chex.assert_rank(metric, expected_ranks=2)
    a = 1 - jnp.cos(2 * jnp.pi * scaled_r)
    b = jnp.sin(2 * jnp.pi * scaled_r)
    cos_term = jnp.einsum("...m,mn,...n->...", a, metric, a)
    sin_term = jnp.einsum("...m,mn,...n->...", b, metric, b)
    return (1 / (2 * jnp.pi)) * jnp.sqrt(cos_term + sin_term)


def make_pbc_feature_layer(lattice, rescale_inputs=False):
    """Sin/cos position features, with smooth periodic pair distances."""
    ndim = lattice.shape[0]
    # Calculate reciprocal vectors, factor 2pi omitted
    reciprocal_vecs = jnp.linalg.inv(lattice)
    lattice_metric = lattice.T @ lattice

    def init() -> Tuple[Tuple[int, int], networks.Param]:
        return (2 * ndim, 2 * ndim + 1), {}

    def apply(offsets, radius, pairs, distances) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Position features in phase coordinates, (scaled_offsets)_i = k_i . offsets
        scaled_offsets = jnp.einsum("il,jkl->jki", reciprocal_vecs, offsets)
        # Pair features in phase coordinates
        scaled_pairs = jnp.einsum("il,jkl->jki", reciprocal_vecs, pairs)
        # Periodized features
        offsets = jnp.concatenate(
            (
                jnp.sin(2 * jnp.pi * scaled_offsets),
                jnp.cos(2 * jnp.pi * scaled_offsets),
            ),
            axis=-1,
        )
        pairs = jnp.concatenate(
            (jnp.sin(2 * jnp.pi * scaled_pairs), jnp.cos(2 * jnp.pi * scaled_pairs)),
            axis=-1,
        )
        # Distance features defined on orthonormal projections
        # Don't take gradients through |0|
        n = pairs.shape[0]
        scaled_pairs += jnp.eye(n)[..., None]
        distances = periodic_norm(lattice_metric, scaled_pairs) * (1.0 - jnp.eye(n))

        position_features = offsets
        position_features = jnp.reshape(
            position_features, [jnp.shape(position_features)[0], -1]
        )
        pair_features = jnp.concatenate((distances[..., None], pairs), axis=2)
        return position_features, pair_features

    return networks.FeatureLayer(init=init, apply=apply)
