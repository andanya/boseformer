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

"""Walker data, geometric features and permutation-equivariant orbitals."""

from typing import Any, Callable, Mapping
import attr
import chex
import jax
import jax.numpy as jnp
from network import envelopes, network_blocks
from network.geometry import Geometry

Param = Mapping[str, jax.Array]
ParamTree = Any
NetworkLike = Callable
LogNetworkLike = Callable
InitLayersFn = Callable
ApplyLayersFn = Callable


@chex.dataclass
class WalkerData:
    """One walker, or a batch: flat particle positions and scalar coupling."""

    positions: jax.Array
    interaction_strength: jax.Array = 0.0


@attr.s(auto_attribs=True)
class FeatureLayer:
    init: Callable
    apply: Callable


@attr.s(auto_attribs=True, kw_only=True)
class BaseNetworkOptions:
    geometry: Geometry
    products: int = 16
    bias_orbitals: bool = False
    envelope: envelopes.Envelope = attr.Factory(envelopes.make_isotropic_envelope)
    feature_layer: FeatureLayer
    rescale_inputs: bool = False
    predict_logits: bool = False
    boson_head: str = "product"


@attr.s(auto_attribs=True)
class Network:
    options: BaseNetworkOptions
    init: Callable
    apply: Callable
    orbitals: Callable


def construct_input_features(positions, center=(0.0, 0.0), ndim=2):
    """Return center offsets, pair displacements, radii and pair distances.

    Offsets and radii have shapes (n_particles, 1, ndim) and
    (n_particles, 1, 1). The center sets the spatial origin of the domain.
    """
    pos = jnp.reshape(positions, (-1, ndim))
    offsets = pos[:, None, :] - jnp.asarray(center)[None, None, :]
    pairs = pos[None, :, :] - pos[:, None, :]
    radius = jnp.linalg.norm(offsets, axis=2, keepdims=True)
    n = pos.shape[0]
    distances = jnp.linalg.norm(pairs + jnp.eye(n)[..., None], axis=-1)
    distances *= 1.0 - jnp.eye(n)
    return offsets, pairs, radius, distances[..., None]


def make_open_features(ndim=2, rescale_inputs=False):
    """Cartesian and radial features relative to the disk center."""

    def init():
        return (ndim + 1, ndim + 1), {}

    def apply(offsets, radius, pairs, distances):
        if rescale_inputs:
            scaled_radius = jnp.log(1 + radius)
            offsets = offsets * scaled_radius / radius
            radius = scaled_radius
            scaled_distances = jnp.log(1 + distances)
            pairs = pairs * scaled_distances / jnp.where(distances > 0, distances, 1.0)
            distances = scaled_distances
        features = jnp.concatenate((radius, offsets), axis=2)
        features = jnp.reshape(features, [jnp.shape(features)[0], -1])
        pair_features = jnp.concatenate((distances, pairs), axis=2)
        return features, pair_features

    return FeatureLayer(init=init, apply=apply)


def make_orbitals(n_particles, options, equivariant_layers):
    """Construct generalized orbitals of shape (products, particles)."""
    layers_init, layers_apply = equivariant_layers

    def init(key):
        key, subkey = jax.random.split(key)
        params = {}
        features, params["layers"] = layers_init(subkey)
        params["envelope"] = options.envelope.init(options.products)
        key, subkey = jax.random.split(key)
        params["orbital"] = [
            network_blocks.init_linear_layer(
                subkey,
                in_dim=features,
                out_dim=options.products,
                include_bias=options.bias_orbitals,
            )
        ]
        return params

    def apply(params, positions, interaction_strength=0.0):
        offsets, pairs, radius, distances = construct_input_features(
            positions, options.geometry.center, options.geometry.ndim
        )
        features = layers_apply(
            params["layers"],
            offsets=offsets,
            radius=radius,
            pairs=pairs,
            distances=distances,
            interaction_strength=interaction_strength,
        )
        orbitals = network_blocks.linear_layer(features, **params["orbital"][0])
        orbitals *= options.envelope.apply(radius, **params["envelope"][0])
        return [jnp.transpose(orbitals.reshape(n_particles, -1), (1, 0))]

    return init, apply
