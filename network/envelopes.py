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
# limitations under the License.

"""Radial orbital envelope for a disk, and constant envelope for a torus."""

from dataclasses import dataclass
from typing import Callable
import jax.numpy as jnp


@dataclass
class Envelope:
    init: Callable
    apply: Callable


def make_isotropic_envelope():
    """Learned exponential decay with distance from the disk center."""

    def init(products):
        # Parameters have shape (radial components, products).
        return [{"pi": jnp.ones((1, products)), "sigma": jnp.ones((1, products))}]

    def apply(radius, pi, sigma):
        return jnp.sum(jnp.exp(-radius * sigma) * pi, axis=1)

    return Envelope(init, apply)


def make_null_envelope():
    """Constant envelope, preserving periodic boundary conditions."""
    return Envelope(lambda products: [{}], lambda radius: jnp.ones((1,)))
