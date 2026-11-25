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

"""Linear layers and stable aggregation of generalized bosonic orbitals."""

import functools
from typing import MutableMapping, Optional, Sequence, Tuple
import chex
import jax
import jax.numpy as jnp


def init_linear_layer(
    key: chex.PRNGKey, in_dim: int, out_dim: int, include_bias: bool = True
) -> MutableMapping[str, jnp.ndarray]:
    """Initialises parameters for a linear layer, x w + b.

    Args:
      key: JAX PRNG state.
      in_dim: input dimension to linear layer.
      out_dim: output dimension (number of hidden units) of linear layer.
      include_bias: if true, include a bias in the linear layer.

    Returns:
      A mapping containing the weight matrix (key 'w') and, if required, bias
      unit (key 'b').
    """
    key1, key2 = jax.random.split(key)
    weight = jax.random.normal(key1, shape=(in_dim, out_dim)) / jnp.sqrt(float(in_dim))
    if include_bias:
        bias = jax.random.normal(key2, shape=(out_dim,))
        return {"w": weight, "b": bias}
    else:
        return {"w": weight}


def linear_layer(
    x: jnp.ndarray, w: jnp.ndarray, b: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """Evaluates a linear layer, x w + b.

    Args:
      x: inputs.
      w: weights.
      b: optional bias.

    Returns:
      x w + b if b is given, x w otherwise.
    """
    y = jnp.dot(x, w)
    return y + b if b is not None else y


def slogproduct(x, eps: float = 1e-7, predict_logits: bool = False):
    """Computes sign and log of product of orbitals.

    Args:
      x: an array.
      eps: small constant for numerical stability.
      predict_logits: if True, treat network outputs as log values directly.

    Returns:
      sign, (natural) logarithm of the product of x.
    """
    if predict_logits:
        sign = 1
        logproduct = jnp.sum(x, axis=-1)
    else:
        sign = jnp.prod(jnp.sign(x), axis=-1)
        logproduct = jnp.sum(jnp.log(jnp.abs(x) + eps), axis=-1)

    return sign, logproduct


def slogsum(x, eps: float = 1e-7):
    """Computes sign and log of sum of orbitals.

    Args:
      x: an array.
      eps: small constant for numerical stability.

    Returns:
      sign, (natural) logarithm of the sum of x along the last axis.
    """
    s = jnp.sum(x, axis=-1)
    sign = jnp.sign(s)
    logsum = jnp.log(jnp.abs(s) + eps)
    return sign, logsum


def logsum_matmul(
    xs: Sequence[jnp.ndarray],
    w: Optional[jnp.ndarray] = None,
    eps: float = 1e-7,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Combines sums and takes dot product with weights in log-domain.

    Analogous to logproduct_matmul but uses sum of orbitals instead of product.

    Args:
      xs: Boseformer orbitals in each product. Either of length 1 with shape
        (nproducts, n_particles)
      w: weight of each product. If none, a uniform weight is assumed.

    Returns:
      sum_i w_i S_i in the log domain, where w_i is the weight of S_i, the i-th
      sum
    """
    phase_in, log_amplitude = functools.reduce(
        lambda a, b: (a[0] * b[0], a[1] + b[1]), [slogsum(x, eps) for x in xs], (1, 0)
    )

    max_log_amplitude = jnp.max(log_amplitude)
    amplitude = phase_in * jnp.exp(log_amplitude - max_log_amplitude)
    if w is None:
        result = jnp.sum(amplitude)
    else:
        result = jnp.matmul(amplitude, w)[0]
    if result.dtype == jnp.complex64 or result.dtype == jnp.complex128:
        phase_out = jnp.angle(result)
    else:
        phase_out = jnp.sign(result)
    log_out = jnp.log(jnp.abs(result)) + max_log_amplitude
    return phase_out, log_out


def logproduct_matmul(
    xs: Sequence[jnp.ndarray],
    w: Optional[jnp.ndarray] = None,
    eps: float = 1e-7,
    predict_logits: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Combines products and takes dot product with weights in log-domain.

    We use the log-sum-exp trick to reduce numerical instabilities.

    Args:
      xs: Boseformer orbitals in each product. Either of length 1 with shape
        (nproducts, n_particles)
      w: weight of each product. If none, a uniform weight is assumed.

    Returns:
      sum_i w_i P_i in the log domain, where w_i is the weight of P_i, the i-th
      product
    """

    phase_in, log_amplitude = functools.reduce(
        lambda a, b: (a[0] * b[0], a[1] + b[1]),
        [slogproduct(x, eps, predict_logits) for x in xs],
        (1, 0),
    )

    max_log_amplitude = jnp.max(log_amplitude)
    amplitude = phase_in * jnp.exp(log_amplitude - max_log_amplitude)
    if w is None:
        result = jnp.sum(amplitude)
    else:
        result = jnp.matmul(amplitude, w)[0]
    if result.dtype == jnp.complex64 or result.dtype == jnp.complex128:
        phase_out = jnp.angle(result)  # result / jnp.abs(result)
    else:
        phase_out = jnp.sign(result)
    log_out = jnp.log(jnp.abs(result)) + max_log_amplitude
    return phase_out, log_out
