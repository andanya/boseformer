# Copyright 2023 DeepMind Technologies Limited.
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

"""Boseformer attention layers and conditional bosonic wavefunctions."""

from typing import Mapping, Optional, Sequence, Tuple
import attr
import chex
from network import envelopes, network_blocks, networks, smooth_jastrow
from network.geometry import Geometry
from network.pbc.feature_layer import make_pbc_feature_layer
import jax
import jax.numpy as jnp


@attr.s(auto_attribs=True, kw_only=True)
class PsiformerOptions(networks.BaseNetworkOptions):
    """Options controlling the Psiformer part of the network architecture.

    Attributes:
      num_layers: Number of self-attention layers.
      num_heads: Number of multihead self-attention heads.
      heads_dim: Embedding dimension for each self-attention head.
      mlp_hidden_dims: Tuple of sizes of hidden dimension of the MLP. Note that
        this does not include the final projection to the embedding dimension.
      use_layer_norm: If true, include a layer norm on both attention and MLP.
    """

    num_layers: int = 2
    num_heads: int = 4
    heads_dim: int = 64
    mlp_hidden_dims: Tuple[int, ...] = (256,)
    use_layer_norm: bool = False
    # Lambda conditioning: 'additive', 'film', or 'none'
    lambda_conditioning_mode: str = "additive"
    # If True, feed log(lambda) to the embedding MLP
    lambda_log_scale_input: bool = False


def make_layer_norm() -> ...:
    """Implementation of LayerNorm."""

    def init(param_shape: int) -> Mapping[str, jnp.ndarray]:
        params = {}
        params["scale"] = jnp.ones(param_shape)
        params["offset"] = jnp.zeros(param_shape)
        return params

    def apply(
        params: networks.ParamTree, inputs: jnp.ndarray, axis: int = -1
    ) -> jnp.ndarray:
        mean = jnp.mean(inputs, axis=axis, keepdims=True)
        variance = jnp.var(inputs, axis=axis, keepdims=True)
        eps = 1e-5
        inv = params["scale"] * jax.lax.rsqrt(variance + eps)
        return inv * (inputs - mean) + params["offset"]

    return init, apply


def make_multi_head_attention(num_heads: int, heads_dim: int) -> ...:
    """Boseformer-style version of MultiHeadAttention."""

    # Linear layer plus reshape final dimensions to num_heads, heads_dim.
    def linear_projection(x: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        y = jnp.dot(x, weights)
        return y.reshape(*x.shape[:-1], num_heads, heads_dim)

    def init(
        key: chex.PRNGKey, q_d: int, kv_d: int, output_channels: Optional[int] = None
    ) -> Mapping[str, jnp.ndarray]:

        # Dimension of attention projection.
        qkv_hiddens = num_heads * heads_dim
        if not output_channels:
            output_channels = qkv_hiddens

        key, *subkeys = jax.random.split(key, num=4)
        params = {}
        params["q_w"] = network_blocks.init_linear_layer(
            subkeys[0], in_dim=q_d, out_dim=qkv_hiddens, include_bias=False
        )["w"]
        params["k_w"] = network_blocks.init_linear_layer(
            subkeys[1], in_dim=kv_d, out_dim=qkv_hiddens, include_bias=False
        )["w"]
        params["v_w"] = network_blocks.init_linear_layer(
            subkeys[2], in_dim=kv_d, out_dim=qkv_hiddens, include_bias=False
        )["w"]

        key, subkey = jax.random.split(key)
        params["attn_output"] = network_blocks.init_linear_layer(
            subkey, in_dim=qkv_hiddens, out_dim=output_channels, include_bias=False
        )["w"]

        return params

    def apply(
        params: networks.ParamTree,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
    ) -> jnp.ndarray:
        """Computes MultiHeadAttention with keys, queries and values.

        Args:
          params: Parameters for attention embeddings.
          query: Shape [..., q_index_dim, q_d]
          key: Shape [..., kv_index_dim, kv_d]
          value: Shape [..., kv_index_dim, kv_d]

        Returns:
          A projection of attention-weighted value projections.
          Shape [..., q_index_dim, output_channels]
        """

        # Projections for q, k, v.
        # Output shape: [..., index_dim, num_heads, heads_dim].
        q = linear_projection(query, params["q_w"])
        k = linear_projection(key, params["k_w"])
        v = linear_projection(value, params["v_w"])

        # Scaled dot-product attention. The operation applies 1/√d_k automatically.
        # This jax function implements GPU-optimized Flash attention automatically.
        attn = jax.nn.dot_product_attention(q, k, v)

        # Concatenate attention matrix of all heads into a single vector.
        # Shape [..., q_index_dim, num_heads * heads_dim]
        attn = jnp.reshape(attn, (*query.shape[:-1], -1))

        # Apply a final projection to get the final embeddings.
        # Output shape: [..., q_index_dim, output_channels]
        return network_blocks.linear_layer(attn, params["attn_output"])

    return init, apply


def make_mlp() -> ...:
    """Construct MLP, with final linear projection to embedding size."""

    def init(
        key: chex.PRNGKey, mlp_hidden_dims: Tuple[int, ...], embed_dim: int
    ) -> Sequence[networks.Param]:
        params = []
        dims_one_in = [embed_dim, *mlp_hidden_dims]
        dims_one_out = [*mlp_hidden_dims, embed_dim]
        for i in range(len(dims_one_in)):
            key, subkey = jax.random.split(key)
            params.append(
                network_blocks.init_linear_layer(
                    subkey,
                    in_dim=dims_one_in[i],
                    out_dim=dims_one_out[i],
                    include_bias=True,
                )
            )
        return params

    def apply(params: Sequence[networks.Param], inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs
        for i in range(len(params)):
            x = jnp.tanh(network_blocks.linear_layer(x, **params[i]))
        return x

    return init, apply


def make_lambda_embedding(attn_dim: int, hidden_dim: int = 64) -> ...:
    """Creates an MLP that maps scalar lambda to an attn_dim-sized token.

    Architecture: scalar -> (1,) -> Linear -> tanh -> Linear -> (attn_dim,)
    """

    def init(key: chex.PRNGKey) -> Sequence[networks.Param]:
        params = []
        key, subkey = jax.random.split(key)
        params.append(
            network_blocks.init_linear_layer(
                subkey, in_dim=1, out_dim=hidden_dim, include_bias=True
            )
        )
        key, subkey = jax.random.split(key)
        params.append(
            network_blocks.init_linear_layer(
                subkey, in_dim=hidden_dim, out_dim=attn_dim, include_bias=True
            )
        )
        return params

    def apply(
        params: Sequence[networks.Param], interaction_strength: jnp.ndarray
    ) -> jnp.ndarray:
        """Maps scalar interaction_strength to a parameter token of size attn_dim."""
        x = jnp.atleast_1d(jnp.asarray(interaction_strength, dtype=jnp.float32))
        x = jnp.tanh(network_blocks.linear_layer(x, **params[0]))
        x = network_blocks.linear_layer(x, **params[1])
        return x  # (attn_dim,)

    return init, apply


def make_cross_attention(attn_dim: int, mode: str = "additive") -> ...:
    """Cross-attention that injects the parameter token into site tokens.

    Args:
      attn_dim: dimension of site tokens.
      mode: 'additive' — x + tanh(Linear(p_lambda))
            'film' — gamma(p_lambda) * x + beta(p_lambda)  (FiLM conditioning)
    """

    def init(key: chex.PRNGKey) -> networks.ParamTree:
        params = {}
        if mode == "film":
            key, subkey = jax.random.split(key)
            params["gamma"] = network_blocks.init_linear_layer(
                subkey, in_dim=attn_dim, out_dim=attn_dim, include_bias=True
            )
            key, subkey = jax.random.split(key)
            params["beta"] = network_blocks.init_linear_layer(
                subkey, in_dim=attn_dim, out_dim=attn_dim, include_bias=True
            )
        else:
            key, subkey = jax.random.split(key)
            params["project"] = network_blocks.init_linear_layer(
                subkey, in_dim=attn_dim, out_dim=attn_dim, include_bias=True
            )
        return params

    def apply(
        params: networks.ParamTree, x: jnp.ndarray, p_lambda: jnp.ndarray
    ) -> jnp.ndarray:
        """Injects lambda-dependent modulation into all site tokens.

        Args:
          x: site tokens, shape (n_particles, attn_dim).
          p_lambda: parameter token, shape (attn_dim,).

        Returns:
          Modulated site tokens, shape (n_particles, attn_dim).
        """
        if mode == "film":
            gamma = 1.0 + jnp.tanh(
                network_blocks.linear_layer(p_lambda, **params["gamma"])
            )
            beta = jnp.tanh(network_blocks.linear_layer(p_lambda, **params["beta"]))
            return gamma * x + beta  # broadcast to (n_particles, attn_dim)
        else:
            ca_bias = jnp.tanh(
                network_blocks.linear_layer(p_lambda, **params["project"])
            )
            return x + ca_bias  # broadcast to (n_particles, attn_dim)

    return init, apply


def make_psiformer_layers(options):
    """Per-layer self attention followed by coupling-dependent modulation."""
    attn_dim = options.num_heads * options.heads_dim
    num_layers = options.num_layers

    # Per-layer self-attention components.
    attention_init, attention_apply = make_multi_head_attention(
        options.num_heads, options.heads_dim
    )
    if options.use_layer_norm:
        layer_norm_init, layer_norm_apply = make_layer_norm()
    mlp_init, mlp_apply = make_mlp()

    # Lambda conditioning modules (shared embedding, per-layer CA).
    conditioning = options.lambda_conditioning_mode != "none"
    if options.lambda_conditioning_mode not in ("additive", "film", "none"):
        raise ValueError("lambda conditioning mode must be additive, film, or none")
    log_scale = options.lambda_log_scale_input
    lambda_embed_init, lambda_embed_apply = make_lambda_embedding(attn_dim)
    cross_attn_init, cross_attn_apply = make_cross_attention(
        attn_dim, mode=options.lambda_conditioning_mode
    )

    def init(key: chex.PRNGKey) -> Tuple[int, networks.ParamTree]:
        """Returns tuple of output dimension from the final layer and parameters."""
        params = {}
        key, subkey = jax.random.split(key)
        feature_dims, params["input"] = options.feature_layer.init()
        one_particle_feature_dim, _ = feature_dims
        # Append a constant bias feature to each particle embedding.
        feature_dim = one_particle_feature_dim + 1

        # Map to Attention dim.
        key, subkey = jax.random.split(key)
        params["embed"] = network_blocks.init_linear_layer(
            subkey, in_dim=feature_dim, out_dim=attn_dim, include_bias=False
        )["w"]

        # Shared lambda embedding.
        key, subkey = jax.random.split(key)
        if conditioning:
            params["lambda_embed"] = lambda_embed_init(subkey)

        # Per-layer params: SA + CA at every layer.
        sa_params = []
        ln_params = []
        mlp_params = []
        ca_params = []
        for _ in range(num_layers):
            key, attn_key, mlp_key, ca_key = jax.random.split(key, 4)
            sa_params.append(
                attention_init(
                    attn_key, q_d=attn_dim, kv_d=attn_dim, output_channels=attn_dim
                )
            )
            if options.use_layer_norm:
                ln_params.append([layer_norm_init(attn_dim), layer_norm_init(attn_dim)])
            mlp_params.append(mlp_init(mlp_key, options.mlp_hidden_dims, attn_dim))
            if conditioning:
                ca_params.append(cross_attn_init(ca_key))

        params["attention"] = sa_params
        params["ln"] = ln_params
        params["mlp"] = mlp_params
        if conditioning:
            params["cross_attn"] = ca_params

        return attn_dim, params

    def apply(
        params,
        *,
        offsets: jnp.ndarray,
        radius: jnp.ndarray,
        pairs: jnp.ndarray,
        distances: jnp.ndarray,
        interaction_strength: jnp.ndarray = 0.0,
    ) -> jnp.ndarray:
        """Apply attention to the geometric particle features."""
        # Only one-particle features are used by the Psiformer.
        position_features, _ = options.feature_layer.apply(
            offsets=offsets,
            radius=radius,
            pairs=pairs,
            distances=distances,
            **params["input"],
        )

        # Append a constant bias feature to each particle's geometric features.
        position_features = jnp.concatenate(
            (
                position_features,
                jnp.ones(
                    (position_features.shape[0], 1), dtype=position_features.dtype
                ),
            ),
            axis=-1,
        )

        features = position_features  # One-particle stream.

        # Embed into attention dimension.
        x = jnp.dot(features, params["embed"])

        # Embed scalar lambda into a parameter token (shared across layers).
        lam = interaction_strength
        if log_scale:
            lam = jnp.log(jnp.maximum(lam, 1e-8))
        if conditioning:
            p_lambda = lambda_embed_apply(params["lambda_embed"], lam)

        # ── Per-layer: SA -> CA(lambda) ──
        for layer in range(num_layers):
            # Self-attention + residual.
            attn_output = attention_apply(params["attention"][layer], x, x, x)
            x = x + attn_output
            if options.use_layer_norm:
                x = layer_norm_apply(params["ln"][layer][0], x)

            # MLP + residual.
            mlp_output = mlp_apply(params["mlp"][layer], x)
            x = x + mlp_output
            if options.use_layer_norm:
                x = layer_norm_apply(params["ln"][layer][1], x)

            # Cross-attention injection of lambda.
            if conditioning:
                x = cross_attn_apply(params["cross_attn"][layer], x, p_lambda)

        return x

    return init, apply


def make_network(
    n_particles,
    geometry=None,
    *,
    products=16,
    bias_orbitals=False,
    rescale_inputs=False,
    num_layers=4,
    num_heads=4,
    heads_dim=64,
    mlp_hidden_dims=(256,),
    use_layer_norm=True,
    predict_logits=False,
    boson_head="product",
    lambda_conditioning_mode="additive",
    lambda_log_scale_input=False,
    smooth_periodic_jastrow=False,
    smooth_periodic_jastrow_rmatch_frac=1.0,
):
    """Build a bosonic model on a disk or torus.

    apply(params, positions, interaction_strength) returns sign and log|psi|.
    """
    geometry = geometry or Geometry()
    if not isinstance(n_particles, int) or n_particles < 1 or products < 1:
        raise ValueError("n_particles and products must be positive integers")
    if boson_head not in ("product", "sum"):
        raise ValueError("boson_head must be product or sum")
    periodic = geometry.kind == "periodic"
    features = (
        make_pbc_feature_layer(geometry.lattice, rescale_inputs=rescale_inputs)
        if periodic
        else networks.make_open_features(geometry.ndim, rescale_inputs)
    )
    envelope = (
        envelopes.make_null_envelope()
        if periodic
        else envelopes.make_isotropic_envelope()
    )
    options = PsiformerOptions(
        geometry=geometry,
        products=products,
        envelope=envelope,
        feature_layer=features,
        bias_orbitals=bias_orbitals,
        rescale_inputs=rescale_inputs,
        num_layers=num_layers,
        num_heads=num_heads,
        heads_dim=heads_dim,
        mlp_hidden_dims=mlp_hidden_dims,
        use_layer_norm=use_layer_norm,
        predict_logits=predict_logits,
        boson_head=boson_head,
        lambda_conditioning_mode=lambda_conditioning_mode,
        lambda_log_scale_input=lambda_log_scale_input,
    )
    if smooth_periodic_jastrow:
        if not periodic:
            raise ValueError("Smooth periodic Jastrow requires a periodic lattice")
        log_jastrow = smooth_jastrow.make_smooth_periodic_log_jastrow(
            lattice=geometry.lattice,
            ndim=geometry.ndim,
            rmatch_frac=smooth_periodic_jastrow_rmatch_frac,
        )
    else:
        log_jastrow = None
    init, orbitals = networks.make_orbitals(
        n_particles, options, make_psiformer_layers(options)
    )

    def apply(params, positions, interaction_strength=0.0):
        values = orbitals(params, positions, interaction_strength)
        result = (
            network_blocks.logsum_matmul(values)
            if boson_head == "sum"
            else network_blocks.logproduct_matmul(values, predict_logits=predict_logits)
        )
        if log_jastrow is not None:
            sign, logabs = result
            result = sign, logabs + log_jastrow(positions, interaction_strength)
        return result

    return networks.Network(options=options, init=init, apply=apply, orbitals=orbitals)
