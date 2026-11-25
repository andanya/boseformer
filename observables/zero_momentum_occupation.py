"""Zero-momentum occupation from signed wavefunction ratios and uniform displacements."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

try:
    from . import _common as common
except ImportError:
    import _common as common


def configure(parser):
    parser.add_argument(
        "--probes",
        type=common.positive_int,
        default=8,
        help="Independent particle replacements per saved walker.",
    )
    parser.add_argument(
        "--fraction",
        action="store_true",
        help="Plot n(0)/N instead of n(0). Both are saved.",
    )


def ratio_evaluator(signed, lattice, n, probes, center=(0.0, 0.0)):
    batch = jax.vmap(signed, in_axes=(None, 0, None))

    @jax.jit
    def evaluate(params, positions, coupling, key):
        old_sign, old_log = batch(params, positions, coupling)
        x = positions.reshape(-1, n, 2)

        def probe(_, carry):
            key, total = carry
            key, particle_key, position_key = jax.random.split(key, 3)
            particle = jax.random.randint(particle_key, (len(x),), 0, n)
            replacement = jax.random.uniform(
                position_key, (len(x), 2)
            ) @ lattice.T + jnp.asarray(center)
            changed = x.at[jnp.arange(len(x)), particle].set(replacement)
            new_sign, new_log = batch(
                params, changed.reshape(positions.shape), coupling
            )
            ratio = new_sign * old_sign * jnp.exp(new_log - old_log)
            return key, total + ratio

        _, ratios = jax.lax.fori_loop(0, probes, probe, (key, jnp.zeros(len(x))))
        return ratios / probes

    return evaluate


def calculate(state, positions, coupling, args):
    lattice = jnp.asarray(common.periodic_lattice(state))
    evaluator = ratio_evaluator(
        state.model.apply,
        lattice,
        state.cfg.n_particles,
        args.probes,
        state.geometry.center,
    )
    flat = positions.reshape(-1, positions.shape[-1])
    values = []
    for start in range(0, len(flat), args.chunk_size):
        part = flat[start : start + args.chunk_size]
        padded = np.pad(part, ((0, args.chunk_size - len(part)), (0, 0)), mode="edge")
        key = jax.random.fold_in(jax.random.PRNGKey(args.seed + 9173), start)
        values.extend(
            np.asarray(evaluator(state.params, jnp.asarray(padded), coupling, key))[
                : len(part)
            ]
        )
    values = np.asarray(values).reshape(positions.shape[:2])
    mean, se = common.mean_se(values.mean(axis=0))
    n = state.cfg.n_particles
    return pd.DataFrame(
        [
            dict(
                occupation=n * mean,
                occupation_se=n * se,
                occupation_fraction=mean,
                occupation_fraction_se=se,
                probes=args.probes,
            )
        ]
    )


def plot(table, args):
    column = "occupation_fraction" if args.fraction else "occupation"
    label = r"$n(0)/N$" if args.fraction else r"$n(0)$"
    common.curve_plot(table, args, column, label, error=column + "_se")


def main(argv=None):
    return common.run(
        "zero_momentum_occupation",
        __doc__,
        calculate,
        plot,
        configure=configure,
        argv=argv,
    )


if __name__ == "__main__":
    main()
