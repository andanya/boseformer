"""Potential energy per particle, including the configured periodic sum or disk wall."""

import pandas as pd

try:
    from . import _common as common
except ImportError:
    import _common as common


def calculate(state, positions, coupling, args):
    values = common.evaluate_chunks(
        common.energy_evaluator(state),
        state.params,
        positions,
        coupling,
        args.chunk_size,
    )[..., 2]
    mean, se = common.mean_se(values.mean(axis=0))
    n = state.cfg.n_particles
    return pd.DataFrame(
        [
            dict(
                potential_energy=mean,
                potential_energy_se=se,
                potential_energy_per_particle=mean / n,
                potential_energy_per_particle_se=se / n,
            )
        ]
    )


def plot(table, args):
    common.curve_plot(
        table,
        args,
        "potential_energy_per_particle",
        r"$\langle V\rangle/N$",
        error="potential_energy_per_particle_se",
    )


def main(argv=None):
    return common.run("potential_energy", __doc__, calculate, plot, argv=argv)


if __name__ == "__main__":
    main()
