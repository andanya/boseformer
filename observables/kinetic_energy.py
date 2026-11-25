"""Kinetic energy per particle from frozen checkpoint sampling."""

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
    )[..., 1]
    mean, se = common.mean_se(values.mean(axis=0))
    n = state.cfg.n_particles
    return pd.DataFrame(
        [
            dict(
                kinetic_energy=mean,
                kinetic_energy_se=se,
                kinetic_energy_per_particle=mean / n,
                kinetic_energy_per_particle_se=se / n,
            )
        ]
    )


def plot(table, args):
    common.curve_plot(
        table,
        args,
        "kinetic_energy_per_particle",
        r"$\langle T\rangle/N$",
        error="kinetic_energy_per_particle_se",
    )


def main(argv=None):
    return common.run("kinetic_energy", __doc__, calculate, plot, argv=argv)


if __name__ == "__main__":
    main()
