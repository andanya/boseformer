"""Total variational energy per particle from frozen checkpoint sampling."""

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
    )[..., 0]
    mean, se = common.mean_se(values.mean(axis=0))
    n = state.cfg.n_particles
    return pd.DataFrame(
        [
            dict(
                energy=mean,
                energy_se=se,
                energy_per_particle=mean / n,
                energy_per_particle_se=se / n,
            )
        ]
    )


def plot(table, args):
    common.curve_plot(
        table, args, "energy_per_particle", r"$E/N$", error="energy_per_particle_se"
    )


def main(argv=None):
    return common.run("energy", __doc__, calculate, plot, argv=argv)


if __name__ == "__main__":
    main()
