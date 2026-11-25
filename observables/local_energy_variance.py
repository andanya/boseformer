"""Variance of unclipped individual local energies, with chain jackknife uncertainty."""

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
    result = common.energy_statistics(values)
    result["error_method"] = (
        "delete-one-chain jackknife; variance of total local energy"
    )
    return pd.DataFrame([result])


def plot(table, args):
    common.curve_plot(
        table,
        args,
        "energy_variance",
        r"$\mathrm{var}(E_L)$",
        error="energy_variance_se",
        log=True,
    )


def main(argv=None):
    return common.run("local_energy_variance", __doc__, calculate, plot, argv=argv)


if __name__ == "__main__":
    main()
