"""Angle-average S(k) over equal-radius shells of allowed reciprocal vectors."""

import pandas as pd

try:
    from . import _common as common, _geometry as geometry
except ImportError:
    import _common as common
    import _geometry as geometry


def configure(parser):
    parser.add_argument("--kmax", type=float)


def calculate(state, positions, coupling, args):
    del coupling
    k, counts, values = geometry.shell_values(
        positions, common.periodic_lattice(state), state.cfg.n_particles, args.kmax
    )
    mean, se = common.mean_se(values.mean(axis=0))
    return pd.DataFrame(
        dict(k=k, shell_vectors=counts, structure_factor=mean, structure_factor_se=se)
    )


def plot(table, args):
    common.curve_plot(
        table,
        args,
        "structure_factor",
        r"$S(k)$",
        error="structure_factor_se",
        x="k",
        xlabel=r"$k$",
    )


def main(argv=None):
    return common.run(
        "structure_factor_radial",
        __doc__,
        calculate,
        plot,
        configure=configure,
        argv=argv,
    )


if __name__ == "__main__":
    main()
