"""Angle-averaged pair correlation with finite-N ideal-gas normalization."""

import numpy as np
import pandas as pd

try:
    from . import _common as common, _geometry as geometry
except ImportError:
    import _common as common
    import _geometry as geometry


def configure(parser):
    parser.add_argument("--bins", type=common.positive_int, default=100)
    parser.add_argument(
        "--rmax", type=float, help="Maximum radius, at most the nearest-cell inradius."
    )


def calculate(state, positions, coupling, args):
    del coupling
    lattice = common.periodic_lattice(state)
    lengths = np.linalg.norm(geometry.translations(lattice), axis=1)
    inradius = lengths[lengths > 0].min() / 2
    rmax = inradius if args.rmax is None else args.rmax
    if not np.isfinite(rmax) or not 0 < rmax <= inradius * (1 + 1e-12):
        raise ValueError(
            f"--rmax must be positive and at most {inradius:g} for complete annuli"
        )
    edges = np.linspace(0, rmax, args.bins + 1)
    n = state.cfg.n_particles
    normalization = (
        len(positions)
        * n
        * (n - 1)
        / 2
        * np.pi
        * np.diff(edges**2)
        / abs(np.linalg.det(lattice))
    )
    chains = []
    for chain in range(positions.shape[1]):
        disp = geometry.pair_displacements(positions[:, chain], lattice)
        counts = np.histogram(np.linalg.norm(disp, axis=-1), bins=edges)[0]
        chains.append(counts / normalization)
    mean, se = common.mean_se(chains)
    return pd.DataFrame(
        dict(
            r=(edges[1:] + edges[:-1]) / 2,
            r_left=edges[:-1],
            r_right=edges[1:],
            g2=mean,
            g2_se=se,
        )
    )


def plot(table, args):
    common.curve_plot(
        table, args, "g2", r"$g_2(r)$", error="g2_se", x="r", xlabel=r"$r$"
    )


def main(argv=None):
    return common.run(
        "g2_radial", __doc__, calculate, plot, configure=configure, argv=argv
    )


if __name__ == "__main__":
    main()
