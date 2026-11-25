"""Static structure factor on the simulation cell's nonzero reciprocal vectors."""

import numpy as np
import pandas as pd

try:
    from . import _common as common, _geometry as geometry
except ImportError:
    import _common as common
    import _geometry as geometry


def configure(parser):
    parser.add_argument(
        "--kmax",
        type=float,
        help="Maximum wave number; default 1.5 times the crystal wave number.",
    )


def calculate(state, positions, coupling, args):
    del coupling
    indices, vectors = geometry.reciprocal_vectors(
        common.periodic_lattice(state), state.cfg.n_particles, args.kmax
    )
    values = geometry.structure_values(positions, vectors)
    mean, se = common.mean_se(values.mean(axis=0))
    return pd.DataFrame(
        dict(
            m=indices[:, 0],
            n=indices[:, 1],
            kx=vectors[:, 0],
            ky=vectors[:, 1],
            k=np.linalg.norm(vectors, axis=1),
            structure_factor=mean,
            structure_factor_se=se,
            structure_factor_per_particle=mean / state.cfg.n_particles,
        )
    )


def plot(table, args):
    plt = common.style()
    for group, output in common.map_groups(table, args):
        fig, ax = plt.subplots(figsize=(6, 5))
        points = ax.scatter(
            group.kx,
            group.ky,
            c=group.structure_factor_per_particle,
            cmap="viridis",
            vmin=0,
            s=45,
        )
        ax.set(xlabel=r"$k_x$", ylabel=r"$k_y$", aspect="equal")
        fig.colorbar(points, ax=ax, label=r"$S(\mathbf{k})/N$")
        common.save_figure(fig, output)


def main(argv=None):
    return common.run(
        "structure_factor_map", __doc__, calculate, plot, configure=configure, argv=argv
    )


if __name__ == "__main__":
    main()
