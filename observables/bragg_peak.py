"""Bragg peak from a complete symmetry star, selected on separate Markov chains."""

import itertools
import numpy as np
import pandas as pd

try:
    from . import _common as common, _geometry as geometry
except ImportError:
    import _common as common
    import _geometry as geometry


def configure(parser):
    parser.add_argument(
        "--shell",
        type=common.positive_int,
        help="Integer reciprocal shell Q; default N.",
    )


def star_vectors(lattice, shell):
    kind = geometry.cell_kind(lattice)
    bound = int(np.ceil(2 * np.sqrt(shell))) + 1
    points = {
        (m, n)
        for m, n in itertools.product(range(-bound, bound + 1), repeat=2)
        if m * m + n * n - (m * n if kind == "hexagonal" else 0) == shell
    }
    stars = []
    while points:
        point = min(points)
        orbit = []
        for _ in range(6 if kind == "hexagonal" else 4):
            orbit.append(point)
            points.discard(point)
            point = (
                (point[0] - point[1], point[0])
                if kind == "hexagonal"
                else (-point[1], point[0])
            )
        stars.append(orbit)
    if not stars:
        raise ValueError(
            f"No reciprocal vectors on shell Q={shell}; choose a commensurate size or --shell"
        )
    return np.asarray(stars) @ (2 * np.pi * np.linalg.inv(lattice))


def crossfit(star_chain_means, seed):
    chains = len(star_chain_means)
    fold = np.random.default_rng(seed).permutation(chains) % 2
    held = np.empty(chains)
    selected = []
    for part in [0, 1]:
        index = np.argmax(star_chain_means[fold != part].mean(axis=0))
        held[fold == part] = star_chain_means[fold == part, index]
        selected.append(int(index))
    return held, selected


def calculate(state, positions, coupling, args):
    del coupling
    shell = args.shell or state.cfg.n_particles
    stars = star_vectors(common.periodic_lattice(state), shell)
    values = geometry.structure_values(positions, stars.reshape(-1, 2))
    means = values.reshape(*values.shape[:2], *stars.shape[:2]).mean(axis=(0, 3))
    held, selected = crossfit(means, args.seed + state.cfg.n_particles)
    mean, se = common.mean_se(held)
    return pd.DataFrame(
        [
            dict(
                bragg_peak=mean,
                bragg_peak_se=se,
                bragg_peak_per_particle=mean / state.cfg.n_particles,
                shell=shell,
                stars=len(stars),
                fold_0_star=selected[0],
                fold_1_star=selected[1],
                error_method="chain standard error conditional on cross-fitted stars",
            )
        ]
    )


def plot(table, args):
    common.curve_plot(
        table, args, "bragg_peak", r"$S(\mathbf{k}_*)$", error="bragg_peak_se"
    )


def main(argv=None):
    return common.run(
        "bragg_peak", __doc__, calculate, plot, configure=configure, argv=argv
    )


if __name__ == "__main__":
    main()
