"""Sample the two-dimensional pair correlation in the physical nearest-image cell."""

import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

try:
    from . import _common as common, _geometry as geometry
except ImportError:
    import _common as common
    import _geometry as geometry


def configure(parser):
    parser.add_argument("--bins", type=common.positive_int, default=80)
    parser.add_argument(
        "--vmax",
        type=float,
        default=4.0,
        help="Upper colour scale, matching the paper default.",
    )


def calculate(state, positions, coupling, args):
    del coupling
    lattice = common.periodic_lattice(state)
    polygon = geometry.cell_polygon(lattice)
    low, high = polygon.min(axis=0), polygon.max(axis=0)
    xe, ye = [np.linspace(low[i], high[i], args.bins + 1) for i in range(2)]
    areas = geometry.pixel_areas(xe, ye, lattice)
    inside = areas > 1e-12 * abs(np.linalg.det(lattice))
    n = state.cfg.n_particles
    normalization = len(positions) * n * (n - 1) * areas / abs(np.linalg.det(lattice))
    estimates = []
    for chain in range(positions.shape[1]):
        disp = geometry.pair_displacements(positions[:, chain], lattice).reshape(-1, 2)
        both = np.concatenate([disp, -disp])
        counts = np.histogram2d(*both.T, bins=(xe, ye))[0]
        estimates.append(
            np.divide(counts, normalization, out=np.zeros_like(counts), where=inside)
        )
    mean, se = common.mean_se(estimates)
    x, y = np.meshgrid((xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2, indexing="ij")
    return pd.DataFrame(
        dict(
            x=x.ravel(),
            y=y.ravel(),
            g2=mean.ravel(),
            g2_se=se.ravel(),
            inside_cell=inside.ravel(),
            bin_area=areas.ravel(),
            dx=xe[1] - xe[0],
            dy=ye[1] - ye[0],
            lattice_00=lattice[0, 0],
            lattice_01=lattice[0, 1],
            lattice_10=lattice[1, 0],
            lattice_11=lattice[1, 1],
        )
    )


def plot(table, args):
    if args.vmax <= 0:
        raise ValueError("--vmax must be positive")
    plt = common.style()
    for group, output in common.map_groups(table, args):
        x, y = np.sort(group.x.unique()), np.sort(group.y.unique())
        row = group.iloc[0]
        lattice = np.array(
            [[row.lattice_00, row.lattice_01], [row.lattice_10, row.lattice_11]]
        )
        image = group.sort_values(["x", "y"]).g2.to_numpy().reshape(len(x), len(y)).T
        mask = (
            group.sort_values(["x", "y"])
            .inside_cell.to_numpy()
            .reshape(len(x), len(y))
            .T
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            np.ma.masked_where(~mask, image),
            origin="lower",
            cmap="inferno",
            vmin=0,
            vmax=args.vmax,
            interpolation="nearest",
            extent=[
                x[0] - row.dx / 2,
                x[-1] + row.dx / 2,
                y[0] - row.dy / 2,
                y[-1] + row.dy / 2,
            ],
        )
        boundary = Polygon(
            geometry.cell_polygon(lattice), fill=False, edgecolor=".35", lw=0.8
        )
        ax.add_patch(boundary)
        im.set_clip_path(boundary)
        ax.set(
            xlabel=r"$r_x$",
            ylabel=r"$r_y$",
            aspect="equal",
            title=rf"$N={int(row.n_particles)},\ \lambda={row['lambda']:g}$",
        )
        fig.colorbar(im, ax=ax, label=r"$g_2(\mathbf{r})$")
        common.save_figure(fig, output)


def main(argv=None):
    return common.run(
        "g2_map", __doc__, calculate, plot, configure=configure, argv=argv
    )


if __name__ == "__main__":
    main()
