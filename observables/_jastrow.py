"""Energy-optimized pure-Jastrow reference, with independent production sampling."""

from pathlib import Path
import jax.numpy as jnp
import numpy as np
import pandas as pd
from network.pbc.feature_layer import periodic_norm
from network.smooth_jastrow import _log_k0

try:
    from . import _common as common
except ImportError:
    import _common as common


def configure(parser):
    parser.add_argument(
        "--baseline-table",
        type=Path,
        help="Optional matching optimized-Jastrow CSV; otherwise optimize and sample it.",
    )
    parser.add_argument(
        "--baseline-grid",
        type=float,
        nargs="+",
        default=[0.04, 0.08, 0.16, 0.32, 0.64, 1.0, 2.0, 8.0],
        help="Positive R0/Lmin values to scan before local quadratic refinement.",
    )
    parser.add_argument("--baseline-scan-samples", type=common.positive_int, default=16)


def make_wavefunction(lattice, n):
    inverse = jnp.asarray(np.linalg.inv(lattice))
    metric = jnp.asarray(lattice.T @ lattice)
    lmin = float(np.linalg.norm(lattice, axis=0).min())
    i, j = jnp.triu_indices(n, k=1)

    def signed(params, positions, coupling):
        x = positions.reshape(n, 2)
        distance = periodic_norm(metric, (x[i] - x[j]) @ inverse.T)
        r0 = params["fraction"] * lmin
        rho = r0 * jnp.tanh(distance / r0)
        logj = _log_k0(2 * jnp.sqrt(coupling / (rho + 1e-12))).sum()
        return jnp.asarray(1.0), logj

    return signed


def from_table(path, n, coupling):
    table = pd.read_csv(path).rename(
        columns={"N": "n_particles", "energy_mean_se": "energy_se"}
    )
    required = {
        "n_particles",
        "lambda",
        "energy_mean",
        "energy_se",
        "energy_variance",
        "energy_variance_se",
        "rmatch_frac",
    }
    if not required <= set(table):
        raise ValueError(
            f"Baseline CSV is missing columns: {sorted(required - set(table))}"
        )
    selected = table[(table.n_particles == n) & np.isclose(table["lambda"], coupling)]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one baseline row for N={n}, lambda={coupling:g}; found {len(selected)}"
        )
    result = {
        key: float(selected.iloc[0][key])
        for key in required - {"n_particles", "lambda"}
    }
    if (
        not np.isfinite(list(result.values())).all()
        or min(
            result["energy_se"], result["energy_variance"], result["energy_variance_se"]
        )
        < 0
        or result["rmatch_frac"] <= 0
    ):
        raise ValueError(
            "Baseline statistics and R0/Lmin must be finite and physically valid"
        )
    result["source"] = str(path)
    return result


def optimize_reference(state, coupling, args):
    lattice = common.periodic_lattice(state)
    if (
        state.cfg.interaction_type != "inverse_cubic"
        or not state.cfg.interaction_ewald
        or state.cfg.interaction_small_length_cutoff != 0
        or coupling <= 0
    ):
        raise ValueError(
            "The paper Jastrow baseline requires positive coupling and bare periodic dipolar Ewald interactions"
        )
    if args.baseline_table:
        return from_table(args.baseline_table, state.cfg.n_particles, coupling)
    grid = np.unique(args.baseline_grid)
    if len(grid) < 3 or not np.isfinite(grid).all() or grid.min() <= 0:
        raise ValueError(
            "--baseline-grid needs at least three distinct positive finite values"
        )
    signed = make_wavefunction(lattice, state.cfg.n_particles)
    evaluator = common.energy_evaluator(state, signed)
    base_seed = args.seed + 400001 + int(coupling * 1000)

    def estimate(fraction, seed, samples):
        # Start half the chains uniformly to complement checkpoint configurations.
        rng = np.random.default_rng(seed)
        selected = state.positions[
            rng.choice(len(state.positions), args.walkers, replace=False)
        ].copy()
        half = args.walkers // 2
        selected[:half] = (
            rng.random((half, state.cfg.n_particles, 2)) @ lattice.T
            + np.asarray(state.geometry.center)
        ).reshape(half, -1)
        params = {"fraction": jnp.asarray(fraction)}
        positions, acceptance = common.sample(
            signed,
            params,
            selected,
            state.geometry,
            coupling,
            args,
            seed=seed,
            width=0.3,
            samples=samples,
        )
        energies = common.evaluate_chunks(
            evaluator, params, positions, coupling, args.chunk_size
        )[..., 0]
        return common.energy_statistics(energies), acceptance

    estimates = []
    for i, fraction in enumerate(grid):
        stats, _ = estimate(fraction, base_seed + i, args.baseline_scan_samples)
        estimates.append(stats["energy_mean"])
    if np.argmin(estimates) == 0:
        lower = grid[0] * np.array([0.25, 0.5])
        extra = [
            estimate(f, base_seed + 100 + i, args.baseline_scan_samples)[0][
                "energy_mean"
            ]
            for i, f in enumerate(lower)
        ]
        grid, estimates = np.concatenate([lower, grid]), extra + estimates
    i = int(np.argmin(estimates))
    chosen = float(grid[i])
    method = "coarse-grid minimum"
    boundary = i in (0, len(grid) - 1)
    if not boundary:
        a, b, _ = np.polyfit(np.log(grid[i - 1 : i + 2]), estimates[i - 1 : i + 2], 2)
        if a > 0 and np.log(grid[i - 1]) < -b / (2 * a) < np.log(grid[i + 1]):
            chosen = float(np.exp(-b / (2 * a)))
            method = "local convex quadratic in log(R0/Lmin)"
    stats, acceptance = estimate(chosen, base_seed + 10001, args.samples)
    stats.update(
        rmatch_frac=chosen,
        R0=chosen * np.linalg.norm(lattice, axis=0).min(),
        optimization_method=method,
        optimization_boundary=boundary,
        scan_fractions=";".join(map(str, grid)),
        scan_energy_means=";".join(map(str, estimates)),
        acceptance=acceptance,
        source="independent pure-Jastrow sampling",
    )
    print(f"Pure Jastrow: R0/Lmin={chosen:g}, E={stats['energy_mean']:g}", flush=True)
    return stats


def comparison(state, positions, coupling, args):
    energies = common.evaluate_chunks(
        common.energy_evaluator(state),
        state.params,
        positions,
        coupling,
        args.chunk_size,
    )[..., 0]
    result = common.energy_statistics(energies)
    reference = optimize_reference(state, coupling, args)
    result.update({f"baseline_{key}": value for key, value in reference.items()})
    result["error_method"] = (
        "independent chain estimates; first-order ratio propagation; conditional on optimized R0"
    )
    return result
