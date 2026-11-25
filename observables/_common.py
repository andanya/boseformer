"""Configuration, frozen-state sampling, statistics, and publication plotting."""

import argparse
import ast
import contextlib
from dataclasses import dataclass
import io
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from network import config, hamiltonian, mcmc, networks, psiformer
from network.geometry import Geometry

COLORS = {7: "#e69f00", 19: "#0072b2", 25: "#009e73", 37: "#8c5aa7", 61: "#8c564b"}


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parser_for(name, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        required=True,
        help="Checkpoint file, or directory containing qmcjax_ckpt_*.npz.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        nargs="+",
        required=True,
        help="Matching Python run config or JSON; one per checkpoint, or one shared config.",
    )
    parser.add_argument(
        "--lambda",
        "--lambdas",
        dest="couplings",
        type=float,
        nargs="+",
        required=True,
        help="Interaction strengths at which to sample.",
    )
    parser.add_argument(
        "--walkers",
        type=positive_int,
        default=64,
        help="Number of Markov chains (default: 64).",
    )
    parser.add_argument(
        "--burn-in",
        type=positive_int,
        default=200,
        help="Discarded Metropolis steps (default: 200).",
    )
    parser.add_argument(
        "--samples",
        type=positive_int,
        default=64,
        help="Saved times per chain (default: 64).",
    )
    parser.add_argument(
        "--thin",
        type=positive_int,
        default=10,
        help="Metropolis steps between saved times (default: 10).",
    )
    parser.add_argument(
        "--chunk-size",
        type=positive_int,
        default=32,
        help="Wavefunction/energy evaluation batch size.",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--training-couplings",
        type=float,
        nargs="+",
        help="Couplings marked red on scalar plots; defaults to the supplied config's training set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("observable-results") / name,
        help="Output prefix for CSV, PDF, and PNG files.",
    )
    parser.add_argument(
        "--x-axis",
        choices=["lambda", "n_particles"],
        default="lambda",
        help="For scalar plots; size scans require multiple matching checkpoints/configs.",
    )
    return parser


def read_config(path, coupling):
    """Load a trusted run script without calling its training entry point."""
    if path.suffix == ".json":
        cfg = config.default()

        def merge(target, source):
            for key, value in source.items():
                if isinstance(value, dict) and key in target:
                    merge(target[key], value)
                else:
                    target[key] = value

        merge(cfg, json.loads(path.read_text()))
        return cfg
    tree = ast.parse(path.read_text(), filename=str(path))
    train_modules, train_functions = {"train"}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            train_modules.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "network.train"
            )
        if isinstance(node, ast.ImportFrom) and node.module == "network":
            train_modules.update(
                a.asname or a.name for a in node.names if a.name == "train"
            )
        if isinstance(node, ast.ImportFrom) and node.module == "network.train":
            train_functions.update(
                a.asname or a.name for a in node.names if a.name == "train"
            )

    class ConfigOnly(ast.NodeTransformer):
        def visit_Call(self, node):
            func = ast.unparse(node.func)
            if func in train_functions | {f"{name}.train" for name in train_modules} | {
                "network.train.train"
            }:
                return ast.copy_location(ast.Constant(value=None), node)
            return self.generic_visit(node)

    tree = ast.fix_missing_locations(ConfigOnly().visit(tree))
    namespace = {"__name__": "__main__", "__file__": str(path.resolve())}
    previous_argv = sys.argv
    try:
        sys.argv = [str(path), str(coupling), "0"]
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(tree, str(path), "exec"), namespace)
    finally:
        sys.argv = previous_argv
    if "cfg" not in namespace:
        raise ValueError(f"{path} must define cfg")
    return config.resolve(namespace["cfg"])


def checkpoint_path(path):
    if path.is_dir():
        candidates = list(path.glob("qmcjax_ckpt_*.npz"))
        if not candidates:
            raise ValueError(f"No qmcjax_ckpt_*.npz checkpoint in {path}")
        path = max(candidates, key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not path.is_file():
        raise ValueError(f"Checkpoint does not exist: {path}")
    return path


@dataclass
class State:
    cfg: object
    geometry: Geometry
    model: object
    params: object
    positions: np.ndarray
    width: float
    path: Path
    step: int


def load_state(checkpoint, cfg):
    """Read parameters and walkers independently of the original device count."""
    path = checkpoint_path(checkpoint)
    geometry = Geometry.from_config(cfg)
    model = psiformer.make_network(
        cfg.n_particles,
        geometry=geometry,
        products=cfg.network.products,
        bias_orbitals=cfg.network.bias_orbitals,
        rescale_inputs=cfg.network.rescale_inputs,
        predict_logits=cfg.network.predict_logits,
        boson_head=cfg.network.boson_head,
        lambda_conditioning_mode=cfg.network.lambda_conditioning.mode,
        lambda_log_scale_input=cfg.network.lambda_conditioning.log_scale_input,
        smooth_periodic_jastrow=cfg.network.smooth_periodic_jastrow,
        smooth_periodic_jastrow_rmatch_frac=cfg.network.smooth_periodic_jastrow_rmatch_frac,
        **cfg.network.psiformer,
    )
    with np.load(path, allow_pickle=True) as saved:
        positions = np.asarray(saved["data"].item()["positions"])
        params = saved["params"].item()
        step = int(saved["t"])
        width = float(np.asarray(saved["mcmc_width"]).mean())
    if positions.ndim == 3:
        replicas = positions.shape[0]

        def unreplicate(value):
            value = np.asarray(value)
            if value.shape[0] != replicas or not all(
                np.array_equal(value[0], v) for v in value[1:]
            ):
                raise ValueError("Checkpoint parameter replicas disagree")
            return jnp.asarray(value[0])

        params = jax.tree_util.tree_map(unreplicate, params)
    elif positions.ndim == 2:
        params = jax.tree_util.tree_map(jnp.asarray, params)
    else:
        raise ValueError(
            "Expected checkpoint walker shape (devices, walkers, 2*N) or (walkers, 2*N)"
        )
    if positions.shape[-1] != 2 * cfg.n_particles:
        raise ValueError("Particle count in config does not match checkpoint")
    positions = positions.reshape(-1, 2 * cfg.n_particles)
    expected = jax.eval_shape(model.init, jax.random.PRNGKey(0))
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(params)
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
    if actual_tree != expected_tree or any(
        a.shape != e.shape for a, e in zip(actual_leaves, expected_leaves)
    ):
        raise ValueError(
            "Model architecture in config does not match checkpoint parameter shapes"
        )
    if not np.isfinite(positions).all() or not all(
        np.isfinite(x).all() for x in actual_leaves
    ):
        raise ValueError("Checkpoint contains non-finite positions or parameters")
    return State(cfg, geometry, model, params, positions, width, path, step)


def sample(
    signed, params, positions, geometry, coupling, args, *, seed, width, samples=None
):
    """Frozen Metropolis chains; proposal adaptation occurs only during burn-in."""
    if args.walkers < 2 or args.walkers > len(positions):
        raise ValueError(
            f"--walkers must be between 2 and {len(positions)} for this checkpoint"
        )
    rng = np.random.default_rng(seed)
    initial = positions[rng.choice(len(positions), args.walkers, replace=False)]
    data = networks.WalkerData(
        positions=jnp.asarray(initial),
        interaction_strength=jnp.full((args.walkers,), coupling),
    )
    batch_log = jax.vmap(lambda p, x, lam: signed(p, x, lam)[1], in_axes=(None, 0, 0))
    step = mcmc.make_mcmc_step(
        batch_log, args.walkers, steps=1, lattice=geometry.lattice
    )

    @jax.jit
    def advance(data, key, width, count):
        def move(_, carry):
            data, key, acceptance = carry
            key, subkey = jax.random.split(key)
            data, accepted = step(params, data, subkey, width)
            return data, key, acceptance + accepted

        data, key, acceptance = jax.lax.fori_loop(
            0, count, move, (data, key, jnp.asarray(0.0))
        )
        return data, key, acceptance / count

    key = jax.random.PRNGKey(seed)
    width = float(width) if np.isfinite(width) and width > 0 else 0.1
    for start in range(0, args.burn_in, 20):
        count = min(20, args.burn_in - start)
        data, key, acceptance = advance(data, key, width, count)
        if start < args.burn_in // 2:
            width = float(np.clip(width * np.exp(float(acceptance) - 0.5), 1e-4, 10.0))
    result, accepts = [], []
    for _ in range(samples or args.samples):
        data, key, acceptance = advance(data, key, width, args.thin)
        result.append(np.asarray(data.positions))
        accepts.append(float(acceptance))
    result = np.stack(result)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite sampled positions")
    return result, float(np.mean(accepts))


def mean_se(chain_means):
    """Standard error across whole-chain means; time points are not independent."""
    values = np.asarray(chain_means, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Need at least two finite chain estimates")
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(values))


def energy_statistics(values):
    """Mean and population variance, with delete-one-chain jackknife errors."""
    values = np.asarray(values, dtype=float)
    mean, se = mean_se(values.mean(axis=0))
    centered = values - mean
    variance = np.mean(centered**2)
    chains = values.shape[1]
    leave_mean = (centered.sum() - centered.sum(axis=0)) / (
        values.shape[0] * (chains - 1)
    )
    leave_var = (np.square(centered).sum() - np.square(centered).sum(axis=0)) / (
        values.shape[0] * (chains - 1)
    ) - leave_mean**2
    variance_se = np.sqrt(
        (chains - 1) / chains * np.square(leave_var - leave_var.mean()).sum()
    )
    return dict(
        energy_mean=float(mean),
        energy_se=float(se),
        energy_variance=float(variance),
        energy_variance_se=float(variance_se),
    )


def energy_evaluator(state, signed=None):
    cfg = state.cfg
    keys = [
        "interaction_small_length_cutoff",
        "interaction_truncation_limit",
        "barrier_sharpness",
        "interaction_cutoff_radius",
        "interaction_tail_correction",
        "interaction_ewald",
        "interaction_ewald_eta",
        "interaction_ewald_eta_scale",
        "interaction_ewald_real_shells",
        "interaction_ewald_recip_shells",
        "interaction_type",
        "short_range_width",
    ]
    options = {key: cfg[key] for key in keys}
    if cfg.interaction_tail_correction and cfg.interaction_tail_g2_path:
        with np.load(cfg.interaction_tail_g2_path) as table:
            options.update(
                interaction_tail_g2_r=table["r"], interaction_tail_g2_values=table["g2"]
            )
    local = hamiltonian.local_energy(
        signed or state.model.apply,
        cfg.n_particles,
        geometry=state.geometry,
        laplacian_method=cfg.optim.laplacian,
        **options,
    )

    def one(params, position, coupling):
        total, parts = local(
            params,
            jax.random.PRNGKey(0),
            networks.WalkerData(positions=position, interaction_strength=coupling),
        )
        return jnp.stack([total, parts.kinetic, parts.potential])

    return jax.jit(jax.vmap(one, in_axes=(None, 0, None)))


def evaluate_chunks(function, params, positions, coupling, chunk_size):
    flat = positions.reshape(-1, positions.shape[-1])
    pieces = []
    for start in range(0, len(flat), chunk_size):
        part = flat[start : start + chunk_size]
        padded = np.pad(part, ((0, chunk_size - len(part)), (0, 0)), mode="edge")
        pieces.append(
            np.asarray(function(params, jnp.asarray(padded), jnp.asarray(coupling)))[
                : len(part)
            ]
        )
    result = np.concatenate(pieces)
    if not np.isfinite(result).all():
        raise ValueError(
            "Non-finite observable estimates; check sampling and model configuration"
        )
    return result.reshape(*positions.shape[:2], *result.shape[1:])


def periodic_lattice(state):
    lattice = state.geometry.lattice
    if lattice is None:
        raise ValueError("This observable requires periodic geometry")
    return np.asarray(lattice)


def style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 14,
            "axes.labelsize": 18,
            "axes.linewidth": 1.1,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.grid": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    return plt


def save_figure(fig, prefix):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ["pdf", "png"]:
        fig.savefig(str(prefix) + "." + extension, dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def curve_plot(
    table, args, column, ylabel, *, error=None, x=None, xlabel=None, log=False
):
    plt = style()
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    x = x or args.x_axis
    groups = ["lambda"] if x == "n_particles" else ["n_particles", "checkpoint"]
    if x not in ("lambda", "n_particles"):
        groups += ["lambda"]
    for key, group in table.groupby(groups, sort=False):
        group = group.sort_values(x)
        n = int(group.n_particles.iloc[0])
        color = COLORS.get(n, "#0072b2")
        label = (
            rf"$N={n}$"
            if x != "n_particles"
            else rf"$\lambda={group['lambda'].iloc[0]:g}$"
        )
        if x not in ("lambda", "n_particles"):
            label += rf", $\lambda={group['lambda'].iloc[0]:g}$"
        if (
            x != "n_particles"
            and table[table.n_particles == n].checkpoint.nunique() > 1
        ):
            directory = Path(group.checkpoint.iloc[0]).parent
            if directory.name.startswith("lambda_"):
                directory = directory.parent
            label += f" ({directory.name})"
        ax.errorbar(
            group[x],
            group[column],
            yerr=group[error] if error else None,
            fmt="o-",
            color=color,
            ms=4,
            lw=1.5,
            capsize=2,
            label=label,
        )
        if x == "lambda" and "training_coupling" in group:
            for trained, point_color in [(True, "#d62728"), (False, "black")]:
                points = group[group.training_coupling == trained]
                ax.plot(points[x], points[column], "o", ms=4, color=point_color)
    ax.set(xlabel=xlabel or (r"$\lambda$" if x == "lambda" else r"$N$"), ylabel=ylabel)
    if log and np.all(table[column] > 0):
        ax.set_yscale("log")
    ax.legend(fontsize=11)
    save_figure(fig, args.output)


def map_groups(table, args):
    groups = list(table.groupby(["checkpoint", "n_particles", "lambda"], sort=False))
    for i, (key, group) in enumerate(groups):
        suffix = "" if len(groups) == 1 else f"_{i + 1}_N{key[1]}_lambda{key[2]:g}"
        yield group, Path(str(args.output) + suffix)


def run(name, description, calculate, plot, *, configure=None, argv=None):
    parser = parser_for(name, description)
    if configure:
        configure(parser)
    args = parser.parse_args(argv)
    try:
        if not np.isfinite(args.couplings).all() or min(args.couplings) < 0:
            raise ValueError("Couplings must be finite and nonnegative")
        if len(args.config) not in (1, len(args.checkpoint)):
            raise ValueError("Supply one --config, or one per --checkpoint")
        frames = []
        for i, checkpoint in enumerate(args.checkpoint):
            cfg_path = args.config[0 if len(args.config) == 1 else i]
            cfg = read_config(cfg_path, args.couplings[0])
            state = load_state(checkpoint, cfg)
            for j, coupling in enumerate(args.couplings):
                print(
                    f"{name}: {state.path}, N={cfg.n_particles}, lambda={coupling:g}",
                    flush=True,
                )
                positions, acceptance = sample(
                    state.model.apply,
                    state.params,
                    state.positions,
                    state.geometry,
                    coupling,
                    args,
                    seed=args.seed + i * 100003 + j * 101,
                    width=state.width,
                )
                frame = calculate(state, positions, coupling, args)
                frame["n_particles"] = cfg.n_particles
                frame["lambda"] = coupling
                frame["checkpoint"] = str(state.path)
                frame["checkpoint_step"] = state.step
                frame["config"] = str(cfg_path)
                frame["chains"] = args.walkers
                frame["samples_per_chain"] = args.samples
                frame["acceptance"] = acceptance
                frame["seed"] = args.seed + i * 100003 + j * 101
                frame["burn_in"] = args.burn_in
                frame["thin"] = args.thin
                frame["training_coupling"] = any(
                    np.isclose(coupling, x)
                    for x in (
                        args.training_couplings
                        or cfg.interaction_strength_training_set
                        or []
                    )
                )
                frames.append(frame)
        table = pd.concat(frames, ignore_index=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(str(args.output) + ".csv", index=False)
        plot(table, args)
        print(f"Saved {args.output}.csv and standalone PDF/PNG plots", flush=True)
        return table
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
