"""Observable normalization, checkpoint sampling, and standalone CSV/plot commands."""

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from network import config, smooth_jastrow
from network.geometry import Geometry
from observables import _common, _geometry, _jastrow
from observables import bragg_peak, feynman_bijl_dispersion, g2_map, g2_radial
from observables import structure_factor_radial, zero_momentum_occupation

CHECKPOINT = Path(__file__).parent / "data/paper_checkpoint_v1.npz"
ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SCRIPTS = [
    "g2_map",
    "bragg_peak",
    "zero_momentum_occupation",
    "energy",
    "kinetic_energy",
    "potential_energy",
    "local_energy_variance",
    "energy_relative_to_jastrow",
    "variance_relative_to_jastrow",
    "structure_factor_map",
    "structure_factor_radial",
    "feynman_bijl_dispersion",
    "g2_radial",
]


@pytest.fixture(scope="module")
def small_config():
    cfg = _common.read_config(ROOT / "scripts/dipole-pbc-hex-N25-pbc.py", 0.7)
    cfg.n_particles = 4
    cfg.lattice = config.make_lattice("hexagonal", np.sqrt(8 * np.pi / np.sqrt(3)))
    cfg.network.products = 2
    cfg.network.psiformer.num_layers = 1
    cfg.network.psiformer.num_heads = 2
    cfg.network.psiformer.heads_dim = 4
    cfg.network.psiformer.mlp_hidden_dims = (8,)
    return cfg


@pytest.fixture(scope="module")
def config_path(tmp_path_factory, small_config):
    path = tmp_path_factory.mktemp("observable-config") / "config.json"
    path.write_text(
        json.dumps(small_config.to_dict(), default=lambda value: value.tolist())
    )
    return path


def spatial_state(kind, n=4):
    lattice = config.make_lattice(kind, 4.0)
    return SimpleNamespace(
        cfg=SimpleNamespace(n_particles=n),
        geometry=Geometry(kind="periodic", lattice=lattice),
    )


@pytest.mark.parametrize("kind", ["square", "hexagonal"])
def test_ideal_gas_pair_and_structure_normalizations(kind):
    state = spatial_state(kind)
    lattice = state.geometry.lattice
    rng = np.random.default_rng(182)
    positions = (rng.random((32, 64, 4, 2)) @ lattice.T).reshape(32, 64, 8)
    radial = g2_radial.calculate(
        state, positions, 0.7, SimpleNamespace(bins=8, rmax=None)
    )
    # The narrow innermost annulus has fewer pairs and a larger counting error.
    expected_counts = (
        32
        * 64
        * 6
        * np.pi
        * (radial.r_right**2 - radial.r_left**2)
        / abs(np.linalg.det(lattice))
    )
    assert np.all(np.abs(radial.g2 - 1) < 5 / np.sqrt(expected_counts))
    mapped = g2_map.calculate(state, positions, 0.7, SimpleNamespace(bins=16))
    np.testing.assert_allclose(mapped.bin_area.sum(), abs(np.linalg.det(lattice)))
    integral = (mapped.g2 * mapped.bin_area).sum() / abs(np.linalg.det(lattice))
    np.testing.assert_allclose(integral, 1.0, atol=1e-12)
    image = mapped.g2.to_numpy().reshape(16, 16)
    np.testing.assert_allclose(image, image[::-1, ::-1], atol=1e-12)
    args = SimpleNamespace(kmax=3.0)
    structure = structure_factor_radial.calculate(state, positions, 0.7, args)
    np.testing.assert_allclose(structure.structure_factor, 1.0, atol=0.12)
    dispersion = feynman_bijl_dispersion.calculate(state, positions, 0.7, args)
    np.testing.assert_allclose(dispersion.omega, dispersion.k**2 / 2, rtol=0.14)
    assert np.all(dispersion.k > 0)
    zero = _geometry.structure_values(positions, np.zeros((1, 2)))
    np.testing.assert_allclose(zero, 4.0)


def test_hexagonal_nearest_image_and_structure_invariances():
    lattice = config.make_lattice("hexagonal", 4.0)
    # Unequal fractional coordinates avoid an exactly degenerate nearest image.
    displacement = np.array([[0.49, 0.46]]) @ lattice.T
    nearest = _geometry.nearest_image(displacement, lattice)
    assert np.linalg.norm(nearest) < np.linalg.norm(displacement)
    np.testing.assert_allclose(
        _geometry.nearest_image(
            displacement + 7 * lattice[:, 0] - 3 * lattice[:, 1], lattice
        ),
        nearest,
    )
    positions = np.random.default_rng(12).random((3, 4, 4, 2)) @ lattice.T
    _, vectors = _geometry.reciprocal_vectors(lattice, 4)
    reference = _geometry.structure_values(positions.reshape(3, 4, 8), vectors)
    shifted = positions[:, :, [2, 0, 3, 1], :] + np.array([1.2, -0.4])
    shifted[:, :, 0, :] += lattice[:, 1]
    np.testing.assert_allclose(
        _geometry.structure_values(shifted.reshape(3, 4, 8), vectors),
        reference,
        atol=1e-12,
    )


@pytest.mark.parametrize("kind", ["square", "hexagonal"])
def test_perfect_crystal_bragg_height(kind):
    state = spatial_state(kind)
    fractional = np.array([[0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5]])
    positions = np.tile(
        (fractional @ state.geometry.lattice.T).reshape(1, 1, 8), (3, 6, 1)
    )
    result = bragg_peak.calculate(
        state, positions, 0.7, SimpleNamespace(shell=None, seed=23)
    )
    np.testing.assert_allclose(result.bragg_peak, 4.0)
    np.testing.assert_allclose(result.bragg_peak_se, 0.0, atol=1e-12)


def test_crossfit_uses_other_chains_for_star_selection():
    values = np.array([[100.0, 0], [0, 100.0], [3.0, 4.0], [7.0, 2.0]])
    held, selected = bragg_peak.crossfit(values, 91)
    folds = np.random.default_rng(91).permutation(4) % 2
    for part in (0, 1):
        expected = values[folds != part].mean(axis=0).argmax()
        assert selected[part] == expected
        np.testing.assert_array_equal(
            held[folds == part], values[folds == part, expected]
        )


def test_occupation_ratio_normalization_and_sign():
    lattice = jnp.eye(2)
    constant = lambda params, pos, coupling: (jnp.asarray(1.0), jnp.asarray(0.0))
    evaluate = zero_momentum_occupation.ratio_evaluator(constant, lattice, 1, 4)
    np.testing.assert_allclose(
        evaluate(None, jnp.zeros((16, 2)), 0.7, jax.random.PRNGKey(0)), 1.0
    )

    def nodal(params, pos, coupling):
        value = jnp.cos(2 * jnp.pi * pos[0])
        return jnp.sign(value), jnp.log(jnp.abs(value))

    evaluate = zero_momentum_occupation.ratio_evaluator(nodal, lattice, 1, 512)
    result = evaluate(None, jnp.zeros((64, 2)), 0.7, jax.random.PRNGKey(3))
    # The integral of a cosine orbital is zero; absolute ratios would be positive.
    assert abs(float(result.mean())) < 0.02


def test_chain_errors_do_not_count_repeated_times_as_independent():
    values = np.tile([1.0, 2.0, 3.0, 4.0], (5, 1))
    stats = _common.energy_statistics(values)
    repeated = _common.energy_statistics(np.repeat(values, 20, axis=0))
    assert stats == pytest.approx(repeated)
    assert stats["energy_mean"] == 2.5
    assert stats["energy_variance"] == 1.25
    assert stats["energy_se"] == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0], ddof=1) / 2)


def test_jastrow_reference_matches_model_factor_and_derivatives():
    lattice = config.make_lattice("hexagonal", 4.0)
    positions = jnp.array([0.1, 0.4, 1.2, 0.8, 2.1, 2.3, 0.2, 2.8])
    fraction = 0.4
    baseline = _jastrow.make_wavefunction(lattice, 4)
    original = smooth_jastrow.make_smooth_periodic_log_jastrow(
        lattice, rmatch_frac=fraction
    )
    logj = lambda x: baseline({"fraction": fraction}, x, 0.7)[1]
    reference = lambda x: original(x, 0.7)
    np.testing.assert_allclose(logj(positions), reference(positions), rtol=2e-6)
    np.testing.assert_allclose(
        jax.grad(logj)(positions), jax.grad(reference)(positions), rtol=2e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        jax.hessian(logj)(positions),
        jax.hessian(reference)(positions),
        rtol=2e-4,
        atol=2e-4,
    )


def test_replicated_checkpoint_load_and_config_mismatch(tmp_path, small_config):
    original = _common.load_state(CHECKPOINT, small_config)
    with np.load(CHECKPOINT, allow_pickle=True) as z:
        params = jax.tree_util.tree_map(
            lambda a: np.repeat(a, 2, axis=0), z["params"].item()
        )
        data = {
            key: np.repeat(value, 2, axis=0) for key, value in z["data"].item().items()
        }
        np.savez(
            tmp_path / "qmcjax_ckpt_000002.npz",
            params=params,
            data=data,
            t=2,
            mcmc_width=z["mcmc_width"],
        )
    replicated = _common.load_state(tmp_path, small_config)
    assert len(replicated.positions) == 24 and replicated.step == 2
    jax.tree_util.tree_map(
        np.testing.assert_array_equal, original.params, replicated.params
    )
    wrong = config.resolve(small_config)
    wrong.n_particles = 5
    with pytest.raises(ValueError, match="Particle count"):
        _common.load_state(CHECKPOINT, wrong)


@pytest.mark.parametrize(
    "training_import,call",
    [
        ("from network import train", "train.train(cfg)"),
        ("import network.train as trainer", "trainer.train(cfg)"),
        ("from network.train import train as train_model", "train_model(cfg)"),
    ],
)
def test_config_loading_suppresses_training(
    tmp_path, monkeypatch, training_import, call
):
    from network import train

    def unexpected(*args, **kwargs):
        raise AssertionError("Loading configuration must not run training")

    monkeypatch.setattr(train, "train", unexpected)
    source = tmp_path / "config.py"
    source.write_text(
        f"from network import config\n{training_import}\n"
        f"cfg = config.default()\ncfg.n_particles = 7\nresult = {call}\n"
    )
    assert _common.read_config(source, 0.7).n_particles == 7


@pytest.mark.parametrize("kind", ["disk", "square"])
def test_energy_components_from_configured_checkpoint(kind, tmp_path):
    from network import psiformer
    from observables import energy, kinetic_energy, potential_energy

    cfg = config.default()
    cfg.n_particles = 2
    cfg.geometry = "disk" if kind == "disk" else "periodic"
    cfg.lattice = None if kind == "disk" else config.make_lattice("square", 4.0)
    cfg.interaction_type = "short_range"
    cfg.short_range_width = 0.5
    cfg.network.products = 2
    cfg.network.psiformer.num_layers = 1
    cfg.network.psiformer.num_heads = 2
    cfg.network.psiformer.heads_dim = 4
    cfg.network.psiformer.mlp_hidden_dims = (8,)
    geometry = Geometry.from_config(cfg)
    model = psiformer.make_network(
        2, geometry=geometry, products=2, **cfg.network.psiformer
    )
    params = model.init(jax.random.PRNGKey(23))
    positions = np.random.default_rng(42).uniform(0.1, 2.0, (4, 4)).astype(np.float32)
    path = tmp_path / "qmcjax_ckpt_000000.npz"
    np.savez(path, t=0, params=params, data={"positions": positions}, mcmc_width=0.1)
    state = _common.load_state(path, cfg)
    args = SimpleNamespace(walkers=4, samples=2, burn_in=2, thin=1, chunk_size=4)
    sampled, _ = _common.sample(
        state.model.apply,
        state.params,
        positions,
        geometry,
        0.7,
        args,
        seed=23,
        width=0.1,
    )
    total = energy.calculate(state, sampled, 0.7, args).energy.iloc[0]
    kinetic = kinetic_energy.calculate(state, sampled, 0.7, args).kinetic_energy.iloc[0]
    potential = potential_energy.calculate(
        state, sampled, 0.7, args
    ).potential_energy.iloc[0]
    np.testing.assert_allclose(total, kinetic + potential, rtol=3e-6, atol=3e-6)


def test_precomputed_jastrow_table(tmp_path):
    path = tmp_path / "baseline.csv"
    pd.DataFrame(
        [
            dict(
                N=4,
                **{"lambda": 0.7},
                energy_mean=12.0,
                energy_mean_se=0.2,
                energy_variance=3.0,
                energy_variance_se=0.1,
                rmatch_frac=0.4,
            )
        ]
    ).to_csv(path, index=False)
    result = _jastrow.from_table(path, 4, 0.7)
    assert result["energy_mean"] == 12.0 and result["energy_se"] == 0.2
    with pytest.raises(ValueError, match="Expected one baseline row"):
        _jastrow.from_table(path, 4, 1.0)


@pytest.mark.parametrize("script", CHECKPOINT_SCRIPTS)
def test_checkpoint_commands_write_tables_and_single_plots(
    script, tmp_path, config_path
):
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    args = [
        "--checkpoint",
        str(CHECKPOINT),
        "--config",
        str(config_path),
        "--lambda",
        ".7",
        "--walkers",
        "4",
        "--samples",
        "2",
        "--burn-in",
        "2",
        "--thin",
        "1",
        "--chunk-size",
        "4",
        "--output",
        str(tmp_path / script),
    ]
    if script in ("g2_map", "g2_radial"):
        args += ["--bins", "12"]
    if script == "zero_momentum_occupation":
        args += ["--probes", "2"]
    if script.endswith("relative_to_jastrow"):
        args += ["--baseline-grid", ".2", ".4", ".8", "--baseline-scan-samples", "2"]
    module = importlib.import_module("observables." + script)
    table = module.main(args)
    assert len(table) and table.acceptance.between(0, 1).all()
    assert np.isfinite(table.select_dtypes(include="number").to_numpy()).all()
    for extension in ["csv", "pdf", "png"]:
        assert (tmp_path / f"{script}.{extension}").stat().st_size > 100
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize(
    "name,column,power",
    [("training_convergence", "energy", 1), ("training_variance", "ewvar", 2)],
)
def test_training_history_commands(name, column, power, tmp_path):
    path = tmp_path / "train_stats.csv"
    pd.DataFrame(
        {
            "step": [0, 1, 1, 2],
            "energy": [8.0, 400.0, 6.0, 4.0],
            "ewvar": [16.0, 800.0, 12.0, 8.0],
        }
    ).to_csv(path, index=False)
    module = importlib.import_module("observables." + name)
    table = module.main(
        [
            "--stats",
            str(path),
            "--n-particles",
            "4",
            "--bin-size",
            "1",
            "--output",
            str(tmp_path / name),
        ]
    )
    np.testing.assert_array_equal(table.step, [0.0, 1.0, 2.0])
    values = [8.0, 6.0, 4.0] if column == "energy" else [16.0, 12.0, 8.0]
    np.testing.assert_allclose(table.value, np.array(values) / 4**power)
    for extension in ["csv", "pdf", "png"]:
        assert (tmp_path / f"{name}.{extension}").is_file()
