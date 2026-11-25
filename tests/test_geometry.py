"""Disk centers, square/hexagonal tori and geometry validation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from network import config, hamiltonian, networks, psiformer
from network.geometry import Geometry
from test_paper_workflow import _script_config, _short_run, _check_run


@pytest.mark.parametrize("kind", ["square", "hexagonal"])
def test_torus_wrapping_and_boson_symmetry(kind):
    geometry = Geometry(
        kind="periodic", lattice=config.make_lattice(kind, 4.0), center=(0.2, -0.3)
    )
    model = psiformer.make_network(
        4,
        geometry,
        products=2,
        num_layers=1,
        num_heads=2,
        heads_dim=4,
        mlp_hidden_dims=(8,),
        predict_logits=True,
        smooth_periodic_jastrow=True,
    )
    params = model.init(jax.random.PRNGKey(4))
    pos = jnp.array([[0.3, 0.2], [1.3, 0.8], [2.1, 1.9], [0.6, 2.7]])
    shifted = pos + jnp.array([[3, -2], [0, 1], [-1, 0], [1, 1]]) @ geometry.lattice.T
    apply = jax.jit(lambda p: model.apply(params, p.reshape(-1), 0.7)[1])
    value = apply(pos)
    np.testing.assert_allclose(apply(shifted), value, atol=3e-5)
    np.testing.assert_allclose(apply(pos[jnp.array([2, 0, 3, 1])]), value, atol=3e-5)
    np.testing.assert_allclose(apply(geometry.wrap(shifted)), value, atol=3e-5)
    frac = (np.asarray(geometry.wrap(shifted)) - geometry.center) @ np.linalg.inv(
        geometry.lattice
    ).T
    assert np.all(frac >= -1e-7) and np.all(frac < 1 + 1e-7)


def test_disk_center_moves_with_coordinates():
    centers = [(0.0, 0.0), (2.0, -1.0)]
    positions = jnp.array([[0.3, 0.2], [1.3, 0.8]])
    values = []
    for center in centers:
        geometry = Geometry(center=center, radius=1.0)
        model = psiformer.make_network(
            2,
            geometry,
            products=2,
            num_layers=1,
            num_heads=2,
            heads_dim=4,
            mlp_hidden_dims=(8,),
        )
        params = model.init(jax.random.PRNGKey(4))
        pos = (positions + jnp.asarray(center)).reshape(-1)
        local_energy = hamiltonian.local_energy(
            model.apply, 2, geometry, interaction_type="short_range"
        )
        data = networks.WalkerData(positions=pos, interaction_strength=jnp.asarray(0.1))
        values.append(
            (model.apply(params, pos, 0.1)[1], local_energy(params, None, data)[0])
        )
    np.testing.assert_allclose(values[0], values[1], rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "periodic"},
        {"kind": "disk", "lattice": np.eye(2)},
        {"kind": "periodic", "lattice": np.zeros((2, 2))},
        {"kind": "periodic", "lattice": np.eye(3)},
        {"radius": 0.0},
        {"center": (np.nan, 0.0)},
    ],
)
def test_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        Geometry(**kwargs)


def test_square_short_range_training(tmp_path):
    cfg = _script_config("dipole-pbc-hex-N25-pbc.py")
    cfg.n_particles = 4
    cfg.lattice = config.make_lattice("square", 4.0)
    cfg.network.products = 2
    cfg.network.psiformer.num_layers = 1
    cfg.network.psiformer.num_heads = 2
    cfg.network.psiformer.heads_dim = 4
    cfg.network.psiformer.mlp_hidden_dims = (8,)
    cfg.interaction_type = "short_range"
    cfg.interaction_ewald = False
    cfg.network.smooth_periodic_jastrow = False
    cfg.short_range_width = 0.5
    cfg.interaction_strength_training_set = [0.1, 0.5]
    cfg.optim.optimizer = "adam"
    _short_run(cfg, tmp_path, 2, small=False)
    original = _check_run(tmp_path, [0, 1], [0.1, 0.5], 4)
    resume = tmp_path / "resumed"
    _short_run(cfg, resume, 3, small=False, restore=tmp_path)
    resumed = _check_run(resume, [2], [0.1, 0.5], 4)
    assert any(
        jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(lambda a, b: np.any(a != b), original, resumed)
        )
    )
    cfg.optim.optimizer = "none"
    cfg.observables.nk0 = True
    cfg.interaction_strength_training_set = None
    cfg.interaction_strength = 0.3
    inference = tmp_path / "inference"
    _short_run(cfg, inference, 1, small=False, restore=resume)
    frozen = _check_run(inference, [0], [0.3], 4, inference=True)
    jax.tree_util.tree_map(np.testing.assert_array_equal, resumed, frozen)
