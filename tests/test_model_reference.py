"""Published-model numerical references and compatibility with saved runs."""

import hashlib
import json
from pathlib import Path
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from network import checkpoint, config, hamiltonian, networks, psiformer
from network.geometry import Geometry
from test_paper_workflow import _script_config, _short_run, _check_run, BASE_COUPLINGS

DATA = Path(__file__).with_name("data")
REFERENCE = json.loads((DATA / "model_reference.json").read_text())


@pytest.mark.parametrize("case", REFERENCE)
def test_published_model_values_and_derivatives(case):
    """Compare model values and derivatives against fixed numerical references."""
    kind, head, mode = case.split("-")
    lattice = None if kind == "disk" else config.make_lattice(kind, 4.0)
    geometry = Geometry(
        kind="disk" if lattice is None else "periodic", lattice=lattice, radius=3.0
    )
    model = psiformer.make_network(
        4,
        geometry=geometry,
        products=2,
        num_layers=1,
        num_heads=2,
        heads_dim=4,
        mlp_hidden_dims=(8,),
        use_layer_norm=True,
        predict_logits=True,
        boson_head=head,
        lambda_conditioning_mode=mode,
        smooth_periodic_jastrow=lattice is not None,
    )
    params = model.init(jax.random.PRNGKey(23))
    positions = jnp.array([0.3, 0.2, 1.3, 0.8, 2.1, 1.9, 0.6, 2.7])
    coupling = jnp.asarray(0.7)
    data = networks.WalkerData(positions=positions, interaction_strength=coupling)
    log_amplitude = lambda pos: model.apply(params, pos, coupling)[1]
    energy_fn = hamiltonian.local_energy(
        model.apply,
        4,
        geometry=geometry,
        interaction_type="short_range" if lattice is None else "inverse_cubic",
        interaction_small_length_cutoff=0.0,
        interaction_ewald=lattice is not None,
        interaction_ewald_real_shells=4,
        interaction_ewald_recip_shells=12,
    )
    energy, parts = jax.jit(energy_fn)(params, jax.random.PRNGKey(0), data)
    expected = REFERENCE[case]
    for actual, key in [
        (jax.jit(log_amplitude)(positions), "log_amplitude"),
        (jax.jit(jax.grad(log_amplitude))(positions), "gradient"),
        (energy, "energy"),
        (parts.kinetic, "kinetic"),
        (parts.potential, "potential"),
    ]:
        np.testing.assert_allclose(actual, expected[key], rtol=3e-5, atol=3e-5)
    # Exact initialization matters for checkpoint layout and reproducible runs.
    leaves = jax.tree_util.tree_leaves(params)
    digest = hashlib.sha256(
        b"".join(np.asarray(x).tobytes() for x in leaves)
    ).hexdigest()
    assert digest == expected["parameter_hash"]


def test_original_checkpoint_can_resume_and_run_inference(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    filename = original / "qmcjax_ckpt_000001.npz"
    shutil.copyfile(DATA / "paper_checkpoint_v1.npz", filename)
    _, old_data, old_params, _, _ = checkpoint.restore(str(filename))
    assert set(dict(old_data)) == {"positions", "interaction_strength"}
    cfg = _script_config("dipole-pbc-hex-N25-pbc.py")
    resumed_path = tmp_path / "resumed"
    _short_run(cfg, resumed_path, 3, restore=original)
    new_params = _check_run(resumed_path, [2], BASE_COUPLINGS, 4)
    assert any(
        jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(lambda a, b: np.any(a != b), old_params, new_params)
        )
    )

    infer = _script_config("infer-dipole-pbc-hex-N25-pbc.py", "26.0")
    infer_path = tmp_path / "infer"
    _short_run(infer, infer_path, 1, restore=original)
    frozen = _check_run(infer_path, [0], [26.0], 4, inference=True)
    jax.tree_util.tree_map(np.testing.assert_array_equal, old_params, frozen)
