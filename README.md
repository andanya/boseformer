# BOSEFORMER

Neural quantum states (NQS) for modeling interacting bosons

**Author:** Daniil Antonenko (antonenko.daniel@gmail.com)

Preprint: [D. S. Antonenko, L. Fu, L. I. Glazman, *"Parametric neural-network wavefunctions across a bosonic phase transition"*, arXiv:2611.33150](https://arxiv.org/pdf/2611.33150). 

## Overview

Boseformer is a universal attention-based neural-network architecture for representing the wavefunctions of particles with bosonic quantum statistics, including superfluid and crystal phases. The model is trained to generate the wavefunction as a function of a Hamiltonian parameter across the quantum phase transition between these phases. Once trained with variational Monte Carlo optimization, it can serve as a foundation model describing the phase transition without the need to retrain it for individual parameter values.

The architecture uses a transformer backbone used in Psiformer ([https://github.com/google-deepmind/ferminet](https://github.com/google-deepmind/ferminet)). Product heads account for bosonic quantum statistics (permutation invariance). 

The typical usage workflow and the results for two-dimensional bosons with dipole-dipole ($1/r^3$) repulsion are shown in the figure below:

![Main plot panels](main-plot-panels.png)

## Environment setup

```bash
git clone https://github.com/andanya/bosenet.git
cd bosenet
uv sync --locked
source .venv/bin/activate
```
GPU runs need a CUDA-compatible NVIDIA driver. For CPU-only setup, use `uv sync --locked --no-group cuda`. Run the examples below from the repository root.

## Usage

```bash
python scripts/dipole-pbc-hex-N25-pbc.py  # Hexagonal periodic cell, 1/r³ interaction (N=25).
python scripts/dipole-pbc-hex-N25-pbc-ft2040.py  # Fine-tune at lambda = 20, 25, 30, 35, 40.
python scripts/infer-dipole-pbc-hex-N25-pbc-ft2040.py 30.0  # Inference with the fine-tuned model at lambda = 30.

python scripts/bosef-10e-rep01.py         # Disk, short-range Gaussian interaction (N=10).
```

The `network` package uses these config fields:

| Field | Choices |
| --- | --- |
| `cfg.n_particles` | Number of bosons |
| `cfg.geometry` | `'periodic'` or `'disk'` |
| `cfg.lattice` | `config.make_lattice('hexagonal', L)` or `config.make_lattice('square', L)`; `None` for a disk |
| `cfg.disk_radius`, `cfg.center` | Disk radius and coordinate origin |
| `cfg.interaction_type` | `'inverse_cubic'` or `'short_range'` (Gaussian) |
| `cfg.interaction_ewald` | Exact periodic dipolar summation |
| `cfg.interaction_strength_training_set` | Couplings for conditional training; `None` for one coupling |
| `cfg.interaction_strength` | Coupling for single-lambda training or inference |
| `cfg.network.lambda_conditioning.mode` | `'additive'` (default), `'film'`, or `'none'` for an unconditioned model |
| `cfg.network.products` | Number of generalized-orbital products |

Geometry determines the input features and orbital envelope. The periodic
paper scripts also enable `cfg.network.smooth_periodic_jastrow`; retain it
when restoring their checkpoints. For a disk, disable Ewald and the periodic
Jastrow, and set `cfg.lattice = None`.

The training curves are saved to `train_stats.csv`
The positions of Monte-Carlo walkers are saved to `walkers_last.host0.npy`.

## License

Apache 2.0
