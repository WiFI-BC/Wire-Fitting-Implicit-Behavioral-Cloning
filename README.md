# WiFI-BC: Wire-Fitting Implicit Behavioral Cloning

Reference implementation for the ICRA submission *Wire-Fitting Implicit Behavioral Cloning*.

Imitation learning forces a choice between task success and inference speed. Explicit policies
are fast but struggle with discontinuous or multimodal action distributions; implicit and
diffusion-based policies model them well but pay for it with an iterative optimization at every
step.

**WiFI-BC** removes the iteration. A *generator* proposes a small set of `N` candidate actions for
the current state, and a *Q-estimator* scores them; the action is the highest-scoring candidate.
Inference is therefore one forward pass through the generator plus `N` parallel evaluations —
no gradient ascent, no denoising chain.

What makes the candidates worth scoring is how they are trained. Borrowing the *structural
maximizability* of wire-fitting interpolation, the training objective biases one candidate toward
the maximum of the score function, so the generator's proposal set is expected to contain the
argmax rather than merely sample near it. The Q-estimator is trained but not architecturally
constrained to peak at the control points, which leaves an escape hatch: when a task needs more
precision than the generator alone provides, the same critic can be used as an energy function and
refined at inference time (DFO or Langevin MCMC), at a cost you choose.

Across seven task settings — synthetic control, simulated manipulation from states and from
pixels, and a real WidowX arm — WiFI-BC is never beaten on success *and* speed at the same time.

## Contents

- [Results](#results)
- [Installation](#installation)
- [Datasets](#datasets)
- [Using WiFI-BC in your own project](#using-wifi-bc-in-your-own-project)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Repository layout](#repository-layout)
- [Citing](#citing)

## Results

Mean ± standard deviation across three training seeds, 100 evaluation episodes each. Inference
cost is the per-step wall-clock time on a single NVIDIA GeForce RTX 5070 Laptop GPU, at a receding
horizon of one step. **Bold** marks the best score per task.

| Method | Particle 16-D<br>Success % | Adroit Pen<br>Return | Franka Kitchen<br>Subtasks (0–4) | Pushing states<br>Success % | Pushing pixels<br>Success % | LIBERO-Goal<br>Success % |
|---|---|---|---|---|---|---|
| BC (MSE) | 3.0 ± 12.6 | 2141 ± 109 | 1.76 ± 0.07 | 98.3 ± 0.5 | 87.0 ± 4.1 | **94.1 ± 1.9** |
| BC (MDN) | — | — | — | **100 ± 0** | 10.0 ± 4.3 | — |
| IBC | **99.0 ± 4.3** | 2586 ± 65 | 3.37 ± 0.01 | **100 ± 0** | **100 ± 0** | 42.0 ± 8.7 |
| DDPM-100 | 67.0 ± 7.2 | 3050 ± 111 | 2.45 ± 0.41 | 99.3 ± 0.6 | 94.0 ± 2.2 | 84.5 ± 3.4 |
| DDIM (5–25 steps) | 71.0 ± 7.1 | **3077 ± 67** | 2.60 ± 0.51 | 99.0 ± 1.7 | 92.7 ± 1.7 | 81.2 ± 4.1 |
| Consistency Policy (1-step) | 0.0 ± 0.0 | 2287 ± 143 | 1.12 ± 0.88 | 4.5 ± 3.5 | 75.0 ± 11.3 | 91.1 ± 1.4 |
| **WiFI-BC (argmax)** | 82.7 ± 3.1 | 2631 ± 110 | 2.28 ± 0.35 | 99.0 ± 1.0 | 94.0 ± 3.0 | 93.9 ± 0.7 |
| **WiFI-BC (+ refinement)** | 84.2 ± 7.2 | — | **3.41 ± 0.19** | **100 ± 0** | 95.7 ± 1.7 | — |

Per-step inference cost (ms), same runs:

| Method | Particle 16-D | Adroit Pen | Franka Kitchen | Pushing states | Pushing pixels | LIBERO-Goal |
|---|---|---|---|---|---|---|
| BC (MSE) | **0.11** | **0.70** | **0.69** | **0.48** | **0.87** | **5.39** |
| IBC | 476.45 | 195.87 | 432.95 | 5.12 | 16.06 | 111.05 |
| DDPM-100 | 41.95 | 58.39 | 67.34 | 57.92 | 43.01 | 58.77 |
| DDIM (5–25 steps) | 6.45 | 4.32 | 4.49 | 4.17 | 3.69 | 14.92 |
| Consistency Policy (1-step) | 0.90 | 1.13 | 1.20 | 1.12 | 1.57 | 5.81 |
| **WiFI-BC (argmax)** | 0.38 | 1.55 | 1.67 | 0.79 | 1.46 | 10.61 |
| **WiFI-BC (+ refinement)** | 204.24 | — | 172.67 | 3.40 | 5.87 | — |

Reading the two tables together is the point. Excluding regression BC — which is fast but fails
outright on the multimodal tasks (3% on Particle 16-D, 10% for the MDN variant on Pushing pixels) —
WiFI-BC's argmax path is the fastest or near-fastest method everywhere while staying within a few
points of the best score. Where refinement is needed it stays competitive: on Franka Kitchen it
takes the best result at 2.5× IBC's speed, and on Pushing states it matches IBC's perfect score.

Two further results in the paper are not reproducible from this repository:

- **Push-T (real WidowX arm).** WiFI-BC reaches 79.9% IoU with argmax and 83.6% with DFO
  refinement, against 73.4% for IBC and 61.0% for BC. Running it needs the physical rig, so the
  robot code is not included here.
- **Point Maze multimodality.** `point_maze_pillar` ships as a configured environment (an obstacle
  sits between the start and the goal, and the scripted expert takes either corridor arbitrarily).
  It backs a qualitative figure rather than a number in the tables.

## Installation

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 or 3.13.

```bash
git clone <repository-url> wifi-bc
cd wifi-bc
uv sync
```

That covers Particle, Adroit Pen, Franka Kitchen and Point Maze. Two tasks need more:

```bash
uv sync --extra pushing      # Simulated Pushing (adds PyBullet and legacy gym)
bash scripts/setup_libero.sh # LIBERO-Goal (clones LIBERO, fetches demos and goal embeddings)
```

The Pushing extras are separate because `pybullet==3.1.6` publishes no wheels for Python 3.12+ and
has to be built from source, which needs `Python.h` — install your distribution's `python3-dev`
package, or let uv use its own managed interpreter, which ships the headers:
`uv run --managed-python ...`.

## Datasets

| Task | Source | Destination |
|---|---|---|
| Particle | [particle.zip](https://storage.googleapis.com/brain-reach-public/ibc_data/particle.zip) (~80 MB) | `datasets/particle/` |
| Pushing (states) | [block_push_states_location.zip](https://storage.googleapis.com/brain-reach-public/ibc_data/block_push_states_location.zip) (~5 MB) | `datasets/block_push/` |
| Pushing (pixels) | [block_push_visual_location.zip](https://storage.googleapis.com/brain-reach-public/ibc_data/block_push_visual_location.zip) | `datasets/block_push/` |
| Adroit Pen, Franka Kitchen | downloaded automatically by Minari on first use | `~/.minari/` |
| LIBERO-Goal | fetched by `scripts/setup_libero.sh` | `third_party/LIBERO/libero/datasets/` |
| Point Maze | generated at training time from a scripted expert | — |

The Particle and Pushing archives come from the [Implicit BC](https://github.com/google-research/ibc)
data release. For example:

```bash
mkdir -p datasets/block_push && cd datasets/block_push
wget https://storage.googleapis.com/brain-reach-public/ibc_data/block_push_states_location.zip
unzip block_push_states_location.zip && rm block_push_states_location.zip && cd ../..
```

## Using WiFI-BC in your own project

A trained policy is a `WiFIBC` object with one method:

```python
from wifi_bc import WiFIBC

policy = WiFIBC.from_checkpoint("checkpoints/pushing", env="pushing")

obs, _ = env.reset()
policy.reset()                      # clears the frame-stack buffer
for _ in range(max_steps):
    action = policy.act(obs)        # np.ndarray -> np.ndarray, in the env's own units
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
```

`from_checkpoint` reads the `control_point_generator.pt`, `q_estimator.pt` and `norm_stats.pt`
that training writes, and takes the network shapes from the named environment's block in
`config/config.json`. To use it outside those environments, pass your own block instead:

```python
policy = WiFIBC.from_checkpoint(
    "path/to/checkpoint",
    env_config={
        "state_dim": 10, "action_dim": 2, "frame_stack": 2,
        "action_bounds": [-1.0, 1.0], "env_id": "MyRobot-v0",
        "model": {"control_points": 20, "cp_width": 256, "cp_depth": 2,
                  "q_width": 128, "q_depth": 8, "q_network_kind": "resnet"},
    },
)
```

Pick the inference variant with `inference_mode`:

```python
WiFIBC.from_checkpoint(path, inference_mode="argmax")    # default — one forward pass, fastest
WiFIBC.from_checkpoint(path, inference_mode="dfo")       # derivative-free refinement of the cloud
WiFIBC.from_checkpoint(path, inference_mode="langevin")  # Langevin MCMC on the critic's energy
```

Frame stacking, observation normalization and the mapping back to the environment's action units
are all handled inside `act`, so pass the raw per-step observation and call `reset()` between
episodes.

## Reproducing the experiments

Every hyperparameter lives in `config/config.json`, which ships with the best-found configuration
for each method on each task — the numbers in the tables above come from these settings. A run is
fully described by the method's script plus `--env`, and nothing else needs editing:

```bash
uv run python -m training.wifi_bc_training --env pushing
uv run python -m envs.evaluate --checkpoint checkpoints --env pushing
```

`envs.evaluate` infers the method from the weight files in the checkpoint directory, so the same
command scores every method and reports the same metrics.

### One command per method × task

| Task | WiFI-BC | IBC | Diffusion Policy | Consistency Policy | BC (MSE) |
|---|---|---|---|---|---|
| Particle 16-D | `--env particle` | `--env particle` | `--env particle` | `--env particle` | `--env particle` |
| Adroit Pen | `--env pen` | `--env pen` | `--env pen` | `--env pen` | `--env pen` |
| Franka Kitchen | `--env kitchen` | `--env kitchen` | `--env kitchen` | `--env kitchen` | `--env kitchen` |
| Pushing (states) | `--env pushing` | — | `--env pushing` | `--env pushing` | `--env pushing` |
| Pushing (pixels) | `--env pushing_pixels` | — | `--env pushing_pixels` | `--env pushing_pixels` | `--env pushing_pixels` |
| LIBERO-Goal | `--env libero_goal_pixels` | `--env libero_goal_pixels` | `--env libero_goal_pixels` | `--env libero_goal_pixels` | `--env libero_goal_pixels` |

with the training scripts

```bash
uv run python -m training.wifi_bc_training             --env <task>
uv run python -m training.ibc_training                 --env <task>
uv run python -m training.diffusion_policy_training    --env <task>
uv run python -m training.consistency_policy_training  --env <task>
uv run python -m training.bc_mse_training              --env <task>
```

A dash marks a combination this repository does not train. The paper still reports IBC on both
Pushing variants, because for each baseline it takes the better of that method's officially
published result and our own reproduction, and for Pushing the published IBC numbers stand; the
inference cost in the second table was measured on this architecture. `ibc_training` therefore has
no Pushing data path.

Every other cell trains from `config/config.json`'s curated hyperparameters for that method and
task. The two dashes in the WiFI-BC results row mean something different: Adroit Pen and
LIBERO-Goal are reported with argmax only, because refinement did not improve them.

Each task reports on its own protocol: success rate for Particle, both Pushing variants and
LIBERO-Goal; episode return for Adroit Pen; subtasks completed for Franka Kitchen. `envs.evaluate`
prints whichever applies, and `--json results.json` writes the full per-seed breakdown.

### Which inference variant a run uses

The `inference_*` keys in a task's `training` block select the variant, so the shipped config
already reproduces the row reported for that task. `inference_langevin_iterations: 0` and
`inference_dfo_iterations: 0` together mean plain argmax. Override them to trade speed against
precision — the Franka Kitchen block is the clearest example of refinement paying for itself.

### Notes

- Training logs to Weights & Biases. Set `WANDB_MODE=offline` (or `disabled`) to skip it.
- `WIFI_BC_CONFIG_PATH` points any entry point at a different config file, so you can sweep
  without editing the shipped one.
- The image-based tasks render with MuJoCo or PyBullet: set `MUJOCO_GL=egl` (or `osmesa`) on a
  headless machine.

## Repository layout

```
wifi_bc/          the algorithm
  models.py         control-point generator, Q estimator, and the pixel variants
  loss.py           InfoNCE, MSE-to-expert, separation and entropy-KDE objectives
  normalizations.py observation normalization and wire-fitting Q-value normalization
  sampling.py       uniform and Langevin MCMC action samplers
  policy.py         WiFIBC — the plug-and-play policy wrapper
  config.py         config loading and per-method resolution

baselines/        the methods WiFI-BC is compared against
  ibc.py            Implicit BC: energy model, Langevin training, DFO/Langevin inference
  diffusion.py      Diffusion Policy (DDPM / DDIM)
  consistency.py    Consistency Policy

envs/             tasks, data and evaluation
  datasets.py             demonstration loaders for every task
  evaluate.py             the single evaluation entry point, for every method
  *_simulation.py         per-task rollout loops
  *_env.py                the environments implemented here
  ibc_block_pushing/      the Pushing simulator, vendored from google-research/ibc

training/         one standalone script per method
scripts/          LIBERO-Goal setup helpers
config/           config.json (hyperparameters) and observation_bounds.json
```

## Citing

See `CITATION.cff`. The paper is under double-blind review, so the authors are withheld.

This repository builds on prior work whose code or data it uses directly: the
[Implicit BC](https://github.com/google-research/ibc) environments and datasets (Apache-2.0, see
`envs/ibc_block_pushing/NOTICE`), the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
benchmark, [D4RL](https://github.com/Farama-Foundation/Minari) via Minari, and
[Diffusion Policy](https://github.com/real-stanford/diffusion_policy).
