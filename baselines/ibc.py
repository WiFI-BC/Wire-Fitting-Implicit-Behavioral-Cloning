"""IBC (Implicit Behavioral Cloning) baseline — energy model, Langevin training,
Langevin/DFO inference.

Reference implementation of Florence et al., "Implicit Behavioral Cloning"
(CoRL 2021), as used for the IBC rows of the paper's results tables. A single
`QEstimator` is trained as an energy function E(s, a) with an InfoNCE loss over
one positive (the expert action) and `NUM_COUNTER_EXAMPLES` negatives drawn by a
Langevin MCMC chain, plus the paper's gradient penalty. At inference the action
is recovered by running the same chain from uniform samples and taking the
lowest-energy one.

Train with `training/ibc_training.py`; evaluate with `envs/evaluate.py`, which
routes checkpoints containing only a `q_estimator.pt` here.

Hyperparameters live in `config/config.json` under each environment's `ibc`
block; `DEFAULT_IBC_HPARAMS` below supplies the paper-faithful defaults for any
key an environment does not override.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from wifi_bc.config import resolve_env_config
from wifi_bc.models import QEstimator
from wifi_bc.normalizations import ObservationNormalizer
from wifi_bc.sampling import sample_langevin

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "config.json"

# Envs that share the D4RL standardize-obs / [-1,1]-action / reward-eval path.
# kitchen rides the same pipeline as pen (D4RLDataset, standardize obs, per-dim
# [-1,1] actions) but reports an extra tasks-completed (0..N) metric.
_D4RL_REWARD_ENVS = ("pen", "kitchen")

# Paper-faithful defaults. Per-environment overrides live in config.json.
DEFAULT_IBC_HPARAMS: dict = {
    "TRAINING_STEPS": 100_000,
    "BATCH_SIZE": 512,
    "LEARNING_RATE": 1e-3,
    "LR_DECAY_RATE": 0.99,
    "LR_DECAY_STEPS": 100,
    "NUM_COUNTER_EXAMPLES": 16,
    "LANGEVIN_TRAIN_ITERATIONS": 100,
    "LANGEVIN_STEPSIZE_INIT": 0.1,
    "LANGEVIN_STEPSIZE_FINAL": 1e-5,
    "LANGEVIN_STEPSIZE_POWER": 2.0,
    "LANGEVIN_NOISE_SCALE": 1.0,
    "LANGEVIN_DELTA_ACTION_CLIP": 0.1,
    "GRADIENT_MARGIN": 1.0,
    "SOFTMAX_TEMPERATURE": 1.0,
    "UNIFORM_BOUNDARY_BUFFER": 0.05,
    "HIDDEN_DIMS": [256, 256],
    # Architecture (IBC paper uses spectral_norm=True for D4RL pen-human).
    "Q_USE_SPECTRAL_NORM": False,
    # Stability
    "trial_seed": 0,
    "nan_abort_threshold": 50,
    # Inference Langevin (paper-faithful defaults)
    "INFERENCE_NUM_SAMPLES": 512,
    "INFERENCE_NUM_ITERATIONS": 100,
    "INFERENCE_LR_INIT": 0.1,
    "INFERENCE_LR_FINAL": 1e-5,
    "INFERENCE_DECAY_POWER": 2.0,
    "INFERENCE_DELTA_CLIP": 0.1,
    "INFERENCE_NOISE_SCALE": 0.1,
}

# Particle uses 50 eval seeds (legacy); pen uses 100 (matches IBC Table 2).
_DEFAULT_NUM_EVAL_SEEDS = {"particle": 50, "pen": 100, "kitchen": 100,
                           "libero_goal_pixels": 50}

# Where train_ibc writes its checkpoints when the caller passes no directory.
CHECKPOINTS_BASE = ROOT_DIR / "checkpoints" / "ibc"


def config_path() -> Path:
    """Read WIFI_BC_CONFIG_PATH at call time, so a caller can point every
    training and evaluation entry point at an alternative config."""
    return Path(os.environ.get("WIFI_BC_CONFIG_PATH") or DEFAULT_CONFIG_PATH)


def load_config() -> dict:
    with open(config_path()) as f:
        return json.load(f)


def compute_dataset_stats(dataset):
    acts = dataset.actions
    return {
        "act_min": acts.min(axis=0).astype(np.float32),
        "act_max": acts.max(axis=0).astype(np.float32),
    }


def inference_config(hparams: dict) -> dict:
    """The Langevin settings `evaluate_ibc_checkpoint` runs inference with.

    Reads the INFERENCE_* hyperparameters (paper-faithful defaults, overridden
    per environment in config.json) into the argument names the sampler uses.
    The `noise_via_stepsize` and `optimize_again` extras mirror the official IBC
    implementation: noise scaling linearly with the stepsize instead of with its
    square root, and a second chain at a constant tiny stepsize, which D4RL's
    published config enables.
    """
    hp = {**DEFAULT_IBC_HPARAMS, **(hparams or {})}
    return {
        "num_samples": int(hp["INFERENCE_NUM_SAMPLES"]),
        "num_iterations": int(hp["INFERENCE_NUM_ITERATIONS"]),
        "lr_init": float(hp["INFERENCE_LR_INIT"]),
        "lr_final": float(hp["INFERENCE_LR_FINAL"]),
        "polynomial_decay_power": float(hp["INFERENCE_DECAY_POWER"]),
        "delta_action_clip": float(hp["INFERENCE_DELTA_CLIP"]),
        "noise_scale": float(hp["INFERENCE_NOISE_SCALE"]),
        "noise_via_stepsize": bool(hp.get("INFERENCE_NOISE_VIA_STEPSIZE", False)),
        "optimize_again": bool(hp.get("INFERENCE_OPTIMIZE_AGAIN", False)),
        "again_stepsize_init": float(hp.get("INFERENCE_AGAIN_STEPSIZE_INIT", 1e-5)),
        "again_stepsize_final": float(hp.get("INFERENCE_AGAIN_STEPSIZE_FINAL", 1e-5)),
        "again_noise_scale": float(hp.get("INFERENCE_AGAIN_NOISE_SCALE", 0.5)),
    }


def normalize_tensor(x, x_min, x_max, device):
    rng = x_max - x_min
    rng = np.where(rng == 0, np.ones_like(rng), rng)
    return (x - torch.from_numpy(x_min).float().to(device)) / torch.from_numpy(rng).float().to(device)


def _finite(x: float) -> float | None:
    """JSON-safe: inf/nan → None so trials.jsonl stays strictly valid JSON."""
    xf = float(x)
    return xf if np.isfinite(xf) else None


# ─── Training-time Langevin counter-example sampler ──────────────────────────


def langevin_counter_examples(
    energy_model,
    obs_norm,
    device,
    hparams,
    action_dim,
    act_range_lo: float = 0.0,
    act_range_hi: float = 1.0,
    energy_fn=None,
):
    """Sample IBC paper-style Langevin counter-examples.

    `act_range_lo` / `act_range_hi` are the action-box bounds in the SAME
    space the model is trained on. Particle uses [0, 1] (dataset actions are
    pre-normalized to [0, 1]), pen uses [-1, 1] (D4RLDataset returns actions
    already in [-1, 1] via per-dim min-max). The UNIFORM_BOUNDARY_BUFFER
    hparam pads both sides identically.
    """
    # energy_fn(obs_expanded, actions) -> (B, N): override for pixel EBMs so
    # the chain runs against precomputed features (late fusion) instead of
    # re-running the conv encoder every iteration (mandatory for pixel envs —
    # see wifi_bc_training's wall-time postmortem).
    if energy_fn is None:
        energy_fn = lambda o, a: energy_model(o, a).squeeze(-1)  # noqa: E731
    B = obs_norm.shape[0]
    act_min = act_range_lo - hparams["UNIFORM_BOUNDARY_BUFFER"]
    act_max = act_range_hi + hparams["UNIFORM_BOUNDARY_BUFFER"]
    n_counter = hparams["NUM_COUNTER_EXAMPLES"]
    actions = torch.rand(B, n_counter, action_dim, device=device) * (act_max - act_min) + act_min
    delta_clip = hparams["LANGEVIN_DELTA_ACTION_CLIP"] * 0.5 * (act_max - act_min)
    obs_expanded = obs_norm.unsqueeze(1).expand(-1, n_counter, -1)

    for p in energy_model.parameters():
        p.requires_grad_(False)

    for k in range(hparams["LANGEVIN_TRAIN_ITERATIONS"]):
        frac = 1.0 - k / max(hparams["LANGEVIN_TRAIN_ITERATIONS"] - 1, 1)
        stepsize = (
            hparams["LANGEVIN_STEPSIZE_FINAL"]
            + (hparams["LANGEVIN_STEPSIZE_INIT"] - hparams["LANGEVIN_STEPSIZE_FINAL"])
            * (frac ** hparams["LANGEVIN_STEPSIZE_POWER"])
        )
        actions = actions.detach().requires_grad_(True)
        energies = energy_fn(obs_expanded, actions)
        grad = torch.autograd.grad(energies.sum(), actions)[0].detach()
        noise = torch.randn_like(actions) * hparams["LANGEVIN_NOISE_SCALE"]
        delta = stepsize * (0.5 * grad + noise)
        delta = torch.clamp(delta, -delta_clip, delta_clip)
        actions = torch.clamp(actions.detach() - delta, act_min, act_max).detach()

    for p in energy_model.parameters():
        p.requires_grad_(True)
    return actions.detach()


# ─── Training ─────────────────────────────────────────────────────────────────


def train_ibc(hparams: dict, active_env: str = "particle",
              save_dir: str | Path | None = None) -> dict:
    """Train an IBC energy model and return a metadata dict.

    `save_dir` receives `q_estimator.pt` (weights + the normalization and
    architecture fields `envs/evaluate.py` needs to rebuild the model) and
    `hparams.json`; it defaults to `checkpoints/ibc/<active_env>`.

    Env branches:
      - particle: ParticleDataset, minmax obs normalization with JSON bounds,
        actions normalized per-dim to [0, 1] inside the train loop.
      - pen/kitchen: D4RLDataset, standardize obs normalization from dataset
        stats (paper-faithful), actions already in [-1, 1] from D4RLDataset.
      - libero_goal_pixels: PixelQEstimator over the two camera streams.
    """
    # Deterministic seeding — same trial_seed ⇒ same training trajectory.
    seed = int(hparams.get("trial_seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = load_config()
    env_cfg = resolve_env_config(cfg, active_env, "ibc")
    frame_stack = env_cfg.get("frame_stack", 1)
    action_dim = env_cfg["action_dim"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}, trial_seed={seed}, active_env={active_env}")
    print(f"  action_dim={action_dim}, frame_stack={frame_stack}")
    print(f"  Steps={hparams['TRAINING_STEPS']}, LR={hparams['LEARNING_RATE']}, "
          f"Temp={hparams['SOFTMAX_TEMPERATURE']}")
    print(f"  Counter-examples={hparams['NUM_COUNTER_EXAMPLES']}, "
          f"Langevin iters={hparams['LANGEVIN_TRAIN_ITERATIONS']}")
    print(f"  Model: {hparams['HIDDEN_DIMS']}, SN={hparams.get('Q_USE_SPECTRAL_NORM', False)}, "
          f"Grad margin={hparams['GRADIENT_MARGIN']}")

    if active_env == "particle":
        from envs.datasets import ParticleDataset
        n_dim = env_cfg.get("n_dim", 2)
        dataset = ParticleDataset(
            env_cfg["data_dir"], n_dim=n_dim, frame_stack=frame_stack,
        )
        # Particle keeps the legacy [0, 1] in-model action range; per-batch
        # normalize_tensor maps raw dataset actions into [0, 1].
        action_in_model_range = (0.0, 1.0)
        per_batch_action_norm = True
        norm_stats = compute_dataset_stats(dataset)
        obs_normalizer = ObservationNormalizer(
            env_id=env_cfg["env_id"], device=device,
            frame_stack=frame_stack, particle_n_dim=n_dim,
        )
    elif active_env in _D4RL_REWARD_ENVS:
        from envs.datasets import D4RLDataset
        # KITCHEN_QPOS_ONLY reproduces the IBC paper's kitchen input content:
        # legacy d4rl obs = qpos only (+ constant goal); the gymnasium port
        # adds 29 velocity dims. [0:9]=robot qpos, [18:39]=object qpos.
        obs_indices = None
        if active_env == "kitchen" and bool(hparams.get("KITCHEN_QPOS_ONLY", False)):
            obs_indices = list(range(0, 9)) + list(range(18, 39))
            print(f"  KITCHEN_QPOS_ONLY: obs -> {len(obs_indices)}-D (qpos only)")
        dataset = D4RLDataset(
            env_cfg["dataset_name"],
            download=True,
            frame_stack=frame_stack,
            normalize_actions=True,
            action_norm_range=(-1.0, 1.0),
            obs_indices=obs_indices,
        )
        # D4RLDataset returns actions already per-dim min-max normalized to
        # [-1, 1] (IBC paper App. B.3). Skip per-batch action normalize.
        action_in_model_range = (-1.0, 1.0)
        per_batch_action_norm = False
        # Obs normalization divisor. IBC's D4RL best.gin sets
        # `compute_dataset_statistics.use_sqrt_std = True`: obs are normalized as
        # (x - mean) / sqrt(std), NOT the textbook (x - mean) / std. This damps
        # the whitening (normalized dims keep std = std**0.5) and is what the
        # paper actually trained on. Default True to match IBC; set
        # USE_SQRT_STD=false to recover plain standardize.
        use_sqrt_std = bool(hparams.get("USE_SQRT_STD", True))
        obs_std_divisor = (
            np.sqrt(dataset.obs_std) if use_sqrt_std else dataset.obs_std
        ).astype(np.float32)
        # Norm stats include obs mean/divisor AND raw act min/max so eval can
        # both standardize obs and denormalize the model's [-1, 1] action
        # back to the env's per-dim native range.
        norm_stats = {
            "obs_mean": dataset.obs_mean.astype(np.float32),
            "obs_std": obs_std_divisor,
            "act_min": dataset.act_min.astype(np.float32),
            "act_max": dataset.act_max.astype(np.float32),
            "action_norm_range": (-1.0, 1.0),
            "frame_stack": frame_stack,
            "env_id": env_cfg["env_id"],
            "use_sqrt_std": use_sqrt_std,
            "obs_indices": obs_indices,
        }
        obs_normalizer = ObservationNormalizer(
            env_id=env_cfg["env_id"], device=device,
            frame_stack=frame_stack,
            obs_mean=dataset.obs_mean,
            obs_std=obs_std_divisor,
        )
    elif active_env == "libero_goal_pixels":
        from envs.datasets import LiberoGoalPixelsDataset
        dataset = LiberoGoalPixelsDataset(
            goal_embeddings_path=env_cfg["goal_embeddings_path"],
            frame_stack=frame_stack,
            max_demos_per_task=env_cfg.get("max_demos_per_task"),
            crop_size=int(hparams.get("IMAGE_CROP", 0)),
            cameras=str(hparams.get("LIBERO_CAMERAS", env_cfg.get("libero_cameras", "agentview+wrist"))),
            use_proprio=bool(hparams.get("LIBERO_USE_PROPRIO", env_cfg.get("libero_use_proprio", True))),
        )
        action_in_model_range = (-1.0, 1.0)
        per_batch_action_norm = False
        norm_stats = {
            "act_min": dataset.act_min.astype(np.float32),
            "act_max": dataset.act_max.astype(np.float32),
            "action_norm_range": (-1.0, 1.0),
            "frame_stack": frame_stack,
            "env_id": env_cfg["env_id"],
            "libero_obs_keys": dataset.libero_obs_keys,
            "goal_embeddings": dataset.goal_embeddings,
            "goal_task_names": dataset.goal_task_names,
            "goal_emb_dim": dataset.goal_emb_dim,
            "proprio_dim": dataset.proprio_dim,
            "cond_dim": dataset.cond_dim,
            "in_channels": dataset.in_channels,
            "state_shape": list(dataset.state_shape),
            "libero_cameras": list(dataset.cameras),
            "image_crop_size": int(hparams.get("IMAGE_CROP", 0)),
        }
        # Conv encoder preprocesses images itself; conditioning fed raw.
        obs_normalizer = None
    else:
        raise ValueError(f"Unsupported active_env for DFO: {active_env}")

    print(f"  Dataset size: {len(dataset)}")

    obs_dim = dataset.state_shape
    act_dim = dataset.action_shape
    # IBC's D4RL best.gin builds the EBM as `MLPEBM.layers='ResNetPreActivation'`
    # (depth 8, width 512, spectral_norm) — NOT a plain MLP. A deep plain MLP
    # with spectral norm trains poorly (no skip connections → vanishing grads),
    # which is why our prior IBC reproduction had a near-random energy surface
    # (NCE ~= ln(K)). Default to resnet to match the paper; width/depth derive
    # from HIDDEN_DIMS (all-equal widths). Set NETWORK_KIND='mlp' to revert.
    network_kind = str(hparams.get("NETWORK_KIND", "resnet"))
    hd = hparams["HIDDEN_DIMS"]
    q_width = int(hd[0])
    if network_kind == "resnet":
        # IBC's `depth` counts DENSE layers; its ResNetPreActivationLayer pairs
        # them into residual units of 2. Our ResNetPreActivationBlock IS one
        # 2-layer unit, so #blocks = dense_layers / 2. HIDDEN_DIMS=[512]*8
        # (= IBC depth 8) -> 4 blocks = 8 dense layers, matching the paper.
        q_depth = max(1, len(hd) // 2)
    else:
        q_depth = len(hd)
    # Official MLPEBM projects to the energy straight after the last resnet
    # block (no trailing activation). Default False = IBC-faithful.
    resnet_final_act = bool(hparams.get("RESNET_FINAL_ACTIVATION", False))
    is_pixel = (active_env == "libero_goal_pixels")
    if is_pixel:
        # Official IBC pixel EBM (pixel_ebm_langevin.gin): ConvMaxpoolEncoder
        # + DenseResnetValue(width=VALUE_WIDTH, blocks=VALUE_NUM_BLOCKS), plus
        # our proprio+goal conditioning concat (cond_dim).
        from wifi_bc.models import PixelQEstimator
        value_width = int(hparams.get("VALUE_WIDTH", 1024))
        value_blocks = int(hparams.get("VALUE_NUM_BLOCKS", 1))
        # Encoder is selectable so IBC can be run on the SAME visual backbone as
        # WiFI-BC / the diffusion proposal. It used to be hardcoded to conv_maxpool, which meant every
        # libero IBC number was produced on a weaker encoder than the algorithms
        # it was being compared against. Defaults preserve the old behaviour
        # exactly, so existing trials remain reproducible.
        enc_kind = str(hparams.get("ENCODER_KIND", "conv_maxpool"))
        enc_pretrained = hparams.get("ENCODER_PRETRAINED", False)
        if isinstance(enc_pretrained, str):
            enc_pretrained = enc_pretrained.lower() in ("imagenet", "true", "1")
        enc_num_kp = int(hparams.get("ENCODER_NUM_KP", 64))
        enc_norm_kind = str(hparams.get("ENCODER_NORM_KIND", "bn"))
        enc_per_camera = bool(hparams.get("ENCODER_PER_CAMERA", False))
        cond_fusion = str(hparams.get("COND_FUSION", "concat"))
        goal_dim = int(getattr(dataset, "goal_emb_dim", 0))
        print(f"  EBM backbone: PIXEL {enc_kind} (pretrained={enc_pretrained}, "
              f"kp={enc_num_kp}, norm={enc_norm_kind}) + DenseResnetValue("
              f"w={value_width}, blocks={value_blocks}) cond={dataset.cond_dim} "
              f"fusion={cond_fusion}")
        energy_model = PixelQEstimator(
            action_dim=dataset.action_shape,
            in_channels=dataset.in_channels,
            encoder_target_height=int(env_cfg.get("encoder_target_height", 128)),
            encoder_target_width=int(env_cfg.get("encoder_target_width", 128)),
            value_width=value_width,
            value_num_blocks=value_blocks,
            cond_dim=dataset.cond_dim,
            encoder_kind=enc_kind,
            encoder_pretrained=enc_pretrained,
            encoder_num_kp=enc_num_kp,
            encoder_norm_kind=enc_norm_kind,
            encoder_per_camera=enc_per_camera,
            cond_fusion=cond_fusion,
            goal_dim=goal_dim,
        ).to(device)
        norm_stats["value_width"] = value_width
        norm_stats["value_num_blocks"] = value_blocks
        # Persist the ACTUAL encoder so eval rebuilds the same tree.
        norm_stats["encoder_kind"] = enc_kind
        norm_stats["encoder_pretrained"] = bool(enc_pretrained)
        norm_stats["encoder_num_kp"] = enc_num_kp
        norm_stats["encoder_norm_kind"] = enc_norm_kind
        norm_stats["encoder_per_camera"] = enc_per_camera
        norm_stats["cond_fusion"] = cond_fusion
        norm_stats["goal_emb_dim"] = goal_dim
    if not is_pixel:
        print(f"  EBM backbone: {network_kind} (width={q_width}, depth={q_depth}, "
              f"final_act={resnet_final_act})")
    energy_model = energy_model if is_pixel else QEstimator(
        state_dim=obs_dim,
        action_dim=act_dim,
        hidden_dims=hd,
        use_spectral_norm=bool(hparams.get("Q_USE_SPECTRAL_NORM", False)),
        network_kind=network_kind,
        width=q_width,
        depth=q_depth,
        resnet_final_activation=resnet_final_act,
    ).to(device)

    optimizer = torch.optim.Adam(energy_model.parameters(), lr=hparams["LEARNING_RATE"])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=hparams["BATCH_SIZE"], shuffle=True, drop_last=True,
    )

    current_lr = hparams["LEARNING_RATE"]
    nan_abort_threshold = int(hparams.get("nan_abort_threshold", 50))
    consecutive_nan_batches = 0

    start_time = time.time()
    step = 0
    log_interval = 500

    last_loss = last_nce = last_gp = last_acc = None

    while step < hparams["TRAINING_STEPS"]:
        for batch in dataloader:
            if step >= hparams["TRAINING_STEPS"]:
                break

            states = batch["state"].float().to(device)
            actions = batch["action"].float().to(device)
            B = states.shape[0]

            if is_pixel:
                # Conv encoder does its own preprocessing; conditioning
                # (proprio + goal) rides the model's _cond attribute.
                energy_model._cond = batch["cond"].float().to(device)
                states_norm = states
            else:
                states_norm = obs_normalizer.normalize(states)
            if per_batch_action_norm:
                actions_norm = normalize_tensor(
                    actions, norm_stats["act_min"], norm_stats["act_max"], device,
                )
            else:
                # D4RLDataset pre-normalized actions to action_in_model_range.
                actions_norm = actions

            if is_pixel:
                # Late fusion: encode ONCE, chain runs on the value head only.
                with torch.no_grad():
                    lv_feats = energy_model.encode(states_norm)
                counter_actions = langevin_counter_examples(
                    energy_model,
                    lv_feats,  # obs arg only supplies batch size to the sampler
                    device, hparams, act_dim,
                    act_range_lo=action_in_model_range[0],
                    act_range_hi=action_in_model_range[1],
                    energy_fn=lambda o, a: energy_model.score(lv_feats, a).squeeze(-1),
                )
            else:
                counter_actions = langevin_counter_examples(
                    energy_model, states_norm, device, hparams, act_dim,
                    act_range_lo=action_in_model_range[0],
                    act_range_hi=action_in_model_range[1],
                )

            n_counter = hparams["NUM_COUNTER_EXAMPLES"]
            all_actions = torch.cat([counter_actions, actions_norm.unsqueeze(1)], dim=1)
            if is_pixel:
                # Encoder trains through InfoNCE: encode WITH grad, late-fuse.
                nce_feats = energy_model.encode(states_norm)
                energies = energy_model.score(nce_feats, all_actions).squeeze(-1)
            else:
                states_expanded = states_norm.unsqueeze(1).expand(-1, n_counter + 1, -1)
                energies = energy_model(states_expanded, all_actions).squeeze(-1)
            logits = -energies / hparams["SOFTMAX_TEMPERATURE"]
            log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
            loss_infonce = -log_probs[:, -1].mean()

            if is_pixel:
                gp_actions = all_actions.detach().requires_grad_(True)
                gp_energies = energy_model.score(nce_feats.detach(), gp_actions)
            else:
                gp_actions = all_actions.detach().reshape(B * (n_counter + 1), -1).requires_grad_(True)
                gp_states = states_expanded.detach().reshape(B * (n_counter + 1), -1)
                gp_energies = energy_model(gp_states, gp_actions)
            grad_gp = torch.autograd.grad(gp_energies.sum(), gp_actions, create_graph=True)[0]
            grad_norms = grad_gp.abs().max(dim=-1).values
            grad_penalty = torch.clamp(grad_norms - hparams["GRADIENT_MARGIN"], min=0).pow(2).mean()

            loss = loss_infonce + grad_penalty
            if torch.isnan(loss):
                consecutive_nan_batches += 1
                # Wipe Adam moments so a single NaN batch doesn't poison subsequent steps.
                optimizer.state.clear()
                if consecutive_nan_batches >= nan_abort_threshold:
                    raise RuntimeError(
                        f"Training diverged: {consecutive_nan_batches} consecutive NaN batches"
                    )
                continue
            consecutive_nan_batches = 0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                best_idx = logits.argmax(dim=1)
                accuracy = (best_idx == n_counter).float().mean().item()

            last_loss = loss.item()
            last_nce = loss_infonce.item()
            last_gp = grad_penalty.item()
            last_acc = accuracy
            step += 1

            if step % hparams["LR_DECAY_STEPS"] == 0:
                current_lr *= hparams["LR_DECAY_RATE"]
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

            if step % log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"  Step {step}/{hparams['TRAINING_STEPS']} | "
                    f"Loss: {loss.item():.4f} (NCE: {loss_infonce.item():.4f}, GP: {grad_penalty.item():.4f}) | "
                    f"Acc: {accuracy:.3f} | LR: {current_lr:.2e} | {elapsed:.1f}s",
                    flush=True,
                )

    total_time = time.time() - start_time
    print(f"  Training completed in {total_time:.1f}s ({total_time / 60:.2f} min)")

    save_dir = Path(save_dir) if save_dir is not None else CHECKPOINTS_BASE / active_env
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "q_estimator.pt"
    torch.save({
        "model_state_dict": energy_model.state_dict(),
        "norm_stats": norm_stats,
        "step": hparams["TRAINING_STEPS"],
        "hparams": hparams,
        "active_env": active_env,
        "action_in_model_range": action_in_model_range,
        # Backbone spec so eval rebuilds the EXACT architecture (resnet keys
        # can't be inferred from a flat bias list like the MLP path).
        "network_kind": network_kind,
        "q_width": q_width,
        "q_depth": q_depth,
        "resnet_final_activation": resnet_final_act,
        "pixel": is_pixel,
    }, ckpt_path)
    # Persist the exact hparams next to the checkpoint for traceability.
    with open(save_dir / "hparams.json", "w") as f:
        json.dump(hparams, f, indent=2, default=str)
    print(f"  Model saved to {ckpt_path}")

    return {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_dir": str(save_dir),
        "duration_seconds": total_time,
        "final_train_loss": last_loss,
        "final_infonce": last_nce,
        "final_grad_penalty": last_gp,
        "final_accuracy": last_acc,
    }


# ─── Evaluation ──────────────────────────────────────────────────────────────


def evaluate_ibc_checkpoint(
    ckpt_path: str,
    langevin_cfg: dict | None = None,
    num_seeds: int | None = None,
    active_env: str | None = None,
) -> dict:
    """Evaluate a DFO checkpoint with paper-faithful Langevin inference.

    `active_env` is taken from the checkpoint metadata when not passed
    explicitly. Particle returns distance-based metrics; the reward-driven envs
    (pen, kitchen) return success_rate, avg/std/median reward and episode
    length — the same keys `envs.evaluate.evaluate` reports for every other
    method, so the two are directly comparable.
    """
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    norm_stats = ckpt.get("norm_stats") if isinstance(ckpt, dict) else None
    if active_env is None:
        active_env = ckpt.get("active_env", "particle") if isinstance(ckpt, dict) else "particle"
    action_in_model_range = (
        tuple(ckpt.get("action_in_model_range", (0.0, 1.0)))
        if isinstance(ckpt, dict)
        else (0.0, 1.0)
    )

    env_cfg = resolve_env_config(cfg, active_env, "ibc")
    # Default to the environment's own inference settings: the checkpoint's own
    # hparams when it recorded them, else the config's block.
    if langevin_cfg is None:
        ckpt_hparams = ckpt.get("hparams") if isinstance(ckpt, dict) else None
        langevin_cfg = inference_config(ckpt_hparams or env_cfg.get("hparams", {}))
    action_dim = int(env_cfg["action_dim"])
    frame_stack = int(env_cfg.get("frame_stack", 1))
    action_bounds = tuple(env_cfg.get("action_bounds", [0, 1]))

    # ── Pixel EBM branch (libero_goal_pixels) ─────────────────────────────
    # Rebuild the PixelQEstimator from the checkpoint's own arch fields and run
    # the render-eval IBC sim (Langevin on encoded features, grouped by task).
    # Returns early — everything below assumes a flat-state QEstimator.
    if (isinstance(ckpt, dict) and ckpt.get("pixel")) or active_env == "libero_goal_pixels":
        from wifi_bc.models import PixelQEstimator
        from envs.libero_goal_pixels_simulation import LiberoGoalPixelsIBCSimulation

        if norm_stats is None or "cond_dim" not in norm_stats:
            raise RuntimeError("pixel DFO eval needs norm_stats with the libero pixel schema.")
        # Encoder fields come from norm_stats (what training actually built).
        # Older checkpoints predate them and fall back to the conv_maxpool
        # defaults, so they still rebuild correctly. pretrained is forced False:
        # the weights come from the state_dict, and a compute node may have no
        # network to fetch ImageNet with.
        model = PixelQEstimator(
            action_dim=action_dim,
            in_channels=int(norm_stats["in_channels"]),
            encoder_target_height=int(env_cfg.get("encoder_target_height", 128)),
            encoder_target_width=int(env_cfg.get("encoder_target_width", 128)),
            value_width=int(norm_stats.get("value_width", 1024)),
            value_num_blocks=int(norm_stats.get("value_num_blocks", 1)),
            cond_dim=int(norm_stats["cond_dim"]),
            encoder_kind=str(norm_stats.get("encoder_kind", "conv_maxpool")),
            encoder_pretrained=False,
            encoder_num_kp=int(norm_stats.get("encoder_num_kp", 64)),
            encoder_norm_kind=str(norm_stats.get("encoder_norm_kind", "bn")),
            encoder_per_camera=bool(norm_stats.get("encoder_per_camera", False)),
            cond_fusion=str(norm_stats.get("cond_fusion", "concat")),
            goal_dim=int(norm_stats.get("goal_emb_dim", 0)),
        )
        model.load_state_dict(sd)
        model.to(device).eval()
        if num_seeds is None:
            num_seeds = int(env_cfg.get("num_eval_seeds", _DEFAULT_NUM_EVAL_SEEDS.get(active_env, 50)))
        sim = LiberoGoalPixelsIBCSimulation(
            energy_net=model, device=device,
            max_episode_steps=int(env_cfg.get("max_episode_steps", 300)),
            frame_stack=frame_stack, norm_stats=norm_stats,
            langevin_cfg=langevin_cfg, num_eval_seeds=num_seeds,
            action_in_model_range=action_in_model_range,
        )
        t0 = time.time()
        succ, rews, eplens, terms = [], [], [], []
        for seed in range(num_seeds):
            r = sim.run_episode(seed=seed)
            succ.append(bool(r["success"]))
            rews.append(float(r["total_reward"]))
            eplens.append(int(r["episode_length"]))
            terms.append(bool(r["terminated"]))
        sim.close()
        return {
            "success_rate": float(np.mean(succ)),
            "success_rate_std": float(np.std(succ)),
            "avg_reward": float(np.mean(rews)),
            "std_reward": float(np.std(rews)),
            "median_reward": float(np.median(rews)),
            "avg_episode_length": float(np.mean(eplens)),
            "num_seeds": int(num_seeds),
            "eval_time_s": time.time() - t0,
        }

    # Backbone spec: prefer the values saved at train time. Older checkpoints
    # (pre-resnet-fix) lack these → default to the plain-MLP reconstruction.
    network_kind = str(ckpt.get("network_kind", "mlp")) if isinstance(ckpt, dict) else "mlp"
    ckpt_q_width = ckpt.get("q_width") if isinstance(ckpt, dict) else None
    ckpt_q_depth = ckpt.get("q_depth") if isinstance(ckpt, dict) else None
    # Pre-flag resnet checkpoints were trained WITH the trailing activation;
    # default True keeps them loadable/correct. New ones save the flag.
    ckpt_final_act = bool(ckpt.get("resnet_final_activation", True)) if isinstance(ckpt, dict) else True

    # Detect spectral norm under either API:
    #   - old (torch.nn.utils.spectral_norm): keys `weight_orig`, `weight_u`, `weight_v`
    #   - new (torch.nn.utils.parametrizations.spectral_norm): `parametrizations.weight.*`
    use_sn = any(
        ("weight_orig" in k) or ("parametrizations.weight" in k)
        for k in sd.keys()
    )

    def _build_eval_model(state_dim_in: int) -> QEstimator:
        """Rebuild the EBM matching the trained backbone.

        resnet: width/depth come from the checkpoint (the flat bias-key
        inference below only works for the plain-MLP Sequential).
        mlp: hidden dims inferred from the `.bias` shapes (output dim of each
        Linear except the final dim-1 energy projection).
        """
        if network_kind == "resnet":
            w = int(ckpt_q_width)
            d = int(ckpt_q_depth)
            return QEstimator(
                state_dim=state_dim_in, action_dim=action_dim,
                hidden_dims=[w] * d, use_spectral_norm=use_sn,
                network_kind="resnet", width=w, depth=d,
                resnet_final_activation=ckpt_final_act,
            )
        bias_indices = sorted(
            {int(k.split(".")[1]) for k in sd
             if k.startswith("network.") and k.endswith(".bias")}
        )
        if not bias_indices:
            raise RuntimeError(
                f"Cannot infer architecture from checkpoint keys: {list(sd.keys())[:5]}..."
            )
        hidden = [int(sd[f"network.{i}.bias"].shape[0]) for i in bias_indices[:-1]]
        return QEstimator(
            state_dim=state_dim_in, action_dim=action_dim,
            hidden_dims=hidden, use_spectral_norm=use_sn,
        )

    if active_env == "particle":
        state_dim = int(env_cfg["state_dim"])
        max_steps = int(cfg.get("simulation", {}).get("max_episode_steps", 50))
        n_dim = int(env_cfg["n_dim"])
        model = _build_eval_model(state_dim * frame_stack)
        model.load_state_dict(sd)
        model.to(device).eval()
        obs_normalizer = ObservationNormalizer(
            env_id=env_cfg["env_id"], device=device,
            frame_stack=frame_stack, particle_n_dim=n_dim,
        )
    elif active_env in _D4RL_REWARD_ENVS:
        max_steps = int(env_cfg.get("max_episode_steps", 100))
        if norm_stats is None or "obs_mean" not in norm_stats:
            raise RuntimeError(
                f"{active_env} DFO eval requires norm_stats with obs_mean/obs_std. "
                "Train with a fresh checkpoint."
            )
        # Input dim from the TRAINED stats, not config state_dim — obs_indices
        # (e.g. KITCHEN_QPOS_ONLY) shrinks the policy input below config's 59.
        state_dim = int(len(np.asarray(norm_stats["obs_mean"]).reshape(-1)))
        model = _build_eval_model(state_dim * frame_stack)
        model.load_state_dict(sd)
        model.to(device).eval()
        # obs_std here is the SAME divisor saved at train time (already
        # sqrt-std when USE_SQRT_STD was on) — eval matches training exactly.
        obs_normalizer = ObservationNormalizer(
            env_id=env_cfg["env_id"], device=device,
            frame_stack=frame_stack,
            obs_mean=np.asarray(norm_stats["obs_mean"], dtype=np.float32),
            obs_std=np.asarray(norm_stats["obs_std"], dtype=np.float32),
        )
    else:
        raise ValueError(f"Unsupported active_env for DFO eval: {active_env}")

    if num_seeds is None:
        num_seeds = int(env_cfg.get("num_eval_seeds", _DEFAULT_NUM_EVAL_SEEDS.get(active_env, 50)))

    buf_sz = 0.05
    lo, hi = float(action_in_model_range[0]), float(action_in_model_range[1])
    norm_min = torch.full((action_dim,), lo - buf_sz, device=device)
    norm_max = torch.full((action_dim,), hi + buf_sz, device=device)

    def denorm(a):
        """Map model-space action in `action_in_model_range` back to env-native.

        Particle: model and env share [0, 1]; norm_stats is a per-dim min-max
        rescale (legacy). Pen: model emits in [-1, 1] and per-dim min-max
        maps to env-native (env actually wraps to [-1, 1] but per-dim min/max
        is tighter — paper's per-dim min-max protocol).
        """
        if norm_stats is None:
            return a
        rng = np.where(
            (norm_stats["act_max"] - norm_stats["act_min"]) == 0, 1.0,
            norm_stats["act_max"] - norm_stats["act_min"],
        )
        if active_env in _D4RL_REWARD_ENVS:
            scale = rng / (hi - lo)
            return norm_stats["act_min"] + (a - lo) * scale
        return a * rng + norm_stats["act_min"]

    # FrankaKitchen-v1 returns a Dict observation; the policy consumes only the
    # proprio/world 'observation' vector (desired_goal is fixed for -complete).
    # obs_indices (saved at train time, e.g. KITCHEN_QPOS_ONLY) selects the
    # same columns the model was trained on.
    _eval_obs_indices = norm_stats.get("obs_indices") if isinstance(norm_stats, dict) else None

    def _obs_vec(o):
        if isinstance(o, dict):
            o = o["observation"]
        v = np.asarray(o, dtype=np.float32)
        if _eval_obs_indices is not None:
            v = v[_eval_obs_indices]
        return v

    seeds = list(range(num_seeds))
    successes: list[bool] = []
    rewards: list[float] = []
    tasks_completed: list[int] = []   # kitchen: subtasks done (0..N), paper metric
    dists_first: list[float] = []
    dists_second: list[float] = []
    ep_lengths: list[int] = []
    terminated_flags: list[bool] = []

    # Lazy env factory keeps the gymnasium-robotics import out of particle runs.
    def _make_env():
        if active_env == "particle":
            from envs.particle_env import ParticleEnv
            return ParticleEnv(n_dim=n_dim, n_steps=max_steps, render_mode=None)
        if active_env == "kitchen":
            # Recover the exact FrankaKitchen-v1 the dataset was recorded with
            # (correct tasks_to_complete + obs layout). Avoids reward_type kwarg
            # mismatch and guarantees the eval task set matches the demos.
            import minari
            ds = minari.load_dataset(env_cfg["dataset_name"], download=True)
            return ds.recover_environment(eval_env=True)
        # pen, door (both AdroitHand-* via gymnasium-robotics)
        import gymnasium as gym
        import gymnasium_robotics
        gym.register_envs(gymnasium_robotics)
        return gym.make(
            env_cfg["env_id"], reward_type="dense",
            max_episode_steps=max_steps, render_mode=None,
        )

    t0 = time.time()
    for seed in seeds:
        env = _make_env()
        obs, _ = env.reset(seed=seed)
        obs = _obs_vec(obs)
        frame_buf = deque(maxlen=frame_stack)
        for _ in range(frame_stack):
            frame_buf.append(obs.copy())

        total_reward = 0.0
        ep_len = 0
        terminated = False
        truncated = False
        info: dict = {}
        any_step_success = False
        ep_tasks_done = 0
        while not (terminated or truncated):
            stacked = np.concatenate(list(frame_buf)) if frame_stack > 1 else frame_buf[-1]
            st = torch.from_numpy(stacked).float().unsqueeze(0).to(device)
            st_n = obs_normalizer.normalize(st)

            samples = sample_langevin(
                energy_function=model, observations=st_n,
                num_samples=int(langevin_cfg["num_samples"]),
                action_min=norm_min, action_max=norm_max,
                num_iterations=int(langevin_cfg["num_iterations"]),
                lr_init=float(langevin_cfg["lr_init"]),
                lr_final=float(langevin_cfg["lr_final"]),
                polynomial_decay_power=float(langevin_cfg.get("polynomial_decay_power", 2.0)),
                delta_action_clip=float(langevin_cfg.get("delta_action_clip", 0.1)),
                noise_scale=float(langevin_cfg["noise_scale"]),
                device=device,
                noise_via_stepsize=bool(langevin_cfg.get("noise_via_stepsize", False)),
            )
            # Official IbcPolicy.optimize_again: a SECOND chain at a tiny
            # constant stepsize (1e-5) polishes the first chain's samples —
            # "a trick for more precise inference" (ibc_policy.py). The D4RL
            # best.gin enables it.
            if bool(langevin_cfg.get("optimize_again", False)) and int(langevin_cfg["num_iterations"]) > 0:
                samples = sample_langevin(
                    energy_function=model, observations=st_n,
                    num_samples=int(langevin_cfg["num_samples"]),
                    action_min=norm_min, action_max=norm_max,
                    num_iterations=int(langevin_cfg["num_iterations"]),
                    lr_init=float(langevin_cfg.get("again_stepsize_init", 1e-5)),
                    lr_final=float(langevin_cfg.get("again_stepsize_final", 1e-5)),
                    polynomial_decay_power=float(langevin_cfg.get("polynomial_decay_power", 2.0)),
                    delta_action_clip=float(langevin_cfg.get("delta_action_clip", 0.1)),
                    noise_scale=float(langevin_cfg.get("again_noise_scale", 0.5)),
                    initial_actions=samples,
                    device=device,
                    noise_via_stepsize=bool(langevin_cfg.get("noise_via_stepsize", False)),
                )
            with torch.no_grad():
                se = st_n.unsqueeze(1).expand(-1, samples.shape[1], -1)
                e = model(se, samples).squeeze(-1)
                best_a = samples[0, e.argmin(dim=-1)[0]].cpu().numpy()
            action = np.clip(denorm(best_a), action_bounds[0], action_bounds[1])

            obs, reward, terminated, truncated, info = env.step(action)
            obs = _obs_vec(obs)
            frame_buf.append(obs.copy())
            total_reward += float(reward)
            ep_len += 1
            # AdroitHandPen/Door emit info["success"] each step while goal
            # pose tolerance holds — track sticky any-step success.
            if active_env == "pen" and bool(info.get("success", info.get("is_success", False))):
                any_step_success = True
            # FrankaKitchen: count of target subtasks completed so far (0..N).
            # episode_task_completions accumulates over the episode.
            if active_env == "kitchen":
                ep_tasks_done = len(info.get("episode_task_completions", []))

        if active_env == "kitchen":
            # Total targets = completed + still-remaining at episode end.
            n_targets = ep_tasks_done + len(info.get("tasks_to_complete", []))
            tasks_completed.append(ep_tasks_done)
            successes.append(n_targets > 0 and ep_tasks_done >= n_targets)
        elif active_env == "pen":
            successes.append(any_step_success)
        else:
            successes.append(bool(info.get("success", False)))
        rewards.append(total_reward)
        dists_first.append(float(info.get("min_dist_to_first_goal", np.inf)))
        dists_second.append(float(info.get("min_dist_to_second_goal", np.inf)))
        ep_lengths.append(ep_len)
        terminated_flags.append(bool(terminated))
        env.close()

    eval_time = time.time() - t0

    # Env-branched metrics: pen/door have no first/second-goal distance;
    # particle records them. All record reward stats.
    if active_env == "kitchen":
        # Headline metric is avg_tasks_completed (0..N), matching IBC Table 2
        # (kitchen-complete = 3.37/4). success_rate = fraction solving ALL tasks.
        return {
            "success_rate": float(np.mean(successes)),
            "success_rate_std": float(np.std(successes)),
            "avg_tasks_completed": float(np.mean(tasks_completed)),
            "std_tasks_completed": float(np.std(tasks_completed)),
            "median_tasks_completed": float(np.median(tasks_completed)),
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "median_reward": float(np.median(rewards)),
            "avg_episode_length": float(np.mean(ep_lengths)),
            "num_seeds": len(seeds),
            "eval_time_s": eval_time,
            "per_seed": [
                {
                    "seed": seeds[i],
                    "success": successes[i],
                    "tasks_completed": tasks_completed[i],
                    "reward": rewards[i],
                    "episode_length": ep_lengths[i],
                    "terminated": terminated_flags[i],
                }
                for i in range(len(seeds))
            ],
        }

    if active_env == "pen":
        return {
            "success_rate": float(np.mean(successes)),
            "success_rate_std": float(np.std(successes)),
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "median_reward": float(np.median(rewards)),
            "avg_episode_length": float(np.mean(ep_lengths)),
            "num_seeds": len(seeds),
            "eval_time_s": eval_time,
            "per_seed": [
                {
                    "seed": seeds[i],
                    "success": successes[i],
                    "reward": rewards[i],
                    "episode_length": ep_lengths[i],
                    "terminated": terminated_flags[i],
                }
                for i in range(len(seeds))
            ],
        }

    finite_first = [d for d in dists_first if np.isfinite(d)]
    finite_second = [d for d in dists_second if np.isfinite(d)]

    return {
        "success_rate": float(np.mean(successes)),
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "median_reward": float(np.median(rewards)),
        "avg_min_dist_first_goal": float(np.mean(finite_first)) if finite_first else None,
        "avg_min_dist_second_goal": float(np.mean(finite_second)) if finite_second else None,
        "median_min_dist_first_goal": float(np.median(finite_first)) if finite_first else None,
        "median_min_dist_second_goal": float(np.median(finite_second)) if finite_second else None,
        "avg_episode_length": float(np.mean(ep_lengths)),
        "num_seeds": len(seeds),
        "eval_time_s": eval_time,
        "per_seed": [
            {
                "seed": seeds[i],
                "success": successes[i],
                "reward": rewards[i],
                "min_dist_first_goal": _finite(dists_first[i]),
                "min_dist_second_goal": _finite(dists_second[i]),
                "episode_length": ep_lengths[i],
                "terminated": terminated_flags[i],
            }
            for i in range(len(seeds))
        ],
    }


# ─── Trial runner ─────────────────────────────────────────────────────────────

