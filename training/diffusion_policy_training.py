"""Diffusion Policy training (capacity-matched WiFI-BC baseline).

Trains a single epsilon-prediction denoiser (`baselines.diffusion.DiffusionDenoiser`,
which reuses the WiFI-BC `QEstimator` trunk) on offline expert (state, action) pairs.
At eval time the SAME checkpoint is sampled with both DDPM and DDIM — see
`envs.evaluate`.

Driven by envs/evaluate.py exactly like training/wifi_bc_training.py
is driven by envs/evaluate.py: config path comes from WIFI_BC_CONFIG_PATH, all
--fixed-params are routed into env_config['training'].

Saves into MODEL_SAVE_DIR:
  - denoiser.pt        (denoiser state_dict)
  - denoiser_ema.pt    (EMA weights, if ema_decay > 0)
  - norm_stats.pt      (obs_mean/std, act_min/max, action_norm_range, state_shape)
                        — SAME schema wifi_bc_training writes, so PushingSimulation's
                        obs-standardize + action-denormalize works unchanged.
"""

import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wifi_bc.config import (default_checkpoint_dir, resolve_active_env,
                            resolve_config_path, resolve_env_config)  # noqa: E402

from baselines.diffusion import build_denoiser, build_diffusion, resolve_dp_params
from wifi_bc.normalizations import ObservationNormalizer

# ── Config (WIFI_BC_CONFIG_PATH overrides the shipped file) ─────────────────
config_path = resolve_config_path()
with open(config_path, "r") as f:
    config = json.load(f)

active_env = resolve_active_env(config)
env_config = resolve_env_config(config, active_env, "diffusion_policy")
training_shared = config.get("training_shared", {})
env_training = env_config.get("training", {})
env_model = env_config.get("model", {})

frame_stack = env_config.get("frame_stack", 1)
action_bounds = tuple(env_config.get("action_bounds", [-1.0, 1.0]))
env_id = env_config.get("env_id", active_env)

training_steps = int(env_training.get("training_steps", training_shared.get("training_steps", 100000)))
batch_size = int(env_training.get("batch_size", training_shared.get("batch_size", 512)))
learning_rate = float(env_training.get("learning_rate", training_shared.get("learning_rate", 1e-3)))
trial_seed = int(env_training.get("trial_seed", 0))
# Mirror wifi_bc_training: pixel datasets decode/stack uint8 images per item, so they
# NEED worker processes or the GPU starves (0 workers => ~unbounded pixel epochs).
# State-based datasets are in-RAM ndarrays and keep 0 workers.
num_workers = int(env_config.get(
    "dataloader_num_workers",
    training_shared.get(
        "num_workers",
        4 if active_env in ("pushing_pixels", "libero_goal_pixels") else 0,
    ),
))
log_interval = int(training_shared.get("log_interval", 1000))
save_interval = int(training_shared.get("save_interval", 10000))
MODEL_SAVE_DIR = training_shared.get(
    "model_save_dir", default_checkpoint_dir("diffusion_policy", active_env)
)

dp = resolve_dp_params(env_config, training_shared)


def load_dataset():
    """Mirror training.wifi_bc_training.load_dataset (state-based envs)."""
    if active_env in ("pen", "kitchen"):
        from envs.datasets import D4RLDataset
        # Mirror wifi_bc_training: kitchen_qpos_only drops the port-added velocity dims
        # -> IBC's input content (robot qpos [0:9] + object qpos [18:39]) = 30-D.
        # Must match WiFI-BC's final kitchen stack or the comparison is unmatched.
        obs_indices = None
        if active_env == "kitchen" and bool(env_training.get("kitchen_qpos_only", False)):
            obs_indices = list(range(0, 9)) + list(range(18, 39))
            print(f"kitchen_qpos_only: obs -> {len(obs_indices)}-D (qpos only)")
        return D4RLDataset(
            env_config["dataset_name"], download=True, frame_stack=frame_stack,
            obs_indices=obs_indices,
            action_chunk=int(env_training.get("action_chunk", 1)),
        )
    elif active_env == "particle":
        from envs.datasets import ParticleDataset
        return ParticleDataset(env_config["data_dir"], n_dim=env_config.get("n_dim", 2), frame_stack=frame_stack)
    elif active_env == "pushing":
        from envs.datasets import PushingDataset
        return PushingDataset(data_dir=env_config["data_dir"], frame_stack=frame_stack)
    elif active_env == "pushing_pixels":
        from envs.datasets import PushingPixelsDataset
        return PushingPixelsDataset(data_dir=env_config["data_dir"], frame_stack=frame_stack)
    elif active_env == "point_maze_pillar":
        from envs.datasets import PointMazePillarDataset
        return PointMazePillarDataset(
            size=20000,
            max_steps_per_episode=env_config.get("max_episode_steps", 400),
            frame_stack=frame_stack,
        )
    elif active_env == "libero_goal_pixels":
        from envs.datasets import LiberoGoalPixelsDataset
        return LiberoGoalPixelsDataset(
            goal_embeddings_path=env_config["goal_embeddings_path"],
            frame_stack=frame_stack,
            max_demos_per_task=env_config.get("max_demos_per_task"),
            crop_size=int(env_training.get("image_crop_size", 0)),
            action_chunk=int(env_training.get("action_chunk", 1)),
            cameras=str(env_config.get("libero_cameras", "agentview+wrist")),
            use_proprio=bool(env_config.get("libero_use_proprio", True)),
        )
    raise ValueError(f"Unknown / unsupported environment for DP: {active_env}")


def build_obs_normalizer(dataset, device):
    """Standardize for the IBC-faithful envs (matches wifi_bc_training)."""
    if active_env in ("pushing_pixels", "libero_goal_pixels"):
        return None  # the pixel encoder does uint8 -> /255 -> resize internally.
    if active_env in ("pushing", "pen", "kitchen"):
        if not hasattr(dataset, "obs_mean") or not hasattr(dataset, "obs_std"):
            raise RuntimeError(f"{active_env} dataset must expose obs_mean/obs_std.")
        return ObservationNormalizer(env_id=env_id, device=device, frame_stack=frame_stack,
                                     obs_mean=dataset.obs_mean, obs_std=dataset.obs_std)
    particle_n_dim = env_config.get("n_dim") if active_env == "particle" else None
    return ObservationNormalizer(env_id=env_id, device=device, frame_stack=frame_stack,
                                 particle_n_dim=particle_n_dim)


def main():
    random.seed(trial_seed)
    np.random.seed(trial_seed)
    torch.manual_seed(trial_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(trial_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"trial_seed={trial_seed} | device={device} | active_env={active_env}")
    print(f"Diffusion Policy (epsilon-pred, Q-estimator trunk). DP params: {json.dumps(dp)}")
    print(f"Training steps: {training_steps} | batch_size: {batch_size} | lr: {learning_rate}")

    print(f"Loading {active_env} dataset...")
    dataset = load_dataset()
    print(f"Dataset size: {len(dataset)} | state_shape={dataset.state_shape} | action_shape={dataset.action_shape}")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=num_workers > 0,
    )
    obs_normalizer = build_obs_normalizer(dataset, device)

    is_pixels = active_env in ("pushing_pixels", "libero_goal_pixels")
    # libero_goal_pixels conditions the denoiser on proprio + goal embedding;
    # pushing_pixels has no conditioning, and cond_dim 0 makes the builder fall
    # through to the plain pixel denoiser, so one call covers both.
    cond_dim = int(getattr(dataset, "cond_dim", 0))
    if is_pixels:
        from baselines.diffusion import build_cond_pixel_denoiser
        in_channels = int(dataset.state_shape[0])  # 3 * n_cams * frame_stack
        enc_h = int(env_config.get("encoder_target_height", 180))
        enc_w = int(env_config.get("encoder_target_width", 240))
        denoiser = build_cond_pixel_denoiser(
            dataset.action_shape, in_channels, dp,
            cond_dim=cond_dim,
            encoder_target_height=enc_h, encoder_target_width=enc_w,
            encoder_kind=env_model.get("encoder_kind", "conv_maxpool"),
            encoder_pretrained=env_model.get("encoder_pretrained", False),
            encoder_num_kp=int(env_model.get("encoder_num_kp", 64)),
            encoder_norm_kind=env_model.get("encoder_norm_kind", "bn"),
            encoder_per_camera=bool(env_model.get("encoder_per_camera", False)),
            device=device,
        )
    else:
        denoiser = build_denoiser(dataset.state_shape, dataset.action_shape, dp, device)
    diffusion = build_diffusion(dp, device, action_bounds)
    n_params = sum(p.numel() for p in denoiser.parameters())
    print(f"Denoiser params: {n_params} (kind={dp['denoiser_network_kind']} "
          f"w={dp['denoiser_width']} d={dp['denoiser_depth']} t_emb={dp['time_emb_dim']}"
          f"{' +ConvMaxpoolEncoder' if is_pixels else ''})")

    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=learning_rate)
    effective_t_max = env_training.get("cosine_t_max", training_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=effective_t_max, eta_min=1e-6)

    ema_decay = dp["ema_decay"]
    ema_denoiser = copy.deepcopy(denoiser) if ema_decay > 0 else None
    if ema_denoiser is not None:
        for p in ema_denoiser.parameters():
            p.requires_grad_(False)

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    start_time = time.time()
    step = 0
    running_loss = 0.0
    running_n = 0

    denoiser.train()
    while step < training_steps:
        for batch in dataloader:
            if step >= training_steps:
                break
            states = batch["state"].float().to(device)
            if obs_normalizer is not None:
                states = obs_normalizer.normalize(states)
            actions = batch["action"].float().to(device)  # already in [-1, 1]

            # libero_goal_pixels: hand the per-state conditioning (proprio +
            # goal embedding) to the denoiser via `._cond`, the same channel the
            # WiFI-BC pixel nets use, so it reaches every forward in this batch
            # without threading it through the diffusion loss. Gated on the
            # BATCH rather than the env name, so any conditioned pixel dataset
            # works. The EMA copy needs it too: it is a full module that gets
            # called at evaluation.
            if "cond" in batch:
                cond = batch["cond"].float().to(device)
                denoiser._cond = cond
                if ema_denoiser is not None:
                    ema_denoiser._cond = cond

            # For pixels, denoiser.forward(images, xt, t) encodes internally.
            loss = diffusion.training_loss(denoiser, states, actions)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if ema_denoiser is not None:
                with torch.no_grad():
                    for ep, p in zip(ema_denoiser.parameters(), denoiser.parameters()):
                        ep.mul_(ema_decay).add_(p, alpha=1.0 - ema_decay)

            running_loss += float(loss.item())
            running_n += 1
            step += 1

            if step % log_interval == 0:
                avg = running_loss / max(running_n, 1)
                elapsed = time.time() - start_time
                # "Loss:" token parsed by envs/evaluate.py.extract_final_metrics.
                print(f"Step {step}/{training_steps} | Loss: {avg:.6f} | {elapsed:.0f}s")
                running_loss = 0.0
                running_n = 0

            if save_interval and step % save_interval == 0:
                torch.save(denoiser.state_dict(), os.path.join(MODEL_SAVE_DIR, "denoiser.pt"))
                if ema_denoiser is not None:
                    torch.save(ema_denoiser.state_dict(), os.path.join(MODEL_SAVE_DIR, "denoiser_ema.pt"))

    # Final checkpoint.
    torch.save(denoiser.state_dict(), os.path.join(MODEL_SAVE_DIR, "denoiser.pt"))
    if ema_denoiser is not None:
        torch.save(ema_denoiser.state_dict(), os.path.join(MODEL_SAVE_DIR, "denoiser_ema.pt"))

    # norm_stats — SAME schema as wifi_bc_training so eval's PushingSimulation reuses it.
    # Not every dataset carries action stats: ParticleDataset has no act_min /
    # act_max (its actions are already in the model range), so reading them
    # unconditionally crashes AFTER a full training run has completed. Emit them
    # when present and leave them out otherwise — eval falls back to the model
    # range exactly as it does for a checkpoint trained before these existed.
    # Write norm_stats ONLY when the dataset has action stats. Emitting a
    # partial file is worse than emitting none: the simulations treat a missing
    # norm_stats.pt as "no denormalization needed" and carry on, but a file that
    # exists and lacks act_min crashes them. ParticleDataset has no act_min (its
    # actions are already in model range), which is why every dpParticle job
    # died in eval with KeyError('act_min') after training successfully.
    if not hasattr(dataset, "act_min"):
        print("Dataset exposes no act_min/act_max; skipping norm_stats.pt "
              "(eval falls back to the model action range).")
        print(f"Done in {time.time() - start_time:.0f}s")
        return
    norm_stats = {
        "act_min": dataset.act_min,
        "act_max": dataset.act_max,
        "action_norm_range": getattr(dataset, "action_norm_range", (-1.0, 1.0)),
        "state_shape": dataset.state_shape,
    }
    if hasattr(dataset, "obs_mean"):
        norm_stats["obs_mean"] = dataset.obs_mean
        norm_stats["obs_std"] = dataset.obs_std
    # Action chunking K — the sims read this to step chunk[k]. 1 = single action.
    norm_stats["action_chunk"] = int(getattr(dataset, "action_chunk", 1) or 1)
    # Kitchen: persist the column selection (kitchen_qpos_only) so eval rebuilds
    # the identical policy input (mirrors wifi_bc_training / the libero state_shape trick).
    if active_env == "kitchen":
        norm_stats["obs_indices"] = getattr(dataset, "obs_indices", None)
    # libero_goal_pixels: persist the pixel + conditioning schema so the render
    # eval rebuilds a byte-identical (image, cond) input and the same encoder.
    # Mirrors the block wifi_bc_training writes, which is what envs/evaluate.py
    # reads for every method.
    if active_env == "libero_goal_pixels":
        norm_stats["libero_obs_keys"] = dataset.libero_obs_keys
        norm_stats["goal_embeddings"] = dataset.goal_embeddings
        norm_stats["goal_task_names"] = dataset.goal_task_names
        norm_stats["goal_emb_dim"] = dataset.goal_emb_dim
        norm_stats["proprio_dim"] = dataset.proprio_dim
        norm_stats["cond_dim"] = dataset.cond_dim
        norm_stats["in_channels"] = dataset.in_channels
        norm_stats["image_hw"] = [dataset._H, dataset._W]
        norm_stats["libero_cameras"] = list(dataset.cameras)
        norm_stats["encoder_target_height"] = env_config.get("encoder_target_height", 128)
        norm_stats["encoder_target_width"] = env_config.get("encoder_target_width", 128)
        norm_stats["encoder_kind"] = env_model.get("encoder_kind", "conv_maxpool")
        norm_stats["encoder_pretrained"] = bool(env_model.get("encoder_pretrained", False))
        norm_stats["encoder_num_kp"] = int(env_model.get("encoder_num_kp", 64))
        norm_stats["encoder_norm_kind"] = env_model.get("encoder_norm_kind", "bn")
        norm_stats["encoder_per_camera"] = bool(env_model.get("encoder_per_camera", False))
        norm_stats["image_crop_size"] = int(env_training.get("image_crop_size", 0))
        norm_stats["state_shape"] = list(dataset.state_shape)
        # The diffusion sampler eval must match what was trained.
        for key in ("num_train_timesteps", "beta_schedule", "prediction_type",
                    "time_emb_dim", "denoiser_network_kind", "denoiser_width",
                    "denoiser_depth", "denoiser_use_spectral_norm"):
            if key in dp:
                norm_stats[key] = dp[key]
    torch.save(norm_stats, os.path.join(MODEL_SAVE_DIR, "norm_stats.pt"))
    print(f"Saved denoiser.pt + norm_stats.pt to {MODEL_SAVE_DIR}")
    print(f"Done in {time.time() - start_time:.0f}s")


if __name__ == "__main__":
    main()
