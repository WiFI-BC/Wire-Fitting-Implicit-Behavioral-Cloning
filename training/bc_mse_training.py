"""Explicit behaviour cloning (MSE regression). A standalone baseline.

Deliberately NOT a configuration of another method. wifi_bc_training reduced to one
control point with generator_infonce_weight=0 is arithmetically the same
objective, but it still constructs and trains a Q estimator, and reporting it as
"BC" invites the reasonable objection that the baseline is the proposed method
wearing a flag. This trains a regression network and nothing else: no control
points, no energy function, no critic, no sampler.

    policy(obs) -> action        one forward pass
    loss = mean( (policy(obs) - a_expert)^2 )

Same contract as the other trainers so the surrounding tooling works unchanged:
reads WIFI_BC_CONFIG_PATH, uses wifi_bc_training's load_dataset (so the data pipeline is
byte-identical to every method it will be compared against), and writes
norm_stats.pt beside the weights.

Checkpoints are `bc_policy.pt` / `bc_policy_ema.pt`. envs.evaluate
recognises that pair and evaluates it with a null critic, exactly as it does for
a plain diffusion policy — one candidate, nothing to rank.
"""

import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import wandb

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wifi_bc.config import (default_checkpoint_dir, resolve_active_env,
                            resolve_config_path, resolve_env_config)  # noqa: E402

from wifi_bc.models import BCPolicy
from wifi_bc.normalizations import ObservationNormalizer

config_path = resolve_config_path()
with open(config_path) as f:
    config = json.load(f)

active_env = resolve_active_env(config)
env_config = resolve_env_config(config, active_env, "bc_mse")
training_shared = config.get("training_shared", {})
env_training = env_config.get("training", {})
env_model = env_config.get("model", {})

from training.wifi_bc_training import load_dataset  # noqa: E402

PIXEL_ENVS = ("pushing_pixels", "libero_goal_pixels")

training_steps = env_training.get("training_steps", training_shared.get("training_steps", 100000))
batch_size = env_training.get("batch_size", training_shared.get("batch_size", 128))
learning_rate = env_training.get("learning_rate", training_shared.get("learning_rate", 1e-3))
MODEL_SAVE_DIR = training_shared.get(
    "model_save_dir", default_checkpoint_dir("bc_mse", active_env)
)
log_interval = training_shared.get("log_interval", 1000)
save_interval = training_shared.get("save_interval", 10000)
scheduler_type = env_training.get("scheduler_type", training_shared.get("scheduler_type", "cosine"))
cosine_t0 = env_training.get("cosine_t0", training_shared.get("cosine_t0", 50000))
cosine_t_max = env_training.get("cosine_t_max", training_shared.get("cosine_t_max", None))
trial_seed = env_training.get("trial_seed", training_shared.get("trial_seed", 0))
ema_decay = float(env_training.get("ema_decay", training_shared.get("ema_decay", 0.0)))
# The regression trunk. bc_width / bc_depth keep it independent of the
# control-point generator's cp_width / cp_depth, so sweeping one never moves
# the other.
bc_width = int(env_model.get("bc_width", env_model.get("cp_width", 256)))
bc_depth = int(env_model.get("bc_depth", env_model.get("cp_depth", 2)))
encoder_lr_scale = float(env_training.get("encoder_lr_scale", 1.0))
env_id = env_config["env_id"]
action_bounds = env_config.get("action_bounds", [-1, 1])
frame_stack = env_config.get("frame_stack", 1)


def main() -> int:
    random.seed(trial_seed); np.random.seed(trial_seed); torch.manual_seed(trial_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(trial_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"trial_seed={trial_seed}  device={device}  env={active_env}")
    print(f"steps={training_steps} batch={batch_size} lr={learning_rate} "
          f"trunk={bc_width}x{bc_depth} ema={ema_decay}")

    wandb.init(project="WiFI-BC",
               config={"algorithm": "bc_mse", "active_env": active_env,
                       "env_config": env_config, "training_shared": training_shared},
               name=f"{active_env}_bcmse_{bc_width}x{bc_depth}_lr{learning_rate}_seed{trial_seed}")

    dataset = load_dataset()
    print(f"Dataset size: {len(dataset)}   action_shape={dataset.action_shape}")
    is_pixels = active_env in PIXEL_ENVS
    cond_dim = int(getattr(dataset, "cond_dim", 0))
    action_dim = int(dataset.action_shape)

    if is_pixels:
        policy = BCPolicy(
            action_dim, in_channels=dataset.state_shape[0], cond_dim=cond_dim,
            width=bc_width, depth=bc_depth,
            action_bounds=(action_bounds[0], action_bounds[1]),
            encoder_target_height=int(env_config.get("encoder_target_height", 180)),
            encoder_target_width=int(env_config.get("encoder_target_width", 240)),
            encoder_feature_dim=int(env_model.get("encoder_feature_dim", 256)),
            encoder_kind=env_model.get("encoder_kind", "conv_maxpool"),
            encoder_pretrained=bool(env_model.get("encoder_pretrained", False)),
            encoder_num_kp=int(env_model.get("encoder_num_kp", 64)),
            encoder_norm_kind=env_model.get("encoder_norm_kind", "bn"),
            encoder_per_camera=bool(env_model.get("encoder_per_camera", False)),
            cond_fusion=env_model.get("cond_fusion", "concat"),
            goal_dim=int(getattr(dataset, "goal_emb_dim", 0) or 0),
        ).to(device)
    else:
        policy = BCPolicy(action_dim, state_dim=int(dataset.state_shape),
                          cond_dim=cond_dim, width=bc_width, depth=bc_depth,
                          action_bounds=(action_bounds[0], action_bounds[1])).to(device)
    print(f"policy params: {sum(p.numel() for p in policy.parameters())/1e6:.2f}M "
          f"(no critic, no control points)")

    ema = copy.deepcopy(policy) if ema_decay > 0 else None
    if ema is not None:
        ema.eval()
        for p in ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_ema():
        src = dict(policy.named_parameters())
        for n, p in ema.named_parameters():
            p.mul_(ema_decay).add_(src[n].detach(), alpha=1 - ema_decay)
        sb = dict(policy.named_buffers())
        for n, b in ema.named_buffers():
            v = sb[n].detach()
            if b.shape != v.shape:
                b.resize_(v.shape)
            b.copy_(v)

    if encoder_lr_scale != 1.0 and policy.encoder is not None:
        enc = {id(p) for p in policy.encoder.parameters()}
        opt = torch.optim.AdamW([
            {"params": [p for p in policy.parameters() if id(p) not in enc], "lr": learning_rate},
            {"params": [p for p in policy.parameters() if id(p) in enc],
             "lr": learning_rate * encoder_lr_scale}])
        print(f"Split LR: encoder x{encoder_lr_scale}")
    else:
        opt = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    t_max = cosine_t_max if cosine_t_max is not None else training_steps
    sched = (torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=cosine_t0, eta_min=1e-6)
             if scheduler_type == "cosine_warm_restarts"
             else torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=1e-6))

    nw = env_config.get("dataloader_num_workers", 4 if is_pixels else 0)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                         num_workers=nw, persistent_workers=nw > 0,
                                         timeout=600 if nw > 0 else 0)
    obs_norm = None
    if not is_pixels:
        pnd = env_config.get("n_dim") if active_env == "particle" else None
        obs_norm = (ObservationNormalizer(env_id=env_id, device=device, frame_stack=frame_stack,
                                          obs_mean=dataset.obs_mean, obs_std=dataset.obs_std)
                    if hasattr(dataset, "obs_mean")
                    else ObservationNormalizer(env_id=env_id, device=device,
                                               frame_stack=frame_stack, particle_n_dim=pnd))

    def save():
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        torch.save(policy.state_dict(), os.path.join(MODEL_SAVE_DIR, "bc_policy.pt"))
        if ema is not None:
            torch.save(ema.state_dict(), os.path.join(MODEL_SAVE_DIR, "bc_policy_ema.pt"))

    def persist_norm_stats():
        if not hasattr(dataset, "act_min"):
            return
        ns = {"act_min": dataset.act_min, "act_max": dataset.act_max,
              "action_norm_range": getattr(dataset, "action_norm_range", (-1.0, 1.0)),
              "frame_stack": frame_stack, "env_id": env_id, "algorithm": "bc_mse",
              "action_chunk": int(getattr(dataset, "action_chunk", 1)),
              "bc_width": bc_width, "bc_depth": bc_depth,
              "cond_dim": cond_dim}
        if not is_pixels:
            # Flat-state envs: record the exact policy input width and the
            # column selection used. kitchen_qpos_only trims the observation to
            # 30-D, well below the config's state_dim of 59, and without this
            # evaluation rebuilt a 59-wide network and failed to load the
            # checkpoint. The other trainers already persist both.
            ns["state_shape"] = int(dataset.state_shape)
            ns["obs_indices"] = getattr(dataset, "obs_indices", None)
        if is_pixels:
            ns.update(in_channels=dataset.state_shape[0], state_shape=list(dataset.state_shape),
                      image_hw=list(dataset.state_shape[1:]),
                      encoder_target_height=env_config.get("encoder_target_height", 180),
                      encoder_target_width=env_config.get("encoder_target_width", 240),
                      encoder_feature_dim=int(env_model.get("encoder_feature_dim", 256)),
                      encoder_kind=env_model.get("encoder_kind", "conv_maxpool"),
                      encoder_pretrained=bool(env_model.get("encoder_pretrained", False)),
                      encoder_num_kp=int(env_model.get("encoder_num_kp", 64)),
                      encoder_norm_kind=env_model.get("encoder_norm_kind", "bn"),
                      encoder_per_camera=bool(env_model.get("encoder_per_camera", False)),
                      cond_fusion=env_model.get("cond_fusion", "concat"))
        if active_env == "libero_goal_pixels":
            ns.update(libero_obs_keys=dataset.libero_obs_keys,
                      libero_cameras=list(dataset.cameras),
                      goal_embeddings=dataset.goal_embeddings,
                      goal_task_names=dataset.goal_task_names,
                      goal_emb_dim=dataset.goal_emb_dim,
                      proprio_dim=dataset.proprio_dim,
                      image_crop_size=int(env_training.get("image_crop_size", 0)))
        if hasattr(dataset, "obs_mean"):
            ns.update(obs_mean=dataset.obs_mean, obs_std=dataset.obs_std)
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        torch.save(ns, os.path.join(MODEL_SAVE_DIR, "norm_stats.pt"))

    persist_norm_stats()
    t0 = time.time(); step = 0
    while step < training_steps:
        for batch in loader:
            if step >= training_steps:
                break
            obs = batch["state"].float().to(device)
            if obs_norm is not None:
                obs = obs_norm.normalize(obs)
            act = batch["action"].float().to(device)
            if "cond" in batch:
                policy._cond = batch["cond"].float().to(device)
            pred = policy(obs).squeeze(1)
            loss = torch.mean((pred - act) ** 2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            if ema is not None:
                update_ema()
            sched.step(); step += 1
            if step % log_interval == 0:
                mae = (pred - act).abs().mean().item()
                # "Loss:" verbatim so envs/evaluate.py.extract_final_metrics
                # picks it up; the whole objective is this one MSE term.
                print(f"Step {step}/{training_steps} | Loss: {loss.item():.5f} | "
                      f"MAE: {mae:.5f} | LR: {sched.get_last_lr()[0]:.2e} | "
                      f"{time.time()-t0:.1f}s")
                wandb.log({"step": step, "loss/mse": loss.item(),
                           "metric/action_mae": mae,
                           "learning_rate": sched.get_last_lr()[0]})
            if step % save_interval == 0:
                save()
    save(); persist_norm_stats()
    print(f"\nTraining completed in {time.time()-t0:.1f}s. Saved to {MODEL_SAVE_DIR}/")
    art = wandb.Artifact("model-checkpoints", type="model")
    art.add_file(os.path.join(MODEL_SAVE_DIR, "bc_policy.pt"))
    if ema is not None:
        art.add_file(os.path.join(MODEL_SAVE_DIR, "bc_policy_ema.pt"))
    wandb.log_artifact(art); wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
