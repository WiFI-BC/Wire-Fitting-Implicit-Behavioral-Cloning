"""Consistency Policy training (Prasad et al., 2024 — arXiv 2405.07503).

Driven by envs/evaluate.py like every other trainer: config from
WIFI_BC_CONFIG_PATH, datasets from training.wifi_bc_training.load_dataset (so
every environment and its camera/proprio switches work unchanged).

The algorithm follows the official implementation (github.com/Aaditya-Prasad/
Consistency-Policy) — see utils/consistency.py for the line-by-line port:
  1. EDM teacher: Karras sigma grid (0.02-80, rho 7, 80 bins), boundary
     preconditioning, log-normal sigma, Karras-weighted pseudo-Huber DSM, AdamW
     (betas 0.95/0.999, wd 1e-6), cosine LR with 500 warmup steps, power-schedule
     EMA used for evaluation.
  2. CTM student: warm-started from the teacher with zero-initialised stop-time
     columns, encoder frozen, dropout 0.2, loss = ctm + dsm (weights 1, 1),
     one teacher Heun step (u = t+1), stop-gradient target net = student copy
     (decay 0), cosine LR sized to 80% of training (p_epochs 400 / 500), raw
     student weights used for evaluation.
Optimisation budget (steps, batch size, learning rate) and the network trunk come
from our Diffusion Policy recipe per environment, so CP is compared against DP
with matched architecture and compute.

Phases (cp_phase):
  both     teacher, then student, in one job
  teacher  teacher only
  student  student only, distilled from the teacher in cp_teacher_dir (for
           environments where teacher + student would not fit one job).
           cp_teacher_dir="auto" picks the newest finished teacher among the
           sibling run directories with the same env, trial_seed and cp_run_tag,
           so a student batch can be written before the teacher runs exist.

cp_student_feature_cache (pixel envs, default off): the student's encoder is
frozen, so on a dataset without augmentation its features are a fixed function
of the sample. Encode every sample once and draw student batches from that table
(same epoch-wise shuffling) instead of re-decoding images each step. Refused for
augmented data (random crop / image_aug / a sample that differs between reads);
the cached rows are checked against a fresh encode before training starts.

Writes into MODEL_SAVE_DIR:
  cp_teacher.pt, cp_teacher_ema.pt, cp_student.pt   model state_dicts
  cp_meta.json      everything needed to rebuild the model and sampler
  norm_stats.pt     the shared norm_stats schema, only when the dataset has action stats
"""
from __future__ import annotations

import copy
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wifi_bc.config import resolve_active_env, resolve_env_config  # noqa: E402

config_path = Path(os.environ.get("WIFI_BC_CONFIG_PATH") or (Path(__file__).resolve().parent.parent / "config" / "config.json"))
with open(config_path) as f:
    config = json.load(f)
active_env = resolve_active_env(config)
env_config = resolve_env_config(config, active_env, "consistency_policy")
training_shared = config.get("training_shared", {})
env_training = env_config.get("training", {})
env_model = env_config.get("model", {})

from training.wifi_bc_training import load_dataset  # noqa: E402  (same WIFI_BC_CONFIG_PATH)
from baselines.consistency import (CPModel, KarrasSchedule, cosine_with_warmup, ema_power_decay,  # noqa: E402
                               ema_update, student_losses, teacher_loss, warm_start_student)
from baselines.diffusion import resolve_dp_params  # noqa: E402
from wifi_bc.normalizations import ObservationNormalizer  # noqa: E402


def g(key, default):
    return env_training.get(key, training_shared.get(key, default))


PIXEL_ENVS = ("pushing_pixels", "libero_goal_pixels")
is_pixels = active_env in PIXEL_ENVS
frame_stack = int(env_config.get("frame_stack", 1))
action_bounds = tuple(float(v) for v in env_config.get("action_bounds", [-1.0, 1.0]))
env_id = env_config.get("env_id", active_env)

trial_seed = int(g("trial_seed", 0))
training_steps = int(g("training_steps", 100000))
teacher_steps = int(g("cp_teacher_steps", training_steps))
student_steps = int(g("cp_student_steps", round(1.25 * teacher_steps)))
phase = str(g("cp_phase", "both"))
teacher_dir = str(g("cp_teacher_dir", ""))
run_tag = str(g("cp_run_tag", ""))
student_feature_cache = bool(g("cp_student_feature_cache", False))
batch_size = int(g("batch_size", 64))
learning_rate = float(g("learning_rate", 1e-4))
# Student (CTM distillation) learning rate. Defaults to the shared learning_rate, but the
# official CP config trains the student at 1e-4 (configs/ctmp_*.yaml): with our Diffusion
# Policy recipe rates (3e-4 .. 1e-3) the flat-state students collapse to an input-ignoring map.
student_learning_rate = float(g("cp_student_learning_rate", learning_rate))
encoder_lr_scale = float(g("encoder_lr_scale", 1.0))
weight_decay = float(g("cp_weight_decay", 1e-6))
lr_warmup = int(g("cp_lr_warmup", 500))
p_fraction = float(g("cp_p_fraction", 0.8))
huber_delta = float(g("cp_huber_delta", -1.0))
dropout = float(g("cp_dropout", 0.2))
w_ctm, w_dsm = float(g("cp_ctm_weight", 1.0)), float(g("cp_dsm_weight", 1.0))
ode_steps_max = int(g("cp_ode_steps_max", 1))
sched = KarrasSchedule(sigma_min=float(g("cp_sigma_min", 0.02)), sigma_max=float(g("cp_sigma_max", 80.0)),
                       rho=float(g("cp_rho", 7.0)), bins=int(g("cp_bins", 80)), sigma_data=float(g("cp_sigma_data", 0.5)))
log_interval = int(training_shared.get("log_interval", 1000))
save_interval = int(training_shared.get("save_interval", 10000))
MODEL_SAVE_DIR = training_shared.get("model_save_dir", "checkpoints")
num_workers = int(env_config.get("dataloader_num_workers",
                                 training_shared.get("num_workers", 4 if is_pixels else 0)))
dp = resolve_dp_params(env_config, training_shared)



def resolve_teacher_dir(spec: str) -> str:
    if spec != "auto":
        return spec
    here = os.path.abspath(MODEL_SAVE_DIR)
    found = []
    for meta_path in Path(here).parent.glob("*/cp_meta.json"):
        run_dir = str(meta_path.parent)
        if os.path.abspath(run_dir) == here or not (meta_path.parent / "cp_teacher.pt").exists():
            continue
        try:
            with open(meta_path) as fh:
                m = json.load(fh)
        except (OSError, ValueError):
            continue
        if (m.get("teacher_complete") and m.get("active_env") == active_env
                and int(m.get("trial_seed", -1)) == trial_seed and str(m.get("run_tag", "")) == run_tag):
            found.append(((meta_path.parent / "cp_teacher.pt").stat().st_mtime, run_dir))
    if not found:
        raise SystemExit(f"cp_teacher_dir=auto: no finished teacher for env={active_env} seed={trial_seed} "
                         f"tag={run_tag!r} under {Path(here).parent}")
    found.sort()
    print(f"cp_teacher_dir=auto -> {found[-1][1]} ({len(found)} candidate(s))")
    return found[-1][1]


if phase == "student":
    teacher_dir = resolve_teacher_dir(teacher_dir)
if phase not in ("both", "teacher", "student"):
    raise SystemExit(f"cp_phase must be both|teacher|student, got {phase!r}")
if phase == "student" and not (teacher_dir and os.path.exists(os.path.join(teacher_dir, "cp_teacher.pt"))):
    raise SystemExit(f"cp_phase=student needs cp_teacher_dir containing cp_teacher.pt (got {teacher_dir!r})")


def main() -> int:
    random.seed(trial_seed); np.random.seed(trial_seed); torch.manual_seed(trial_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(trial_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    print(f"Consistency Policy | env={active_env} seed={trial_seed} device={device} phase={phase}")
    print(f"teacher_steps={teacher_steps} student_steps={student_steps} batch={batch_size} lr={learning_rate} "
          f"student_lr={student_learning_rate} "
          f"head={dp['denoiser_network_kind']}({dp['denoiser_width']}x{dp['denoiser_depth']}) t_emb={dp['time_emb_dim']}")
    print(f"sigma [{sched.sigma_min}, {sched.sigma_max}] rho={sched.rho} bins={sched.bins} sigma_data={sched.sigma_data} "
          f"huber_delta={huber_delta} dropout={dropout} ctm/dsm={w_ctm}/{w_dsm} ode_steps_max={ode_steps_max} "
          f"run_tag={run_tag!r} student_feature_cache={student_feature_cache}")

    dataset = load_dataset()
    action_dim = int(dataset.action_shape)
    cond_dim = int(getattr(dataset, "cond_dim", 0))
    print(f"Dataset: {len(dataset)} samples, state_shape={dataset.state_shape}, action_dim={action_dim}, cond_dim={cond_dim}")
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                         num_workers=num_workers, persistent_workers=num_workers > 0)

    if is_pixels:
        obs_normalizer = None
    elif hasattr(dataset, "obs_mean"):
        obs_normalizer = ObservationNormalizer(env_id=env_id, device=device, frame_stack=frame_stack,
                                               obs_mean=dataset.obs_mean, obs_std=dataset.obs_std)
    else:
        obs_normalizer = ObservationNormalizer(env_id=env_id, device=device, frame_stack=frame_stack,
                                               particle_n_dim=env_config.get("n_dim") if active_env == "particle" else None)

    encoder_kwargs = None
    if is_pixels:
        in_channels = int(dataset.state_shape[0])
        encoder_kwargs = dict(
            encoder_kind=env_model.get("encoder_kind", "conv_maxpool"),
            encoder_target_height=int(env_config.get("encoder_target_height", 180)),
            encoder_target_width=int(env_config.get("encoder_target_width", 240)),
            encoder_feature_dim=int(env_model.get("encoder_feature_dim", 256)),
            encoder_pretrained=env_model.get("encoder_pretrained", False),
            encoder_num_kp=int(env_model.get("encoder_num_kp", 64)),
            encoder_norm_kind=env_model.get("encoder_norm_kind", "bn"),
            encoder_per_camera=bool(env_model.get("encoder_per_camera", False)))

    def build(two_times: bool, p_drop: float) -> CPModel:
        common = dict(cond_dim=cond_dim, time_emb_dim=int(dp["time_emb_dim"]), network_kind=dp["denoiser_network_kind"],
                      width=int(dp["denoiser_width"]), depth=int(dp["denoiser_depth"]), two_times=two_times, dropout=p_drop)
        if is_pixels:
            return CPModel(action_dim, in_channels=in_channels, encoder_kwargs=encoder_kwargs, **common).to(device)
        return CPModel(action_dim, state_dim=int(dataset.state_shape), **common).to(device)

    meta = dict(algorithm="consistency_policy", phase=phase, active_env=active_env, trial_seed=trial_seed,
                run_tag=run_tag, teacher_complete=phase == "student", action_dim=action_dim,
                action_bounds=list(action_bounds), pixel=is_pixels, cond_dim=cond_dim,
                state_dim=None if is_pixels else int(dataset.state_shape),
                in_channels=in_channels if is_pixels else None, encoder_kwargs=encoder_kwargs,
                time_emb_dim=int(dp["time_emb_dim"]), network_kind=dp["denoiser_network_kind"],
                width=int(dp["denoiser_width"]), depth=int(dp["denoiser_depth"]), dropout=dropout,
                sigma_min=sched.sigma_min, sigma_max=sched.sigma_max, rho=sched.rho, bins=sched.bins,
                sigma_data=sched.sigma_data, huber_delta=huber_delta, ode_steps_max=ode_steps_max,
                ctm_weight=w_ctm, dsm_weight=w_dsm, chaining_default="D:27,54", teacher_steps=teacher_steps,
                student_steps=student_steps, student_learning_rate=student_learning_rate, frame_stack=frame_stack, student_feature_cache=student_feature_cache,
                action_chunk=int(getattr(dataset, "action_chunk", 1) or 1), teacher_dir=teacher_dir or None)
    def write_meta() -> None:
        with open(os.path.join(MODEL_SAVE_DIR, "cp_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
    write_meta()
    persist_norm_stats(dataset, cond_dim)

    lo, hi = action_bounds
    stream = _forever(loader)

    def next_batch(model: CPModel):
        b = next(stream)
        states = b["state"].float().to(device)
        if obs_normalizer is not None:
            states = obs_normalizer.normalize(states)
        x0 = (b["action"].float().to(device) - lo) / (hi - lo) * 2.0 - 1.0
        model._cond = b["cond"].float().to(device) if cond_dim else None
        return states, x0

    # ── 1. EDM teacher ────────────────────────────────────────────────────────
    if phase in ("both", "teacher"):
        teacher = build(False, 0.0)
        params = _param_groups(teacher, learning_rate, encoder_lr_scale)
        opt = torch.optim.AdamW(params, lr=learning_rate, betas=(0.95, 0.999), eps=1e-8, weight_decay=weight_decay)
        lrs = torch.optim.lr_scheduler.LambdaLR(opt, cosine_with_warmup(lr_warmup, teacher_steps))
        teacher_ema = copy.deepcopy(teacher).eval()
        teacher_ema.requires_grad_(False)
        print(f"Teacher params: {sum(p.numel() for p in teacher.parameters())}")
        t0, run = time.time(), 0.0
        for step in range(1, teacher_steps + 1):
            teacher.train()
            states, x0 = next_batch(teacher)
            loss = teacher_loss(sched, teacher.head, teacher.encode(states), x0, huber_delta)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); lrs.step()
            ema_update(teacher_ema, teacher, ema_power_decay(step))
            run += float(loss.item())
            if step % log_interval == 0:
                print(f"[teacher] Step {step}/{teacher_steps} | Loss: {run / log_interval:.6f} | "
                      f"LR: {lrs.get_last_lr()[0]:.2e} | {time.time() - t0:.0f}s", flush=True)
                run = 0.0
            if save_interval and step % save_interval == 0:
                _save_teacher(teacher, teacher_ema)
        _save_teacher(teacher, teacher_ema)
        meta["teacher_complete"] = True
        write_meta()
        print(f"Teacher done in {time.time() - t0:.0f}s")
        if phase == "teacher":
            return 0
    else:
        teacher = build(False, 0.0)
        teacher.load_state_dict(torch.load(os.path.join(teacher_dir, "cp_teacher.pt"), map_location=device, weights_only=True))
        for name in ("cp_teacher.pt", "cp_teacher_ema.pt"):
            src = os.path.join(teacher_dir, name)
            if os.path.exists(src) and os.path.abspath(teacher_dir) != os.path.abspath(MODEL_SAVE_DIR):
                shutil.copy2(src, os.path.join(MODEL_SAVE_DIR, name))
        print(f"Loaded teacher from {teacher_dir}")

    # ── 2. CTM student ────────────────────────────────────────────────────────
    # Official warm start uses the teacher's raw (non-EMA) weights, as does the
    # teacher inside the CTM loss.
    teacher.eval()
    teacher.requires_grad_(False)
    student = build(True, dropout)
    warm_start_student(student, teacher)
    if student.encoder is not None:
        student.encoder.eval()
        student.encoder.requires_grad_(False)
    target = copy.deepcopy(student)
    target.requires_grad_(False)
    trainable = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=student_learning_rate, betas=(0.95, 0.999), eps=1e-8, weight_decay=weight_decay)
    lrs = torch.optim.lr_scheduler.LambdaLR(opt, cosine_with_warmup(lr_warmup, max(1, int(p_fraction * student_steps))))
    print(f"Student trainable params: {sum(p.numel() for p in trainable)} (encoder frozen: {student.encoder is not None})")
    cache = build_feature_cache(student, dataset, device, cond_dim) if student_feature_cache else None
    cache_batches = _index_batches(cache[0].shape[0], batch_size, device) if cache is not None else None
    t0, run_c, run_d = time.time(), 0.0, 0.0
    for step in range(1, student_steps + 1):
        student.train(); target.train()
        if student.encoder is not None:
            student.encoder.eval(); target.encoder.eval()
        if cache is not None:
            idx = next(cache_batches)
            feats, x0 = cache[0][idx], cache[1][idx]
        else:
            states, x0 = next_batch(student)
            with torch.no_grad():
                feats = student.encode(states)   # frozen encoder == teacher's: one encode serves all three nets
        l_ctm, l_dsm = student_losses(sched, student.head, target.head, teacher.head, feats, x0, huber_delta, ode_steps_max)
        loss = w_ctm * l_ctm + w_dsm * l_dsm
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); lrs.step()
        ema_update(target, student, 0.0)      # official initial_ema_decay = 0.0
        run_c += float(l_ctm.item()); run_d += float(l_dsm.item())
        if step % log_interval == 0:
            print(f"[student] Step {step}/{student_steps} | Loss: {(w_ctm * run_c + w_dsm * run_d) / log_interval:.6f} | "
                  f"ctm {run_c / log_interval:.6f} dsm {run_d / log_interval:.6f} | LR: {lrs.get_last_lr()[0]:.2e} | "
                  f"{time.time() - t0:.0f}s", flush=True)
            run_c = run_d = 0.0
        if save_interval and step % save_interval == 0:
            torch.save(student.state_dict(), os.path.join(MODEL_SAVE_DIR, "cp_student.pt"))
    torch.save(student.state_dict(), os.path.join(MODEL_SAVE_DIR, "cp_student.pt"))
    print(f"Student done in {time.time() - t0:.0f}s. Saved to {MODEL_SAVE_DIR}")
    return 0


def _forever(loader):
    while True:
        for b in loader:
            yield b


def _index_batches(n: int, bs: int, device):
    """Epoch-wise shuffled index batches with drop_last, like the DataLoader it replaces."""
    if n < bs:
        raise SystemExit(f"cp_student_feature_cache: {n} samples < batch_size {bs}")
    while True:
        perm = torch.randperm(n, device=device)
        for i in range(0, n - bs + 1, bs):
            yield perm[i:i + bs]


@torch.no_grad()
def build_feature_cache(model: CPModel, dataset, device, cond_dim: int):
    if not is_pixels or model.encoder is None:
        raise SystemExit("cp_student_feature_cache is only for pixel environments")
    if (int(env_training.get("image_crop_size", 0) or 0) > 0 or bool(getattr(dataset, "augment", False))
            or bool(env_config.get("image_aug", False))):
        raise SystemExit("cp_student_feature_cache needs a deterministic dataset; this one augments "
                         "(image_crop_size / image_aug), so its features are not fixed")
    probe = sorted({0, len(dataset) // 2, len(dataset) - 1})
    for i in probe:
        a, b = dataset[i], dataset[i]
        if any(not np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a):
            raise SystemExit(f"cp_student_feature_cache: dataset[{i}] differs between two reads (augmentation?)")
    t0 = time.time()
    lo, hi = action_bounds
    model.encoder.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=num_workers)
    feats, acts = [], []
    for b in loader:
        model._cond = b["cond"].float().to(device) if cond_dim else None
        feats.append(model.encode(b["state"].float().to(device)))
        acts.append((b["action"].float().to(device) - lo) / (hi - lo) * 2.0 - 1.0)
    feats, acts = torch.cat(feats), torch.cat(acts)
    if feats.shape[0] != len(dataset):
        raise SystemExit(f"cp_student_feature_cache: cached {feats.shape[0]} rows for {len(dataset)} samples")
    for i in probe:
        smp = dataset[i]
        model._cond = torch.as_tensor(np.asarray(smp["cond"])).float()[None].to(device) if cond_dim else None
        f = model.encode(torch.as_tensor(np.asarray(smp["state"])).float()[None].to(device))[0]
        if not torch.allclose(f, feats[i], rtol=1e-3, atol=1e-4):
            raise SystemExit(f"cp_student_feature_cache: cached row {i} != fresh encode "
                             f"(max abs diff {(f - feats[i]).abs().max().item():.2e})")
    print(f"Student feature cache: {feats.shape[0]} samples x {feats.shape[1]} dims "
          f"({feats.numel() * 4 / 2**20:.0f} MiB) built in {time.time() - t0:.0f}s; matches a fresh encode")
    return feats, acts


def _param_groups(model: CPModel, lr: float, enc_scale: float):
    if model.encoder is None or enc_scale == 1.0:
        return [{"params": list(model.parameters()), "lr": lr}]
    enc = list(model.encoder.parameters())
    ids = {id(p) for p in enc}
    return [{"params": enc, "lr": lr * enc_scale}, {"params": [p for p in model.parameters() if id(p) not in ids], "lr": lr}]


def _save_teacher(teacher: CPModel, teacher_ema: CPModel) -> None:
    torch.save(teacher.state_dict(), os.path.join(MODEL_SAVE_DIR, "cp_teacher.pt"))
    torch.save(teacher_ema.state_dict(), os.path.join(MODEL_SAVE_DIR, "cp_teacher_ema.pt"))


def persist_norm_stats(dataset, cond_dim: int) -> None:
    """the shared norm_stats schema, so every simulation's obs/action handling works unchanged."""
    if not hasattr(dataset, "act_min"):
        return  # e.g. ParticleDataset: a partial norm_stats.pt would crash eval; the sim falls back to model range
    ns = {"act_min": dataset.act_min, "act_max": dataset.act_max,
          "action_norm_range": getattr(dataset, "action_norm_range", (-1.0, 1.0)),
          "frame_stack": frame_stack, "env_id": env_id,
          "action_chunk": int(getattr(dataset, "action_chunk", 1) or 1),
          "algorithm": "consistency_policy", "cond_dim": cond_dim, "state_shape": dataset.state_shape}
    if is_pixels:
        ss = list(getattr(dataset, "state_shape", []) or [])
        ns.update({"in_channels": int(ss[0]), "image_hw": [int(ss[1]), int(ss[2])], "state_shape": ss,
                   "encoder_target_height": env_config.get("encoder_target_height", 180),
                   "encoder_target_width": env_config.get("encoder_target_width", 240),
                   "encoder_feature_dim": int(env_model.get("encoder_feature_dim", 256)),
                   "encoder_kind": env_model.get("encoder_kind", "conv_maxpool"),
                   "encoder_pretrained": env_model.get("encoder_pretrained", False),
                   "encoder_num_kp": int(env_model.get("encoder_num_kp", 64)),
                   "encoder_norm_kind": env_model.get("encoder_norm_kind", "bn"),
                   "encoder_per_camera": bool(env_model.get("encoder_per_camera", False)),
                   "cond_fusion": env_model.get("cond_fusion", "concat")})
    if active_env == "kitchen":
        ns["obs_indices"] = getattr(dataset, "obs_indices", None)
    if active_env == "libero_goal_pixels":
        ns.update({"libero_obs_keys": dataset.libero_obs_keys, "libero_cameras": list(dataset.cameras),
                   "goal_embeddings": dataset.goal_embeddings, "goal_task_names": dataset.goal_task_names,
                   "goal_emb_dim": dataset.goal_emb_dim, "proprio_dim": dataset.proprio_dim,
                   "image_crop_size": int(env_training.get("image_crop_size", 0))})
    if hasattr(dataset, "obs_mean"):
        ns["obs_mean"], ns["obs_std"] = dataset.obs_mean, dataset.obs_std
    torch.save(ns, os.path.join(MODEL_SAVE_DIR, "norm_stats.pt"))


if __name__ == "__main__":
    raise SystemExit(main())
