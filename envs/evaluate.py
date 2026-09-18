"""Policy evaluation: load a trained checkpoint and measure task success.

One entry point, `evaluate`, covers every method in the paper — WiFI-BC, IBC,
Diffusion Policy, Consistency Policy and explicit BC (MSE) — because the method
is inferred from which weight files the checkpoint directory contains:

    control_point_generator.pt + q_estimator.pt   WiFI-BC
    q_estimator.pt only                           IBC (energy model, DFO/Langevin)
    denoiser.pt                                   Diffusion Policy
    cp_meta.json + cp_student/teacher.pt          Consistency Policy
    bc_policy.pt                                  explicit BC (MSE)

Every method is scored through the same simulation classes with the same seeds
and episode counts, so the numbers in the README's results table are produced by
this file for all rows.

Usage:
    uv run python -m envs.evaluate --checkpoint checkpoints/wifi_bc/particle
    uv run python -m envs.evaluate --checkpoint <dir> --env pushing --episodes 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# torch>=2.6 defaults torch.load to weights_only=True, which rejects the numpy
# arrays inside our norm_stats.pt (obs mean/std, action min/max, and the LIBERO
# goal-embedding matrix). add_safe_globals doesn't reliably fix it under numpy
# 2.x: the pickle stores the global as `numpy.core.multiarray._reconstruct`,
# but on numpy 2.x the real callable's module is `numpy._core.multiarray`, so
# torch's allowlist match misses. We trust our own checkpoints, so force every
# torch.load in this process to weights_only=False.
_ORIG_TORCH_LOAD = torch.load


def _trusted_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _ORIG_TORCH_LOAD(*args, **kwargs)


torch.load = _trusted_torch_load

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def effective_langevin_config(env_config: dict) -> dict:
    """Merge env_training langevin_* overrides onto env_model.langevin_config defaults.

    Returns a dict keyed by the native sample_langevin arg names (lr_init, etc.),
    so callers in both training and evaluation share one source of truth.
    """
    base = dict(env_config.get("model", {}).get("langevin_config", {}))
    training = env_config.get("training", {})
    overrides = {
        "num_iterations": "langevin_num_iterations",
        "lr_init": "langevin_lr_init",
        "lr_final": "langevin_lr_final",
        "noise_scale": "langevin_noise_scale",
        "delta_action_clip": "langevin_delta_clip",
        "polynomial_decay_power": "langevin_decay_power",
    }
    for native_key, training_key in overrides.items():
        if training_key in training:
            base[native_key] = training[training_key]
    return base


def effective_inference_langevin_config(env_config: dict) -> dict:
    """Inference-time Langevin config: training defaults overridden by inference_* keys.

    Lets callers run aggressive paper-faithful Langevin during training
    (for hard negatives) while using a gentler inference chain to refine
    actions on WiFI-BC's narrow-trained Q surface. Falls back to training values
    for any key not overridden, so an empty inference_* set = same as training.
    """
    cfg = effective_langevin_config(env_config)
    training = env_config.get("training", {})
    overrides = {
        "lr_init": "inference_langevin_lr_init",
        "lr_final": "inference_langevin_lr_final",
        "noise_scale": "inference_langevin_noise_scale",
        "delta_action_clip": "inference_langevin_delta_clip",
        "polynomial_decay_power": "inference_langevin_decay_power",
    }
    for native_key, inf_key in overrides.items():
        if inf_key in training:
            cfg[native_key] = training[inf_key]
    return cfg


class _NullCritic(torch.nn.Module):
    """Constant-score stand-in for a policy that has no critic.

    A plain diffusion policy is evaluated with a single candidate, so there is
    nothing to rank — but every simulation calls `q.encode(obs)` then
    `q.score(features, actions)` and takes an argmax. Returning zeros makes that
    argmax pick the one candidate, which leaves the diffusion sample untouched
    and keeps the simulations free of any special case.
    """

    def encode(self, images):
        return torch.zeros(images.shape[0], 1, device=images.device)

    def score(self, features, action):
        # (B, N, A) -> (B, N, 1);  (B, A) -> (B, 1)
        shape = action.shape[:-1] + (1,)
        return torch.zeros(shape, device=action.device)

    def forward(self, state, action):
        return self.score(None, action)


def _cp_weights_path(checkpoint_dir, env_config):
    """Which Consistency Policy weights to score: inference_cp_mode 'student' (default when a
    student exists) or 'teacher' (its EMA weights, as the official teacher is evaluated)."""
    et = env_config.get("training", {})
    student = os.path.join(checkpoint_dir, "cp_student.pt")
    mode = str(et.get("inference_cp_mode", "student" if os.path.exists(student) else "teacher"))
    if mode == "student":
        return student
    ema = os.path.join(checkpoint_dir, "cp_teacher_ema.pt")
    return ema if os.path.exists(ema) else os.path.join(checkpoint_dir, "cp_teacher.pt")


def _build_cp_generator(checkpoint_dir, weights_path, env_config, device, action_dim):
    """Rebuild a consistency_policy_training.py checkpoint from cp_meta.json and expose its sampler
    as a `control_point_generator(state) -> (B, 1, A)`: student = one jump T -> 0 (chained when
    inference_cp_chaining is e.g. 'D:27,54'), teacher = Heun over the full sigma grid."""
    from baselines.consistency import ConsistencyPolicyGenerator, CPModel, KarrasSchedule
    with open(os.path.join(checkpoint_dir, "cp_meta.json")) as fh:
        meta = json.load(fh)
    if int(meta["action_dim"]) != int(action_dim):
        raise RuntimeError(f"cp_meta action_dim {meta['action_dim']} != evaluation action_dim {action_dim}")
    is_student = os.path.basename(weights_path).startswith("cp_student")
    common = dict(cond_dim=int(meta.get("cond_dim", 0)), time_emb_dim=int(meta["time_emb_dim"]),
                  network_kind=meta["network_kind"], width=int(meta["width"]), depth=int(meta["depth"]),
                  two_times=is_student, dropout=float(meta.get("dropout", 0.0)) if is_student else 0.0)
    if meta["pixel"]:
        ek = dict(meta.get("encoder_kwargs") or {})
        ek["encoder_pretrained"] = False  # weights come from the state_dict; never download on a compute node
        model = CPModel(int(action_dim), in_channels=int(meta["in_channels"]), encoder_kwargs=ek, **common)
    else:
        model = CPModel(int(action_dim), state_dim=int(meta["state_dim"]), **common)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device).eval()
    sched = KarrasSchedule(sigma_min=meta["sigma_min"], sigma_max=meta["sigma_max"], rho=meta["rho"],
                           bins=meta["bins"], sigma_data=meta["sigma_data"])
    chaining = env_config.get("training", {}).get("inference_cp_chaining", "none") if is_student else None
    gen = ConsistencyPolicyGenerator(model, sched, int(action_dim), tuple(meta.get("action_bounds", [-1.0, 1.0])),
                                     mode="student" if is_student else "teacher", chaining=chaining)
    print(f"Consistency Policy: {'student' if is_student else 'teacher (Heun-' + str(meta['bins']) + ')'} "
          f"from {os.path.basename(weights_path)}"
          + (f", chaining={chaining}" if is_student else "") + f", action_dim={action_dim}")
    return gen.to(device)


def _build_diffusion_generator(weights_path, env_config, norm_stats, action_dim,
                           control_points, action_bounds, device, *, pixel,
                           plain_dp=False,
                           in_channels=None, cond_dim=0, state_dim=None,
                           encoder_target_height=180, encoder_target_width=240,
                           encoder_feature_dim=256, encoder_kind="conv_maxpool",
                           encoder_num_kp=64, encoder_norm_kind="bn",
                           encoder_per_camera=False):
    """Load a diffusion denoiser and expose it as a control-point generator.

    A diffusion policy only changes WHERE the candidate cloud comes from, so it
    is wrapped in the `cp_gen(state) -> (B, N, A)` signature the whole
    evaluation stack already speaks. Every simulation class then works on a
    diffusion checkpoint unchanged, and the single sample it draws is scored by
    the null critic.

    norm_stats is the authority on the sampler that was trained (it is what the
    trainer actually used), falling back to the config's training block.
    """
    from baselines.diffusion import (build_cond_pixel_denoiser, build_denoiser,
                                     build_diffusion, resolve_dp_params,
                                     DiffusionControlPointGenerator)

    dp = resolve_dp_params(env_config)
    for key in ("num_train_timesteps", "beta_schedule", "prediction_type",
                "time_emb_dim", "denoiser_network_kind", "denoiser_width",
                "denoiser_depth", "denoiser_use_spectral_norm"):
        if key in norm_stats:
            dp[key] = norm_stats[key]

    if pixel:
        denoiser = build_cond_pixel_denoiser(
            action_dim, in_channels, dp, cond_dim=cond_dim,
            encoder_target_height=encoder_target_height,
            encoder_target_width=encoder_target_width,
            encoder_feature_dim=encoder_feature_dim,
            encoder_kind=encoder_kind,
            # Weights come from the state_dict; never fetch ImageNet on a
            # compute node that may have no network.
            encoder_pretrained=False,
            encoder_num_kp=encoder_num_kp,
            encoder_norm_kind=encoder_norm_kind,
            encoder_per_camera=encoder_per_camera,
            device=device)
    else:
        denoiser = build_denoiser(state_dim, action_dim, dp, device=device)
    denoiser.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True))
    denoiser.to(device).eval()

    et = env_config.get("training", {})
    # Eval-only cloud size; falls back to the trained-against value.
    control_points = int(et.get("inference_control_points", control_points))
    if plain_dp:
        # No critic exists, so more than one candidate could not be chosen
        # between. One sample IS the diffusion policy's action.
        control_points = 1
    # Eval sampler knobs. Default the step count to the first entry the trainer
    # recorded in ddim_eval_steps, so a trial evaluates at the schedule it was
    # set up for rather than a hardcoded guess.
    ddim_default = norm_stats.get("ddim_eval_steps", dp.get("ddim_eval_steps", [10]))
    num_steps = int(et.get("inference_dp_iters",
                           (ddim_default[0] if ddim_default else 10)))
    method = str(et.get("inference_dp_method", "ddim"))
    eta = float(et.get("inference_dp_eta",
                       norm_stats.get("ddim_eta", dp.get("ddim_eta", 0.0))))
    print(f"Diffusion CP generator — {control_points} candidates via "
          f"{method} x{num_steps} (eta={eta}), action_dim={action_dim}")
    gen = DiffusionControlPointGenerator(
        denoiser, build_diffusion(dp, device, action_bounds),
        control_points, action_dim, num_steps=num_steps, eta=eta, method=method)
    return gen.to(device).eval()


def evaluate(checkpoint_dir: str, config: dict) -> dict:
    """Load the policy in *checkpoint_dir* and measure its task performance.

    The method is inferred from the weight files present (see the module
    docstring). Returns a dict with `success_rate`, `avg_reward` and per-seed
    detail; the exact extra keys depend on the environment (kitchen adds
    `avg_tasks_completed`, particle adds goal distances).
    """
    from wifi_bc.models import ControlPointGenerator, QEstimator
    from wifi_bc.sampling import sample_langevin

    active_env = config.get("active_env", "particle")
    env_config = config["environments"][active_env]
    sim_config = config.get("simulation", {})

    # Pick the right simulation class. `pushing` uses the vendored IBC env
    # (PyBullet + gym), which lives behind the `pushing` optional-extras and
    # is NOT installed unless the `pushing` extra was requested. Keep the
    # import lazy so a particle or pen run never touches the pushing deps.
    if active_env == "pushing":
        from envs.pushing_simulation import PushingSimulation
        SimulationCls = PushingSimulation
    elif active_env == "pushing_pixels":
        from envs.pushing_pixels_simulation import PushingPixelsSimulation
        SimulationCls = PushingPixelsSimulation
    elif active_env == "pen":
        from envs.pen_human_v2_simulation import PenHumanV2Simulation
        SimulationCls = PenHumanV2Simulation
    elif active_env == "kitchen":
        from envs.kitchen_simulation import KitchenSimulation
        SimulationCls = KitchenSimulation
    elif active_env == "libero_goal_pixels":
        from envs.libero_goal_pixels_simulation import LiberoGoalPixelsSimulation
        SimulationCls = LiberoGoalPixelsSimulation
    elif active_env == "point_maze_pillar":
        from envs.point_maze_pillar_simulation import PointMazePillarSimulation
        SimulationCls = PointMazePillarSimulation
    else:
        from envs.particle_simulation import ParticleSimulation
        SimulationCls = ParticleSimulation

    state_dim = env_config["state_dim"]
    action_dim = env_config["action_dim"]
    frame_stack = env_config.get("frame_stack", 1)
    action_bounds = tuple(env_config.get("action_bounds", [0, 1]))
    n_dim = env_config.get("n_dim", 2)
    em = env_config["model"]
    control_points = em["control_points"]
    num_hidden_layers = em["num_hidden_layers"]
    num_neurons = em["num_neurons"]
    use_spectral_norm = em.get("use_spectral_norm", False)
    # Per-net architecture (mirrors training/wifi_bc_training.py).
    q_network_kind = em.get("q_network_kind", "mlp")
    q_width = em.get("q_width", num_neurons)
    q_depth = em.get("q_depth", num_hidden_layers)
    q_use_spectral_norm = em.get("q_use_spectral_norm", use_spectral_norm)
    cp_network_kind = em.get("cp_network_kind", "mlp")
    cp_width = em.get("cp_width", num_neurons)
    cp_depth = em.get("cp_depth", num_hidden_layers)
    cp_use_spectral_norm = em.get("cp_use_spectral_norm", False)
    cp_output_activation = em.get("cp_output_activation", "tanh")
    # Per-env override wins over the shared simulation.max_episode_steps.
    # Pushing needs 100 (IBC paper BlockPush-v0); particle uses the global 50.
    max_episode_steps = env_config.get(
        "max_episode_steps", sim_config.get("max_episode_steps", 50)
    )
    # IBC Table 3 reports simulated pushing over 100 evaluation episodes per
    # training seed. Keep this env-scoped so particle's established eval count
    # does not change.
    num_seeds = int(
        env_config.get(
            "num_eval_seeds",
            sim_config.get("num_seeds", len(sim_config.get("default_seeds", [0]))),
        )
    )
    if num_seeds <= 0:
        raise ValueError("simulation.num_seeds must be >= 1")
    seeds = list(range(num_seeds))

    inference_langevin_iterations = int(
        env_config.get("training", {}).get("inference_langevin_iterations", 0)
    )
    # CP-DFO refinement (WiFI-BC-specific, no IBC analog). Takes precedence
    # over inference Langevin when > 0, so a trial can opt into either path
    # without changing the rest of the recipe.
    inference_dfo_iterations = int(
        env_config.get("training", {}).get("inference_dfo_iterations", 0)
    )
    inference_dfo_iteration_std = float(
        env_config.get("training", {}).get("inference_dfo_iteration_std", 0.1)
    )
    inference_dfo_iteration_std_decay = float(
        env_config.get("training", {}).get("inference_dfo_iteration_std_decay", 0.7)
    )
    inference_dfo_num_uniform = int(
        env_config.get("training", {}).get("inference_dfo_num_uniform", 0)
    )
    # Elitist CP-DFO: the untouched initial candidates re-enter every resample and
    # the final argmax, so jitter can never erase a precise control point. The
    # plain procedure jitters every candidate each iteration and on particle-16D
    # scores 0% even with a critic whose argmax gets 86.7%.
    inference_dfo_elitist = bool(env_config.get("training", {}).get("inference_dfo_elitist", False))
    if inference_dfo_elitist and inference_dfo_iterations > 0 and active_env in ("pushing_pixels", "libero_goal_pixels"):
        raise NotImplementedError("inference_dfo_elitist is implemented for flat-state envs only; "
                                  "the pixel DFO loops have not been patched")
    # Ablation control: discard the learned proposal at evaluation and draw the
    # same number of candidates uniformly from the action box. Isolates how much
    # of the method's performance comes from the control-point generator rather
    # than from scoring a small candidate set at all. Training is untouched.
    inference_uniform_proposal = bool(
        env_config.get("training", {}).get("inference_uniform_proposal", False)
    )
    # Effective langevin hyperparams for INFERENCE chain. Starts from training
    # Langevin config (env_model.langevin_config + langevin_* training overrides),
    # then applies any inference_langevin_* overrides on top. Lets eval use
    # gentler step sizes than training while keeping training paper-faithful.
    langevin_cfg = effective_inference_langevin_config(env_config)
    # Official-IBC-faithful chain extras (audit fixes; see ibc-repro-fixes).
    inference_langevin_noise_via_stepsize = bool(
        env_config.get("training", {}).get("inference_langevin_noise_via_stepsize", False)
    )
    inference_langevin_again_iterations = int(
        env_config.get("training", {}).get("inference_langevin_again_iterations", 0)
    )
    inference_langevin_again_noise_scale = float(
        env_config.get("training", {}).get("inference_langevin_again_noise_scale", 0.5)
    )
    inference_langevin_top_k = int(
        env_config.get("training", {}).get("inference_langevin_top_k", 0)
    )

    # diffusion_policy_training.py writes a denoiser where WiFI-BC writes a
    # control-point generator. Everything else about the evaluation is
    # unchanged, so detect it by which proposal file is on disk and swap only
    # the generator build below.
    dp_raw_path = os.path.join(checkpoint_dir, "denoiser.pt")
    dp_ema_path = os.path.join(checkpoint_dir, "denoiser_ema.pt")
    is_diffusion = os.path.exists(dp_raw_path) or os.path.exists(dp_ema_path)

    # bc_mse_training.py writes a single regression network and no critic. It is
    # the explicit-BC baseline, so there is exactly one candidate per state:
    # evaluate it like a plain diffusion policy (null critic, cloud of one) but
    # with a BCPolicy in place of the sampler.
    bc_raw_path = os.path.join(checkpoint_dir, "bc_policy.pt")
    bc_ema_path = os.path.join(checkpoint_dir, "bc_policy_ema.pt")
    is_bc = os.path.exists(bc_raw_path) or os.path.exists(bc_ema_path)

    # consistency_policy_training.py writes an EDM teacher and/or a CTM student plus
    # cp_meta.json, and no critic: evaluate like a plain diffusion policy (cloud of
    # one, null critic) with the sampler chosen by inference_cp_mode.
    is_cp = os.path.exists(os.path.join(checkpoint_dir, "cp_meta.json")) and any(
        os.path.exists(os.path.join(checkpoint_dir, f)) for f in ("cp_student.pt", "cp_teacher.pt", "cp_teacher_ema.pt"))

    # IBC writes ONLY an energy model: no proposal network of any kind. Its
    # candidates come from uniform sampling refined by Langevin, which is a
    # different inference procedure rather than a different generator, so it
    # gets its own evaluator in baselines/ibc.py.
    is_ibc = not (is_diffusion or is_bc or is_cp) and not os.path.exists(
        os.path.join(checkpoint_dir, "control_point_generator.pt")
    ) and os.path.exists(os.path.join(checkpoint_dir, "q_estimator.pt"))
    if is_ibc:
        from baselines.ibc import evaluate_ibc_checkpoint
        print("IBC energy model detected (q_estimator.pt, no proposal network): "
              "evaluating with uniform-init Langevin inference.")
        # IBC has its own inference settings (a sample count and the official
        # implementation's chain extras), so it reads them from the env's `ibc`
        # block rather than reusing the WiFI-BC refinement config above.
        return evaluate_ibc_checkpoint(
            os.path.join(checkpoint_dir, "q_estimator.pt"),
            num_seeds=num_seeds,
            active_env=active_env,
        )

    if is_bc:
        cp_raw_path, cp_ema_path = bc_raw_path, bc_ema_path
    elif is_cp:
        cp_raw_path = cp_ema_path = _cp_weights_path(checkpoint_dir, env_config)
    elif is_diffusion:
        cp_raw_path, cp_ema_path = dp_raw_path, dp_ema_path
    else:
        cp_raw_path = os.path.join(checkpoint_dir, "control_point_generator.pt")
        cp_ema_path = os.path.join(checkpoint_dir, "control_point_generator_ema.pt")
    q_raw_path = os.path.join(checkpoint_dir, "q_estimator.pt")
    q_ema_path = os.path.join(checkpoint_dir, "q_estimator_ema.pt")

    # A PLAIN diffusion policy (diffusion_policy_training.py) writes a denoiser
    # and NO critic, so requiring q_estimator.pt rejected it before anything was
    # loaded — which is why every dpParticle job recorded "Checkpoints not
    # found" after training successfully for 300k steps. Evaluate it with a
    # cloud of ONE (nothing to rank) and a constant-score stub critic, so the
    # simulations' `cp_gen(obs)` then `q.score(...)` call pattern works unchanged
    # and argmax over a single candidate trivially returns it.
    is_plain_dp = (is_diffusion or is_bc or is_cp) and not (
        os.path.exists(q_raw_path) or os.path.exists(q_ema_path))

    if is_bc or is_cp:
        # One candidate, no energy to climb: any refinement loop would be a
        # no-op that still costs a forward pass per iteration, and top-k over a
        # single action is meaningless. Hard-off so a stray config value cannot
        # silently change what "BC" means.
        control_points = 1
        inference_dfo_iterations = 0
        inference_langevin_iterations = 0
        inference_langevin_again_iterations = 0

    eval_ema_decay = float(env_config.get("training", {}).get("ema_decay", 0.0))
    use_ema = (
        eval_ema_decay > 0.0
        and os.path.exists(cp_ema_path)
        and (is_plain_dp or os.path.exists(q_ema_path))
    )
    cp_path = cp_ema_path if use_ema else cp_raw_path
    q_path = q_ema_path if use_ema else q_raw_path
    norm_stats_path = os.path.join(checkpoint_dir, "norm_stats.pt")

    if not os.path.exists(cp_path) or (not is_plain_dp and not os.path.exists(q_path)):
        return {
            "success_rate": 0.0,
            "avg_reward": 0.0,
            "error": f"Checkpoints not found in {checkpoint_dir}",
        }
    if is_cp:
        print(f"Consistency Policy detected (cp_meta.json): scoring {os.path.basename(cp_raw_path)} with a null critic.")
    elif is_bc:
        print("Explicit BC policy detected (bc_policy, no critic): evaluating "
              "its single regressed action with a null critic.")
    elif is_plain_dp:
        print("Plain diffusion policy detected (denoiser, no critic): evaluating "
              "with a 1-candidate cloud and a null critic.")
    if eval_ema_decay > 0.0 and not use_ema:
        print(
            f"Warning: EMA requested (decay={eval_ema_decay}) but paired EMA "
            "checkpoints are missing; evaluating raw weights."
        )
    elif use_ema:
        print(f"Evaluating WiFI-BC EMA weights (decay={eval_ema_decay}).")

    # Presence of norm_stats.pt = ibc_with_cps (actions normalized to [0,1]
    # before the Q estimator sees them).
    norm_stats = None
    if os.path.exists(norm_stats_path):
        norm_stats = torch.load(norm_stats_path, weights_only=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if active_env in ("pushing_pixels", "libero_goal_pixels"):
        from wifi_bc.models import PixelControlPointGenerator, PixelQEstimator
        # in_channels MUST come from the checkpoint's norm_stats when present:
        # the config's static state_dim assumes frame_stack=1 (6 channels), but
        # fs=2 trials trained with 12 — rebuilding from config broke every
        # frame_stack=2 eval in Dstandardlibero (head 1432 vs 920 mismatch).
        in_channels = int((norm_stats or {}).get("in_channels", state_dim[0]))
        enc_h = int(env_config.get("encoder_target_height", 180))
        enc_w = int(env_config.get("encoder_target_width", 240))
        value_width = int(em.get("value_width", 1024))
        value_num_blocks = int(em.get("value_num_blocks", 1))
        # libero_goal_pixels conditions on proprio+goal (cond_dim from norm_stats).
        cond_dim = int(norm_stats["cond_dim"]) if (norm_stats and "cond_dim" in norm_stats) else 0
        # Encoder architecture: read from norm_stats (what training used), falling
        # back to the config model block for older checkpoints.
        _ns = norm_stats or {}
        encoder_kind = _ns.get("encoder_kind", em.get("encoder_kind", "conv_maxpool"))
        # Always rebuild with pretrained=False: the trained weights come from the
        # checkpoint's state_dict anyway, and compute nodes may have no network
        # to download ImageNet weights (they'd be overwritten regardless).
        encoder_pretrained = False
        encoder_num_kp = int(_ns.get("encoder_num_kp", em.get("encoder_num_kp", 64)))
        encoder_norm_kind = _ns.get("encoder_norm_kind", em.get("encoder_norm_kind", "bn"))
        encoder_per_camera = bool(_ns.get("encoder_per_camera", em.get("encoder_per_camera", False)))
        cond_fusion = _ns.get("cond_fusion", em.get("cond_fusion", "concat"))
        goal_dim = int(_ns.get("goal_emb_dim", 0))
        # Action chunking: model output = action_dim * K per CP.
        action_chunk = int(_ns.get("action_chunk", 1) or 1)
        action_dim_eff = action_dim * action_chunk
        if is_bc:
            from wifi_bc.models import BCPolicy
            cp_gen = BCPolicy(
                action_dim_eff,
                in_channels=in_channels,
                cond_dim=cond_dim,
                width=int(_ns.get("bc_width", em.get("bc_width", em.get("cp_width", 256)))),
                depth=int(_ns.get("bc_depth", em.get("bc_depth", em.get("cp_depth", 2)))),
                action_bounds=(action_bounds[0], action_bounds[1]),
                encoder_target_height=enc_h,
                encoder_target_width=enc_w,
                encoder_feature_dim=int(_ns.get("encoder_feature_dim", 256)),
                encoder_kind=encoder_kind,
                encoder_pretrained=encoder_pretrained,
                encoder_num_kp=encoder_num_kp,
                encoder_norm_kind=encoder_norm_kind,
                encoder_per_camera=encoder_per_camera,
                cond_fusion=cond_fusion,
                goal_dim=goal_dim,
            )
            cp_gen.load_state_dict(torch.load(cp_path, map_location=device, weights_only=True))
            cp_gen.to(device).eval()
        elif is_cp:
            cp_gen = _build_cp_generator(checkpoint_dir, cp_path, env_config, device, action_dim_eff)
        elif is_diffusion:
            cp_gen = _build_diffusion_generator(
                cp_path, env_config, norm_stats or {}, action_dim_eff,
                control_points, action_bounds, device,
                pixel=True, plain_dp=is_plain_dp,
                in_channels=in_channels, cond_dim=cond_dim,
                encoder_target_height=enc_h, encoder_target_width=enc_w,
                encoder_feature_dim=int(_ns.get("encoder_feature_dim", 256)),
                encoder_kind=encoder_kind, encoder_num_kp=encoder_num_kp,
                encoder_norm_kind=encoder_norm_kind,
                encoder_per_camera=encoder_per_camera,
            )
        else:
            cp_gen = PixelControlPointGenerator(
                output_dim=action_dim_eff,
                control_points=control_points,
                hidden_dims=[cp_width] * cp_depth,
                action_bounds=action_bounds,
                network_kind=cp_network_kind,
                width=cp_width,
                depth=cp_depth,
                use_spectral_norm=cp_use_spectral_norm,
                in_channels=in_channels,
                encoder_target_height=enc_h,
                encoder_target_width=enc_w,
                cond_dim=cond_dim,
                encoder_kind=encoder_kind,
                encoder_pretrained=encoder_pretrained,
                encoder_num_kp=encoder_num_kp,
                encoder_norm_kind=encoder_norm_kind,
                encoder_per_camera=encoder_per_camera,
                cond_fusion=cond_fusion,
                goal_dim=goal_dim,
                output_activation=cp_output_activation,
            )
            cp_gen.load_state_dict(torch.load(cp_path, map_location=device, weights_only=True))
            cp_gen.to(device).eval()

        q_est = PixelQEstimator(
            action_dim=action_dim_eff,
            in_channels=in_channels,
            encoder_target_height=enc_h,
            encoder_target_width=enc_w,
            value_width=value_width,
            value_num_blocks=value_num_blocks,
            cond_dim=cond_dim,
            encoder_kind=encoder_kind,
            encoder_pretrained=encoder_pretrained,
            encoder_num_kp=encoder_num_kp,
            encoder_norm_kind=encoder_norm_kind,
            encoder_per_camera=encoder_per_camera,
            cond_fusion=cond_fusion,
            goal_dim=goal_dim,
        )
        if is_plain_dp:
            q_est = _NullCritic().to(device).eval()
        else:
            q_est.load_state_dict(torch.load(q_path, map_location=device, weights_only=True))
            q_est.to(device).eval()
    else:
        # libero_goal bakes the goal embedding into the state AFTER frame-stacking,
        # so its input dim is NOT state_dim*frame_stack. Same for kitchen when
        # kitchen_qpos_only shrank the obs below config's state_dim. Read the
        # exact length the dataset used straight from norm_stats.
        if active_env in ("kitchen",) and norm_stats is not None and "state_shape" in norm_stats:
            flat_input_dim = int(norm_stats["state_shape"])
        else:
            flat_input_dim = state_dim * frame_stack
        # Action chunking: the model was trained on K-step chunk targets, so
        # its output/action dim is action_dim*K (norm_stats carries K).
        flat_action_chunk = int((norm_stats or {}).get("action_chunk", 1) or 1)
        flat_action_dim = action_dim * flat_action_chunk
        if is_bc:
            from wifi_bc.models import BCPolicy
            _ns = norm_stats or {}
            cp_gen = BCPolicy(
                flat_action_dim,
                state_dim=flat_input_dim,
                cond_dim=int(_ns.get("cond_dim", 0)),
                width=int(_ns.get("bc_width", em.get("bc_width", em.get("cp_width", 256)))),
                depth=int(_ns.get("bc_depth", em.get("bc_depth", em.get("cp_depth", 2)))),
                action_bounds=(action_bounds[0], action_bounds[1]),
            )
            cp_gen.load_state_dict(torch.load(cp_path, map_location=device, weights_only=True))
            cp_gen.to(device).eval()
        elif is_cp:
            cp_gen = _build_cp_generator(checkpoint_dir, cp_path, env_config, device, flat_action_dim)
        elif is_diffusion:
            cp_gen = _build_diffusion_generator(
                cp_path, env_config, norm_stats or {}, flat_action_dim,
                control_points, action_bounds, device,
                pixel=False, plain_dp=is_plain_dp, state_dim=flat_input_dim,
            )
        else:
            cp_gen = ControlPointGenerator(
                input_dim=flat_input_dim,
                output_dim=flat_action_dim,
                control_points=control_points,
                hidden_dims=[cp_width] * cp_depth,
                action_bounds=action_bounds,
                network_kind=cp_network_kind,
                width=cp_width,
                depth=cp_depth,
                use_spectral_norm=cp_use_spectral_norm,
                output_activation=cp_output_activation,
            )
            cp_gen.load_state_dict(
                torch.load(cp_path, map_location=device, weights_only=True)
            )
            cp_gen.to(device).eval()

        q_est = QEstimator(
            state_dim=flat_input_dim,
            action_dim=flat_action_dim,
            hidden_dims=[q_width] * q_depth,
            use_spectral_norm=q_use_spectral_norm,
            network_kind=q_network_kind,
            width=q_width,
            depth=q_depth,
            resnet_final_activation=bool(em.get("q_resnet_final_activation", True)),
        )
        if is_plain_dp:
            q_est = _NullCritic()
        else:
            q_est.load_state_dict(
                torch.load(q_path, map_location=device, weights_only=True)
            )
        q_est.to(device).eval()

    # ── Pixel envs: dedicated late-fused DFO / Langevin refinement ────────
    # The flat-state wrappers below assume `obs.unsqueeze(1).expand(-1, N, -1)`
    # is cheap — that's true for vector obs, but for images it would re-encode
    # the (1, C, H, W) tensor N (DFO) or 100 (Langevin) times PER ENV STEP.
    # Instead we encode ONCE per step, cache the 256-D features, and run the
    # refinement inner loop against PixelQEstimator.score(features, actions).
    # This is what IBC's `late_fusion = True` config flag does upstream.
    if active_env == "pushing_pixels":
        if inference_dfo_iterations > 0:
            _dfo_iters = inference_dfo_iterations
            _dfo_std0 = inference_dfo_iteration_std
            _dfo_decay = inference_dfo_iteration_std_decay
            _dfo_n_uniform = inference_dfo_num_uniform

            class PixelDFORefinedSimulation(SimulationCls):
                """Pixel-aware CP-DFO refinement (encode once per env step).

                Same algorithm as DFORefinedSimulation below — initial pop = CP
                cloud (+ optional uniform safety samples); each iter resamples
                via category-ordered softmax(Q) and jitters — but the Q forward
                calls run against cached image features instead of re-encoding.
                """

                def select_action(self, observation, return_q_range: bool = False):
                    obs_tensor = self._obs_to_tensor(observation)  # (1, C, H, W) uint8

                    with torch.no_grad():
                        features = self.q_estimator.encode(obs_tensor)  # (1, F)
                        cps = self.control_point_generator(obs_tensor)  # (1, N_cp, A)

                        if _dfo_n_uniform > 0:
                            unif = torch.empty(
                                1, _dfo_n_uniform, cps.shape[-1], device=self.device
                            ).uniform_(float(action_bounds[0]), float(action_bounds[1]))
                            candidates = torch.cat([cps, unif], dim=1)
                        else:
                            candidates = cps.clone()

                        N = candidates.shape[1]
                        std = float(_dfo_std0)
                        for it in range(_dfo_iters):
                            log_probs = self.q_estimator.score(features, candidates).squeeze(-1)  # (1, N)
                            probs = torch.softmax(log_probs.squeeze(0), dim=-1)
                            idx = torch.multinomial(probs, N, replacement=True)
                            counts = torch.bincount(idx, minlength=N)
                            repeat_idx = torch.repeat_interleave(
                                torch.arange(N, device=self.device), counts
                            )
                            candidates = candidates[:, repeat_idx, :]
                            if it < _dfo_iters - 1:
                                candidates = candidates + torch.randn_like(candidates) * std
                                candidates = candidates.clamp(
                                    float(action_bounds[0]), float(action_bounds[1])
                                )
                                std *= _dfo_decay
                        # Re-score after the final reorder so argmax index aligns
                        # with the (reordered) candidates tensor — same fix as
                        # the flat-state DFORefinedSimulation.
                        final_log_probs = self.q_estimator.score(features, candidates).squeeze(-1)
                        sel = final_log_probs.argmax(dim=1)
                        action_normalized = candidates[0, sel[0], :].cpu().numpy()
                        q_range = (final_log_probs.min().item(), final_log_probs.max().item())

                    action = np.clip(action_normalized, action_bounds[0], action_bounds[1])
                    action = self._denormalize_action(action)
                    if return_q_range:
                        return action, q_range
                    return action

            sim_cls = PixelDFORefinedSimulation

        elif inference_langevin_iterations > 0:
            class PixelLangevinRefinedSimulation(SimulationCls):
                """Pixel-aware Langevin refinement (encode once per env step).

                Encodes the (1, C, H, W) image once into 256-D features, picks
                the argmax-Q CP as the starting action, then runs Langevin MCMC
                on actions against the cached features. The energy_function
                ignores `sample_langevin`'s expanded-obs argument and uses the
                closed-over `features` tensor instead — that's how we get the
                speedup vs the flat-state wrapper.
                """

                def select_action(self, observation, return_q_range: bool = False):
                    obs_tensor = self._obs_to_tensor(observation)  # (1, C, H, W) uint8

                    with torch.no_grad():
                        features = self.q_estimator.encode(obs_tensor)  # (1, F)
                        cps = self.control_point_generator(obs_tensor)  # (1, N_cp, A)
                        q_values = self.q_estimator.score(features, cps).squeeze(-1)  # (1, N)
                        best_idx = q_values.argmax(dim=1)
                        q_range = (q_values.min().item(), q_values.max().item())
                        best_cp = cps[0, best_idx[0], :].view(1, 1, -1).clone()  # (1, 1, A)

                    act_min_t = torch.full(
                        (cps.shape[-1],), float(action_bounds[0]), device=self.device
                    )
                    act_max_t = torch.full(
                        (cps.shape[-1],), float(action_bounds[1]), device=self.device
                    )

                    for p in self.q_estimator.parameters():
                        p.requires_grad_(False)

                    # Closed over `features` — the loop uses the cached encoding,
                    # not sample_langevin's expanded `obs_lv` arg (we just need
                    # to accept its signature).
                    def _neg_energy_fn(obs_lv, actions_lv):
                        return -self.q_estimator.score(features, actions_lv).squeeze(-1)

                    refined = sample_langevin(
                        energy_function=_neg_energy_fn,
                        observations=features,  # (1, F) — expanded internally, ignored by our fn
                        num_samples=1,
                        action_min=act_min_t,
                        action_max=act_max_t,
                        num_iterations=inference_langevin_iterations,
                        lr_init=float(langevin_cfg.get("lr_init", 0.1)),
                        lr_final=float(langevin_cfg.get("lr_final", 1e-5)),
                        polynomial_decay_power=float(
                            langevin_cfg.get("polynomial_decay_power", 2.0)
                        ),
                        delta_action_clip=float(
                            langevin_cfg.get("delta_action_clip", 0.1)
                        ),
                        noise_scale=float(langevin_cfg.get("noise_scale", 1.0)),
                        initial_actions=best_cp,
                        device=self.device,
                    )

                    for p in self.q_estimator.parameters():
                        p.requires_grad_(True)

                    action = refined[0, 0, :].cpu().numpy()
                    action = np.clip(action, action_bounds[0], action_bounds[1])
                    action = self._denormalize_action(action)
                    if return_q_range:
                        return action, q_range
                    return action

            sim_cls = PixelLangevinRefinedSimulation

        else:
            sim_cls = SimulationCls

    # libero_goal_pixels: the same late-fused refinement, but its simulation has
    # a different select_action contract — it takes the raw obs dict, builds
    # (image, cond) itself and must set `_cond` on both nets before any forward.
    # Without this branch the inference_langevin_* / inference_dfo_* overrides
    # were silently ignored on libero and every "refined" run returned the
    # unrefined number.
    elif active_env == "libero_goal_pixels" and (
            inference_langevin_iterations > 0 or inference_dfo_iterations > 0):
        _lv_iters = inference_langevin_iterations
        _dfo_iters_l = inference_dfo_iterations
        _dfo_std0_l = inference_dfo_iteration_std
        _dfo_decay_l = inference_dfo_iteration_std_decay

        class LiberoRefinedSimulation(SimulationCls):
            """Encode once, then refine the chosen control point on cached features.

            DFO takes precedence over Langevin when both are set, matching the
            flat-state wrappers.
            """

            def select_action(self, live_obs):
                img_t, cond_t = self._build_inputs(live_obs)
                self.control_point_generator._cond = cond_t
                self.q_estimator._cond = cond_t
                A = None
                with torch.no_grad():
                    feats = self.q_estimator.encode(img_t)
                    cps = self.control_point_generator(img_t)
                    A = cps.shape[-1]
                    qv = self.q_estimator.score(feats, cps).squeeze(-1)

                amin = torch.full((A,), float(action_bounds[0]), device=self.device)
                amax = torch.full((A,), float(action_bounds[1]), device=self.device)

                if _dfo_iters_l > 0:
                    with torch.no_grad():
                        cand = cps.clone()
                        N = cand.shape[1]
                        std = float(_dfo_std0_l)
                        for it in range(_dfo_iters_l):
                            lp = self.q_estimator.score(feats, cand).squeeze(-1)
                            idx = torch.multinomial(
                                torch.softmax(lp.squeeze(0), dim=-1), N, replacement=True)
                            cand = cand[:, idx, :]
                            if it < _dfo_iters_l - 1:
                                cand = (cand + torch.randn_like(cand) * std).clamp(
                                    float(action_bounds[0]), float(action_bounds[1]))
                                std *= _dfo_decay_l
                        final = self.q_estimator.score(feats, cand).squeeze(-1)
                        act = cand[0, int(final.argmax(dim=1)[0]), :].cpu().numpy()
                else:
                    best = cps[0, int(qv.argmax(dim=1)[0]), :].view(1, 1, -1).clone()
                    for prm in self.q_estimator.parameters():
                        prm.requires_grad_(False)

                    def _neg_energy_fn(obs_lv, actions_lv):
                        return -self.q_estimator.score(feats, actions_lv).squeeze(-1)

                    refined = sample_langevin(
                        energy_function=_neg_energy_fn, observations=feats,
                        num_samples=1, action_min=amin, action_max=amax,
                        num_iterations=_lv_iters,
                        lr_init=float(langevin_cfg.get("lr_init", 0.1)),
                        lr_final=float(langevin_cfg.get("lr_final", 1e-5)),
                        polynomial_decay_power=float(langevin_cfg.get("polynomial_decay_power", 2.0)),
                        delta_action_clip=float(langevin_cfg.get("delta_action_clip", 0.1)),
                        noise_scale=float(langevin_cfg.get("noise_scale", 1.0)),
                        initial_actions=best, device=self.device)
                    for prm in self.q_estimator.parameters():
                        prm.requires_grad_(True)
                    act = refined[0, 0, :].cpu().numpy()

                act = np.clip(act, action_bounds[0], action_bounds[1])
                return self._denormalize_action(act)

        sim_cls = LiberoRefinedSimulation


    elif inference_dfo_iterations > 0:
        # ── CP-DFO refinement (WiFI-BC inference). Cheaper than Langevin: no
        # autograd, only N small-batch forward passes through the Q-net.
        # Initial population = CP cloud (+ optional N_uniform random
        # samples). Each iter: score → category-ordered resample with
        # softmax(Q) → small Gaussian jitter → clip. Mirrors IBC's
        # `iterative_dfo` mechanics (see `bench_inference.iterative_dfo_pass`)
        # but with a model-trained initial population.
        _dfo_iters = inference_dfo_iterations
        _dfo_std0 = inference_dfo_iteration_std
        _dfo_decay = inference_dfo_iteration_std_decay
        _dfo_n_uniform = inference_dfo_num_uniform
        _dfo_elitist = inference_dfo_elitist

        class DFORefinedSimulation(SimulationCls):
            """Refines the CP cloud with iterative DFO before acting."""

            def select_action(self, observation, return_q_range: bool = False):
                obs_tensor = (
                    torch.tensor(observation, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
                obs_tensor = self.obs_normalizer.normalize(obs_tensor)

                with torch.no_grad():
                    cps = self.control_point_generator(obs_tensor)  # (1, N_cp, D)

                    # Action normalization helper (matches the Langevin path).
                    def _norm(a):
                        if self._act_min_t is not None:
                            return (a - self._act_min_t) / self._act_rng_t
                        return a

                    # Mix in uniform safety samples if requested.
                    if _dfo_n_uniform > 0:
                        unif = torch.empty(
                            1, _dfo_n_uniform, cps.shape[-1], device=self.device
                        ).uniform_(float(action_bounds[0]), float(action_bounds[1]))
                        candidates = torch.cat([cps, unif], dim=1)
                    else:
                        candidates = cps.clone()

                    N = candidates.shape[1]
                    obs_expanded = obs_tensor.unsqueeze(1).expand(-1, N, -1)
                    std = float(_dfo_std0)
                    _orig = candidates.clone()
                    for it in range(_dfo_iters):
                        # Elitist: the initial candidates re-enter every resample.
                        pool = torch.cat([candidates, _orig], dim=1) if _dfo_elitist else candidates
                        log_probs = self.q_estimator(
                            obs_tensor.unsqueeze(1).expand(-1, pool.shape[1], -1), _norm(pool)
                        ).squeeze(-1)
                        probs = torch.softmax(log_probs.squeeze(0), dim=-1)
                        # IBC-style category-ordered resample.
                        idx = torch.multinomial(probs, N, replacement=True)
                        if _dfo_elitist:
                            candidates = pool[:, idx, :]
                        else:
                            counts = torch.bincount(idx, minlength=N)
                            repeat_idx = torch.repeat_interleave(
                                torch.arange(N, device=self.device), counts
                            )
                            candidates = candidates[:, repeat_idx, :]
                        if it < _dfo_iters - 1:
                            candidates = candidates + torch.randn_like(candidates) * std
                            candidates = candidates.clamp(
                                float(action_bounds[0]), float(action_bounds[1])
                            )
                            std *= _dfo_decay
                    # FIX: re-score AFTER the final reorder so argmax index lines
                    # up with the (now reordered) candidates tensor. The previous
                    # version used log_probs from the iteration's pre-reorder
                    # scoring and indexed into the reordered candidates, picking
                    # the wrong action when softmax mass was spread (the bug was
                    # masked on pushing where Q is sharply peaked).
                    if _dfo_elitist:
                        candidates = torch.cat([candidates, _orig], dim=1)
                        obs_expanded = obs_tensor.unsqueeze(1).expand(-1, candidates.shape[1], -1)
                    final_log_probs = self.q_estimator(obs_expanded, _norm(candidates)).squeeze(-1)
                    sel = final_log_probs.argmax(dim=1)
                    action_normalized = candidates[0, sel[0], :].cpu().numpy()
                    q_range = (final_log_probs.min().item(), final_log_probs.max().item())

                action = np.clip(action_normalized, action_bounds[0], action_bounds[1])
                # _denormalize_action maps model-space (e.g., [-1, 1] for pushing)
                # back to env-action space. It's a no-op for particle where
                # _raw_act_min is None, but REQUIRED for pushing.
                action = self._denormalize_action(action)
                if return_q_range:
                    return action, q_range
                return action

        sim_cls = DFORefinedSimulation
    elif inference_langevin_iterations > 0:
        class LangevinRefinedParticleSimulation(SimulationCls):
            """Refines the CP CLOUD with official-IBC-faithful Langevin MCMC.

            Upgraded after the IBC audit (memory: ibc-repro-fixes; these chain
            details tripled our in-env IBC's kitchen score):
              - Chains start from ALL control points (model-trained proposals),
                not just the argmax CP — WiFI-BC's analog of IBC's 512 uniform
                inits, but ~5x fewer and already near-modal, so short chains
                suffice (efficiency is the point of WiFI-BC).
              - Optional noise_via_stepsize (official langevin_step): noise
                shrinks linearly with stepsize -> chain end is a pure polish.
              - Optional second chain at constant 1e-5 stepsize (official
                IbcPolicy.optimize_again).
              - Final action = argmax Q over the REFINED cloud (greedy, same
                as official GreedyPolicy mode).
            """

            def select_action(self, observation, return_q_range: bool = False):
                obs_tensor = (
                    torch.tensor(observation, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
                obs_tensor = self.obs_normalizer.normalize(obs_tensor)

                with torch.no_grad():
                    cps = self.control_point_generator(obs_tensor)  # (1, N, D)
                    # top_k > 0: refine only the k best CPs by initial Q
                    # (k=1 = single chain from the argmax CP). 0 = whole cloud.
                    if inference_langevin_top_k > 0:
                        if self._act_min_t is not None:
                            cp_q_in = (cps - self._act_min_t) / self._act_rng_t
                        else:
                            cp_q_in = cps
                        obs_exp0 = obs_tensor.unsqueeze(1).expand(-1, cps.shape[1], -1)
                        q0 = self.q_estimator(obs_exp0, cp_q_in).squeeze(-1)  # (1, N)
                        k = min(inference_langevin_top_k, cps.shape[1])
                        top_idx = q0.topk(k, dim=1).indices  # (1, k)
                        cps = torch.gather(
                            cps, 1, top_idx.unsqueeze(-1).expand(-1, -1, cps.shape[-1])
                        )

                act_min_t = torch.full(
                    (cps.shape[-1],), float(action_bounds[0]), device=self.device
                )
                act_max_t = torch.full(
                    (cps.shape[-1],), float(action_bounds[1]), device=self.device
                )

                for p in self.q_estimator.parameters():
                    p.requires_grad_(False)

                _norm_min = self._act_min_t
                _norm_rng = self._act_rng_t

                def _neg_energy_fn(obs_lv, actions_lv):
                    if _norm_min is not None:
                        a_in = (actions_lv - _norm_min) / _norm_rng
                    else:
                        a_in = actions_lv
                    return -self.q_estimator(obs_lv, a_in).squeeze(-1)

                refined = sample_langevin(
                    energy_function=_neg_energy_fn,
                    observations=obs_tensor,
                    num_samples=cps.shape[1],
                    action_min=act_min_t,
                    action_max=act_max_t,
                    num_iterations=inference_langevin_iterations,
                    lr_init=float(langevin_cfg.get("lr_init", 0.1)),
                    lr_final=float(langevin_cfg.get("lr_final", 1e-5)),
                    polynomial_decay_power=float(
                        langevin_cfg.get("polynomial_decay_power", 2.0)
                    ),
                    delta_action_clip=float(
                        langevin_cfg.get("delta_action_clip", 0.1)
                    ),
                    noise_scale=float(langevin_cfg.get("noise_scale", 1.0)),
                    initial_actions=cps.clone(),
                    device=self.device,
                    noise_via_stepsize=inference_langevin_noise_via_stepsize,
                )
                if inference_langevin_again_iterations > 0:
                    refined = sample_langevin(
                        energy_function=_neg_energy_fn,
                        observations=obs_tensor,
                        num_samples=cps.shape[1],
                        action_min=act_min_t,
                        action_max=act_max_t,
                        num_iterations=inference_langevin_again_iterations,
                        lr_init=1e-5,
                        lr_final=1e-5,
                        polynomial_decay_power=float(
                            langevin_cfg.get("polynomial_decay_power", 2.0)
                        ),
                        delta_action_clip=float(
                            langevin_cfg.get("delta_action_clip", 0.1)
                        ),
                        noise_scale=inference_langevin_again_noise_scale,
                        initial_actions=refined,
                        device=self.device,
                        noise_via_stepsize=inference_langevin_noise_via_stepsize,
                    )

                for p in self.q_estimator.parameters():
                    p.requires_grad_(True)

                # Greedy over the refined cloud (re-scored post-refinement so
                # the argmax indexes the actions actually being returned).
                with torch.no_grad():
                    obs_expanded = obs_tensor.unsqueeze(1).expand(-1, refined.shape[1], -1)
                    if _norm_min is not None:
                        ref_for_q = (refined - _norm_min) / _norm_rng
                    else:
                        ref_for_q = refined
                    q_values = self.q_estimator(obs_expanded, ref_for_q).squeeze(-1)
                    best_idx = q_values.argmax(dim=1)
                    q_range = (q_values.min().item(), q_values.max().item())
                    action = refined[0, best_idx[0], :].cpu().numpy()

                action = np.clip(action, action_bounds[0], action_bounds[1])
                # Denormalize to the env's native action box when the
                # simulation declares a non-identity inverse (Pushing). For
                # ParticleSimulation this is a no-op (action_bounds = [0, 1]
                # is already the env action box).
                action = self._denormalize_action(action)
                if return_q_range:
                    return action, q_range
                return action

        sim_cls = LangevinRefinedParticleSimulation
    else:
        sim_cls = SimulationCls

    # PushingSimulation has no n_dim arg (1-block/1-target, fixed schema)
    # but has its own goal_dist_tolerance knob (IBC paper used 0.02).
    if inference_uniform_proposal:
        # Wrap rather than branch at each call site: every downstream consumer
        # (plain argmax, CP-DFO, inference-time Langevin, pixel and state envs)
        # goes through cp_gen(obs) -> (B, N, A), so replacing the module here
        # covers all of them and keeps the candidate count identical.
        class _UniformProposal(torch.nn.Module):
            def __init__(self, inner, n, action_dim, lo, hi):
                super().__init__()
                self.inner, self.n, self.action_dim = inner, int(n), int(action_dim)
                self.lo, self.hi = float(lo), float(hi)

            def forward(self, obs, *a, **k):
                b = obs.shape[0]
                dev = obs.device if hasattr(obs, "device") else None
                return torch.empty(
                    b, self.n, self.action_dim, device=dev
                ).uniform_(self.lo, self.hi)

            def encode(self, *a, **k):
                return self.inner.encode(*a, **k)

        cp_gen = _UniformProposal(
            cp_gen, control_points, action_dim, action_bounds[0], action_bounds[1]
        ).to(device).eval()
        print(f"ABLATION: learned proposal replaced by {control_points} uniform "
              f"candidates in [{action_bounds[0]}, {action_bounds[1]}]")

    sim_kwargs: dict = dict(
        control_point_generator=cp_gen,
        q_estimator=q_est,
        device=device,
        max_episode_steps=max_episode_steps,
        render_mode=None,
        frame_stack=frame_stack,
        norm_stats=norm_stats,
    )
    if active_env == "pushing":
        sim_kwargs["goal_dist_tolerance"] = float(
            env_config.get("goal_dist_tolerance", 0.02)
        )
    elif active_env == "pushing_pixels":
        # Single-target physics — same 0.02 tolerance as states variant.
        sim_kwargs["goal_dist_tolerance"] = float(
            env_config.get("goal_dist_tolerance", 0.02)
        )
    elif active_env in ("pen", "kitchen"):
        # Adroit D4RL + FrankaKitchen — no goal_dist_tolerance / n_dim knobs.
        # Receding horizon (execute R of the K-step chunk then replan).
        # 0 = execute all K (pure chunking). Eval-time-only knob.
        sim_kwargs["execute_horizon"] = int(
            env_config.get("training", {}).get("action_execute_horizon", 0)
        )
    elif active_env == "libero_goal_pixels":
        # Render eval grouped by task needs the eval-episode count to map seeds
        # task-major (avoids per-episode EGL env churn).
        sim_kwargs["num_eval_seeds"] = int(
            env_config.get("num_eval_seeds", len(seeds))
        )
    elif active_env == "point_maze_pillar":
        # No extra knobs — PointMazePillarSimulation only takes the base
        # kwargs already in sim_kwargs (start/goal/pillar are fixed
        # constants in the env itself, not config-driven).
        pass
    else:
        sim_kwargs["n_dim"] = n_dim
    sim = sim_cls(**sim_kwargs)

    all_results = []
    for seed in seeds:
        result = sim.run_episode(seed=seed)
        all_results.append(result)
    sim.close()

    def _finite(x: float) -> float | None:
        """JSON-safe: inf/nan → None so trials.jsonl stays strictly valid JSON."""
        xf = float(x)
        return xf if np.isfinite(xf) else None

    successes = [bool(r.get("success", False)) for r in all_results]
    rewards = [float(r.get("total_reward", 0.0)) for r in all_results]
    ep_lengths = [int(r.get("episode_length", 0)) for r in all_results]
    terminated_flags = [bool(r.get("terminated", False)) for r in all_results]

    if active_env == "kitchen":
        # FrankaKitchen headline metric = avg_tasks_completed (0..N), matching
        # IBC Table 2 (kitchen-complete = 3.37/4). success = solved ALL tasks.
        tasks_done = [int(r.get("tasks_completed", 0)) for r in all_results]
        return {
            "success_rate": float(np.mean(successes)),
            "success_rate_std": float(np.std(successes)),
            "avg_tasks_completed": float(np.mean(tasks_done)),
            "std_tasks_completed": float(np.std(tasks_done)),
            "median_tasks_completed": float(np.median(tasks_done)),
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "median_reward": float(np.median(rewards)),
            "avg_episode_length": float(np.mean(ep_lengths)),
            "num_seeds": len(seeds),
            "per_seed": [
                {
                    "seed": seeds[i],
                    "success": successes[i],
                    "tasks_completed": tasks_done[i],
                    "reward": rewards[i],
                    "episode_length": ep_lengths[i],
                    "terminated": terminated_flags[i],
                }
                for i in range(len(seeds))
            ],
        }

    if active_env in ("pen", "libero_goal_pixels"):
        # Adroit D4RL human tasks AND LIBERO-Goal report success_rate as the
        # headline metric (LIBERO's canonical number is per-suite success rate;
        # the env emits a binary success info bit). avg_reward is logged too but
        # is secondary for libero_goal. per_seed here is per-eval-episode; for
        # LIBERO the sim cycles tasks across episodes (see LiberoSimulation).
        return {
            "success_rate": float(np.mean(successes)),
            "success_rate_std": float(np.std(successes)),
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "median_reward": float(np.median(rewards)),
            "avg_episode_length": float(np.mean(ep_lengths)),
            "num_seeds": len(seeds),
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

    if active_env in ("pushing", "pushing_pixels"):
        # Single-target pushing (states OR pixels) — single goal, same metric
        # layout so the trial logs / analyzer queries stay uniform across the
        # two observation modalities.
        dists_target = [float(r.get("min_dist_to_target", np.inf)) for r in all_results]
        finite_target = [d for d in dists_target if np.isfinite(d)]
        return {
            "success_rate": float(np.mean(successes)),
            "success_rate_std": float(np.std(successes)),
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "median_reward": float(np.median(rewards)),
            "avg_min_dist_to_target": float(np.mean(finite_target)) if finite_target else None,
            "std_min_dist_to_target": float(np.std(finite_target)) if finite_target else None,
            "median_min_dist_to_target": float(np.median(finite_target)) if finite_target else None,
            "avg_episode_length": float(np.mean(ep_lengths)),
            "num_seeds": len(seeds),
            "per_seed": [
                {
                    "seed": seeds[i],
                    "success": successes[i],
                    "reward": rewards[i],
                    "min_dist_to_target": _finite(dists_target[i]),
                    "episode_length": ep_lengths[i],
                    "terminated": terminated_flags[i],
                }
                for i in range(len(seeds))
            ],
        }

    dists_first = [float(r.get("min_dist_to_first_goal", np.inf)) for r in all_results]
    dists_second = [float(r.get("min_dist_to_second_goal", np.inf)) for r in all_results]
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


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _load_config(path: str | None = None) -> dict:
    cfg_path = Path(
        path
        or os.environ.get("WIFI_BC_CONFIG_PATH")
        or (Path(__file__).resolve().parent.parent / "config" / "config.json")
    )
    with open(cfg_path) as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained policy and report its task performance."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Directory holding the trained weights.")
    parser.add_argument("--env", default=None,
                        help="Environment to evaluate on (default: config's active_env).")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of evaluation episodes (default: the env's protocol).")
    parser.add_argument("--config", default=None,
                        help="Config file to read (default: config/config.json).")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="Also write the full metrics dict to this JSON file.")
    args = parser.parse_args()

    config = _load_config(args.config)
    if args.env:
        if args.env not in config["environments"]:
            parser.error(f"unknown env {args.env!r}; "
                         f"choose from {sorted(config['environments'])}")
        config["active_env"] = args.env
    if args.episodes is not None:
        config["environments"][config["active_env"]]["num_eval_seeds"] = args.episodes

    results = evaluate(args.checkpoint, config)

    print()
    print(f"Environment : {config['active_env']}")
    print(f"Checkpoint  : {args.checkpoint}")
    if results.get("error"):
        print(f"ERROR       : {results['error']}")
        return 1
    print(f"Episodes    : {results.get('num_seeds')}")
    print(f"Success rate: {results.get('success_rate', 0.0):.3f}")
    print(f"Avg reward  : {results.get('avg_reward', 0.0):.3f}")
    if results.get("avg_tasks_completed") is not None:
        print(f"Avg tasks   : {results['avg_tasks_completed']:.3f}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Full metrics written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
