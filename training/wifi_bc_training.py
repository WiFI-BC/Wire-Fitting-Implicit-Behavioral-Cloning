"""Train WiFI-BC — the paper's method.

Jointly trains the control-point generator (MSE to the expert action + InfoNCE
against the critic) and the Q estimator (IBC InfoNCE over generator control
points, uniform samples and Langevin-refined hard negatives).

    uv run python -m training.wifi_bc_training --env pushing

Every hyperparameter comes from `config/config.json`, so a run is fully
described by the config plus the environment name. Writes
`control_point_generator.pt`, `q_estimator.pt` and `norm_stats.pt`; evaluate the
result with `uv run python -m envs.evaluate --checkpoint <dir>`.
"""

import copy
import os
import random
import sys
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wifi_bc.config import resolve_active_env, resolve_env_config  # noqa: E402
from wifi_bc.models import (
    ControlPointGenerator,
    QEstimator,
    PixelControlPointGenerator,
    PixelQEstimator,
)
from wifi_bc.loss import lossInfoNCE, lossMSE, lossSeparation, lossEntropyKDE
from wifi_bc.normalizations import ObservationNormalizer
from wifi_bc.sampling import sample_langevin

# Load config. WIFI_BC_CONFIG_PATH overrides the shipped file, so a run can be
# pointed at an alternative config without editing it in place.
config_path = Path(
    os.environ.get("WIFI_BC_CONFIG_PATH")
    or (Path(__file__).resolve().parent.parent / "config" / "config.json")
)
with open(config_path, "r") as f:
    config = json.load(f)

# Get active environment
active_env = resolve_active_env(config)
env_config = resolve_env_config(config, active_env, "wifi_bc")
training_shared = config.get("training_shared", {})
env_training = env_config.get("training", {})
env_model = env_config.get("model", {})

# Training parameters (merge env-specific with shared, env-specific takes priority)
training_steps = env_training.get("training_steps", training_shared.get("training_steps", 100000))
batch_size = env_training.get("batch_size", training_shared.get("batch_size", 128))
learning_rate = env_training.get("learning_rate", training_shared.get("learning_rate", 1e-3))

# WiFI-BC IBC Loss parameters
separation_weight = env_training.get("separation_weight", training_shared.get("separation_weight", 0.1))
mse_weight = env_training.get("mse_weight", training_shared.get("mse_weight", 1.0))
info_nce_weight = env_training.get("info_nce_weight", training_shared.get("info_nce_weight", 1.0))
generator_infonce_weight = env_training.get(
    "generator_infonce_weight",
    training_shared.get("generator_infonce_weight", 0.05),
)

MODEL_SAVE_DIR = training_shared.get("model_save_dir", "checkpoints")
log_interval = training_shared.get("log_interval", 1000)
save_interval = training_shared.get("save_interval", 10000)

sampling_method = env_training.get("sampling_method", training_shared.get("sampling_method", "uniform"))
top_k_control_points = env_training.get(
    "top_k_control_points",
    training_shared.get("top_k_control_points", 64),
)

langevin_config = env_model.get("langevin_config", {})
# Each langevin hyperparam: env_training.langevin_* override wins over
# env_model.langevin_config.* default. Keeps envs/evaluate.py's SEARCH_SPACE
# entries (langevin_lr_init, ..., langevin_decay_power) effective without
# touching the nested config block.
langevin_num_iterations = env_training.get(
    "langevin_num_iterations",
    langevin_config.get("num_iterations", 50),
)
langevin_lr_init = env_training.get("langevin_lr_init", langevin_config.get("lr_init", 0.1))
langevin_lr_final = env_training.get("langevin_lr_final", langevin_config.get("lr_final", 1e-5))
langevin_decay_power = env_training.get(
    "langevin_decay_power", langevin_config.get("polynomial_decay_power", 2.0)
)
langevin_delta_clip = env_training.get(
    "langevin_delta_clip", langevin_config.get("delta_action_clip", 0.1)
)
langevin_noise_scale = env_training.get(
    "langevin_noise_scale", langevin_config.get("noise_scale", 1.0)
)
# Training-only IBC sampler fidelity knobs. Defaults preserve every existing
# WiFI-BC run: textbook sqrt(stepsize) noise and the exact action box.
langevin_noise_via_stepsize = bool(
    env_training.get(
        "langevin_noise_via_stepsize",
        training_shared.get("langevin_noise_via_stepsize", False),
    )
)
langevin_boundary_buffer = float(
    env_training.get(
        "langevin_boundary_buffer",
        training_shared.get("langevin_boundary_buffer", 0.0),
    )
)
if langevin_boundary_buffer < 0.0:
    raise ValueError("langevin_boundary_buffer must be >= 0")

# IBC counter-example mixture (Florence et al., 2021, §4.1).
# Estimator's InfoNCE negatives = top-k generator CPs (existing) + uniform random
# + Langevin-refined hard negatives. The Langevin negatives are sampled from
# uniform initialisations and pushed UP the Q surface (i.e., toward expert-like
# actions that aren't the expert) via gradient ascent on Q.
num_uniform_negatives = env_training.get(
    "num_uniform_negatives", training_shared.get("num_uniform_negatives", 32)
)
num_langevin_negatives = env_training.get(
    "num_langevin_negatives", training_shared.get("num_langevin_negatives", 32)
)

# Langevin negative starting distribution. "uniform" (default, paper-faithful)
# starts each chain at a uniformly-random action and ascends Q. "cps" starts
# each chain at a randomly-picked CP from the generator output (with optional
# Gaussian jitter) and ascends Q — finds high-Q points in the *neighborhood
# of CPs* rather than anywhere in the action box.
langevin_init_kind = env_training.get(
    "langevin_init_kind", training_shared.get("langevin_init_kind", "uniform")
)
if langevin_init_kind not in ("uniform", "cps"):
    raise ValueError(
        f"langevin_init_kind must be 'uniform' or 'cps', got {langevin_init_kind!r}"
    )
langevin_init_jitter = float(
    env_training.get("langevin_init_jitter", training_shared.get("langevin_init_jitter", 0.0))
)

# Noisy-expert hard negatives (estimator-only — kept out of generator's
# InfoNCE because expert+noise would conflict with the generator's MSE pull).
# Curriculum: σ linearly interpolates from σ_start (broad, gross structure)
# at step 0 to σ_final (precise) at step=training_steps. Floor at σ_final
# avoids degeneracy as training nears completion.
noisy_expert_count = int(
    env_training.get(
        "noisy_expert_count", training_shared.get("noisy_expert_count", 0)
    )
)
noisy_expert_sigma_start = float(
    env_training.get(
        "noisy_expert_sigma_start",
        training_shared.get(
            "noisy_expert_sigma_start",
            env_training.get("noisy_expert_std", training_shared.get("noisy_expert_std", 0.1)),
        ),
    )
)
noisy_expert_sigma_final = float(
    env_training.get(
        "noisy_expert_sigma_final",
        training_shared.get("noisy_expert_sigma_final", 0.02),
    )
)

# IBC gradient penalty (Florence et al., 2021, App. B; Gulrajani et al., 2017).
# Bounds ||∇_a E(s,a)|| around a margin to give the energy local curvature.
# - "hinge" (IBC paper): max(0, ||grad|| - margin)^2  — bounds gradients above margin.
#                        Inactive while ||grad|| < margin (typical at initialization).
# - "target" (WGAN-GP): (||grad|| - margin)^2 — pushes gradients toward exactly margin
#                       in BOTH directions. Always fires; more aggressive shaping.
gradient_penalty_weight = env_training.get(
    "gradient_penalty_weight", training_shared.get("gradient_penalty_weight", 0.0)
)
gradient_penalty_margin = env_training.get(
    "gradient_penalty_margin", training_shared.get("gradient_penalty_margin", 1.0)
)
gradient_penalty_form = env_training.get(
    "gradient_penalty_form", training_shared.get("gradient_penalty_form", "hinge")
)
gradient_penalty_norm = env_training.get(
    "gradient_penalty_norm", training_shared.get("gradient_penalty_norm", "l2")
).lower()
if gradient_penalty_form not in ("hinge", "target"):
    raise ValueError(
        f"gradient_penalty_form must be 'hinge' or 'target', got {gradient_penalty_form!r}"
    )
if gradient_penalty_norm not in ("l2", "linf"):
    raise ValueError(
        f"gradient_penalty_norm must be 'l2' or 'linf', got {gradient_penalty_norm!r}"
    )

# Exponential moving averages are generic optimization stabilization, not a
# change to WiFI-BC's CP-proposal + Q-ranking policy. Zero keeps legacy behavior.
ema_decay = float(env_training.get("ema_decay", training_shared.get("ema_decay", 0.0)))
if not 0.0 <= ema_decay < 1.0:
    raise ValueError("ema_decay must satisfy 0 <= ema_decay < 1")

# Best-checkpoint selection: periodically eval in the env DURING training and
# keep the highest-reward weights, instead of the (possibly-collapsed) final
# weights. Motivated by pen's high-ceiling/unstable-endpoint behaviour (same
# config gives final reward 863..4985). Gated OFF by default so existing batches
# are unaffected. When ema_decay>0 the EMA weights are the ones evaluated/kept.
best_ckpt = bool(env_training.get("best_ckpt", training_shared.get("best_ckpt", False)))
best_ckpt_eval_interval = int(env_training.get("best_ckpt_eval_interval", 20000))
best_ckpt_eval_seeds = int(env_training.get("best_ckpt_eval_seeds", 20))

# Deterministic seeding & NaN recovery — both fight the ~33% training-divergence rate.
trial_seed = env_training.get("trial_seed", training_shared.get("trial_seed", 0))
nan_abort_threshold = env_training.get(
    "nan_abort_threshold", training_shared.get("nan_abort_threshold", 50)
)

# Separation loss epsilon: must be << action-space diameter so overlapping control
# points are strongly repelled.  Default 1.0 is too large for particle's [0,1]^2.
separation_epsilon = env_training.get("separation_epsilon", training_shared.get("separation_epsilon", 1.0))
separation_loss_type = env_training.get("separation_loss", training_shared.get("separation_loss", "separation"))
entropy_bandwidth = env_training.get("entropy_bandwidth", training_shared.get("entropy_bandwidth", 0.1))

# S1: Separate LR for estimator (defaults to same as generator)
estimator_learning_rate = env_training.get(
    "estimator_learning_rate",
    training_shared.get("estimator_learning_rate", learning_rate),
)

# S2: Scheduler type — "cosine" (default) or "cosine_warm_restarts"
scheduler_type = env_training.get(
    "scheduler_type",
    training_shared.get("scheduler_type", "cosine"),
)
cosine_t0 = env_training.get(
    "cosine_t0",
    training_shared.get("cosine_t0", 50000),
)

# S4: InfoNCE logit clamp — lower values keep gradients flowing
infonce_logit_clamp = env_training.get(
    "infonce_logit_clamp",
    training_shared.get("infonce_logit_clamp", 50.0),
)
# Control points closer than this L2 radius to the expert action are scored as
# additional InfoNCE POSITIVES rather than negatives. With a precise generator
# (cp_output_activation="linear") the nearest CP sits ~0.01 from the expert, so
# plain InfoNCE trains the critic to reject the correct answer. 0 = off: the
# unmodified lossInfoNCE call runs, bit-identical to before this option existed.
infonce_positive_cp_radius = float(env_training.get(
    "infonce_positive_cp_radius", training_shared.get("infonce_positive_cp_radius", 0.0)
))

# S6: Spectral norm on estimator
use_spectral_norm = env_model.get(
    "use_spectral_norm",
    training_shared.get("use_spectral_norm", False),
)

# S7: Override cosine T_max (defaults to training_steps)
cosine_t_max = env_training.get(
    "cosine_t_max",
    training_shared.get("cosine_t_max", None),
)

# Model parameters
control_points = env_model.get("control_points", 50)
num_hidden_layers = env_model.get("num_hidden_layers", 8)
num_neurons = env_model.get("num_neurons", 512)

# Per-net architecture overrides. If a *_network_kind is set, that net uses
# the new (kind, width, depth) plumbing; otherwise it falls back to the legacy
# plain MLP defined by hidden_dims=[num_neurons]*num_hidden_layers.
q_network_kind = env_model.get("q_network_kind", "mlp")
q_width = env_model.get("q_width", num_neurons)
q_depth = env_model.get("q_depth", num_hidden_layers)
q_use_spectral_norm = env_model.get("q_use_spectral_norm", use_spectral_norm)

cp_network_kind = env_model.get("cp_network_kind", "mlp")
cp_width = env_model.get("cp_width", num_neurons)
cp_depth = env_model.get("cp_depth", num_hidden_layers)
cp_output_activation = env_model.get("cp_output_activation", "tanh")
cp_use_spectral_norm = env_model.get("cp_use_spectral_norm", False)

# Environment parameters
env_id = env_config["env_id"]
state_dim = env_config["state_dim"]
action_dim = env_config["action_dim"]
action_bounds = env_config.get("action_bounds", [-1, 1])
frame_stack = env_config.get("frame_stack", 1)


def load_dataset(split="train"):
    """Load the appropriate dataset based on active_env.

    `split` ("train"/"val"/"all") only affects the zarr_video PushT dataset,
    which supports an episode-level held-out split; other datasets ignore it.
    """
    if active_env in ("pen", "kitchen"):
        from envs.datasets import D4RLDataset
        dataset_name = env_config["dataset_name"]
        # kitchen carries a Dict obs; D4RLDataset extracts the 'observation' field.
        # kitchen_qpos_only drops the port-added velocity dims -> the IBC
        # paper's input content (robot qpos [0:9] + object qpos [18:39]).
        obs_indices = None
        if active_env == "kitchen" and bool(env_training.get("kitchen_qpos_only", False)):
            obs_indices = list(range(0, 9)) + list(range(18, 39))
            print(f"kitchen_qpos_only: obs -> {len(obs_indices)}-D (qpos only)")
        return D4RLDataset(
            dataset_name, download=True, frame_stack=frame_stack,
            obs_indices=obs_indices,
            action_chunk=int(env_config.get("training", {}).get("action_chunk", 1)),
        )
    elif active_env == "particle":
        from envs.datasets import ParticleDataset
        data_dir = env_config["data_dir"]
        n_dim = env_config.get("n_dim", 2)
        return ParticleDataset(data_dir, n_dim=n_dim, frame_stack=frame_stack)
    elif active_env == "pushing":
        from envs.datasets import PushingDataset
        data_dir = env_config["data_dir"]
        return PushingDataset(data_dir=data_dir, frame_stack=frame_stack)
    elif active_env == "pushing_pixels":
        from envs.datasets import PushingPixelsDataset
        data_dir = env_config["data_dir"]
        return PushingPixelsDataset(
            data_dir=data_dir, frame_stack=frame_stack,
            action_chunk=int(env_config.get("training", {}).get("action_chunk", 1)),
        )
    elif active_env == "libero_goal_pixels":
        from envs.datasets import LiberoGoalPixelsDataset
        return LiberoGoalPixelsDataset(
            goal_embeddings_path=env_config["goal_embeddings_path"],
            frame_stack=frame_stack,
            max_demos_per_task=env_config.get("max_demos_per_task"),
            crop_size=int(env_config.get("training", {}).get("image_crop_size", 0)),
            action_chunk=int(env_config.get("training", {}).get("action_chunk", 1)),
            cameras=str(env_config.get("libero_cameras", "agentview+wrist")),
            use_proprio=bool(env_config.get("libero_use_proprio", True)),
        )
    elif active_env == "point_maze_pillar":
        from envs.datasets import PointMazePillarDataset
        return PointMazePillarDataset(
            size=20000,
            max_steps_per_episode=env_config.get("max_episode_steps", 400),
            frame_stack=frame_stack,
        )
    else:
        raise ValueError(f"Unknown environment: {active_env}")


def main():
    global learning_rate

    # Deterministic seeding — same trial_seed → same training trajectory across reps.
    random.seed(trial_seed)
    np.random.seed(trial_seed)
    torch.manual_seed(trial_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(trial_seed)
    print(f"trial_seed={trial_seed} (deterministic seeding applied)")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Active environment: {active_env}")
    print(f"Training steps: {training_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate (generator): {learning_rate}")
    print(f"Learning rate (estimator): {estimator_learning_rate}")
    print(f"Scheduler: {scheduler_type}")
    print(f"InfoNCE logit clamp: {infonce_logit_clamp}")
    print(f"Spectral norm (estimator): {use_spectral_norm}")
    print(f"Top-k control points as counter examples: {top_k_control_points}")
    print(f"IBC uniform negatives: {num_uniform_negatives}")
    print(f"IBC Langevin negatives: {num_langevin_negatives} (iters={langevin_num_iterations}, "
          f"lr={langevin_lr_init}, noise={langevin_noise_scale}, clip={langevin_delta_clip}, "
          f"init_kind={langevin_init_kind}, init_jitter={langevin_init_jitter}, "
          f"noise_via_stepsize={langevin_noise_via_stepsize}, "
          f"boundary_buffer={langevin_boundary_buffer})")
    print(f"Noisy expert (estimator-only): count={noisy_expert_count} "
          f"sigma_start={noisy_expert_sigma_start} sigma_final={noisy_expert_sigma_final}")
    print(f"Gradient penalty: weight={gradient_penalty_weight}, margin={gradient_penalty_margin}, "
          f"form={gradient_penalty_form}, norm={gradient_penalty_norm}")
    print(f"EMA decay: {ema_decay} ({'enabled' if ema_decay > 0.0 else 'disabled'})")
    print(f"NaN abort threshold (consecutive bad batches): {nan_abort_threshold}")
    print(f"Frame stack: {frame_stack}")
    
    # Initialize Weights & Biases
    wandb_run_name = f"{active_env}_combined_cp{control_points}_lr{learning_rate}"
    wandb.init(
        project="WiFI-BC",
        config={
            "active_env": active_env,
            "env_config": env_config,
            "training_shared": training_shared,
        },
        name=wandb_run_name,
    )
    
    # Load dataset
    print(f"Loading {active_env} dataset...")
    dataset = load_dataset()
    print(f"Dataset size: {len(dataset)}")

    if active_env == "particle" and hasattr(dataset, "_episode_starts"):
        episode_starts = int(dataset._episode_starts.sum())
        tfrecord_count = len(getattr(dataset, "tfrecord_files", []))
        avg_episode_length = len(dataset) / max(episode_starts, 1)
        print(
            f"Particle dataset episodes: {episode_starts} | "
            f"Avg samples/episode: {avg_episode_length:.2f}"
        )
        if tfrecord_count and episode_starts <= tfrecord_count:
            raise RuntimeError(
                "Particle dataset episode boundary parsing looks wrong: "
                f"detected {episode_starts} episode starts across {tfrecord_count} TFRecord files. "
                "This usually means step_type decoding failed and frame stacking would mix episodes."
            )
    
    # Create models
    if active_env in ("pushing_pixels", "libero_goal_pixels"):
        # Image-conditioned models with vendored IBC ConvMaxpoolEncoder.
        # dataset.state_shape is (C, H, W); only C and the encoder target
        # resolution are passed to the model (the encoder bilinearly resizes
        # any input to target_h × target_w internally, matching IBC's
        # image_prepro.preprocess).
        in_channels = dataset.state_shape[0]  # 3 * n_cams * frame_stack
        enc_h = env_config.get("encoder_target_height", 180)
        enc_w = env_config.get("encoder_target_width", 240)
        value_width = env_model.get("value_width", 1024)
        value_num_blocks = env_model.get("value_num_blocks", 1)
        # libero_goal_pixels conditions the pixel nets on proprio + goal embed
        # (dataset.cond_dim); pushing_pixels has no conditioning (cond_dim=0).
        cond_dim = int(getattr(dataset, "cond_dim", 0))
        # Image encoder: IBC ConvMaxpool (default) or ImageNet-pretrained
        # ResNet-18 + SpatialSoftmax (LIBERO-standard BC encoder).
        encoder_kind = env_model.get("encoder_kind", "conv_maxpool")
        encoder_pretrained = bool(env_model.get("encoder_pretrained", True))
        encoder_num_kp = int(env_model.get("encoder_num_kp", 64))
        # ResNet norm strategy: bn | gn | bn_frozen (see models.py rationale —
        # raw BN's train/eval stat mismatch is hostile to EBM training).
        encoder_norm_kind = env_model.get("encoder_norm_kind", "bn")
        encoder_per_camera = bool(env_model.get("encoder_per_camera", False))
        cond_fusion = env_model.get("cond_fusion", "concat")
        goal_dim = int(getattr(dataset, "goal_emb_dim", 0))
        # share_encoder: one conv trunk for BOTH nets instead of two. WiFI-BC's
        # default (separate trunks, IBC's convention) makes every inference pay
        # the encoder twice; sharing halves that. The ESTIMATOR owns the trunk,
        # so it is built first and the generator adopts it -- see the optimizer,
        # grad-clip and EMA branches below, all of which must skip the
        # generator's now-borrowed encoder parameters to avoid double-updating.
        share_encoder = bool(env_model.get("share_encoder", False))
        print(
            f"Q estimator:  PIXEL value=DenseResnetValue(w={value_width}, "
            f"blocks={value_num_blocks}) in_ch={in_channels} enc={encoder_kind} "
            f"{enc_h}x{enc_w} cond={cond_dim}"
        )
        estimator = PixelQEstimator(
            action_dim=dataset.action_shape,
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
        ).to(device)
        print(
            f"CP generator: PIXEL kind={cp_network_kind} width={cp_width} "
            f"depth={cp_depth} in_ch={in_channels} enc={encoder_kind} "
            f"{enc_h}x{enc_w} cond={cond_dim} "
            f"trunk={'SHARED with Q estimator' if share_encoder else 'own'}"
        )
        control_point_generator = PixelControlPointGenerator(
            share_encoder_from=(estimator if share_encoder else None),
            output_dim=dataset.action_shape,
            control_points=control_points,
            hidden_dims=[cp_width for _ in range(cp_depth)],
            action_bounds=(action_bounds[0], action_bounds[1]),
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
        ).to(device)
    else:
        print(f"CP generator: kind={cp_network_kind} width={cp_width} depth={cp_depth} sn={cp_use_spectral_norm} "
              f"out={cp_output_activation}")
        control_point_generator = ControlPointGenerator(
            input_dim=dataset.state_shape,
            output_dim=dataset.action_shape,
            control_points=control_points,
            hidden_dims=[cp_width for _ in range(cp_depth)],
            action_bounds=(action_bounds[0], action_bounds[1]),
            network_kind=cp_network_kind,
            width=cp_width,
            depth=cp_depth,
            use_spectral_norm=cp_use_spectral_norm,
            output_activation=cp_output_activation,
        ).to(device)

        q_resnet_final_act = bool(env_model.get("q_resnet_final_activation", True))
        print(f"Q estimator:  kind={q_network_kind} width={q_width} depth={q_depth} "
              f"sn={q_use_spectral_norm} final_act={q_resnet_final_act}")
        estimator = QEstimator(
            state_dim=dataset.state_shape,
            action_dim=dataset.action_shape,
            hidden_dims=[q_width for _ in range(q_depth)],
            use_spectral_norm=q_use_spectral_norm,
            network_kind=q_network_kind,
            width=q_width,
            depth=q_depth,
            resnet_final_activation=q_resnet_final_act,
        ).to(device)

    _shares_encoder = bool(getattr(control_point_generator, "shares_encoder", False))
    ema_generator = copy.deepcopy(control_point_generator) if ema_decay > 0.0 else None
    ema_estimator = copy.deepcopy(estimator) if ema_decay > 0.0 else None
    if _shares_encoder and ema_generator is not None:
        # deepcopy gave the two EMA models INDEPENDENT trunk copies, which would
        # drift apart and stop mirroring the (single) live trunk. Re-point the
        # generator's EMA at the estimator's EMA trunk so there is exactly one,
        # and skip it when averaging the generator (update_ema's skip_prefix)
        # so it is not decayed twice per step.
        ema_generator.encoder = ema_estimator.encoder
    for ema_model in (ema_generator, ema_estimator):
        if ema_model is not None:
            ema_model.eval()
            for parameter in ema_model.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def update_ema(ema_model: nn.Module, source_model: nn.Module,
                   skip_prefix: str | None = None) -> None:
        """Average parameters and copy buffers (BN/SN state) from source.

        `skip_prefix` leaves a submodule alone -- used for a SHARED trunk, which
        the estimator's own update already covers.
        """
        source_parameters = dict(source_model.named_parameters())
        for name, ema_parameter in ema_model.named_parameters():
            if skip_prefix is not None and name.startswith(skip_prefix):
                continue
            ema_parameter.mul_(ema_decay).add_(
                source_parameters[name].detach(), alpha=1.0 - ema_decay
            )
        source_buffers = dict(source_model.named_buffers())
        for name, ema_buffer in ema_model.named_buffers():
            if skip_prefix is not None and name.startswith(skip_prefix):
                continue
            src = source_buffers[name].detach()
            # Some encoder buffers (e.g. the SpatialSoftmax coordinate grid on
            # the ResNet-18 encoder) are materialized LAZILY on the first
            # forward — after this EMA model was deep-copied — so the EMA copy
            # still holds a 0-sized placeholder. Adopt the source shape once;
            # these are deterministic grids, so copying is exact.
            if ema_buffer.shape != src.shape:
                ema_buffer.resize_(src.shape)
            ema_buffer.copy_(src)

    def save_checkpoints() -> None:
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        torch.save(
            control_point_generator.state_dict(),
            os.path.join(MODEL_SAVE_DIR, "control_point_generator.pt"),
        )
        torch.save(estimator.state_dict(), os.path.join(MODEL_SAVE_DIR, "q_estimator.pt"))
        if ema_generator is not None and ema_estimator is not None:
            torch.save(
                ema_generator.state_dict(),
                os.path.join(MODEL_SAVE_DIR, "control_point_generator_ema.pt"),
            )
            torch.save(
                ema_estimator.state_dict(),
                os.path.join(MODEL_SAVE_DIR, "q_estimator_ema.pt"),
            )

    # Helper: call estimator with (state, candidate_actions). For pixels we
    # pass un-expanded (B, C, H, W) state + (B, N, A) actions so the model's
    # late-fusion path encodes the image ONCE per state and broadcasts the
    # 256-D features over the N candidates. For flat states we expand the
    # state to (B, N, D) the way the legacy code did.
    def q_score_candidates(state: torch.Tensor, actions_bna: torch.Tensor) -> torch.Tensor:
        if state.ndim == 4:  # image (B, C, H, W)
            return estimator(state, actions_bna)
        states_expanded = state.unsqueeze(1).expand(-1, actions_bna.shape[1], -1)
        return estimator(states_expanded, actions_bna)
    
    # With a SHARED trunk the estimator owns it: the generator's optimizer and
    # grad-clip see HEAD parameters only. The generator's loss still backprops
    # into the trunk -- there is a single total_loss.backward(), so both losses'
    # gradients accumulate there -- it just does not step or clip it, which
    # would otherwise apply two AdamW updates (at two different LRs, with two
    # separate moment states) to the same weights every iteration.
    def _generator_named_params():
        for name, parameter in control_point_generator.named_parameters():
            if _shares_encoder and name.startswith("encoder."):
                continue
            yield name, parameter

    def _generator_opt_params():
        return [p for _, p in _generator_named_params()]

    def _generator_clip_params():
        return _generator_opt_params()

    if _shares_encoder:
        _n_all = sum(1 for _ in control_point_generator.parameters())
        print(f"Shared trunk: estimator owns it; generator optimizer holds "
              f"{len(_generator_opt_params())}/{_n_all} of its tensors")

    # Split LR for pixel envs: pretrained conv trunks want a much smaller LR
    # than freshly-initialized heads (encoder_lr_scale, default 1.0 = off).
    encoder_lr_scale = float(env_config.get("training", {}).get("encoder_lr_scale", 1.0))
    if encoder_lr_scale != 1.0 and hasattr(control_point_generator, "encoder"):
        def _split_groups(model, base_lr, include_encoder: bool = True):
            enc_ids = {id(p) for p in model.encoder.parameters()}
            enc = ([p for p in model.parameters() if id(p) in enc_ids]
                   if include_encoder else [])
            rest = [p for p in model.parameters() if id(p) not in enc_ids]
            groups = [{"params": rest, "lr": base_lr}]
            if enc:
                groups.append({"params": enc, "lr": base_lr * encoder_lr_scale})
            return groups
        print(f"Split LR: encoder x{encoder_lr_scale}")
        optimizer_generator = torch.optim.AdamW(
            _split_groups(control_point_generator, learning_rate,
                          include_encoder=not _shares_encoder))
        optimizer_estimator = torch.optim.AdamW(_split_groups(estimator, estimator_learning_rate))
    else:
        optimizer_generator = torch.optim.AdamW(_generator_opt_params(), lr=learning_rate)
        optimizer_estimator = torch.optim.AdamW(estimator.parameters(), lr=estimator_learning_rate)

    # Learning Rate Schedules
    effective_t_max = cosine_t_max if cosine_t_max is not None else training_steps
    if scheduler_type == "cosine_warm_restarts":
        scheduler_generator = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_generator, T_0=cosine_t0, eta_min=1e-6
        )
        scheduler_estimator = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_estimator, T_0=cosine_t0, eta_min=1e-6
        )
    else:
        scheduler_generator = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_generator, T_max=effective_t_max, eta_min=1e-6
        )
        scheduler_estimator = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_estimator, T_max=effective_t_max, eta_min=1e-6
        )
    
    # Pixels need multi-worker decode to keep the GPU fed (~2-3ms JPEG decode
    # per frame × frame_stack × batch_size adds up on a single thread). Flat
    # envs keep num_workers=0 since their dataset is fully in RAM as ndarrays.
    num_workers = env_config.get(
        "dataloader_num_workers",
        4 if active_env == "pushing_pixels" else 0,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        # Surface a stuck worker as a crash (requeue-able) instead of an
        # indefinite silent hang that burns the whole GPU allocation.
        timeout=600 if num_workers > 0 else 0,
    )

    # Optional held-out validation set (episode-level split) for a live
    # generalization signal. Only the zarr_video PushT dataset supports it;
    # gated on val_frac>0 so all other envs are unaffected.
    val_loader = None
    val_interval = int(env_training.get("val_interval", save_interval))
    if (str(env_config.get("data_format", "")) == "zarr_video"
            and float(env_config.get("val_frac", 0.0)) > 0.0):
        val_dataset = load_dataset(split="val")
        # Single-process on purpose: iterating a SECOND persistent-worker loader
        # inside the training loop (on top of the train loader's workers) can
        # deadlock/spin. The val set is small and read once every val_interval,
        # so main-process loading is fine and removes the hazard entirely.
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        print(
            f"Validation: {len(val_dataset)} held-out transitions "
            f"({len(val_dataset._val_episodes)} episodes); val action-MAE "
            f"logged every {val_interval} steps."
        )

    # Observation normalizer.
    # For pushing we run in IBC-paper-faithful "standardize" mode using
    # per-dim mean/std computed from the dataset (matches `get_normalizers.py`
    # in google-research/ibc). Other envs keep their hand-authored min-max
    # bounds. The pushing stats also feed `norm_stats.pt` so the eval-time
    # PushingSimulation can recreate the exact same normalizer.
    particle_n_dim = env_config.get("n_dim") if active_env == "particle" else None
    if active_env in ("pushing_pixels", "libero_goal_pixels"):
        # The ConvMaxpoolEncoder handles its own preprocessing (uint8 → float
        # → /255 → bilinear resize) on every forward, matching IBC's
        # image_prepro.preprocess. So we skip the standardize/minmax
        # ObservationNormalizer entirely here: setting it to None makes the
        # batch loop branch and pass states straight through. (libero_goal_pixels
        # proprio+goal conditioning is fed raw via the model's _cond attribute.)
        obs_normalizer = None
        print("Observation normalizer: NONE (pixel encoder handles preprocessing)")
    elif active_env in ("pushing", "pen", "kitchen"):
        if not hasattr(dataset, "obs_mean") or not hasattr(dataset, "obs_std"):
            raise RuntimeError(
                f"{active_env} dataset must expose `obs_mean`/`obs_std` for standardize "
                f"normalization. Refresh utils/datasets.py."
            )
        obs_normalizer = ObservationNormalizer(
            env_id=env_id,
            device=device,
            frame_stack=frame_stack,
            obs_mean=dataset.obs_mean,
            obs_std=dataset.obs_std,
        )
        print("Observation normalizer: standardize (per-dim mean/std from dataset)")
    else:
        obs_normalizer = ObservationNormalizer(
            env_id=env_id,
            device=device,
            frame_stack=frame_stack,
            particle_n_dim=particle_n_dim,
        )
        print("Observation normalizer: minmax")
    
    # Number of generated control points used as counter examples.
    k_cp_counter_examples = max(1, min(top_k_control_points, control_points))

    # IBC action-space bounds tensors (used by uniform/Langevin negatives).
    action_min_tensor = torch.full((dataset.action_shape,), action_bounds[0], device=device)
    action_max_tensor = torch.full((dataset.action_shape,), action_bounds[1], device=device)
    action_range_tensor = action_max_tensor - action_min_tensor

    def persist_norm_stats() -> None:
        """Save norm_stats.pt for the eval-time simulation.

        Everything here is dataset/config-derived, so we save BEFORE training
        starts (and again at the end, harmless overwrite). Rationale: periodic
        checkpoints are written every save_interval steps, but a wall-time-killed
        job used to leave them un-evaluable because norm_stats.pt only appeared
        after training completed. Saving up front makes partial runs salvageable.
        """
        if active_env not in (
            "pushing", "pushing_pixels", "pen", "kitchen", "libero_goal_pixels",
        ):
            return
        norm_stats = {
            "act_min": dataset.act_min,
            "act_max": dataset.act_max,
            "action_norm_range": getattr(dataset, "action_norm_range", (-1.0, 1.0)),
            "frame_stack": frame_stack,
            "env_id": env_id,
            # Action chunking (1 = off). Eval reads this to rebuild the model
            # with action_dim*K and to execute chunks open-loop.
            "action_chunk": int(getattr(dataset, "action_chunk", 1)),
            # CP selection at eval: "argmax" (default) or "sample" from
            # softmax(Q/temperature) over the CP cloud. Read by the sim's
            # select_action so all eval paths share it.
            "cp_selection": str(env_training.get("cp_selection", "argmax")),
            "cp_selection_temperature": float(env_training.get("cp_selection_temperature", 1.0)),
        }
        # libero_goal_pixels: persist the pixel + conditioning schema so the
        # render-eval sim rebuilds an identical (image, cond) input.
        if active_env == "libero_goal_pixels":
            norm_stats["libero_obs_keys"] = dataset.libero_obs_keys      # proprio keys
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
            norm_stats["state_shape"] = list(dataset.state_shape)
            # Encoder architecture — eval must rebuild the exact same encoder.
            norm_stats["encoder_kind"] = env_model.get("encoder_kind", "conv_maxpool")
            norm_stats["encoder_pretrained"] = bool(env_model.get("encoder_pretrained", True))
            norm_stats["encoder_num_kp"] = int(env_model.get("encoder_num_kp", 64))
            norm_stats["encoder_norm_kind"] = env_model.get("encoder_norm_kind", "bn")
            norm_stats["encoder_per_camera"] = bool(env_model.get("encoder_per_camera", False))
            norm_stats["cond_fusion"] = env_model.get("cond_fusion", "concat")
            norm_stats["action_chunk"] = int(env_config.get("training", {}).get("action_chunk", 1))
            # Eval must center-crop to the train-time random-crop size.
            norm_stats["image_crop_size"] = int(env_config.get("training", {}).get("image_crop_size", 0))
        # Pixel dataset doesn't expose obs_mean/obs_std (the conv encoder does
        # its own [0,1] scaling + bilinear resize on every forward, matching
        # IBC's image_prepro.preprocess). Only persist these when present.
        if hasattr(dataset, "obs_mean"):
            norm_stats["obs_mean"] = dataset.obs_mean
            norm_stats["obs_std"] = dataset.obs_std
        # Kitchen: persist the column selection (kitchen_qpos_only) + exact
        # input length so eval rebuilds the same policy input (mirrors the
        # libero state_shape mechanism).
        if active_env == "kitchen":
            norm_stats["obs_indices"] = getattr(dataset, "obs_indices", None)
            norm_stats["state_shape"] = dataset.state_shape
        # LIBERO-Goal: persist the exact obs schema + per-task goal embeddings
        # so the eval simulation rebuilds a byte-identical state vector and
        # looks up the right goal per task. `state_shape` lets eval skip fragile
        # state_dim*frame_stack arithmetic (goal dims aren't frame-stacked).
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        torch.save(norm_stats, os.path.join(MODEL_SAVE_DIR, "norm_stats.pt"))
        print(f"norm_stats.pt saved (act range {dataset.act_min} → {dataset.act_max})")

    # Save norm stats up front — makes wall-killed runs (periodic checkpoints)
    # evaluable without retraining.
    persist_norm_stats()

    # ── Best-checkpoint env-eval helper (only used when best_ckpt) ────────────
    def _build_eval_sim(cp_model, q_model, ns):
        common = dict(
            control_point_generator=cp_model, q_estimator=q_model, device=str(device),
            max_episode_steps=int(env_config.get("max_episode_steps", 100)),
            frame_stack=frame_stack, norm_stats=ns,
        )
        if active_env == "pen":
            from envs.pen_human_v2_simulation import PenHumanV2Simulation as S
        elif active_env == "kitchen":
            from envs.kitchen_simulation import KitchenSimulation as S
        elif active_env == "pushing":
            from envs.pushing_simulation import PushingSimulation as S
        else:
            return None
        import inspect as _inspect
        sig = _inspect.signature(S.__init__)
        return S(**{k: v for k, v in common.items() if k in sig.parameters})

    def _eval_reward(cp_model, q_model, ns, n_seeds) -> float | None:
        sim = _build_eval_sim(cp_model, q_model, ns)
        if sim is None:
            return None
        rewards = []
        for s in range(n_seeds):
            rewards.append(float(sim.run_episode(seed=s).get("total_reward", 0.0)))
        if hasattr(sim, "close"):
            sim.close()
        return float(np.mean(rewards)) if rewards else None

    eval_norm_stats = None
    if best_ckpt:
        _ns_path = os.path.join(MODEL_SAVE_DIR, "norm_stats.pt")
        if os.path.exists(_ns_path):
            eval_norm_stats = torch.load(_ns_path, weights_only=False)
        print(f"Best-checkpoint selection: ON (every {best_ckpt_eval_interval} steps, "
              f"{best_ckpt_eval_seeds} seeds, keep max reward)")
    best_reward = float("-inf")
    best_val_mae = float("inf")  # tracked + logged to wandb for model selection

    # ── Held-out validation helpers (action-MAE via the deploy-matching
    #    argmax-Q-over-CP-cloud selection). No-ops unless val_loader is set. ──
    @torch.no_grad()
    def _argmax_action_mae(cp_model, q_model, states_t, actions_t, cond_t=None):
        if cond_t is not None:
            cp_model._cond = cond_t
            q_model._cond = cond_t
        preds = cp_model(states_t)                                # (B, Ncp, A)
        qv = q_score_candidates(states_t, preds).squeeze(-1)      # (B, Ncp)
        best = preds[
            torch.arange(states_t.shape[0], device=states_t.device),
            qv.argmax(dim=1),
        ]                                                         # (B, A)
        return (best - actions_t).abs().mean().item()

    @torch.no_grad()
    def _val_action_mae(cp_model, q_model):
        was_training = cp_model.training
        cp_model.eval(); q_model.eval()
        tot, n = 0.0, 0
        for vb in val_loader:
            vs = vb["state"].float().to(device)
            va = vb["action"].float().to(device)
            vc = vb["cond"].float().to(device) if "cond" in vb else None
            bs = vs.shape[0]
            tot += _argmax_action_mae(cp_model, q_model, vs, va, vc) * bs
            n += bs
        if was_training:
            cp_model.train(); q_model.train()
        return tot / max(n, 1)

    # Training timing
    start_time = time.time()
    step = 0
    consecutive_nan_batches = 0
    
    # Cycle through dataloader indefinitely until steps are reached
    while step < training_steps:
        for batch in dataloader:
            if step >= training_steps:
                break
            
            states = batch['state'].float().to(device)
            if obs_normalizer is not None:
                states = obs_normalizer.normalize(states)
            actions = batch['action'].float().to(device)
            B = states.shape[0]

            # libero_goal_pixels: feed the per-state conditioning (proprio +
            # goal embedding) to the pixel nets via their `_cond` attribute, so
            # every forward in this batch (CP gen, Q est, Langevin negatives,
            # gradient penalty) sees the same (B, cond_dim) vector without
            # threading it through each call site.
            # Gated on the BATCH, not the env name, so any conditioned pixel
            # dataset works (libero_goal_pixels proprio+goal, pusht EEF x/y).
            if 'cond' in batch:
                cond = batch['cond'].float().to(device)
                control_point_generator._cond = cond
                estimator._cond = cond
            
            # ==================== Generator Loss (MSE + Separation) ====================
            predicted_actions = control_point_generator(states)
            # Both lossMSE and lossSeparation return SUMS over the batch.
            # We divide by B to make them MEANs over the batch, matching InfoNCE.
            loss_mse = mse_weight * (lossMSE(predicted_actions, actions) / B)
            if separation_loss_type == "entropy":
                loss_sep = separation_weight * lossEntropyKDE(predicted_actions, bandwidth=entropy_bandwidth)
            elif separation_loss_type == "separation":
                loss_sep = separation_weight * (lossSeparation(predicted_actions, epsilon=separation_epsilon) / B)
            else:
                raise ValueError(f"Unknown separation_loss '{separation_loss_type}'. Expected 'separation' or 'entropy'.")
            loss_generator = loss_mse + loss_sep
            
            # ==================== Estimator Training (Direct InfoNCE) ====================
            # Use top-k generated control points directly as counter examples.
            # Detach control points so InfoNCE loss gradients only flow to estimator, not generator.
            predicted_actions_detached = predicted_actions.detach()
            with torch.no_grad():
                cp_q_values = q_score_candidates(states, predicted_actions_detached).squeeze(-1)
                topk_idx = torch.topk(cp_q_values, k=k_cp_counter_examples, dim=1).indices

            gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, predicted_actions_detached.shape[2])
            cp_counter_samples = torch.gather(predicted_actions_detached, dim=1, index=gather_idx)

            # ─── IBC counter-example mixture ─────────────────────────────────
            # Florence et al. 2021, §4.1: estimator should see hard negatives from
            # multiple sources, not just the generator's own outputs. Mix:
            #   (a) top-k CPs (above) — actions the generator says are good
            #   (b) uniform random — easy/medium negatives covering the action box
            #   (c) Langevin-refined — start at uniform, ascend Q to find HARD
            #       negatives the *current estimator* believes are good.
            extra_neg_chunks: list[torch.Tensor] = []

            if num_uniform_negatives > 0:
                uniform_neg = torch.rand(
                    B, num_uniform_negatives, dataset.action_shape, device=device
                ) * action_range_tensor + action_min_tensor
                extra_neg_chunks.append(uniform_neg)

            langevin_neg = None
            if num_langevin_negatives > 0 and langevin_num_iterations > 0:
                # Freeze estimator params during MCMC: gradients flow only to actions.
                for p in estimator.parameters():
                    p.requires_grad_(False)

                if states.ndim == 4:
                    # PIXELS — LATE FUSION IS MANDATORY HERE. The chain calls the
                    # energy at every iteration; going through estimator.forward
                    # would re-run the conv encoder num_iterations× per training
                    # step (with ResNet-18 that alone made steps ~3.6 s — jobs
                    # blew the 48 h wall at 44k/150k steps). Encode ONCE, run the
                    # chain against the value head only. Features are constants
                    # w.r.t. the chain (params frozen, grads flow to actions).
                    with torch.no_grad():
                        _lv_feats = estimator.encode(states)  # (B, F)

                    def _neg_energy_fn(obs_expanded_lv, actions_batch):
                        # obs arg ignored — features precomputed above.
                        return -estimator.score(_lv_feats, actions_batch).squeeze(-1)
                else:
                    def _neg_energy_fn(obs_expanded_lv, actions_batch):
                        # Q ascent ≡ descent on -Q (sample_langevin descends).
                        return -estimator(obs_expanded_lv, actions_batch).squeeze(-1)

                langevin_action_min = action_min_tensor - langevin_boundary_buffer
                langevin_action_max = action_max_tensor + langevin_boundary_buffer

                # Build the chain's starting distribution: uniform (paper) or CP-anchored.
                if langevin_init_kind == "cps":
                    # Sample one starting CP per chain, with replacement; optional
                    # Gaussian jitter so chains starting at the same CP diverge.
                    pool = predicted_actions_detached  # (B, N_cp, A)
                    pick_idx = torch.randint(
                        0, pool.shape[1], (B, num_langevin_negatives), device=device
                    )
                    pick_idx_exp = pick_idx.unsqueeze(-1).expand(-1, -1, pool.shape[2])
                    initial_actions = torch.gather(pool, dim=1, index=pick_idx_exp)
                    if langevin_init_jitter > 0.0:
                        initial_actions = initial_actions + torch.randn_like(initial_actions) * langevin_init_jitter
                    initial_actions = torch.clamp(
                        initial_actions,
                        langevin_action_min.squeeze(0),
                        langevin_action_max.squeeze(0),
                    )
                else:
                    initial_actions = None  # sample_langevin will draw uniform starts

                langevin_neg = sample_langevin(
                    energy_function=_neg_energy_fn,
                    observations=states,
                    num_samples=num_langevin_negatives,
                    action_min=langevin_action_min,
                    action_max=langevin_action_max,
                    num_iterations=langevin_num_iterations,
                    lr_init=langevin_lr_init,
                    lr_final=langevin_lr_final,
                    polynomial_decay_power=langevin_decay_power,
                    delta_action_clip=langevin_delta_clip,
                    noise_scale=langevin_noise_scale,
                    initial_actions=initial_actions,
                    device=device,
                    noise_via_stepsize=langevin_noise_via_stepsize,
                )

                for p in estimator.parameters():
                    p.requires_grad_(True)

                extra_neg_chunks.append(langevin_neg.detach())

            # ─── Estimator-only hard negatives (#4 noisy expert curriculum) ────
            # Kept in a SEPARATE list so they don't leak into the generator's
            # InfoNCE: the generator should still see only [expert, CPs], otherwise
            # the noisy-expert "negatives" would push CPs away from the very
            # region MSE is trying to land them in.
            estimator_only_neg_chunks: list[torch.Tensor] = []
            if noisy_expert_count > 0:
                # Linear σ curriculum: broad early (learn gross structure),
                # precise late (learn sharp peaks at expert).
                progress = min(1.0, max(0.0, step / max(1, training_steps - 1)))
                sigma = noisy_expert_sigma_start + progress * (noisy_expert_sigma_final - noisy_expert_sigma_start)
                expert_expanded = actions.unsqueeze(1).expand(-1, noisy_expert_count, -1)
                noisy_expert = expert_expanded + torch.randn_like(expert_expanded) * sigma
                noisy_expert = torch.clamp(
                    noisy_expert, action_min_tensor.squeeze(0), action_max_tensor.squeeze(0)
                )
                estimator_only_neg_chunks.append(noisy_expert)

            if extra_neg_chunks or estimator_only_neg_chunks:
                counter_samples = torch.cat(
                    [cp_counter_samples] + extra_neg_chunks + estimator_only_neg_chunks, dim=1
                )
            else:
                counter_samples = cp_counter_samples

            
            # Concatenate expert action (index 0) with counter-examples
            all_actions = torch.cat([actions.unsqueeze(1), counter_samples], dim=1)

            # Direct energy evaluation for InfoNCE — late-fused for pixels.
            energies = q_score_candidates(states, all_actions).squeeze(-1)

            # InfoNCE loss: expert action should have the highest Q value (lowest energy equivalent)
            if infonce_positive_cp_radius > 0.0:
                # cp_counter_samples occupy columns 1..k of all_actions (ahead of
                # uniform / Langevin / noisy-expert negatives). Multi-positive
                # InfoNCE over the same clamped logits lossInfoNCE uses.
                k_cp = cp_counter_samples.shape[1]
                near_expert = (cp_counter_samples - actions.unsqueeze(1)).norm(dim=-1) < infonce_positive_cp_radius
                pos_mask = torch.zeros_like(energies, dtype=torch.bool)
                pos_mask[:, 0] = True
                pos_mask[:, 1:1 + k_cp] = near_expert
                clamped = energies.clamp(-infonce_logit_clamp, infonce_logit_clamp)
                loss_estimator = -(
                    torch.logsumexp(clamped.masked_fill(~pos_mask, float("-inf")), dim=1)
                    - torch.logsumexp(clamped, dim=1)
                ).mean()
            else:
                loss_estimator = lossInfoNCE(energies, logit_clamp=infonce_logit_clamp)

            # ─── Gradient penalty on the estimator (IBC App. B / WGAN-GP style) ─
            # Bounds ||∇_a E(s, a)|| around `gradient_penalty_margin` so the energy
            # has local curvature instead of an unbounded slope. Applied to the
            # full action set (expert + negatives) so it shapes Q everywhere it's
            # actually evaluated by the InfoNCE loss.
            if gradient_penalty_weight > 0.0:
                gp_actions = all_actions.detach().clone().requires_grad_(True)
                gp_energies = q_score_candidates(states, gp_actions).squeeze(-1)
                gp_grad = torch.autograd.grad(
                    outputs=gp_energies.sum(),
                    inputs=gp_actions,
                    create_graph=True,
                )[0]
                gp_grad_flat = gp_grad.flatten(start_dim=2)
                if gradient_penalty_norm == "linf":
                    grad_norms = gp_grad_flat.abs().amax(dim=-1)
                else:
                    grad_norms = gp_grad_flat.norm(dim=-1)
                if gradient_penalty_form == "hinge":
                    # IBC-faithful: only penalize gradients ABOVE the margin.
                    penalty = torch.clamp(
                        grad_norms - gradient_penalty_margin, min=0.0
                    ).pow(2).mean()
                else:  # "target" — WGAN-GP: drive gradients toward the margin from both sides.
                    penalty = (grad_norms - gradient_penalty_margin).pow(2).mean()
                loss_gradient_penalty = gradient_penalty_weight * penalty
            else:
                loss_gradient_penalty = torch.tensor(0.0, device=device)

            # Generator receives InfoNCE with opposite sign.
            # Rebuild counter samples from non-detached control points so gradients reach generator,
            # while freezing estimator parameters so this branch updates only the generator.
            # IMPORTANT: noisy-expert (estimator_only_neg_chunks) is intentionally
            # excluded here — see comment above. We re-expand states to match the
            # smaller action set since states_expanded was sized for the estimator path.
            cp_counter_samples_for_generator = torch.gather(predicted_actions, dim=1, index=gather_idx)
            if extra_neg_chunks:
                counter_samples_for_generator = torch.cat(
                    [cp_counter_samples_for_generator] + extra_neg_chunks, dim=1,
                )
            else:
                counter_samples_for_generator = cp_counter_samples_for_generator

            all_actions_for_generator = torch.cat([actions.unsqueeze(1), counter_samples_for_generator], dim=1)
            for param in estimator.parameters():
                param.requires_grad_(False)
            energies_for_generator = q_score_candidates(states, all_actions_for_generator).squeeze(-1)
            loss_infonce_generator = lossInfoNCE(energies_for_generator, logit_clamp=infonce_logit_clamp)
            for param in estimator.parameters():
                param.requires_grad_(True)

            if (
                torch.isnan(loss_estimator)
                or torch.isnan(loss_infonce_generator)
                or torch.isnan(loss_generator)
                or torch.isnan(loss_gradient_penalty)
            ):
                consecutive_nan_batches += 1
                # Wipe Adam moment estimates so a single NaN batch doesn't poison
                # the optimizer state and silently break the rest of training.
                optimizer_generator.state.clear()
                optimizer_estimator.state.clear()
                if consecutive_nan_batches >= nan_abort_threshold:
                    print(
                        f"NaN loss for {consecutive_nan_batches} consecutive batches "
                        f"(>= threshold {nan_abort_threshold}). Aborting training."
                    )
                    raise RuntimeError(
                        f"Training diverged: {consecutive_nan_batches} consecutive NaN batches"
                    )
                if consecutive_nan_batches % 10 == 1:
                    print(f"NaN loss detected (run {consecutive_nan_batches}); cleared optimizer state, continuing.")
                continue
            consecutive_nan_batches = 0

            # ==================== Update Models ====================
            optimizer_estimator.zero_grad()
            optimizer_generator.zero_grad()

            loss_estimator_total = info_nce_weight * loss_estimator + loss_gradient_penalty
            loss_generator_total = loss_generator - generator_infonce_weight * loss_infonce_generator
            total_loss = loss_generator_total + loss_estimator_total
            total_loss.backward()

            # Gradient clipping. With a SHARED trunk the generator's parameter
            # list still contains the borrowed encoder, so clip the generator
            # HEAD only -- the estimator's clip already bounds the trunk, and
            # clipping it twice would scale those grads by the product of two
            # independently-computed factors.
            torch.nn.utils.clip_grad_norm_(_generator_clip_params(), 1.0)
            torch.nn.utils.clip_grad_norm_(estimator.parameters(), 1.0)

            optimizer_estimator.step()
            optimizer_generator.step()
            if ema_generator is not None and ema_estimator is not None:
                update_ema(ema_generator, control_point_generator,
                           skip_prefix="encoder." if _shares_encoder else None)
                update_ema(ema_estimator, estimator)
            scheduler_generator.step()
            scheduler_estimator.step()

            step += 1

            # ── Best-checkpoint: periodic env-eval, keep max-reward weights ──
            if best_ckpt and step % best_ckpt_eval_interval == 0:
                use_ema_eval = ema_generator is not None and ema_estimator is not None
                eval_cp = ema_generator if use_ema_eval else control_point_generator
                eval_q = ema_estimator if use_ema_eval else estimator
                if not use_ema_eval:
                    control_point_generator.eval(); estimator.eval()
                r = _eval_reward(eval_cp, eval_q, eval_norm_stats, best_ckpt_eval_seeds)
                if not use_ema_eval:
                    control_point_generator.train(); estimator.train()
                if r is not None and r > best_reward:
                    best_reward = r
                    save_checkpoints()
                    print(f"[best-ckpt] step {step}: reward {r:.1f} -> NEW BEST, saved")
                elif r is not None:
                    print(f"[best-ckpt] step {step}: reward {r:.1f} (best {best_reward:.1f})")

            # ── Held-out validation: live generalization signal ──────────────
            if val_loader is not None and step % val_interval == 0:
                use_ema_eval = ema_generator is not None and ema_estimator is not None
                eval_cp = ema_generator if use_ema_eval else control_point_generator
                eval_q = ema_estimator if use_ema_eval else estimator
                val_mae = _val_action_mae(eval_cp, eval_q)
                # Paired TRAIN MAE on the current batch, same argmax path / same
                # (EMA) weights / eval mode, for a direct train-vs-val read.
                if not use_ema_eval:
                    control_point_generator.eval(); estimator.eval()
                train_cond = cond if "cond" in batch else None
                train_mae = _argmax_action_mae(
                    eval_cp, eval_q, states, actions, train_cond
                )
                if not use_ema_eval:
                    control_point_generator.train(); estimator.train()
                best_val_mae = min(best_val_mae, val_mae)
                print(
                    f"[val] step {step}: action_MAE train={train_mae:.4f} "
                    f"val={val_mae:.4f} gap={val_mae - train_mae:+.4f} "
                    f"best_val={best_val_mae:.4f}"
                )
                wandb.log({
                    "step": step,
                    "val/action_mae": val_mae,
                    "val/action_mae_train": train_mae,
                    "val/action_mae_gap": val_mae - train_mae,
                    "val/action_mae_best": best_val_mae,
                })

            # Logging
            if step % log_interval == 0:
                current_lr = scheduler_generator.get_last_lr()[0]

                # Compute accuracy of estimator + CP-cloud / Q-ranking diagnostics
                with torch.no_grad():
                    best_idx = energies.argmax(dim=1)
                    accuracy = (best_idx == 0).float().mean().item()

                    # ─── CP coverage / Q ranking diagnostics ────────────────
                    # Decomposes the failure mode into:
                    #   - cp_to_expert_min:   does the CP cloud reach the expert?
                    #     (large = generator coverage problem)
                    #   - cp_to_expert_qbest: does Q's argmax pick a near-expert CP?
                    #     (large while min is small = Q ranking problem)
                    #   - cp_ranking_gap:     qbest - closest, the part Q gets wrong
                    #   - q_pick_closest_frac: fraction where Q's argmax IS the
                    #     closest-to-expert CP (1.0 means perfect ranking)
                    cp_to_expert = (predicted_actions_detached - actions.unsqueeze(1)).norm(dim=-1)  # (B, N_cp)
                    closest_cp_idx = cp_to_expert.argmin(dim=1)  # (B,)
                    closest_cp_to_expert = cp_to_expert.min(dim=1).values.mean().item()
                    q_argmax_idx = cp_q_values.argmax(dim=1)  # (B,)
                    qbest_cp_to_expert = cp_to_expert.gather(
                        1, q_argmax_idx.unsqueeze(-1)
                    ).squeeze(-1).mean().item()
                    q_pick_closest_frac = (q_argmax_idx == closest_cp_idx).float().mean().item()

                elapsed = time.time() - start_time
                print(f"Step {step}/{training_steps} | Total: {total_loss.item():.4f} "
                      f"(MSE: {loss_mse.item():.4f}, "
                      f"Sep: {loss_sep.item():.4f}, "
                      f"EST: {loss_estimator.item():.4f}, "
                      f"GP: {loss_gradient_penalty.item():.4f}, "
                      f"GEN_INF_ONCE(-): {loss_infonce_generator.item():.4f}, "
                      f"Acc: {accuracy:.3f}) | "
                      f"cp→a*: closest={closest_cp_to_expert:.4f} "
                      f"qbest={qbest_cp_to_expert:.4f} "
                      f"pick={q_pick_closest_frac:.3f} | "
                      f"LR: {current_lr:.2e} | {elapsed:.1f}s")

                log_dict = {
                    "step": step,
                    "loss/total": total_loss.item(),
                    "loss/generator": loss_generator_total.item(),
                    "loss/estimator": loss_estimator.item(),
                    "loss/gradient_penalty": loss_gradient_penalty.item(),
                    "loss/infonce_generator_opposite": loss_infonce_generator.item(),
                    "loss/mse": loss_mse.item(),
                    "loss/separation": loss_sep.item(),
                    "metric/accuracy": accuracy,
                    "metric/cp_to_expert_min": closest_cp_to_expert,
                    "metric/cp_to_expert_qbest": qbest_cp_to_expert,
                    "metric/cp_ranking_gap": qbest_cp_to_expert - closest_cp_to_expert,
                    "metric/q_pick_closest_frac": q_pick_closest_frac,
                    "learning_rate": current_lr,
                }
                wandb.log(log_dict)
            
            # Save checkpoint (overwrite the same file each interval so we
            # don't accumulate one .pt per step across long runs).
            # Skip when best_ckpt: the best-reward eval owns the saved weights;
            # an interval save here would clobber them with a non-best snapshot.
            if step % save_interval == 0 and not best_ckpt:
                save_checkpoints()
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.2f} min)")

    # Save trained models. When best_ckpt is ON, the checkpoint already holds the
    # best-reward weights found during training — do NOT overwrite with the final
    # (possibly collapsed) weights. Fallback: if no eval ever ran (interval >
    # training_steps), save the final weights so the run is still evaluable.
    if best_ckpt:
        if best_reward == float("-inf"):
            save_checkpoints()
            print("[best-ckpt] no eval ran; saved final weights as fallback")
        else:
            print(f"[best-ckpt] keeping best-reward checkpoint (reward {best_reward:.1f})")
    else:
        save_checkpoints()

    # Persist normalization stats for pushing — the eval-time PushingSimulation
    # uses these to recreate the exact same obs-standardize + action-denorm
    # transforms that training used. Mirrors `get_normalizers.py` in
    # google-research/ibc (stats computed from data, frozen, applied at eval).
    persist_norm_stats()

    # Remove stale smoothing param if exists
    smoothing_param_path = os.path.join(MODEL_SAVE_DIR, "smoothing_param.pt")
    if os.path.exists(smoothing_param_path):
        os.remove(smoothing_param_path)
        print(f"Removed stale {smoothing_param_path}")

    print(f"Models saved to {MODEL_SAVE_DIR}/")
    
    # Log model artifacts to W&B
    artifact = wandb.Artifact("model-checkpoints", type="model")
    artifact.add_file(os.path.join(MODEL_SAVE_DIR, "control_point_generator.pt"))
    artifact.add_file(os.path.join(MODEL_SAVE_DIR, "q_estimator.pt"))
    if ema_generator is not None and ema_estimator is not None:
        artifact.add_file(os.path.join(MODEL_SAVE_DIR, "control_point_generator_ema.pt"))
        artifact.add_file(os.path.join(MODEL_SAVE_DIR, "q_estimator_ema.pt"))
    wandb.log_artifact(artifact)
    
    # Log final metrics
    wandb.summary["total_training_time_min"] = total_time / 60
    
    wandb.finish()


if __name__ == "__main__":
    main()
