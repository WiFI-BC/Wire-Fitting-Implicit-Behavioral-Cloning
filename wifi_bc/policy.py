"""`WiFIBC` — a trained WiFI-BC policy behind a plain `.act(state) -> action`.

The training script in `training/wifi_bc_training.py` is a config-driven loop
(data loading, optimizer, logging, checkpointing) and is not meant to be
imported. This is the piece you drop into your own stack:

    from wifi_bc import WiFIBC

    policy = WiFIBC.from_checkpoint("checkpoints/pushing")
    action = policy.act(observation)          # np.ndarray -> np.ndarray

`from_checkpoint` reads the same `control_point_generator.pt`, `q_estimator.pt`
and `norm_stats.pt` that training writes, so nothing else has to be kept in
sync. The environment block it needs (network shapes, action bounds) comes from
`config/config.json` by default; pass `env_config=` to supply your own.

Three inference modes, all reported in the paper:

    argmax    score the generator's control points with the critic and take the
              best one. One forward pass through each network — the fast path.
    dfo       iteratively resample and jitter the cloud, keeping high-Q
              candidates (derivative-free optimization).
    langevin  run Langevin MCMC on the critic's energy surface, initialized at
              the control points.

    policy = WiFIBC.from_checkpoint(path, inference_mode="dfo")

`argmax` is the default. The refinement modes cost more per step and are worth
it when the critic is a better energy function than the generator is a
proposal — see the paper's Franka Kitchen result.

Observations are normalized exactly as in training (the normalizer is rebuilt
from `norm_stats.pt` when present, else from `config/observation_bounds.json`),
and actions are returned in the environment's own units.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from wifi_bc.config import load_config, resolve_env_config
from wifi_bc.models import ControlPointGenerator, QEstimator
from wifi_bc.normalizations import ObservationNormalizer
from wifi_bc.sampling import sample_langevin

INFERENCE_MODES = ("argmax", "dfo", "langevin")


class WiFIBC:
    """A trained control-point generator plus its Q estimator, ready to act.

    Construct it with `WiFIBC.from_checkpoint(...)` rather than directly unless
    you already hold both networks.
    """

    def __init__(
        self,
        control_point_generator: torch.nn.Module,
        q_estimator: torch.nn.Module,
        *,
        action_bounds: tuple[float, float] = (-1.0, 1.0),
        obs_normalizer: ObservationNormalizer | None = None,
        act_min: np.ndarray | None = None,
        act_max: np.ndarray | None = None,
        action_space: str = "model",
        frame_stack: int = 1,
        input_dim: int | None = None,
        inference_mode: str = "argmax",
        dfo_iterations: int = 3,
        dfo_std: float = 0.1,
        dfo_std_decay: float = 0.5,
        dfo_num_uniform: int = 0,
        langevin_iterations: int = 25,
        langevin_config: dict | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if inference_mode not in INFERENCE_MODES:
            raise ValueError(
                f"inference_mode must be one of {INFERENCE_MODES}, got {inference_mode!r}"
            )
        self.device = torch.device(device)
        self.cp_gen = control_point_generator.to(self.device).eval()
        self.q_estimator = q_estimator.to(self.device).eval()
        self.action_bounds = (float(action_bounds[0]), float(action_bounds[1]))
        self.obs_normalizer = obs_normalizer
        self.frame_stack = int(frame_stack)
        self.inference_mode = inference_mode
        self.dfo_iterations = int(dfo_iterations)
        self.dfo_std = float(dfo_std)
        self.dfo_std_decay = float(dfo_std_decay)
        self.dfo_num_uniform = int(dfo_num_uniform)
        self.langevin_iterations = int(langevin_iterations)
        self.langevin_config = dict(langevin_config or {})

        # `act_min`/`act_max` are the dataset's per-dim action range, and which
        # of the two conversions they drive depends on the environment's action
        # convention — `action_space` names it:
        #
        #   "model" (the default, and what every env but particle trains with)
        #       the generator emits actions in the model box (e.g. [-1, 1]) and
        #       the critic was trained on that same box, so the critic needs no
        #       conversion and the box is mapped back to environment units on
        #       the way out.
        #   "env"  (particle)
        #       the generator already emits environment-space actions, so
        #       nothing is mapped on the way out, but the critic was trained on
        #       min-max-normalized actions and needs them converted.
        #
        # Getting this backwards silently returns plausible-looking actions in
        # the wrong units, so it is a named argument rather than a guess.
        if action_space not in ("model", "env"):
            raise ValueError(f"action_space must be 'model' or 'env', got {action_space!r}")
        self.action_space = action_space
        self._act_min = None if act_min is None else np.asarray(act_min, dtype=np.float32)
        self._act_max = None if act_max is None else np.asarray(act_max, dtype=np.float32)
        if self._act_min is not None and self._act_max is not None:
            rng = self._act_max - self._act_min
            self._act_rng = np.where(rng == 0, 1.0, rng).astype(np.float32)
            self._act_min_t = torch.from_numpy(self._act_min).to(self.device)
            self._act_rng_t = torch.from_numpy(self._act_rng).to(self.device)
        else:
            self._act_rng = None
            self._act_min_t = None
            self._act_rng_t = None

        # Width of the (frame-stacked) policy input. Lets `act` accept either a
        # single raw observation or one the caller stacked themselves.
        self._input_dim = input_dim
        self._frames: list[np.ndarray] = []

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        env: str | None = None,
        env_config: dict | None = None,
        config: dict | None = None,
        inference_mode: str = "argmax",
        use_ema: bool | None = None,
        device: str | torch.device | None = None,
        **overrides: Any,
    ) -> "WiFIBC":
        """Rebuild a policy from a directory written by `training/wifi_bc_training.py`.

        `env` names the environment block to read shapes from; it defaults to
        the config's `active_env`. Pass `env_config` to bypass the config file
        entirely. `use_ema` picks the EMA weights when they exist (the default
        when they do). Any further keyword goes to `__init__`, so the inference
        knobs can be overridden per call.
        """
        checkpoint_dir = Path(checkpoint_dir)
        if env_config is None:
            config = config if config is not None else load_config()
            env = env or config.get("active_env")
            env_config = resolve_env_config(config, env, "wifi_bc")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        norm_stats_path = checkpoint_dir / "norm_stats.pt"
        norm_stats = (
            torch.load(norm_stats_path, map_location="cpu", weights_only=False)
            if norm_stats_path.exists()
            else {}
        )

        cp_path, q_path = cls._weight_paths(checkpoint_dir, use_ema)

        em = env_config.get("model", {})
        et = env_config.get("training", {})
        frame_stack = int(env_config.get("frame_stack", 1))
        action_bounds = tuple(env_config.get("action_bounds", [-1.0, 1.0]))

        # norm_stats is the authority on the input/output shapes: an env whose
        # observation was trimmed (kitchen) or whose actions were chunked has a
        # model that does not match a naive state_dim * frame_stack.
        state_dim = int(norm_stats.get("state_shape", env_config["state_dim"] * frame_stack))
        action_chunk = int(norm_stats.get("action_chunk", 1) or 1)
        action_dim = int(env_config["action_dim"]) * action_chunk

        cp_width = int(em.get("cp_width", em.get("num_neurons", 256)))
        cp_depth = int(em.get("cp_depth", em.get("num_hidden_layers", 2)))
        q_width = int(em.get("q_width", em.get("num_neurons", 256)))
        q_depth = int(em.get("q_depth", em.get("num_hidden_layers", 2)))

        cp_gen = ControlPointGenerator(
            input_dim=state_dim,
            output_dim=action_dim,
            control_points=int(em["control_points"]),
            hidden_dims=[cp_width] * cp_depth,
            action_bounds=action_bounds,
            network_kind=em.get("cp_network_kind", "mlp"),
            width=cp_width,
            depth=cp_depth,
            use_spectral_norm=bool(em.get("cp_use_spectral_norm", False)),
            output_activation=em.get("cp_output_activation", "tanh"),
        )
        cp_gen.load_state_dict(torch.load(cp_path, map_location=device, weights_only=True))

        q_est = QEstimator(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[q_width] * q_depth,
            use_spectral_norm=bool(em.get("q_use_spectral_norm", em.get("use_spectral_norm", False))),
            network_kind=em.get("q_network_kind", "mlp"),
            width=q_width,
            depth=q_depth,
            resnet_final_activation=bool(em.get("q_resnet_final_activation", True)),
        )
        q_est.load_state_dict(torch.load(q_path, map_location=device, weights_only=True))

        normalizer = cls._build_normalizer(env_config, norm_stats, frame_stack, device)

        params = dict(
            action_bounds=action_bounds,
            obs_normalizer=normalizer,
            act_min=norm_stats.get("act_min"),
            act_max=norm_stats.get("act_max"),
            action_space="env" if float(action_bounds[0]) == 0.0 else "model",
            frame_stack=frame_stack,
            input_dim=state_dim,
            inference_mode=inference_mode,
            dfo_iterations=int(et.get("inference_dfo_iterations", 3) or 3),
            dfo_std=float(et.get("inference_dfo_iteration_std", 0.1)),
            dfo_std_decay=float(et.get("inference_dfo_iteration_std_decay", 0.5)),
            dfo_num_uniform=int(et.get("inference_dfo_num_uniform", 0)),
            langevin_iterations=int(et.get("inference_langevin_iterations", 25) or 25),
            langevin_config=cls._langevin_config(env_config),
            device=device,
        )
        params.update(overrides)
        return cls(cp_gen, q_est, **params)

    @staticmethod
    def _weight_paths(checkpoint_dir: Path, use_ema: bool | None) -> tuple[Path, Path]:
        cp_ema = checkpoint_dir / "control_point_generator_ema.pt"
        q_ema = checkpoint_dir / "q_estimator_ema.pt"
        have_ema = cp_ema.exists() and q_ema.exists()
        if use_ema is None:
            use_ema = have_ema
        if use_ema:
            if not have_ema:
                raise FileNotFoundError(f"no EMA weights in {checkpoint_dir}")
            return cp_ema, q_ema
        cp, q = (checkpoint_dir / "control_point_generator.pt",
                 checkpoint_dir / "q_estimator.pt")
        for path in (cp, q):
            if not path.exists():
                raise FileNotFoundError(f"missing {path}")
        return cp, q

    @staticmethod
    def _build_normalizer(env_config, norm_stats, frame_stack, device):
        """Rebuild training's observation normalizer.

        Training persists obs mean/std for the standardizing envs; everything
        else min-max normalizes against `config/observation_bounds.json`.
        """
        return ObservationNormalizer(
            env_id=env_config.get("env_id"),
            device=str(device),
            frame_stack=frame_stack,
            particle_n_dim=env_config.get("n_dim"),
            obs_mean=norm_stats.get("obs_mean"),
            obs_std=norm_stats.get("obs_std"),
        )

    @staticmethod
    def _langevin_config(env_config: dict) -> dict:
        """Inference Langevin settings: the model's defaults with any
        `inference_langevin_*` training override applied on top."""
        cfg = dict(env_config.get("model", {}).get("langevin_config", {}))
        training = env_config.get("training", {})
        for native, key in (
            ("lr_init", "inference_langevin_lr_init"),
            ("lr_final", "inference_langevin_lr_final"),
            ("noise_scale", "inference_langevin_noise_scale"),
            ("delta_action_clip", "inference_langevin_delta_clip"),
            ("polynomial_decay_power", "inference_langevin_decay_power"),
        ):
            if key in training:
                cfg[native] = training[key]
        return cfg

    # ── acting ──────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear the frame-stack buffer. Call at the start of each episode."""
        self._frames = []

    @torch.no_grad()
    def act(self, observation: np.ndarray | Sequence[float]) -> np.ndarray:
        """One action for one observation, in the environment's own units.

        Pass the raw per-step observation: frame stacking is handled here, so
        `reset()` between episodes is all the bookkeeping required. An
        observation already the width of the stacked input is used as-is.
        """
        obs = self._stack(np.asarray(observation, dtype=np.float32).reshape(-1))
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        if self.obs_normalizer is not None:
            obs_t = self.obs_normalizer.normalize(obs_t)

        candidates = self.cp_gen(obs_t)  # (1, N, A)
        if self.inference_mode == "dfo":
            action = self._act_dfo(obs_t, candidates)
        elif self.inference_mode == "langevin":
            action = self._act_langevin(obs_t, candidates)
        else:
            action = self._argmax(obs_t, candidates)

        action = np.clip(action, self.action_bounds[0], self.action_bounds[1])
        return self._denormalize_action(action)

    def _stack(self, obs: np.ndarray) -> np.ndarray:
        if self.frame_stack <= 1:
            return obs
        if self._input_dim is not None and obs.size == self._input_dim:
            return obs  # caller already stacked
        if not self._frames:
            self._frames = [obs.copy() for _ in range(self.frame_stack)]
        else:
            self._frames.append(obs.copy())
            self._frames = self._frames[-self.frame_stack:]
        return np.concatenate(self._frames)

    def _q(self, obs_t: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Critic scores for a candidate cloud, shape (1, N)."""
        obs_expanded = obs_t.unsqueeze(1).expand(-1, actions.shape[1], -1)
        return self.q_estimator(obs_expanded, self._normalize_action(actions)).squeeze(-1)

    def _normalize_action(self, actions: torch.Tensor) -> torch.Tensor:
        """Actions as the critic was trained to score them."""
        if self.action_space == "model" or self._act_min_t is None:
            return actions
        return (actions - self._act_min_t) / self._act_rng_t

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Model-box action back to the environment's own units."""
        if self.action_space == "env" or self._act_min is None:
            return action
        lo, hi = self.action_bounds
        if hi == lo:
            return action
        scale = self._act_rng / (hi - lo)
        return (self._act_min + (np.asarray(action, dtype=np.float32) - lo) * scale).astype(np.float32)

    def _argmax(self, obs_t: torch.Tensor, candidates: torch.Tensor) -> np.ndarray:
        best = self._q(obs_t, candidates).argmax(dim=1)
        return candidates[0, best[0], :].cpu().numpy()

    def _act_dfo(self, obs_t: torch.Tensor, candidates: torch.Tensor) -> np.ndarray:
        """Derivative-free refinement: resample the cloud toward high Q, jitter,
        shrink the jitter, repeat. The final argmax is re-scored after the last
        resample so the index matches the candidates it selects from."""
        lo, hi = self.action_bounds
        if self.dfo_num_uniform > 0:
            extra = torch.empty(
                1, self.dfo_num_uniform, candidates.shape[-1], device=self.device
            ).uniform_(lo, hi)
            candidates = torch.cat([candidates, extra], dim=1)

        n = candidates.shape[1]
        std = self.dfo_std
        for it in range(self.dfo_iterations):
            probs = torch.softmax(self._q(obs_t, candidates).squeeze(0), dim=-1)
            idx = torch.multinomial(probs, n, replacement=True)
            candidates = candidates[:, idx, :]
            if it < self.dfo_iterations - 1:
                candidates = (candidates + torch.randn_like(candidates) * std).clamp(lo, hi)
                std *= self.dfo_std_decay
        return self._argmax(obs_t, candidates)

    def _act_langevin(self, obs_t: torch.Tensor, candidates: torch.Tensor) -> np.ndarray:
        """Langevin MCMC on the critic's energy surface, started at the control
        points, then argmax over the refined cloud."""
        lo, hi = self.action_bounds
        action_dim = candidates.shape[-1]
        act_min = torch.full((action_dim,), lo, device=self.device)
        act_max = torch.full((action_dim,), hi, device=self.device)

        def neg_energy(obs_lv, actions_lv):
            return -self.q_estimator(obs_lv, self._normalize_action(actions_lv)).squeeze(-1)

        cfg = self.langevin_config
        with torch.enable_grad():
            refined = sample_langevin(
                energy_function=neg_energy,
                observations=obs_t,
                num_samples=candidates.shape[1],
                action_min=act_min,
                action_max=act_max,
                num_iterations=self.langevin_iterations,
                lr_init=float(cfg.get("lr_init", 0.1)),
                lr_final=float(cfg.get("lr_final", 1e-5)),
                polynomial_decay_power=float(cfg.get("polynomial_decay_power", 2.0)),
                delta_action_clip=float(cfg.get("delta_action_clip", 0.1)),
                noise_scale=float(cfg.get("noise_scale", 1.0)),
                initial_actions=candidates.clone(),
            )
        return self._argmax(obs_t, refined.detach())
