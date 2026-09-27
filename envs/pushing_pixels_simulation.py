"""Simulation runner for the IBC BlockPush env in RGB-observation mode.

Sibling of `pushing_simulation.py` for the IBC paper's "Block Pushing —
Single target, Images" task. Mirrors `PushingSimulation`'s overall shape
(same env id namespace, same per-seed eval, same denormalized actions) but
with two important differences:

  1. Observations are images (H, W, 3) uint8 — frame-stacking is channel-wise
     (stacks two frames into (H, W, 3*frame_stack)) rather than the
     flat-concatenation `BaseSimulation` does for vector obs.
  2. `self.obs_normalizer` is never invoked. The conv encoder
     (`wifi_bc.models.ConvMaxpoolEncoder`) does its own preprocessing
     (uint8→float / 255, bilinear resize to 180×240). The base-class
     normalizer is left in place only because BaseSimulation.__init__
     constructs it unconditionally; a stub entry in
     `observation_bounds.json` keeps the construction from crashing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .base_simulation import BaseSimulation
from .pushing_pixels_env import PushingPixelsEnv


class PushingPixelsSimulation(BaseSimulation):
    """Evaluator for trained WiFI-BC pixel policies on PushingPixels-v0."""

    def __init__(
        self,
        control_point_generator: torch.nn.Module,
        q_estimator: torch.nn.Module,
        device: str = "cpu",
        max_episode_steps: int = 100,
        render_mode: Optional[str] = None,
        frame_stack: int = 1,
        norm_stats: Optional[dict] = None,
        goal_dist_tolerance: float = 0.02,
        execute_horizon: int = 0,
    ) -> None:
        super().__init__(
            env_id="PushingPixels-v0",
            control_point_generator=control_point_generator,
            q_estimator=q_estimator,
            device=device,
            max_episode_steps=max_episode_steps,
            frame_stack=frame_stack,
        )
        self.render_mode = render_mode
        self.goal_dist_tolerance = goal_dist_tolerance
        self.norm_stats = norm_stats
        # Action chunking: the model emits K*2 per CP. execute_horizon R is the
        # eval-time receding-horizon knob — execute the first R actions of the
        # chunk then replan. 0 (or >=K) keeps pure open-loop chunk execution.
        # Mirrors PenHumanV2Simulation.
        self.action_chunk = int((norm_stats or {}).get("action_chunk", 1) or 1)
        # Cloud-selection rule. Only pen_human_v2 honoured these before, so
        # every pixel result to date was a hard argmax.
        _nsd = norm_stats or {}
        self.cp_selection = str(_nsd.get("cp_selection", "argmax"))
        self.cp_selection_temperature = float(_nsd.get("cp_selection_temperature", 1.0) or 1.0)
        self.cp_score_norm = str(_nsd.get("cp_score_norm", "none"))
        _eh = int(execute_horizon or 0)
        self.execute_horizon = (
            self.action_chunk if _eh <= 0 else min(_eh, self.action_chunk)
        )

        # See PushingSimulation: these `_act_*_t` are reserved for the legacy
        # ibc_with_cps action remapping. Set to None to short-circuit the
        # LangevinRefinedParticleSimulation wrapper in envs/evaluate.py.
        self._act_min_t = None
        self._act_rng_t = None
        if norm_stats is not None:
            self._raw_act_min = np.asarray(norm_stats["act_min"], dtype=np.float32)
            self._raw_act_max = np.asarray(norm_stats["act_max"], dtype=np.float32)
            lo_hi = norm_stats.get("action_norm_range", (-1.0, 1.0))
            self._act_lo = float(lo_hi[0])
            self._act_hi = float(lo_hi[1])
        else:
            self._raw_act_min = None
            self._raw_act_max = None
            self._act_lo = None
            self._act_hi = None

    def create_env(self) -> PushingPixelsEnv:
        return PushingPixelsEnv(
            n_steps=self.max_episode_steps,
            render_mode=self.render_mode,
            goal_dist_tolerance=self.goal_dist_tolerance,
        )

    def _denormalize_action(self, action_normalized: np.ndarray) -> np.ndarray:
        if self._raw_act_min is None:
            return action_normalized
        scale = (self._raw_act_max - self._raw_act_min) / (self._act_hi - self._act_lo)
        return (
            self._raw_act_min + (np.asarray(action_normalized, dtype=np.float32) - self._act_lo) * scale
        ).astype(np.float32)

    # ── Frame-stacking overrides (channel-wise for images) ──────────────────
    # BaseSimulation's defaults concat 1D vectors. Images need channel-wise
    # stack: two (H, W, 3) frames → one (H, W, 6) frame.

    def _get_stacked_obs(self) -> np.ndarray:
        if self.frame_stack <= 1:
            return self._frame_buffer[-1]
        return np.concatenate(list(self._frame_buffer), axis=-1)  # (H, W, 3*fs)

    def _obs_to_tensor(self, stacked_obs_hwc: np.ndarray) -> torch.Tensor:
        """(H, W, 3*fs) uint8 → (1, 3*fs, H, W) uint8 tensor on device."""
        # Channels-last → channels-first for the conv encoder.
        chw = np.transpose(stacked_obs_hwc, (2, 0, 1))  # (3*fs, H, W)
        return torch.from_numpy(chw).unsqueeze(0).to(self.device)  # (1, 3*fs, H, W)

    def select_action(self, observation: np.ndarray, return_q_range: bool = False):
        """Pick best CP under Q. Encoder runs ONCE per state (late fusion).

        Args:
            observation: (H, W, 3*frame_stack) uint8 image stack as returned
                by `_get_stacked_obs`.
        """
        obs_tensor = self._obs_to_tensor(observation)  # (1, C, H, W) uint8

        with torch.no_grad():
            control_points = self.control_point_generator(obs_tensor)  # (1, N, action_dim)
            # Late fusion: encode once, broadcast features over the N CPs.
            features = self.q_estimator.encode(obs_tensor)  # (1, F)
            q_values = self.q_estimator.score(features, control_points).squeeze(-1)  # (1, N)
            from envs.cp_selection import select_from_cloud
            idx = select_from_cloud(q_values, control_points, self.cp_selection,
                                    self.cp_selection_temperature, self.cp_score_norm)
            action_normalized = control_points[0, idx, :].cpu().numpy()
            q_range = (q_values.min().item(), q_values.max().item())

        action = self._denormalize_action(action_normalized)
        if return_q_range:
            return action, q_range
        return action

    def run_episode(self, seed: int | None = None) -> dict:
        if self.env is None:
            self.env = self.create_env()

        obs, _ = self.env.reset(seed=seed)
        stacked_obs = self._reset_frame_buffer(obs)

        total_reward = 0.0
        episode_length = 0
        min_dist = float("inf")
        done = False
        info: dict = {}

        while not done:
            # (K*2,) denormalized when chunking; (2,) when action_chunk == 1.
            chunk = self.select_action(stacked_obs)
            steps = np.asarray(chunk).reshape(self.action_chunk, -1)
            for k in range(self.execute_horizon):
                obs, reward, terminated, truncated, info = self.env.step(steps[k])
                stacked_obs = self._update_frame_buffer(obs)
                total_reward += float(reward)
                episode_length += 1
                min_dist = min(min_dist, float(info.get("block_to_target_distance", np.inf)))
                done = terminated or truncated
                if done:
                    break

        return {
            "episode_length": episode_length,
            "total_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
            "success": bool(info.get("success", False)),
            "min_dist_to_target": min_dist,
            "final_dist_to_target": float(info.get("block_to_target_distance", np.inf)),
        }

    def get_summary(self) -> dict[str, float]:
        summary = super().get_summary()
        if not self.results:
            return summary
        successes = [r.get("success", False) for r in self.results]
        summary["success_rate"] = float(np.mean(successes))
        summary["success_rate_std"] = float(np.std(successes))
        dists = [r.get("min_dist_to_target", np.inf) for r in self.results]
        finite_dists = [d for d in dists if np.isfinite(d)]
        summary["avg_min_dist_to_target"] = float(np.mean(finite_dists)) if finite_dists else float("inf")
        summary["std_min_dist_to_target"] = float(np.std(finite_dists)) if finite_dists else float("inf")
        return summary


class PushingPixelsIBCSimulation(PushingPixelsSimulation):
    """Original-IBC (pure pixel EBM) evaluation on PushingPixels-v0.

    Reuses PushingPixelsSimulation's env, channel-wise frame stacking and
    action denormalization, but selects actions the way IBC does: no control
    points at all, just Langevin MCMC over the energy net's action argument.

    As in the LIBERO pixel evaluator, the image is encoded ONCE per env step
    and the chain runs against the value head on those features (late fusion),
    so a 100-iteration chain costs one conv forward, not a hundred.
    """

    def __init__(
        self,
        energy_net: torch.nn.Module,
        device: str = "cpu",
        max_episode_steps: int = 100,
        frame_stack: int = 1,
        norm_stats: Optional[dict] = None,
        langevin_cfg: Optional[dict] = None,
        goal_dist_tolerance: float = 0.02,
        action_in_model_range: tuple[float, float] = (-1.0, 1.0),
        uniform_boundary_buffer: float = 0.05,
    ) -> None:
        super().__init__(
            control_point_generator=None,
            q_estimator=energy_net,
            device=device,
            max_episode_steps=max_episode_steps,
            frame_stack=frame_stack,
            norm_stats=norm_stats,
            goal_dist_tolerance=goal_dist_tolerance,
        )
        self.energy_net = energy_net
        self.langevin_cfg = dict(langevin_cfg or {})
        lo, hi = float(action_in_model_range[0]), float(action_in_model_range[1])
        buf = float(uniform_boundary_buffer)
        # The chain is allowed slightly outside the action box, matching the
        # official implementation's uniform_boundary_buffer; the env clips.
        adim = int(self.env_action_dim)
        self._amin = torch.full((adim,), lo - buf, device=device)
        self._amax = torch.full((adim,), hi + buf, device=device)

    @property
    def env_action_dim(self) -> int:
        """Action width the energy net was trained on."""
        if self._raw_act_min is not None:
            return int(np.asarray(self._raw_act_min).reshape(-1).shape[0])
        return 2  # PushingPixels-v0 is a 2-D end-effector setpoint

    def select_action(self, observation: np.ndarray, return_q_range: bool = False):
        from wifi_bc.sampling import sample_langevin

        obs_tensor = self._obs_to_tensor(observation)  # (1, C, H, W) uint8
        c = self.langevin_cfg
        with torch.no_grad():
            feats = self.energy_net.encode(obs_tensor)  # (1, F) — once per step

        samples = sample_langevin(
            energy_function=lambda o, a: self.energy_net.score(feats, a).squeeze(-1),
            observations=obs_tensor,  # batch-size / device carrier only
            num_samples=int(c.get("num_samples", 512)),
            action_min=self._amin,
            action_max=self._amax,
            num_iterations=int(c.get("num_iterations", 100)),
            lr_init=float(c.get("lr_init", 0.1)),
            lr_final=float(c.get("lr_final", 1e-5)),
            polynomial_decay_power=float(c.get("polynomial_decay_power", 2.0)),
            delta_action_clip=float(c.get("delta_action_clip", 0.1)),
            noise_scale=float(c.get("noise_scale", 0.1)),
            device=self.device,
            noise_via_stepsize=bool(c.get("noise_via_stepsize", False)),
        )
        with torch.no_grad():
            e = self.energy_net.score(feats, samples).squeeze(-1)  # (1, N)
            best = samples[0, e.argmin(dim=-1)[0]].cpu().numpy()
            q_range = (float(-e.max()), float(-e.min()))

        action = self._denormalize_action(best)
        if return_q_range:
            return action, q_range
        return action
