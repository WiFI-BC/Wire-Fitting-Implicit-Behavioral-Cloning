"""Evaluation loop for `Reach-v0`.

The thinnest possible BaseSimulation subclass: it names the environment and
says how to construct it. Action selection, frame stacking and the episode loop
are all inherited, which is what a new flat-state environment should need.
"""

from __future__ import annotations

from typing import Optional

from .base_simulation import BaseSimulation
from .reach_env import ReachEnv


class ReachSimulation(BaseSimulation):
    def __init__(
        self,
        control_point_generator,
        q_estimator,
        device: str = "cpu",
        max_episode_steps: int = 50,
        render_mode: Optional[str] = None,
        frame_stack: int = 1,
        norm_stats: Optional[dict] = None,
        goal_radius: float = 0.05,
    ) -> None:
        super().__init__(
            env_id="Reach-v0",
            control_point_generator=control_point_generator,
            q_estimator=q_estimator,
            device=device,
            max_episode_steps=max_episode_steps,
            frame_stack=frame_stack,
        )
        self.render_mode = render_mode
        self.norm_stats = norm_stats
        self.goal_radius = goal_radius
        # Reserved by the refinement wrappers in envs/evaluate.py; None means
        # the critic scores actions in the same space the generator emits.
        self._act_min_t = None
        self._act_rng_t = None

    def create_env(self) -> ReachEnv:
        return ReachEnv(
            n_steps=self.max_episode_steps,
            goal_radius=self.goal_radius,
            render_mode=self.render_mode,
        )
