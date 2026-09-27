"""Dataset classes for loading training data from various sources.

Supports frame stacking: concatenating N consecutive observations into a single
state vector to give the model temporal context.
"""

import os
import glob
from typing import Optional
import numpy as np
from torch.utils.data import Dataset
import minari

try:
    # Import torch's lazy `_dynamo` submodule BEFORE TensorFlow. TF pulls in
    # jaxlib, and with jaxlib already resident the first import of
    # torch._dynamo segfaults the process. That import is triggered lazily by
    # the first `torch.optim.*` construction, so without this line every
    # training script on a TFRecord task (particle, pushing, pushing_pixels)
    # dies with SIGSEGV the moment it builds its optimizer — after the dataset
    # has loaded, which makes it look like a data problem. Importing it here,
    # ahead of TF, is the whole fix.
    import torch._dynamo  # noqa: F401

    # TF preallocates the whole GPU at first device touch by default, which
    # starves PyTorch. Setting allow-growth before import keeps TF on the GPU
    # for fast tf.data pipeline ops while only reserving what it actually uses
    # (typically <1 GB for TFRecord parsing).
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    import tensorflow as tf
    # Force TF to CPU. We only use TF for:
    #   - TFRecord parsing in __init__ (CPU op)
    #   - tf.io.decode_image in PushingPixelsDataset.__getitem__ (CPU op)
    # If TF grabs CUDA, DataLoader workers forked after PyTorch's CUDA init
    # crash with `CUDA_ERROR_NOT_INITIALIZED` when they touch the (post-fork
    # broken) CUDA context. set_visible_devices([], "GPU") prevents that
    # without affecting PyTorch's GPU access. The try/except handles the
    # case where TF has already initialized GPUs (then this is a no-op).
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def stack_frames(observations: np.ndarray, episode_starts: np.ndarray, frame_stack: int) -> np.ndarray:
    """Stack consecutive frames into a single observation vector.
    
    For each timestep t, the stacked observation is:
        [obs[t - (frame_stack-1)], obs[t - (frame_stack-2)], ..., obs[t]]
    
    At episode boundaries, earlier frames are filled by repeating the first
    observation of the episode (zero-padding alternative would lose position info).
    
    Args:
        observations: Array of shape (N, obs_dim) with all observations.
        episode_starts: Array of shape (N,) with True at the start of each episode.
        frame_stack: Number of frames to stack.
        
    Returns:
        Stacked observations of shape (N, obs_dim * frame_stack).
    """
    if frame_stack <= 1:
        return observations
    
    n_samples, obs_dim = observations.shape
    stacked = np.zeros((n_samples, obs_dim * frame_stack), dtype=observations.dtype)
    
    for i in range(n_samples):
        frames = []
        for k in range(frame_stack - 1, -1, -1):  # oldest to newest
            idx = i - k
            # Check if we crossed an episode boundary
            if idx < 0 or np.any(episode_starts[idx + 1:i + 1]) if idx < i else False:
                # Pad with the earliest available frame in this episode
                # Find episode start
                ep_start = i
                while ep_start > 0 and not episode_starts[ep_start]:
                    ep_start -= 1
                idx = max(idx, ep_start)
            elif idx < 0:
                idx = 0
            frames.append(observations[idx])
        stacked[i] = np.concatenate(frames)
    
    return stacked


def build_chunked_actions(raw_actions: np.ndarray, episode_starts: np.ndarray, K: int) -> np.ndarray:
    """Turn per-step actions into K-step chunk targets (action chunking).

    Sample t's target becomes [a_t, ..., a_{t+K-1}] flattened to (N, K*A).
    Windows never cross an episode boundary: indices past the episode's last
    step repeat that final action (same padding policy as the LIBERO pixel
    dataset, where chunking was first validated). Call BEFORE computing action
    stats / normalization so act_min/max cover the chunked vector.
    """
    if K <= 1:
        return raw_actions
    n, a = raw_actions.shape
    episode_id = np.cumsum(episode_starts) - 1
    # For each step, the absolute index of its episode's LAST step.
    last_of_ep = np.empty(n, dtype=np.int64)
    ep_last: dict[int, int] = {}
    for i in range(n - 1, -1, -1):
        e = int(episode_id[i])
        if e not in ep_last:
            ep_last[e] = i
        last_of_ep[i] = ep_last[e]
    chunks = np.empty((n, K, a), dtype=np.float32)
    for k in range(K):
        idx = np.minimum(np.arange(n) + k, last_of_ep)
        chunks[:, k] = raw_actions[idx]
    return chunks.reshape(n, K * a)


class D4RLDataset(Dataset):
    """Minari D4RL dataset wrapper with IBC-paper-faithful normalization.

    IBC paper (Florence et al. 2021, App. B.1 / B.3) normalizes:
      - observations: per-dim zero-mean unit-variance (standardize), and
      - actions:      per-dim min-max to `action_norm_range` (default [-1, 1]).

    Stats are computed from the UNSTACKED observations / raw actions so they
    apply to one frame at a time; ObservationNormalizer repeats them
    `frame_stack` times when consuming stacked obs. `act_min`/`act_max` are
    exposed for the eval-time simulation to invert via
    `unnormalize_action()` before stepping the env (mirrors PushingDataset).
    """

    def __init__(
        self,
        root: str,
        download: bool = True,
        frame_stack: int = 1,
        normalize_actions: bool = True,
        action_norm_range: tuple[float, float] = (-1.0, 1.0),
        obs_indices: list[int] | None = None,
        action_chunk: int = 1,
    ):
        self.dataset_name = root
        self.action_chunk = max(1, int(action_chunk))
        self.dataset = self._load_dataset(root, download=download)
        self.frame_stack = frame_stack
        self.normalize_actions = normalize_actions
        self.action_norm_range = action_norm_range
        # Optional column selection on the raw observation vector, applied
        # BEFORE stats/stacking. Used to reproduce the IBC paper's kitchen
        # input: legacy d4rl kitchen obs = robot qpos(9)+obj qpos(21)+goal(30,
        # constant for -complete). The gymnasium-robotics port instead emits
        # qpos+QVEL (59-D); selecting [0:9]+[18:39] recovers the paper's
        # informative content (velocities add 29 noisy dims on 4.2k samples).
        self.obs_indices = list(obs_indices) if obs_indices is not None else None

        all_observations = []
        all_actions = []
        episode_starts = []

        for ep in self.dataset.iterate_episodes():
            # FrankaKitchen episodes carry a Dict observation
            # {observation, achieved_goal, desired_goal}; the policy trains on
            # the flat 'observation' vector (goal is fixed for -complete).
            ep_obs = ep.observations
            if isinstance(ep_obs, dict):
                ep_obs = np.asarray(ep_obs["observation"])
            if self.obs_indices is not None:
                ep_obs = ep_obs[:, self.obs_indices]
            obs = ep_obs[:-1]  # exclude terminal observation
            acts = ep.actions
            starts = np.zeros(len(obs), dtype=bool)
            starts[0] = True
            all_observations.append(obs)
            all_actions.append(acts)
            episode_starts.append(starts)

        self.observations = np.concatenate(all_observations).astype(np.float32)
        raw_actions = np.concatenate(all_actions).astype(np.float32)
        self._episode_starts = np.concatenate(episode_starts)

        # Action chunking: replace per-step targets with K-step windows BEFORE
        # stats so normalization covers the full (K*A) chunk vector.
        if self.action_chunk > 1:
            raw_actions = build_chunked_actions(
                raw_actions, self._episode_starts, self.action_chunk
            )

        # ─── Dataset statistics (paper-faithful, from raw obs/actions) ──────
        # Obs stats are computed on UNSTACKED obs — ObservationNormalizer tiles
        # them frame_stack times. Small std floor avoids div-by-zero on any
        # degenerate dim (Adroit obs dims are healthy in practice; defensive).
        self.obs_mean = self.observations.mean(axis=0).astype(np.float32)
        self.obs_std = (self.observations.std(axis=0) + 1e-6).astype(np.float32)
        self.act_min = raw_actions.min(axis=0).astype(np.float32)
        self.act_max = raw_actions.max(axis=0).astype(np.float32)

        if normalize_actions:
            lo, hi = float(action_norm_range[0]), float(action_norm_range[1])
            denom = self.act_max - self.act_min
            denom = np.where(denom == 0, np.ones_like(denom), denom)
            self.actions = (
                lo + (raw_actions - self.act_min) * (hi - lo) / denom
            ).astype(np.float32)
        else:
            self.actions = raw_actions

        if frame_stack > 1:
            self.observations = stack_frames(
                self.observations, self._episode_starts, frame_stack
            )

        self.state_shape = self.observations.shape[1]  # obs_dim * frame_stack
        self.action_shape = self.actions.shape[1]

    @staticmethod
    def _load_dataset(root: str, download: bool):
        """Load from Minari cache, downloading into it when the dataset is missing."""
        try:
            return minari.load_dataset(root, download=False)
        except Exception:
            if not download:
                raise
            print(f"Minari dataset {root!r} not found locally; downloading...")
            return minari.load_dataset(root, download=True)

    def unnormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        """Inverse of the action normalization applied in __init__.

        Use this at env.step time to convert the model's output (in
        `action_norm_range`) back to the env's native action box.
        """
        if not self.normalize_actions:
            return np.asarray(normalized_action, dtype=np.float32)
        lo, hi = float(self.action_norm_range[0]), float(self.action_norm_range[1])
        scale = (self.act_max - self.act_min) / (hi - lo)
        return (
            self.act_min + (np.asarray(normalized_action, dtype=np.float32) - lo) * scale
        ).astype(np.float32)

    def __getitem__(self, index):
        return {'state': self.observations[index], 'action': self.actions[index]}

    def __len__(self):
        return len(self.observations)


class ParticleDataset(Dataset):
    """Dataset for loading particle environment demonstrations from TFRecord files.
    
    The particle environment observation consists of:
    - pos_agent (n_dim): Agent position
    - vel_agent (n_dim): Agent velocity  
    - pos_first_goal (n_dim): First goal position
    - pos_second_goal (n_dim): Second goal position
    
    Total observation dim: 4 * n_dim (before stacking)
    After stacking: 4 * n_dim * frame_stack
    Action dim: n_dim (position setpoint)
    """
    
    def __init__(self, data_dir: str, n_dim: int = 2, frame_stack: int = 1):
        """Initialize the particle dataset.
        
        Args:
            data_dir: Directory containing TFRecord files.
            n_dim: Dimensionality of the particle environment (1, 2, 3, ..., 32).
            frame_stack: Number of consecutive frames to stack into one observation.
        """
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required to load particle TFRecord files. "
                "Install with: pip install tensorflow"
            )
        
        self.data_dir = data_dir
        self.n_dim = n_dim
        self.frame_stack = frame_stack
        self._base_obs_dim = 4 * n_dim  # single-frame observation dimension
        self.action_shape = n_dim       # position setpoint
        
        # Find all matching TFRecord files
        pattern = os.path.join(data_dir, f"{n_dim}d_oracle_particle_*.tfrecord")
        self.tfrecord_files = sorted(glob.glob(pattern))
        
        if not self.tfrecord_files:
            raise FileNotFoundError(
                f"No TFRecord files found matching pattern: {pattern}\n"
                f"Available files in {data_dir}: {os.listdir(data_dir)[:10]}..."
            )
        
        # Load all data into memory
        self.observations, self.actions, self._episode_starts = self._load_all_data()
        
        # Apply frame stacking
        if frame_stack > 1:
            self.observations = stack_frames(self.observations, self._episode_starts, frame_stack)
        
        self.state_shape = self.observations.shape[1]  # obs_dim * frame_stack
        
    def _parse_tfrecord(self, serialized_example):
        """Parse a single TFRecord example."""
        feature_description = {
            'observation/pos_agent': tf.io.FixedLenFeature([self.n_dim], tf.float32),
            'observation/vel_agent': tf.io.FixedLenFeature([self.n_dim], tf.float32),
            'observation/pos_first_goal': tf.io.FixedLenFeature([self.n_dim], tf.float32),
            'observation/pos_second_goal': tf.io.FixedLenFeature([self.n_dim], tf.float32),
            'action': tf.io.FixedLenFeature([self.n_dim], tf.float32),
        }
        
        try:
            example = tf.io.parse_single_example(serialized_example, feature_description)
            return example
        except tf.errors.InvalidArgumentError:
            return None

    @staticmethod
    def _decode_step_type(feature) -> int | None:
        """Decode TF-Agents step_type from a TF Example feature.

        In these particle TFRecords, step_type may be stored either as an
        int64_list scalar or as raw bytes (little-endian integer).
        Returns None when the value cannot be decoded.
        """
        try:
            if feature.int64_list.value:
                return int(feature.int64_list.value[0])
            if feature.bytes_list.value:
                raw = feature.bytes_list.value[0]
                # TF-Agents often serializes small scalar ints into raw bytes.
                return int.from_bytes(raw, byteorder="little", signed=False)
        except Exception:
            return None
        return None
    
    def _load_all_data(self):
        """Load all data from TFRecord files into numpy arrays.
        
        Returns:
            observations, actions, episode_starts arrays.
        """
        all_observations = []
        all_actions = []
        episode_starts = []
        
        for tfrecord_file in self.tfrecord_files:
            raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
            is_first_in_episode = True
            
            for raw_record in raw_dataset:
                try:
                    example = tf.train.Example()
                    example.ParseFromString(raw_record.numpy())
                    features = example.features.feature
                    
                    pos_agent = np.array(features['observation/pos_agent'].float_list.value, dtype=np.float32)
                    vel_agent = np.array(features['observation/vel_agent'].float_list.value, dtype=np.float32)
                    pos_first_goal = np.array(features['observation/pos_first_goal'].float_list.value, dtype=np.float32)
                    pos_second_goal = np.array(features['observation/pos_second_goal'].float_list.value, dtype=np.float32)
                    
                    observation = np.concatenate([pos_agent, vel_agent, pos_first_goal, pos_second_goal])
                    action = np.array(features['action'].float_list.value, dtype=np.float32)
                    
                    all_observations.append(observation)
                    all_actions.append(action)
                    
                    # Detect episode boundaries via step_type (0=FIRST) or file boundaries.
                    # Important: step_type is byte-encoded in these TFRecords.
                    is_start = is_first_in_episode
                    try:
                        if 'step_type' in features:
                            step_type = self._decode_step_type(features['step_type'])
                            if step_type is not None:
                                is_start = (step_type == 0)
                    except Exception:
                        pass
                    episode_starts.append(is_start)
                    is_first_in_episode = False
                    
                except Exception:
                    continue
        
        if not all_observations:
            raise ValueError(
                f"No valid records found in TFRecord files. "
                f"Files checked: {self.tfrecord_files}"
            )
        
        return (
            np.array(all_observations),
            np.array(all_actions),
            np.array(episode_starts, dtype=bool)
        )
    
    def __getitem__(self, index):
        return {
            'state': self.observations[index],
            'action': self.actions[index]
        }
    
    def __len__(self):
        return len(self.observations)


if __name__ == "__main__":
    # Test D4RL dataset
    print("Testing D4RLDataset...")
    dataset = D4RLDataset('D4RL/pen/human-v2', download=True)
    print(f"Dataset length: {len(dataset), len(dataset.observations), len(dataset.actions)}")
    sample = dataset[0]
    print(f"Sample state shape: {sample['state'].shape}, action shape: {sample['action'].shape}")

    # Test D4RL with frame stacking
    print("\nTesting D4RLDataset with frame_stack=3...")
    dataset_stacked = D4RLDataset('D4RL/pen/human-v2', download=True, frame_stack=3)
    print(f"Stacked state shape: {dataset_stacked.state_shape}")

    # Test Particle dataset
    print("\nTesting ParticleDataset...")
    particle_ds = ParticleDataset("datasets/particle", n_dim=2)
    print(f"Dataset length: {len(particle_ds)}")
    sample = particle_ds[0]
    print(f"Sample state shape: {sample['state'].shape}, action shape: {sample['action'].shape}")

    # Test Particle with frame stacking
    print("\nTesting ParticleDataset with frame_stack=3...")
    particle_stacked = ParticleDataset("datasets/particle", n_dim=2, frame_stack=3)
    print(f"Stacked state shape: {particle_stacked.state_shape}")
    sample = particle_stacked[0]
    print(f"Sample state shape: {sample['state'].shape}")
class PointMazePillarDataset(Dataset):
    """Synthetic dataset for PointMazePillar-v0 — the "realistic" complement
    to DummyBimodalDataset, generated by actually rolling the oracle through
    the real MuJoCo env (envs.point_maze_pillar_env), not a hand-coded
    approximation of its dynamics.

    Fixed start (-2, 0) and goal (2, 0), pillar centered at the origin.
    Oracle: a 3-stage PD waypoint controller — (1) go straight up/down to the
    open corridor row at y=+-2, (2) cross to the goal's x, (3) descend into
    the goal — with a fresh per-EPISODE coin flip choosing +2 (top corridor)
    or -2 (bottom corridor). Since start/goal never change, every episode is
    the exact same ambiguous decision (confirmed with user — this is
    intentional, not an oversight: it keeps every snapshot informative
    without needing dummy_bimodal's `ambiguous_frac` rebalancing at all).

    Action: 2D force in [-1, 1]^2 (native PointMaze action space) — the
    demonstrated force at each step, i.e. the PD controller's OWN output,
    clipped exactly as the env would clip it. This is what the policy is
    trained to imitate.

    DAgger-style recovery augmentation (`position_noise_prob`): the clean
    oracle transitions between waypoints well before drifting near a wall,
    so plain rollouts barely visit "already overshot, pinned against the
    wall" states — a trained BC policy that drifts even slightly off the
    oracle's exact path lands somewhere it never saw a demonstration for,
    and (observed directly via rollout on a real checkpoint) gets stuck
    repeating the wrong action forever. Periodically teleporting the agent
    to a random nearby offset (still recomputing the CORRECT oracle action
    for wherever it lands, including near walls) fixes this by teaching
    recovery, not just the nominal path.
    """

    def __init__(
        self,
        size: int = 20000,
        max_steps_per_episode: int = 400,
        waypoint_tolerance: float = 0.35,
        detour_y: float = 2.0,
        kp: float = 6.0,
        kv: float = 3.0,
        expert_noise_std: float = 0.02,
        position_noise_prob: float = 0.01,
        position_noise_std: float = 0.4,
        frame_stack: int = 1,
    ):
        import mujoco

        from envs.point_maze_pillar_env import PointMazePillarEnv

        self.frame_stack = frame_stack

        all_observations = []
        all_actions = []
        episode_starts = []

        total_samples = 0
        rng = np.random.default_rng(seed=42)
        env = PointMazePillarEnv(max_episode_steps=max_steps_per_episode)
        maze_unwrapped = env._inner.unwrapped

        def teleport(new_pos):
            maze_unwrapped.data.qpos[:2] = new_pos
            maze_unwrapped.data.qvel[:2] = 0.0
            mujoco.mj_forward(maze_unwrapped.model, maze_unwrapped.data)

        while total_samples < size:
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            goal = obs[4:6].copy()
            start = obs[:2].copy()
            side = 1.0 if rng.random() < 0.5 else -1.0
            waypoints = [
                np.array([start[0], side * detour_y], dtype=np.float32),
                np.array([goal[0], side * detour_y], dtype=np.float32),
                goal,
            ]
            wp_idx = 0

            ep_obs = []
            ep_acts = []

            for step_i in range(max_steps_per_episode):
                # Recovery augmentation: occasionally teleport to a nearby
                # random offset (clipped to the maze's outer bounds) BEFORE
                # computing this step's action, so the demonstrated action
                # is the correct recovery for the perturbed position, not a
                # continuation of the clean path.
                # Never teleport once on final approach to the goal (wp_idx
                # == last stage) — early tests showed frequent teleports
                # there repeatedly knock the agent out of the tight
                # (0.45-unit) success radius, so most demonstrations timed
                # out instead of reaching the goal.
                if step_i > 0 and wp_idx < len(waypoints) - 1 and rng.random() < position_noise_prob:
                    offset = rng.normal(0, position_noise_std, size=2)
                    new_pos = np.clip(obs[:2] + offset, -2.9, 2.9).astype(np.float64)
                    # Never teleport INTO the pillar (max(|x|,|y|) < 1.6 is
                    # solid wall) — MuJoCo's constraint solver would shove
                    # the agent out violently next step, producing a bogus
                    # "recovery" action unrelated to the oracle's intent.
                    if max(abs(new_pos[0]), abs(new_pos[1])) > 1.6:
                        teleport(new_pos)
                        obs = np.concatenate([new_pos.astype(np.float32), [0.0, 0.0], goal])

                pos = obs[:2]
                vel = obs[2:4]
                target = waypoints[wp_idx]
                if wp_idx < len(waypoints) - 1 and np.linalg.norm(pos - target) < waypoint_tolerance:
                    wp_idx += 1
                    target = waypoints[wp_idx]

                force = kp * (target - pos) - kv * vel
                noise = rng.normal(0, expert_noise_std, size=2)
                action = np.clip(force + noise, -1.0, 1.0).astype(np.float32)

                ep_obs.append(obs.copy())
                ep_acts.append(action)

                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break

            ep_starts = np.zeros(len(ep_obs), dtype=bool)
            ep_starts[0] = True

            all_observations.append(np.array(ep_obs))
            all_actions.append(np.array(ep_acts))
            episode_starts.append(ep_starts)
            total_samples += len(ep_obs)

        env.close()

        self.observations = np.concatenate(all_observations)[:size]
        self.actions = np.concatenate(all_actions)[:size]
        self._episode_starts = np.concatenate(episode_starts)[:size]

        if frame_stack > 1:
            self.observations = stack_frames(
                self.observations, self._episode_starts, frame_stack
            )

        self.state_shape = self.observations.shape[1]
        self.action_shape = self.actions.shape[1]

    def __getitem__(self, index):
        return {'state': self.observations[index], 'action': self.actions[index]}

    def __len__(self):
        return len(self.observations)


class PushingDataset(Dataset):
    """Dataset for the IBC paper's Simulated Pushing task (single target).

    Loads the official `block_push_states_location` TFRecord oracle dataset
    published with Florence et al. 2021 (Implicit Behavioral Cloning) —
    download instructions in the IBC README:
        https://storage.googleapis.com/brain-reach-public/ibc_data/block_push_states_location.zip

    State layout (10D before frame-stacking) — MUST stay aligned with the
    canonical ordering used by `envs.pushing_env.OBS_KEYS_AND_DIMS`:
        [block_translation (2), block_orientation (1),
         effector_translation (2), effector_target_translation (2),
         target_translation (2), target_orientation (1)]
    Action (2D): xArm planar position delta (data-driven range
        [-0.0255, -0.0209] → [0.0287, 0.0427]).
    """

    # Canonical key order. Sync with envs.pushing_env.OBS_KEYS_AND_DIMS.
    _FEATURE_KEYS = (
        ("observation/block_translation", 2),
        ("observation/block_orientation", 1),
        ("observation/effector_translation", 2),
        ("observation/effector_target_translation", 2),
        ("observation/target_translation", 2),
        ("observation/target_orientation", 1),
    )

    # Glob pattern for the IBC TFRecord shards. Subclasses (multimodal) override.
    _TFRECORD_GLOB = "oracle_push_*.tfrecord"
    # Short human-readable name of the IBC zip to point users at when the glob
    # finds no files. Subclasses override.
    _DATASET_ZIP_NAME = "block_push_states_location.zip"

    def __init__(
        self,
        data_dir: str = "datasets/block_push/block_push_states_location",
        frame_stack: int = 1,
        max_samples: Optional[int] = None,
        normalize_actions: bool = True,
        action_norm_range: tuple[float, float] = (-1.0, 1.0),
    ):
        """Load the IBC block_push oracle dataset.

        Args:
            data_dir: Directory of `oracle_push_*.tfrecord` files.
            frame_stack: Concatenate the previous (frame_stack - 1) obs into
                the current observation. IBC paper uses 2.
            max_samples: Optional cap (default: load all 75k transitions).
            normalize_actions: When True, return actions linearly mapped to
                `action_norm_range`. This matches the IBC pipeline
                (`compute_dataset_statistics.min_max_actions=True` in
                `pushing_states/mlp_ebm_langevin.gin`) — the network operates
                in normalized action space, denormalized back to raw effector
                deltas only at env.step time. Stats persist in attributes
                `act_min` / `act_max` so callers can denormalize.
            action_norm_range: Linear target range for action normalization.
                Default `(-1, 1)` matches IBC; pass `(0, 1)` for the
                ibc_with_cps convention.
        """
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required to load IBC block_push TFRecord files. "
                "Install with: uv add tensorflow"
            )

        self.frame_stack = frame_stack
        self.data_dir = data_dir
        self.normalize_actions = normalize_actions
        self.action_norm_range = action_norm_range
        self._base_obs_dim = sum(d for _, d in self._FEATURE_KEYS)  # 10

        pattern = os.path.join(data_dir, self._TFRECORD_GLOB)
        self.tfrecord_files = sorted(glob.glob(pattern))
        if not self.tfrecord_files:
            raise FileNotFoundError(
                f"No TFRecord files match {pattern}. Did you download "
                f"{self._DATASET_ZIP_NAME}?"
            )

        self.observations, raw_actions, self._episode_starts = self._load_all_data(
            max_samples=max_samples
        )

        # ─── Dataset statistics (paper-faithful: from raw obs/actions) ──────
        # Computed on UNSTACKED obs so they apply to one frame at a time.
        # The ObservationNormalizer will repeat them frame_stack times.
        self.obs_mean = self.observations.mean(axis=0).astype(np.float32)
        # Small floor on std avoids divide-by-zero for any degenerate dim
        # (block_orientation in particular has near-uniform coverage so std
        # is healthy; this is defensive).
        self.obs_std = (self.observations.std(axis=0) + 1e-6).astype(np.float32)
        self.act_min = raw_actions.min(axis=0).astype(np.float32)
        self.act_max = raw_actions.max(axis=0).astype(np.float32)

        # ─── Action normalization (paper-faithful) ───────────────────────────
        # Linearly map per-dim from [act_min, act_max] → action_norm_range.
        # The reverse map is `_unnormalize_action` for use at env.step time.
        if normalize_actions:
            lo, hi = float(action_norm_range[0]), float(action_norm_range[1])
            denom = (self.act_max - self.act_min)
            # Guard near-degenerate dims (shouldn't happen for pushing but
            # cheap insurance for future datasets).
            denom = np.where(denom == 0, np.ones_like(denom), denom)
            self.actions = (lo + (raw_actions - self.act_min) * (hi - lo) / denom).astype(np.float32)
        else:
            self.actions = raw_actions

        if frame_stack > 1:
            self.observations = stack_frames(
                self.observations, self._episode_starts, frame_stack
            )

        self.state_shape = self.observations.shape[1]
        self.action_shape = self.actions.shape[1]

    def unnormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        """Inverse of the action normalization applied in __init__.

        Use this at env.step time to convert the model's output (in
        `action_norm_range`) back to a raw effector delta in the env's
        native action box.
        """
        if not self.normalize_actions:
            return np.asarray(normalized_action, dtype=np.float32)
        lo, hi = float(self.action_norm_range[0]), float(self.action_norm_range[1])
        scale = (self.act_max - self.act_min) / (hi - lo)
        return (self.act_min + (np.asarray(normalized_action, dtype=np.float32) - lo) * scale).astype(np.float32)

    @staticmethod
    def _decode_step_type(feature) -> Optional[int]:
        """Decode tf-agents step_type, which is stored as 1-byte raw bytes."""
        try:
            if feature.int64_list.value:
                return int(feature.int64_list.value[0])
            if feature.bytes_list.value:
                raw = feature.bytes_list.value[0]
                return int.from_bytes(raw, byteorder="little", signed=False)
        except Exception:
            return None
        return None

    def _load_all_data(self, max_samples: Optional[int] = None):
        all_obs: list[np.ndarray] = []
        all_acts: list[np.ndarray] = []
        ep_starts: list[bool] = []
        total = 0

        for tfrecord_file in self.tfrecord_files:
            raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
            is_first_in_file = True
            for raw_record in raw_dataset:
                try:
                    example = tf.train.Example()
                    example.ParseFromString(raw_record.numpy())
                    features = example.features.feature

                    chunks = []
                    for key, dim in self._FEATURE_KEYS:
                        vals = np.asarray(
                            features[key].float_list.value, dtype=np.float32
                        )
                        if vals.shape[0] != dim:
                            raise ValueError(
                                f"Feature {key} has shape {vals.shape}, expected ({dim},)"
                            )
                        chunks.append(vals)
                    obs = np.concatenate(chunks)
                    action = np.asarray(
                        features["action"].float_list.value, dtype=np.float32
                    )

                    # Episode boundary detection. tf-agents step_type:
                    # 0=FIRST, 1=MID, 2=LAST. Treat 0 as start.
                    is_start = is_first_in_file
                    st_val = None
                    if "step_type" in features:
                        st_val = self._decode_step_type(features["step_type"])
                        if st_val is not None:
                            is_start = (st_val == 0)
                    is_first_in_file = False

                    # SKIP terminal rows. tf-agents Trajectory stores a row
                    # for the LAST step where the action is a placeholder /
                    # boundary value, not what the expert actually executed
                    # from the terminal state. Training a policy on
                    # (terminal_obs → boundary_action) introduces a
                    # ~episode-count fraction of noisy supervision and
                    # corrupts the BC objective.
                    if st_val == 2:  # LAST
                        continue

                    all_obs.append(obs)
                    all_acts.append(action)
                    ep_starts.append(is_start)
                    total += 1
                    if max_samples is not None and total >= max_samples:
                        break
                except Exception:
                    continue
            if max_samples is not None and total >= max_samples:
                break

        if not all_obs:
            raise ValueError(f"No valid records found in {self.tfrecord_files}")

        return (
            np.array(all_obs, dtype=np.float32),
            np.array(all_acts, dtype=np.float32),
            np.array(ep_starts, dtype=bool),
        )

    def __getitem__(self, index):
        return {"state": self.observations[index], "action": self.actions[index]}

    def __len__(self):
        return len(self.observations)


class PushingPixelsDataset(Dataset):
    """Dataset for the IBC paper's Simulated Pushing task (Single target, IMAGES).

    Loads the official `block_push_visual_location` TFRecord oracle dataset
    published with Florence et al. 2021 (Implicit Behavioral Cloning):
        https://storage.googleapis.com/brain-reach-public/ibc_data/block_push_visual_location.zip

    Unzip into `datasets/block_push/block_push_visual_location/` (oracle_*.tfrecord
    files at the top level — flatten any nested folder if needed).

    Storage strategy: LAZY. We scan all TFRecords at __init__ and keep the
    JPEG-encoded `observation/rgb` bytes (~14 KB/frame) in a Python list +
    the float actions and episode-start flags in numpy arrays. JPEG decode
    happens per __getitem__ call. RAM footprint:
        ~100k frames × ~14 KB = ~1.4 GB encoded
        + ~100k × 8 bytes (action) = ~800 KB
    Decode is a few ms per call so num_workers≥4 in the DataLoader keeps the
    pipeline GPU-bound.

    __getitem__ returns:
        state:  (3*frame_stack, H, W) uint8 channel-stacked image
                H=240, W=320 native env resolution. The conv encoder
                (wifi_bc.models.ConvMaxpoolEncoder) does its own bilinear
                resize to (180, 240) internally.
        action: (2,) float32 in `action_norm_range` (default [-1, 1]).

    Action normalization mirrors PushingDataset (min-max from raw oracle
    actions). The `act_min`/`act_max` and `action_norm_range` attrs are
    exposed for the eval-time simulation to invert.
    """

    _IMAGE_KEY = "observation/rgb"
    _ACTION_KEY = "action"
    _STEP_TYPE_KEY = "step_type"
    _TFRECORD_GLOB = "oracle_*.tfrecord"
    _DATASET_ZIP_NAME = "block_push_visual_location.zip"
    _IMAGE_HEIGHT = 240
    _IMAGE_WIDTH = 320
    _IMAGE_CHANNELS = 3

    def __init__(
        self,
        data_dir: str = "datasets/block_push/block_push_visual_location",
        frame_stack: int = 1,
        max_samples: Optional[int] = None,
        normalize_actions: bool = True,
        action_norm_range: tuple[float, float] = (-1.0, 1.0),
        action_chunk: int = 1,
    ):
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required to load IBC block_push TFRecord files. "
                "Install with: uv add tensorflow"
            )

        self.frame_stack = frame_stack
        self.data_dir = data_dir
        self.normalize_actions = normalize_actions
        self.action_norm_range = action_norm_range
        # K-step action chunking (1 = off). action_shape becomes K*2.
        self.action_chunk = max(1, int(action_chunk))

        pattern = os.path.join(data_dir, self._TFRECORD_GLOB)
        self.tfrecord_files = sorted(glob.glob(pattern))
        if not self.tfrecord_files:
            raise FileNotFoundError(
                f"No TFRecord files match {pattern}. Did you download "
                f"{self._DATASET_ZIP_NAME}?"
            )

        (
            self._encoded_rgb,
            raw_actions,
            self._episode_starts,
        ) = self._scan_all(max_samples=max_samples)

        # Action chunking: replace per-step targets with K-step windows BEFORE
        # stats so normalization covers the full (K*A) chunk vector. Windows
        # never cross an episode boundary (see build_chunked_actions).
        if self.action_chunk > 1:
            raw_actions = build_chunked_actions(
                raw_actions, self._episode_starts, self.action_chunk
            )

        self.act_min = raw_actions.min(axis=0).astype(np.float32)
        self.act_max = raw_actions.max(axis=0).astype(np.float32)

        if normalize_actions:
            lo, hi = float(action_norm_range[0]), float(action_norm_range[1])
            denom = self.act_max - self.act_min
            denom = np.where(denom == 0, np.ones_like(denom), denom)
            self.actions = (
                lo + (raw_actions - self.act_min) * (hi - lo) / denom
            ).astype(np.float32)
        else:
            self.actions = raw_actions

        # Pre-compute, for each step i, the indices to read for frame-stacking.
        # At episode boundaries the earliest frames are repeated (same policy
        # as `stack_frames` for flat obs — keeps position information rather
        # than zero-padding).
        self._stack_indices = self._build_stack_index_map()

        # Per-frame uint8 image is the model-facing "state". We expose its
        # shape so the training-script reads `dataset.state_shape` the same
        # way it does for flat datasets.
        self.state_shape = (
            self._IMAGE_CHANNELS * frame_stack,
            self._IMAGE_HEIGHT,
            self._IMAGE_WIDTH,
        )
        self.action_shape = self.actions.shape[1]

    def unnormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        if not self.normalize_actions:
            return np.asarray(normalized_action, dtype=np.float32)
        lo, hi = float(self.action_norm_range[0]), float(self.action_norm_range[1])
        scale = (self.act_max - self.act_min) / (hi - lo)
        return (
            self.act_min + (np.asarray(normalized_action, dtype=np.float32) - lo) * scale
        ).astype(np.float32)

    @staticmethod
    def _decode_step_type(feature) -> Optional[int]:
        try:
            if feature.int64_list.value:
                return int(feature.int64_list.value[0])
            if feature.bytes_list.value:
                raw = feature.bytes_list.value[0]
                return int.from_bytes(raw, byteorder="little", signed=False)
        except Exception:
            return None
        return None

    def _scan_all(self, max_samples: Optional[int] = None):
        encoded_rgb: list[bytes] = []
        all_acts: list[np.ndarray] = []
        ep_starts: list[bool] = []
        total = 0

        for tfrecord_file in self.tfrecord_files:
            raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
            is_first_in_file = True
            for raw_record in raw_dataset:
                try:
                    example = tf.train.Example()
                    example.ParseFromString(raw_record.numpy())
                    features = example.features.feature

                    is_start = is_first_in_file
                    st_val = None
                    if self._STEP_TYPE_KEY in features:
                        st_val = self._decode_step_type(features[self._STEP_TYPE_KEY])
                        if st_val is not None:
                            is_start = (st_val == 0)
                    is_first_in_file = False

                    # SKIP terminal rows: same logic as PushingDataset — the
                    # last step's action is a tf-agents boundary placeholder,
                    # not the executed expert action.
                    if st_val == 2:  # LAST
                        continue

                    rgb_bytes = features[self._IMAGE_KEY].bytes_list.value[0]
                    action = np.asarray(
                        features[self._ACTION_KEY].float_list.value, dtype=np.float32
                    )

                    encoded_rgb.append(rgb_bytes)
                    all_acts.append(action)
                    ep_starts.append(is_start)
                    total += 1
                    if max_samples is not None and total >= max_samples:
                        break
                except Exception:
                    continue
            if max_samples is not None and total >= max_samples:
                break

        if not encoded_rgb:
            raise ValueError(f"No valid records found in {self.tfrecord_files}")

        return (
            encoded_rgb,  # list[bytes]
            np.array(all_acts, dtype=np.float32),
            np.array(ep_starts, dtype=bool),
        )

    def _build_stack_index_map(self) -> np.ndarray:
        """For each step i, return the list of frame indices to channel-stack.

        Mirrors the boundary-repeat behavior of envs.datasets.stack_frames:
        the earliest indices are clamped to the first frame of the episode.
        Returns shape (N, frame_stack), int64.
        """
        n = len(self._encoded_rgb)
        fs = self.frame_stack
        # Episode id per step — cumulative count of episode starts.
        episode_id = np.cumsum(self._episode_starts).astype(np.int64) - 1
        # Episode-start absolute index per step.
        starts_abs = np.where(self._episode_starts)[0]
        # For each step, the absolute index of its episode start:
        ep_start_for_step = starts_abs[episode_id]

        stack = np.empty((n, fs), dtype=np.int64)
        for k in range(fs):
            # Offset k means "k frames before current" (k=fs-1 → current frame
            # in the channel-stack order, matching stack_frames' convention
            # of [oldest, ..., newest]).
            offset = fs - 1 - k
            raw = np.arange(n) - offset
            # Clamp to the episode start of the current step.
            stack[:, k] = np.maximum(raw, ep_start_for_step)
        return stack

    def _decode_jpeg(self, idx: int) -> np.ndarray:
        """Decode one frame's bytes → (H, W, 3) uint8 ndarray."""
        img = tf.io.decode_image(self._encoded_rgb[idx], channels=3).numpy()
        return img.astype(np.uint8)

    def __getitem__(self, index):
        # Decode and channel-stack `frame_stack` frames; channels-first layout
        # so the conv encoder gets (C, H, W) per sample directly.
        idxs = self._stack_indices[index]
        frames = [self._decode_jpeg(int(i)) for i in idxs]  # each (H, W, 3)
        # Channel-wise stack: [(H, W, 3), (H, W, 3)] → (H, W, 6) → (6, H, W).
        stacked = np.concatenate(frames, axis=-1)  # (H, W, 3*fs)
        stacked = np.transpose(stacked, (2, 0, 1))  # (3*fs, H, W)
        return {"state": stacked, "action": self.actions[index]}

    def __len__(self):
        return len(self._encoded_rgb)


class LiberoGoalPixelsDataset(Dataset):
    """LIBERO-Goal multi-task PIXEL dataset (standard protocol).

    Standard LIBERO obs: per-camera RGB (agentview + eye-in-hand, 128x128x3)
    channel-stacked, PLUS low-dim proprio (ee_pos + gripper + joint = 12), and a
    per-task language (goal) embedding. No object-state (privileged) — the policy
    infers objects from pixels.

    __getitem__ returns:
        state:  (3*2*frame_stack, H, W) uint8  — [agentview, wrist] channel-stack
        cond:   (proprio_dim*frame_stack + goal_emb_dim,) float32  — proprio | goal
        action: (7,) float32 in [-1, 1]

    The conv encoder (wifi_bc.models.ConvMaxpoolEncoder) does its own /255 + resize.
    Actions are min-max normalized to [-1, 1]; act_min/max exposed for eval denorm.

    Images held in RAM as uint8 (≈7 GB for the full suite) — needs a 32 GB node;
    keep DataLoader num_workers=0.
    """

    _IMAGE_KEYS = ("agentview_rgb", "eye_in_hand_rgb")
    # Proprio keys that have an exact live-env match (see envs.libero); NO
    # ee_ori/ee_states (euler, no live key) and NO object-state (privileged).
    _PROPRIO_KEYS = ("ee_pos", "gripper_states", "joint_states")
    _H = 128
    _W = 128

    def __init__(
        self,
        goal_embeddings_path: str,
        frame_stack: int = 1,
        max_demos_per_task: Optional[int] = None,
        max_samples: Optional[int] = None,
        normalize_actions: bool = True,
        action_norm_range: tuple[float, float] = (-1.0, 1.0),
        crop_size: int = 0,
        action_chunk: int = 1,
        cameras: str = "agentview+wrist",
        use_proprio: bool = True,
    ):
        try:
            import h5py  # noqa: F401
        except ImportError as e:
            raise ImportError("h5py required for LIBERO demos (uv sync --extra libero).") from e
        import h5py
        from envs.libero import get_task_infos, load_goal_embeddings
        # libero_cameras: "agentview+wrist" (default, historical) or "agentview"
        # (third-person only, the OpenVLA LIBERO protocol used for Octo / Diffusion
        # Policy / OpenVLA). Order is canonical: agentview first, then wrist.
        requested = tuple(c.strip() for c in str(cameras).split("+") if c.strip())
        self.cameras = tuple(c for c in ("agentview", "wrist") if c in requested)
        if not self.cameras or len(self.cameras) != len(requested):
            raise ValueError(f"cameras must be a '+'-joined subset of agentview, wrist; got {cameras!r}")
        self._use_wrist = "wrist" in self.cameras
        # use_proprio=False: conditioning is the goal embedding ONLY (image +
        # language, the OpenVLA LIBERO input set). libero_obs_keys becomes [] and
        # proprio_dim 0, which every norm_stats writer already records, so eval
        # rebuilds a goal-only cond with no further change.
        self.use_proprio = bool(use_proprio)

        # Random-crop augmentation (train-time only; eval center-crops to the
        # same size — see LiberoGoalPixelsSimulation). 0 = off. Standard pixel-BC
        # trick (robomimic / Diffusion Policy use ~90% crops; 116 of 128 here).
        # Note: both cameras + all stacked frames share ONE crop offset per
        # sample (they're channel-stacked); per-camera independent crops would
        # be marginally stronger aug but need a layout change.
        self.crop_size = int(crop_size)
        if self.crop_size and not (0 < self.crop_size <= self._H):
            raise ValueError(f"crop_size must be in (0, {self._H}]; got {crop_size}")
        self._rng = np.random.default_rng(0)
        # Action chunking (DP-style): each sample's target is the next K
        # actions concatenated (K*A vector); episode tails pad by repeating the
        # last action. Models treat the chunk as one big action; eval executes
        # it open-loop. K=1 keeps legacy single-step behavior.
        self.action_chunk = max(1, int(action_chunk))
        self.frame_stack = frame_stack
        self.normalize_actions = normalize_actions
        self.action_norm_range = action_norm_range

        emb_names, emb_matrix, _ = load_goal_embeddings(goal_embeddings_path)
        name_to_emb = {n: emb_matrix[i] for i, n in enumerate(emb_names)}
        self.goal_emb_dim = int(emb_matrix.shape[1])

        task_infos = get_task_infos()
        self.goal_task_names = [t["name"] for t in task_infos]
        self.goal_embeddings = np.stack(
            [name_to_emb[t["name"]] for t in task_infos]
        ).astype(np.float32)

        agv: list[np.ndarray] = []   # per-frame (H,W,3) uint8
        wrist: list[np.ndarray] = []
        proprio: list[np.ndarray] = []
        acts: list[np.ndarray] = []
        starts: list[bool] = []
        task_ids: list[int] = []
        self.libero_obs_keys = list(self._PROPRIO_KEYS) if self.use_proprio else []
        total = 0

        for t in task_infos:
            demo_file = t["demo_file"]
            if not os.path.exists(demo_file):
                raise FileNotFoundError(f"Missing LIBERO demo: {demo_file}")
            with h5py.File(demo_file, "r") as f:
                data = f["data"]
                demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[-1]))
                if max_demos_per_task is not None:
                    demo_keys = demo_keys[:max_demos_per_task]
                for dk in demo_keys:
                    obsg = data[dk]["obs"]
                    a = np.asarray(obsg["agentview_rgb"], dtype=np.uint8)        # (T,H,W,3)
                    # Wrist frames are only loaded when used (third-person-only halves image RAM).
                    w = np.asarray(obsg["eye_in_hand_rgb"], dtype=np.uint8) if self._use_wrist else None
                    pr = np.concatenate(
                        [np.asarray(obsg[k], dtype=np.float32).reshape(a.shape[0], -1)
                         for k in self._PROPRIO_KEYS], axis=1)               # (T,12)
                    ac = np.asarray(data[dk]["actions"], dtype=np.float32)
                    n = min(len(a), len(pr), len(ac), *( [len(w)] if w is not None else [] ))
                    for i in range(n):
                        agv.append(a[i]); proprio.append(pr[i]); acts.append(ac[i])
                        if w is not None:
                            wrist.append(w[i])
                        starts.append(i == 0); task_ids.append(t["index"])
                    total += n
                    if max_samples is not None and total >= max_samples:
                        break
            if max_samples is not None and total >= max_samples:
                break

        self._agv = np.stack(agv)        # (N,H,W,3) uint8
        self._wrist = np.stack(wrist) if wrist else None
        self._proprio = np.stack(proprio).astype(np.float32)   # (N,12)
        raw_actions = np.stack(acts).astype(np.float32)
        self._episode_starts = np.asarray(starts, dtype=bool)
        self._task_ids = np.asarray(task_ids, dtype=np.int64)
        if max_samples is not None:
            self._agv = self._agv[:max_samples]
            if self._wrist is not None:
                self._wrist = self._wrist[:max_samples]
            self._proprio = self._proprio[:max_samples]; raw_actions = raw_actions[:max_samples]
            self._episode_starts = self._episode_starts[:max_samples]
            self._task_ids = self._task_ids[:max_samples]

        if not self.use_proprio:
            self._proprio = self._proprio[:, :0]
        self.proprio_dim = int(self._proprio.shape[1])

        if self.action_chunk > 1:
            K = self.action_chunk
            n = len(raw_actions)
            episode_id = np.cumsum(self._episode_starts) - 1
            chunks = np.empty((n, K, raw_actions.shape[1]), dtype=np.float32)
            for k in range(K):
                idx = np.minimum(np.arange(n) + k, n - 1)
                # Don't cross episode boundaries: clamp to the last step of the
                # current episode (repeat-last-action padding).
                same_ep = episode_id[idx] == episode_id
                idx = np.where(same_ep, idx, -1)
                # For crossed indices walk back to this episode's final step.
                if (idx < 0).any():
                    last_of_ep = np.zeros(n, dtype=np.int64)
                    ep_last = {}
                    for i in range(n - 1, -1, -1):
                        e = episode_id[i]
                        if e not in ep_last:
                            ep_last[e] = i
                        last_of_ep[i] = ep_last[e]
                    idx = np.where(idx < 0, last_of_ep, idx)
                chunks[:, k] = raw_actions[idx]
            raw_actions = chunks.reshape(n, K * raw_actions.shape[1])

        self.act_min = raw_actions.min(axis=0).astype(np.float32)
        self.act_max = raw_actions.max(axis=0).astype(np.float32)
        if normalize_actions:
            lo, hi = float(action_norm_range[0]), float(action_norm_range[1])
            denom = np.where((self.act_max - self.act_min) == 0, 1.0, self.act_max - self.act_min)
            self.actions = (lo + (raw_actions - self.act_min) * (hi - lo) / denom).astype(np.float32)
        else:
            self.actions = raw_actions

        self._stack_idx = self._build_stack_index_map()
        self.in_channels = 3 * len(self.cameras) * frame_stack
        self.cond_dim = self.proprio_dim * frame_stack + self.goal_emb_dim
        out_hw = self.crop_size if self.crop_size else self._H
        self.state_shape = (self.in_channels, out_hw, out_hw)
        self.action_shape = self.actions.shape[1]

    def unnormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        if not self.normalize_actions:
            return np.asarray(normalized_action, dtype=np.float32)
        lo, hi = float(self.action_norm_range[0]), float(self.action_norm_range[1])
        scale = (self.act_max - self.act_min) / (hi - lo)
        return (self.act_min + (np.asarray(normalized_action, dtype=np.float32) - lo) * scale).astype(np.float32)

    def _build_stack_index_map(self) -> np.ndarray:
        n = len(self._agv)
        fs = self.frame_stack
        episode_id = np.cumsum(self._episode_starts).astype(np.int64) - 1
        starts_abs = np.where(self._episode_starts)[0]
        ep_start_for_step = starts_abs[episode_id]
        stack = np.empty((n, fs), dtype=np.int64)
        for k in range(fs):
            offset = fs - 1 - k
            raw = np.arange(n) - offset
            stack[:, k] = np.maximum(raw, ep_start_for_step)
        return stack

    def __getitem__(self, index):
        idxs = self._stack_idx[index]
        frames = []
        for i in idxs:                       # oldest -> newest
            if "agentview" in self.cameras:
                frames.append(self._agv[int(i)])
            if self._use_wrist:
                frames.append(self._wrist[int(i)])
        stacked = np.concatenate(frames, axis=-1)        # (H,W,3*n_cams*fs)
        if self.crop_size:
            s = self.crop_size
            oy = int(self._rng.integers(0, stacked.shape[0] - s + 1))
            ox = int(self._rng.integers(0, stacked.shape[1] - s + 1))
            stacked = stacked[oy:oy + s, ox:ox + s]
        stacked = np.transpose(stacked, (2, 0, 1)).copy()  # (C,S,S) uint8
        proprio_stack = np.concatenate([self._proprio[int(i)] for i in idxs]).astype(np.float32)
        goal = self.goal_embeddings[self._task_ids[index]]
        cond = np.concatenate([proprio_stack, goal]).astype(np.float32)
        return {"state": stacked, "cond": cond, "action": self.actions[index]}

    def __len__(self):
        return len(self._agv)


class ReachDataset(Dataset):
    """Scripted-expert demonstrations for `Reach-v0`.

    Generated on the fly rather than downloaded, so the "bring your own
    environment" walkthrough in the README needs no data files. The expert
    drives straight at the goal at full speed, which makes the task unimodal
    and easy — the point is the plumbing, not the difficulty.
    """

    def __init__(self, size: int = 20000, frame_stack: int = 1,
                 goal_radius: float = 0.05, step_size: float = 0.1, seed: int = 0):
        from envs.reach_env import ACTION_DIM, OBS_DIM

        rng = np.random.default_rng(seed)
        obs, acts, starts = [], [], []
        while len(obs) < size:
            agent = rng.uniform(0, 1, 2).astype(np.float32)
            goal = rng.uniform(0, 1, 2).astype(np.float32)
            starts.append(len(obs))
            for _ in range(50):
                if len(obs) >= size:
                    break
                delta = goal - agent
                dist = float(np.linalg.norm(delta))
                if dist < goal_radius:
                    break
                # Full-speed unit step toward the goal, clipped to the box.
                action = np.clip(delta / max(dist, 1e-8), -1.0, 1.0).astype(np.float32)
                obs.append(np.concatenate([agent, goal]).astype(np.float32))
                acts.append(action)
                agent = np.clip(agent + action * step_size, 0.0, 1.0).astype(np.float32)

        self.observations = np.asarray(obs, dtype=np.float32)
        self.actions = np.asarray(acts, dtype=np.float32)
        self._episode_starts = np.asarray(starts, dtype=np.int64)
        if frame_stack > 1:
            self.observations = stack_frames(self.observations, self._episode_starts, frame_stack)

        self.state_shape = self.observations.shape[1]
        self.action_shape = ACTION_DIM
        # The trainers read these to build norm_stats; actions are already in
        # the model's [-1, 1] box, so the min/max map is the identity.
        self.act_min = self.actions.min(axis=0).astype(np.float32)
        self.act_max = self.actions.max(axis=0).astype(np.float32)
        self.action_norm_range = (-1.0, 1.0)
        self.obs_mean = self.observations.mean(axis=0)[:OBS_DIM].astype(np.float32)
        self.obs_std = (self.observations.std(axis=0)[:OBS_DIM] + 1e-6).astype(np.float32)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, index):
        return {"state": self.observations[index], "action": self.actions[index]}
