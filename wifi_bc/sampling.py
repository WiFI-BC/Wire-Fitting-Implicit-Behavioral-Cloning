"""Sampling utilities for generating counter-example actions.

Supports:
- Uniform sampling within bounds
- Langevin MCMC with polynomial learning rate decay (IBC paper, Florence et al., 2021)
"""

import numpy as np
import torch
from typing import Union, List, Callable


def sample_uniform(
    k: int,
    batch_size: int,
    mins: Union[List[float], np.ndarray],
    maxs: Union[List[float], np.ndarray],
    seed: int = 42
) -> np.ndarray:
    """
    Generate k random samples uniformly distributed within specified bounds.

    Args:
        k: Number of samples to generate.
        mins: Minimum values for each dimension. Can be a list or numpy array.
        maxs: Maximum values for each dimension. Can be a list or numpy array.
        seed: Optional random seed for reproducibility.

    Returns:
        np.ndarray: Array of shape (batch_size, k, n_dims) containing the random samples.

    Raises:
        ValueError: If mins and maxs have different lengths or if any min > max.

    Example:
        >>> samples = sample_uniform(k=100, batch_size=32, mins=[0, -1, 0], maxs=[1, 1, 10])
        >>> samples.shape
        (32, 100, 3)
    """
    mins = np.asarray(mins)
    maxs = np.asarray(maxs)

    if mins.shape != maxs.shape:
        raise ValueError(f"mins and maxs must have the same shape. Got {mins.shape} and {maxs.shape}")

    if np.any(mins > maxs):
        raise ValueError("All min values must be less than or equal to corresponding max values")

    if seed is not None:
        np.random.seed(seed)

    n_dims = len(mins)
    samples = np.random.uniform(low=mins, high=maxs, size=(batch_size, k, n_dims))

    return samples  # (B, N, D)


def _polynomial_decay(lr_init: float, lr_final: float, power: float, step: int, num_steps: int) -> float:
    """Compute learning rate with polynomial decay schedule.
    
    lr(t) = (lr_init - lr_final) * (1 - t / T)^power + lr_final
    
    Args:
        lr_init: Initial learning rate.
        lr_final: Final learning rate.
        power: Polynomial decay power (2.0 for quadratic).
        step: Current step (0-indexed).
        num_steps: Total number of steps.
        
    Returns:
        Learning rate at the given step.
    """
    progress = min(step / max(num_steps - 1, 1), 1.0)
    return (lr_init - lr_final) * ((1.0 - progress) ** power) + lr_final


def sample_langevin(
    energy_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    observations: torch.Tensor,
    num_samples: int,
    action_min: Union[float, torch.Tensor],
    action_max: Union[float, torch.Tensor],
    num_iterations: int = 50,
    lr_init: float = 0.1,
    lr_final: float = 1e-5,
    polynomial_decay_power: float = 2.0,
    delta_action_clip: float = 0.1,
    noise_scale: float = 1.0,
    initial_actions: torch.Tensor = None,
    return_trajectories: bool = False,
    device: torch.device = None,
    noise_via_stepsize: bool = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, List[torch.Tensor]]]:
    """Generate counter-example actions using Langevin MCMC with polynomial LR decay.

    Implements the Langevin dynamics sampling from the Implicit Behavioral Cloning
    paper (Florence et al., 2021), Section B.3, with the full set of parameters.

    Args:
        energy_function: Neural network E(obs, action) -> energy.
                         Takes (B, N, D_obs) and (B, N, D_action), returns (B, N).
        observations: Observations tensor (B, D_obs).
        num_samples: Number of counter-example samples per observation (N).
        action_min: Minimum action bound (scalar or tensor).
        action_max: Maximum action bound (scalar or tensor).
        num_iterations: Number of Langevin MCMC iterations (default: 50).
        lr_init: Initial learning rate (default: 0.1).
        lr_final: Final learning rate (default: 1e-5).
        polynomial_decay_power: Decay power for LR schedule (default: 2.0).
        delta_action_clip: Maximum absolute change per iteration (default: 0.1).
        noise_scale: Scale of Gaussian noise (default: 1.0).
        initial_actions: Optional starting actions. If None, samples uniformly.
        return_trajectories: If True, returns (final_samples, list_of_traj_steps).
        device: Torch device. If None, uses observations.device.

    Returns:
        torch.Tensor: Counter-example actions of shape (B, N, D_action).
        (Optional) tuple: (actions, trajectories) if return_trajectories is True.
    """
    if device is None:
        device = observations.device
    
    batch_size = observations.shape[0]
    
    # Infer action dimension from bounds
    if isinstance(action_min, torch.Tensor):
        action_dim = action_min.shape[-1]
    elif isinstance(action_max, torch.Tensor):
        action_dim = action_max.shape[-1]
    else:

        raise ValueError(
            "action_min or action_max must be a tensor to infer action_dim."
        )
    
    # Initialize actions
    if initial_actions is not None:
        if initial_actions.shape != (batch_size, num_samples, action_dim):
             raise ValueError(f"initial_actions shape mismatch. Expected {(batch_size, num_samples, action_dim)}, got {initial_actions.shape}")
        actions = initial_actions.clone().to(device)
    else:
        # Initialize uniformly within bounds
        actions = torch.rand(batch_size, num_samples, action_dim, device=device)
        actions = actions * (action_max - action_min) + action_min
    
    # Expand observations to match samples: (B, *obs_shape) -> (B, N, *obs_shape).
    # Shape-agnostic so this works for flat 1-D obs AND for image obs (B, C, H, W)
    # where the model uses late-fused encoding (the resulting (B, N, C, H, W)
    # tensor is a stride-0 view — no extra memory — and PixelQEstimator.forward
    # detects ndim==5 and slices dim=1 to recover the un-expanded image).
    obs_trailing = [-1] * (observations.ndim - 1)
    obs_expanded = observations.unsqueeze(1).expand(-1, num_samples, *obs_trailing)
    
    trajectories = []
    if return_trajectories:
        trajectories.append(actions.detach().clone())

    # Langevin dynamics with polynomial LR decay
    for k in range(num_iterations):
        actions = actions.detach().requires_grad_(True)
        
        # Compute energy and gradient
        energy = energy_function(obs_expanded, actions)  # (B, N)
        grad = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=actions,
            create_graph=False,
            retain_graph=False
        )[0]  # (B, N, D_action)
        
        # Current learning rate (polynomial decay)
        lr_k = _polynomial_decay(lr_init, lr_final, polynomial_decay_power, k, num_iterations)
        
        # Langevin update. Two noise conventions:
        #   noise_via_stepsize=False (legacy/textbook): noise std = noise_scale*sqrt(lr)
        #   noise_via_stepsize=True  (official IBC mcmc.py langevin_step):
        #     delta = stepsize * (0.5*grad + noise_scale*N(0,1)) — noise scales
        #     LINEARLY with stepsize, so the chain end (lr→1e-5) is a pure
        #     gradient polish instead of a residual jitter.
        if noise_via_stepsize:
            noise = torch.randn_like(actions) * noise_scale * lr_k
        else:
            noise = torch.randn_like(actions) * noise_scale * (lr_k ** 0.5)
        delta = -(lr_k / 2.0) * grad + noise
        
        # Clip the delta (per-step action change)
        delta = torch.clamp(delta, -delta_action_clip, delta_action_clip)
        
        # Apply update
        actions = actions.detach() + delta
        
        # Clip to action bounds
        actions = torch.clamp(actions, action_min, action_max)

        if return_trajectories:
            trajectories.append(actions.detach().clone())
    
    if return_trajectories:
        return actions.detach(), trajectories
    return actions.detach()
