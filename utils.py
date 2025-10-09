"""
utils.py

Utility functions for Reinforcement Learning experiments, including environment creation,
state encoding, seed generation, argument validation, data processing, and plotting helpers.
"""

import gymnasium as gym
import numpy as np
import torch.nn.functional as F
import torch
import random
from typing import Optional, Any, List
from matplotlib.ticker import FuncFormatter

def encode_state(state: Any, obs_space: gym.Space) -> torch.Tensor:
    """
    Encodes a Gym state as a PyTorch tensor.

    Args:
        state: The state to encode.
        obs_space: Gym observation space (Discrete or Box).

    Returns:
        torch.Tensor: Encoded state, one-hot if discrete, float tensor if continuous.
    """
    if isinstance(obs_space, gym.spaces.Discrete):
        return F.one_hot(torch.tensor(state), num_classes=obs_space.n).float()
    else:
        return torch.tensor(state, dtype=torch.float32)
    
def last_nonzero_index(arr: np.ndarray) -> int:
    """
    Finds the index of the last non-zero element in a 1D numpy array.

    Args:
        arr (np.ndarray): Input array.

    Returns:
        int: Index of the last non-zero element. If all elements are zero, returns len(arr).
    """
    rev_nz = np.flatnonzero(arr[::-1])
    return len(arr) - rev_nz[0] if rev_nz.size > 0 else len(arr)

def make_env(env_name: str, env_kwargs: dict = None, render_mode: Optional[str] = None) -> gym.Env:
    """
    Creates a Gym environment instance with optional rendering and environment-specific defaults.

    Args:
        env_name (str): Name of the Gym environment (e.g., "CartPole-v1").
        env_kwargs (dict, optional): Extra kwargs for gym.make(). Defaults to None.
        render_mode (str, optional): Rendering mode (e.g., "human"). Defaults to None.

    Returns:
        gym.Env: Instantiated Gym environment.
    """
    if env_kwargs is None:
        env_kwargs = {}

    if env_name == "LunarLander-v3":
        default_lunar_kwargs = {
            "continuous": False,
            "gravity": -10.0,
            "enable_wind": False,
            "wind_power": 15.0,
            "turbulence_power": 1.5,
        }
        merged_kwargs = {**default_lunar_kwargs, **env_kwargs}
    else:
        merged_kwargs = {**env_kwargs}

    if render_mode is not None:
        merged_kwargs["render_mode"] = render_mode

    return gym.make(env_name, **merged_kwargs)


def get_random_seeds(n: int, max_seeds: int = 10, base_seed: int = 2025) -> List[int]:
    """
    Generates a reproducible list of random seeds.

    Args:
        n (int): Number of seeds to generate.
        max_seeds (int): Maximum allowed seeds. Defaults to 10.
        base_seed (int): Seed for random generator. Defaults to 2025.

    Returns:
        List[int]: List of random seeds.

    Raises:
        ValueError: If n <= 0 or n > max_seeds.
    """
    if n <= 0:
        raise ValueError("Number of seeds must be positive.")
    
    random.seed(base_seed)
    SEEDS = random.sample(range(1, 10000), max_seeds)
    
    if n > max_seeds:
        raise ValueError(f"Requested {n} seeds, max allowed {max_seeds}.")
    
    return SEEDS[:n]


def validate_args(env_ids: list[str], environments, num_samples=None, bootstrap_heads=None):
    """
    Validates environment IDs and optional OPS-VBQN or BootstrapDQN arguments.

    Args:
        env_ids (list[str]): List of environment IDs to validate.
        environments (list[str]): Valid environment options.
        num_samples (list[int], optional): List of posterior samples for OPS-VBQN.
        bootstrap_heads (list[int], optional): List of bootstrap heads for BootstrapDQN.

    Raises:
        ValueError: If any validation fails.
    """
    if not env_ids:
        raise ValueError("No environment IDs provided. Please specify at least one environment.")
    invalid_envs = [e for e in env_ids if e not in environments]
    if invalid_envs:
        raise ValueError(f"Invalid environment IDs: {invalid_envs}. Valid options: {environments}")

    if num_samples is not None and len(num_samples) == 0:
        raise ValueError("num_samples must be a non-empty list for OPS-VBQN.")

    if bootstrap_heads is not None and len(bootstrap_heads) == 0:
        raise ValueError("bootstrap_heads must be a non-empty list for BootstrapDQN.")


def exponential_moving_average(data: np.ndarray, alpha: float) -> np.ndarray:
    """
    Computes the exponential moving average (EMA) of a 1D numpy array.

    Args:
        data (np.ndarray): Input data array.
        alpha (float): Smoothing factor (0 < alpha <= 1).

    Returns:
        np.ndarray: EMA of the data.
    """

    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema


def get_scaled_formatter(scale_factor=None):
    """
    Returns a matplotlib FuncFormatter for axes with optional scaling.

    Args:
        scale_factor (int or None): Value to divide axis by (e.g., 1_000 for 10^3).
                                     If None, no scaling is applied.

    Returns:
        matplotlib.ticker.FuncFormatter: Formatter for axis labels.
    """
    def _format(x, _):
        if scale_factor and scale_factor != 0:
            val = x / scale_factor
            return f"{int(val)}" if val.is_integer() else f"{val:.1f}"
        else:
            return f"{x:.0f}"  # No scaling, plain int format

    return FuncFormatter(_format)


def format_large_number(x, scale_factor=None) -> str:
    """
    Formats a single number with optional scaling.

    Args:
        x (float|int): Value to format.
        scale_factor (int|None): Value to divide by (e.g., 1_000 for 10^3).

    Returns:
        str: Formatted number as string.
    """
    if scale_factor and scale_factor != 0:
        val = x / scale_factor
        return f"{int(val)}" if val.is_integer() else f"{val:.1f}"
    else:
        return f"{x:.2f}"  # No scaling