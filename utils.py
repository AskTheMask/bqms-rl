import gymnasium as gym
import numpy as np
import torch.nn.functional as F
import torch
import random
from typing import Optional, Any
from matplotlib.ticker import FuncFormatter

def encode_state(state: Any, obs_space: gym.Space) -> torch.Tensor:
    """
    Encodes a state based on the observation space type. Supports both discrete and continuous spaces.
    """
    if isinstance(obs_space, gym.spaces.Discrete):
        return F.one_hot(torch.tensor(state), num_classes=obs_space.n).float()
    else:
        return torch.tensor(state, dtype=torch.float32)
    
def last_nonzero_index(arr: np.ndarray) -> int:
    """
    Returns the index of the last non-zero element in the array.
    If all elements are zero, returns the length of the array.
    """
    rev_nz = np.flatnonzero(arr[::-1])
    return len(arr) - rev_nz[0] if rev_nz.size > 0 else len(arr)



def make_env(env_name: str, env_kwargs: dict = None, render_mode: Optional[str] = None) -> gym.Env:
    """
    Creates a Gym environment instance with optional rendering and environment-specific defaults.

    Args:
        env_name: Name of the Gym environment (e.g., "CartPole-v1").
        env_kwargs: Extra keyword arguments to pass to gym.make().
        render_mode: Optional rendering mode (e.g., "human", "rgb_array").

    Returns:
        Instantiated Gym environment.
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


def get_random_seeds(n: int, max_seeds: int = 10, base_seed: int = 2025):
    if n <= 0:
        raise ValueError("Number of seeds must be positive.")
    
    random.seed(base_seed)
    SEEDS = random.sample(range(1, 10000), max_seeds)
    
    if n > max_seeds:
        raise ValueError(f"Requested {n} seeds, max allowed {max_seeds}.")
    
    return SEEDS[:n]





def validate_args(env_ids: list[str], environments, num_samples=None, bootstrap_heads=None):
    if not env_ids:
        raise ValueError("No environment IDs provided. Please specify at least one environment.")
    invalid_envs = [e for e in env_ids if e not in environments]
    if invalid_envs:
        raise ValueError(f"Invalid environment IDs: {invalid_envs}. Valid options: {environments}")

    if num_samples is not None and len(num_samples) == 0:
        raise ValueError("num_samples must be a non-empty list for BQMS.")

    if bootstrap_heads is not None and len(bootstrap_heads) == 0:
        raise ValueError("bootstrap_heads must be a non-empty list for BootstrapDQN.")


def exponential_moving_average(data: np.ndarray, alpha: float) -> np.ndarray:
    """Compute exponential moving average of a 1D data array."""

    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema


def get_scaled_formatter(scale_factor=None):
    """
    Returns a matplotlib FuncFormatter for axes with optional scaling.
    
    Parameters:
        scale_factor (int or None): Value to divide by (e.g., 1_000 for 10^3).
                                    If None, no scaling is applied.
    """
    def _format(x, _):
        if scale_factor and scale_factor != 0:
            val = x / scale_factor
            return f"{int(val)}" if val.is_integer() else f"{val:.1f}"
        else:
            return f"{x:.0f}"  # No scaling, plain int format

    return FuncFormatter(_format)


def format_large_number(x, scale_factor=None):
    """
    Formats a single number with optional scaling.
    
    Parameters:
        x (float/int): Value to format
        scale_factor (int or None): Value to divide by (e.g., 1_000 for 10^3).
    
    Returns:
        str: Formatted number as string.
    """
    if scale_factor and scale_factor != 0:
        val = x / scale_factor
        return f"{int(val)}" if val.is_integer() else f"{val:.1f}"
    else:
        return f"{x:.2f}"  # No scaling