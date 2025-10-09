"""
data_handler.py

Provides utility functions for saving and loading trained models, episodic data,
and benchmark results. This module standardizes file organization and I/O operations
for reproducibility and consistency across experiments.
"""

import os
import json
import torch
import numpy as np
from typing import Optional, Type, Dict, Any, List

def _maybe_add_variant_dir(
        base_path: str, 
        alg_name: str, 
        posterior_samples: Optional[int] = None, 
        bootstrap_heads: Optional[int] = None
    ) -> str:
    """
    Appends a variant-specific subdirectory to the base path if applicable.

    Depending on the algorithm type, this function adds a directory suffix
    corresponding to the number of posterior samples (for OPS-VBQN) or bootstrap
    heads (for BootstrapDQN). If no variant applies, the base path is returned unchanged.

    Args:
        base_path (str): Base directory path for saving or loading data.
        alg_name (str): Name of the algorithm ("OPS-VBQN", "BootstrapDQN", or others).
        posterior_samples (int, optional): Number of posterior samples for OPS-VBQN.
        bootstrap_heads (int, optional): Number of bootstrap heads for BootstrapDQN.

    Returns:
        str: The modified or unmodified directory path.
    """
    if alg_name == "OPS-VBQN" and posterior_samples is not None:
        return os.path.join(base_path, f"post_{posterior_samples}")
    elif alg_name == "BootstrapDQN" and bootstrap_heads is not None:
        return os.path.join(base_path, f"heads_{bootstrap_heads}")
    return base_path


# === MODEL PATHS ===

def get_model_path(
        env_name: str, 
        alg_name: str, 
        seed: int, 
        posterior_samples: Optional[int] = None, 
        bootstrap_heads: Optional[int] = None
    ) -> str:
    """
    Constructs the file path for saving or loading a model.

    The path is built from the environment name, algorithm name, and seed.
    If the algorithm has variants (OPS-VBQN or BootstrapDQN), the corresponding
    subdirectory for posterior samples or bootstrap heads is added automatically.

    Args:
        env_name (str): Name of the environment.
        alg_name (str): Name of the algorithm.
        seed (int): Random seed used for the model.
        posterior_samples (int, optional): Number of posterior samples (for OPS-VBQN).
        bootstrap_heads (int, optional): Number of bootstrap heads (for BootstrapDQN).

    Returns:
        str: Full file path to the model.
    """
    base_path = os.path.join("models", env_name, alg_name)
    base_path = _maybe_add_variant_dir(base_path, alg_name, posterior_samples, bootstrap_heads)
    return os.path.join(base_path, f"seed{seed}.pt")


def save_model(
        model: torch.nn.Module, 
        env_name: str, 
        alg_name: str, 
        seed: int, 
        posterior_samples: Optional[int] = None, 
        bootstrap_heads: Optional[int] = None
    ) -> None:
    """
    Saves the PyTorch model's state dictionary to a structured file path.

    The file path is determined by environment name, algorithm name, random seed,
    and optionally posterior samples or bootstrap heads. Necessary directories
    are created automatically if they do not exist.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        env_name (str): Name of the environment (e.g., "CartPole-v1").
        alg_name (str): Name of the algorithm (e.g., "DQN", "OPS-VBQN").
        seed (int): Random seed used in training, included in the file name.
        posterior_samples (Optional[int], default=None): Number of posterior samples
            (for OPS-VBQN) to include in the directory structure.
        bootstrap_heads (Optional[int], default=None): Number of bootstrap heads
            (for BootstrapDQN) to include in the directory structure.

    Returns:
        None
    """
    path = get_model_path(env_name, alg_name, seed, posterior_samples, bootstrap_heads)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(
        model_class: Type[torch.nn.Module], 
        env_name: str, 
        alg_name: str, 
        seed: int, 
        in_features: int, 
        hidden_layer_size: int, 
        out_features: int,
        posterior_samples: Optional[int] = None, 
        bootstrap_heads: Optional[int] = None
    ) -> torch.nn.Module:
    """
    Loads a PyTorch model from a saved state dictionary.

    The model is instantiated based on the specified class and architecture parameters,
    then its weights are loaded from the appropriate file path determined by environment,
    algorithm, seed, and optionally posterior samples or bootstrap heads.

    Args:
        model_class (Type[torch.nn.Module]): The class of the model to instantiate.
        env_name (str): Name of the environment (e.g., "CartPole-v1").
        alg_name (str): Name of the algorithm (e.g., "DQN", "OPS-VBQN").
        seed (int): Random seed used in training, included in the file path.
        in_features (int): Number of input features for the model.
        hidden_layer_size (int): Size of the hidden layer(s) in the model.
        out_features (int): Number of output features (typically the number of actions).
        posterior_samples (Optional[int], default=None): Number of posterior samples
            (for OPS-VBQN) included in the directory structure.
        bootstrap_heads (Optional[int], default=None): Number of bootstrap heads
            (for BootstrapDQN) included in the directory structure.

    Returns:
        torch.nn.Module: An instance of the model with loaded weights.
    """
    path = get_model_path(env_name, alg_name, seed, posterior_samples, bootstrap_heads)

    # 🔁 Instantiate the correct model based on type
    if model_class.__name__ == "BootstrapDQN":
        if bootstrap_heads is None:
            raise ValueError("bootstrap_heads must be specified when loading a BootstrapDQN model.")
        model = model_class(
            in_features=in_features,
            hidden_layer_size=hidden_layer_size,
            out_features=out_features,
            bootstrap_heads=bootstrap_heads
        )
    else:
        model = model_class(
            in_features=in_features,
            hidden_layer_size=hidden_layer_size,
            out_features=out_features
        )

    model.load_state_dict(torch.load(path))
    return model


# === DATA PATHS ===

def get_returns_path(
        env_name: str, 
        alg_name: str, 
        seed: int, 
        posterior_samples: Optional[int] = None, 
        bootstrap_heads: Optional[int] = None
    ) -> str:
    """
    Constructs the file path for saving or loading episodic returns.

    The path is based on the environment name, algorithm, seed, and optionally
    posterior samples or bootstrap heads, following the structured directory hierarchy.

    Args:
        env_name (str): Name of the environment (e.g., "CartPole-v1").
        alg_name (str): Name of the algorithm (e.g., "DQN", "OPS-VBQN").
        seed (int): Random seed used in training, included in the file name.
        posterior_samples (Optional[int], default=None): Number of posterior samples
            for OPS-VBQN; used to create a variant-specific subdirectory.
        bootstrap_heads (Optional[int], default=None): Number of bootstrap heads
            for BootstrapDQN; used to create a variant-specific subdirectory.

    Returns:
        str: Full file path for the returns file (e.g., "data/CartPole-v1/DQN/seed1_returns.npy").
    """
    base_path = os.path.join("data", env_name, alg_name)
    base_path = _maybe_add_variant_dir(base_path, alg_name, posterior_samples, bootstrap_heads)
    return os.path.join(base_path, f"seed{seed}_returns.npy")


def get_benchmarks_path(
        env_name: str, 
        alg_name: str, 
        posterior_samples: Optional[int] = None, 
        bootstrap_heads: Optional[int] = None
    ) -> str:
    """
    Constructs the file path for saving or loading benchmark results.

    The path is based on the environment name, algorithm, and optionally
    posterior samples or bootstrap heads, following the structured directory hierarchy.

    Args:
        env_name (str): Name of the environment (e.g., "CartPole-v1").
        alg_name (str): Name of the algorithm (e.g., "DQN", "OPS-VBQN").
        posterior_samples (Optional[int], default=None): Number of posterior samples
            for OPS-VBQN; used to create a variant-specific subdirectory.
        bootstrap_heads (Optional[int], default=None): Number of bootstrap heads
            for BootstrapDQN; used to create a variant-specific subdirectory.

    Returns:
        str: Full file path for the benchmark results JSON file
            (e.g., "data/CartPole-v1/DQN/benchmarks.json").
    """
    base_path = os.path.join("data", env_name, alg_name)
    base_path = _maybe_add_variant_dir(base_path, alg_name, posterior_samples, bootstrap_heads)
    return os.path.join(base_path, "benchmarks.json")


def save_seed_data(
    env_name: str,
    alg_name: str,
    seed: int,
    benchmark_score: Optional[float] = None,
    returns: Optional[np.ndarray] = None,
    cumulative_regret: Optional[float] = None,
    episodes_to_solve: Optional[int] = None,
    posterior_samples: Optional[int] = None,
    bootstrap_heads: Optional[int] = None
    ) -> None:
    """
    Saves episodic returns and benchmark metrics for a specific seed.

    Creates necessary directories if they do not exist. Saves the `returns`
    as a NumPy array and updates benchmark metrics (JSON) for the given
    environment, algorithm, and seed. Supports OPS-VBQN and BootstrapDQN
    variants via `posterior_samples` and `bootstrap_heads`.

    Args:
        env_name (str): Name of the environment (e.g., "CartPole-v1").
        alg_name (str): Name of the algorithm (e.g., "DQN", "OPS-VBQN").
        seed (int): Random seed used for this experiment.
        benchmark_score (Optional[float], default=None): Average benchmark score
            for the seed.
        returns (Optional[np.ndarray], default=None): Episodic returns to save.
        cumulative_regret (Optional[float], default=None): Cumulative regret
            for the seed.
        episodes_to_solve (Optional[int], default=None): Number of episodes
            the agent used to solve the task.
        posterior_samples (Optional[int], default=None): Number of posterior samples
            for OPS-VBQN (if applicable).
        bootstrap_heads (Optional[int], default=None): Number of bootstrap heads
            for BootstrapDQN (if applicable).

    Returns:
        None
    """
    folder_path = os.path.dirname(get_returns_path(env_name, alg_name, seed, posterior_samples, bootstrap_heads))
    os.makedirs(folder_path, exist_ok=True)

    # Only save returns if provided
    if returns is not None:
        returns_path = get_returns_path(env_name, alg_name, seed, posterior_samples, bootstrap_heads)
        np.save(returns_path, returns)

    # Load existing benchmark JSON or create new
    benchmarks_path = get_benchmarks_path(env_name, alg_name, posterior_samples, bootstrap_heads)
    if os.path.exists(benchmarks_path):
        with open(benchmarks_path, "r") as f:
            benchmarks = json.load(f)
    else:
        benchmarks = {}

    seed_key = f"seed{seed}"

    # Ensure the benchmark entry exists
    if seed_key not in benchmarks:
        benchmarks[seed_key] = {}

    # Update metrics individually if provided
    if benchmark_score is not None:
        benchmarks[seed_key]["benchmark"] = benchmark_score

    if cumulative_regret is not None:
        benchmarks[seed_key]["cumulative_regret"] = cumulative_regret

    if episodes_to_solve is not None:
        benchmarks[seed_key]["episodes_to_solve"] = episodes_to_solve

    # Save updated JSON
    with open(benchmarks_path, "w") as f:
        json.dump(benchmarks, f, indent=2)



# === LOAD ALL BENCHMARKS ===

def load_all_benchmarks(data_root: str="data") -> Dict[str, Dict[str, Any]]:
    """
    Loads benchmark results for all environments and algorithms from the specified data directory.

    Traverses the folder structure to collect benchmarks stored in JSON files. Handles
    algorithm-specific subfolders for OPS-VBQN (`post_{num_samples}`) and BootstrapDQN
    (`heads_{K}`), as well as standard algorithms (like DQN) without subfolders.

    Args:
        data_root (str, default="data"): Root directory containing environment subfolders.

    Returns:
        dict: Nested dictionary containing benchmark data organized as:
            {
                env_name: {
                    alg_name: {posterior_key: benchmarks, ...} or benchmarks_dict
                }, ...
            }
            - For OPS-VBQN: `posterior_key` corresponds to the posterior sample size.
            - For BootstrapDQN: `posterior_key` corresponds to the number of bootstrap heads.
            - For standard algorithms: directly maps `alg_name` to the benchmarks dictionary.
    """
    all_data = {}
    for env_name in os.listdir(data_root):
        env_path = os.path.join(data_root, env_name)
        if not os.path.isdir(env_path):
            continue

        all_data[env_name] = {}

        for alg_name in os.listdir(env_path):
            alg_path = os.path.join(env_path, alg_name)

            # === Handle OPS-VBQN with posterior sample subfolders ===
            if alg_name == "OPS-VBQN":
                all_data[env_name][alg_name] = {}

                for post_dir in os.listdir(alg_path):
                    post_path = os.path.join(alg_path, post_dir)
                    if not os.path.isdir(post_path):
                        continue

                    benchmarks_path = os.path.join(post_path, "benchmarks.json")
                    if os.path.exists(benchmarks_path):
                        with open(benchmarks_path, "r") as f:
                            benchmarks = json.load(f)
                        posterior_key = post_dir.replace("post_", "")
                        all_data[env_name][alg_name][posterior_key] = benchmarks

            # === Handle Bootstrapped DQN with heads_K subfolders ===
            elif alg_name == "BootstrapDQN":
                all_data[env_name][alg_name] = {}

                for heads_dir in os.listdir(alg_path):
                    heads_path = os.path.join(alg_path, heads_dir)
                    if not os.path.isdir(heads_path):
                        continue

                    benchmarks_path = os.path.join(heads_path, "benchmarks.json")
                    if os.path.exists(benchmarks_path):
                        with open(benchmarks_path, "r") as f:
                            benchmarks = json.load(f)
                        head_key = heads_dir.replace("heads_", "")
                        all_data[env_name][alg_name][head_key] = benchmarks

            # === Standard algorithms (like DQN) ===
            else:
                benchmarks_path = os.path.join(alg_path, "benchmarks.json")
                if os.path.exists(benchmarks_path):
                    with open(benchmarks_path, "r") as f:
                        benchmarks = json.load(f)
                    all_data[env_name][alg_name] = benchmarks

    return all_data




# === LOAD RETURNS ===

def load_returns_for_env(
        env_name: str, 
        seeds: Optional[List[int]] = None, 
        data_root: str="data", 
        posterior_samples: Optional[int] = None, 
        bootstrap_heads: Optional[int] = None
    ) -> Dict[str, Any]:
    """
    Loads episodic returns for a given environment across algorithms and seeds.

    Handles standard algorithms (like DQN), OPS-VBQN (with posterior sample variants),
    and BootstrapDQN (with bootstrap head variants). Returns are loaded from the
    structured benchmark directories.

    Args:
        env_name (str): Name of the environment (e.g., "CartPole-v1").
        seeds (list[int], optional): List of seeds to load. If None, loads all available seeds.
        data_root (str, default="data"): Root directory where benchmark data is stored.
        posterior_samples (int, optional): Specific posterior sample size for OPS-VBQN.
        bootstrap_heads (int, optional): Specific number of bootstrap heads for BootstrapDQN.

    Returns:
        dict: Nested dictionary containing returns:
            - Top-level keys are algorithm names (str).
            - Second-level keys exist for OPS-VBQN or BootstrapDQN variants (str), if multiple variants exist.
            - Leaf dictionaries map seed integers to np.ndarray of episodic returns.
    """
    env_path = os.path.join(data_root, env_name)
    if not os.path.isdir(env_path):
        raise ValueError(f"Environment folder '{env_name}' not found.")

    alg_data = {}

    for alg_name in os.listdir(env_path):
        alg_path = os.path.join(env_path, alg_name)
        if not os.path.isdir(alg_path):
            continue

        # === Handle OPS-VBQN ===
        if alg_name == "OPS-VBQN":
            alg_data[alg_name] = {}

            if posterior_samples is not None:
                post_dir = f"post_{posterior_samples}"
                post_path = os.path.join(alg_path, post_dir)
                seed_returns = _load_seed_returns(post_path, seeds)
                if seed_returns:
                    alg_data[alg_name] = seed_returns
            else:
                for post_dir in os.listdir(alg_path):
                    post_path = os.path.join(alg_path, post_dir)
                    if not os.path.isdir(post_path):
                        continue
                    seed_returns = _load_seed_returns(post_path, seeds)
                    if seed_returns:
                        alg_data[alg_name][post_dir] = seed_returns

        # === Handle Bootstrapped DQN ===
        elif alg_name == "BootstrapDQN":
            alg_data[alg_name] = {}

            if bootstrap_heads is not None:
                heads_dir = f"heads_{bootstrap_heads}"
                heads_path = os.path.join(alg_path, heads_dir)
                seed_returns = _load_seed_returns(heads_path, seeds)
                if seed_returns:
                    alg_data[alg_name] = seed_returns
            else:
                for heads_dir in os.listdir(alg_path):
                    heads_path = os.path.join(alg_path, heads_dir)
                    if not os.path.isdir(heads_path):
                        continue
                    seed_returns = _load_seed_returns(heads_path, seeds)
                    if seed_returns:
                        alg_data[alg_name][heads_dir] = seed_returns

        # === Standard algorithms (e.g., DQN) ===
        else:
            seed_returns = _load_seed_returns(alg_path, seeds)
            if seed_returns:
                alg_data[alg_name] = seed_returns

    return alg_data


def _load_seed_returns(folder_path: str, seeds: Optional[List[int]]) -> Dict[int, np.ndarray]:
    """
    Load episodic returns for specific seeds from a given folder.

    Args:
        folder_path (str): Path to the folder containing saved returns files.
        seeds (Optional[list[int]]): List of seeds to load. If `None`, all seeds in the folder are loaded.

    Returns:
        dict[int, np.ndarray]: Dictionary mapping each seed to its corresponding 
        episodic returns array. Returns an empty dictionary if the folder does not exist 
        or if no matching files are found.
    """
    if not os.path.isdir(folder_path):
        return {}

    seed_returns = {}
    for file in os.listdir(folder_path):
        if file.endswith("_returns.npy") and file.startswith("seed"):
            seed_str = file[len("seed"):file.index("_returns")]
            try:
                seed = int(seed_str)
            except ValueError:
                continue
            if seeds is None or seed in seeds:
                path = os.path.join(folder_path, file)
                seed_returns[seed] = np.load(path)

    return seed_returns


