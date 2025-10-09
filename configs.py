"""
configs.py

Defines environment- and algorithm-specific configuration settings for the
Reinforcement Learning Benchmark Suite. Handles creation of directory
structures for saving models and results, and provides a helper function
to instantiate agents with environment-appropriate hyperparameters.
"""


import os
from agent import Agent
from typing import List, Optional, Type
import torch
import gymnasium as gym

def setup_project_dirs(
        env_names: List[str], 
        algo_names: List[str], 
        posterior_samples: Optional[List[int]] = None, 
        bootstrap_heads: Optional[List[int]] = None,
        base_model_dir: Optional[str] = "models", 
        base_data_dir: Optional[str] = "data"
        ) -> None:
    
    """
    Creates a structured directory hierarchy for saving models and benchmark data.

    For each environment–algorithm combination, subfolders are created under the
    model and data directories. If `OPS-VBQN` or `BootstrapDQN` are used, their
    respective posterior sample or bootstrap head configurations are also included
    in the directory structure.

    Args:
        env_names (list[str]): List of environment names (e.g., ["CartPole-v1", "LunarLander-v3"]).
        algo_names (list[str]): List of algorithm names (e.g., ["DQN", "OPS-VBQN", "BootstrapDQN"]).
        posterior_samples (list[int], optional): Posterior sample sizes for OPS-VBQN runs.
        bootstrap_heads (list[int], optional): Bootstrap head counts for BootstrapDQN runs.
        base_model_dir (str, optional, default="models"): Root directory where trained models are saved.
        base_data_dir (str, optional, default="data"): Root directory where benchmark results are stored.

    Notes:
        - Existing directories are preserved (created only if missing).
        - Nested folders are automatically generated for algorithm-specific configurations.

    Returns:
        None
    """


    os.makedirs(base_model_dir, exist_ok=True)
    os.makedirs(base_data_dir, exist_ok=True)

    for env in env_names:
        env_data_path = os.path.join(base_data_dir, env)
        os.makedirs(env_data_path, exist_ok=True)

        env_model_path = os.path.join(base_model_dir, env)
        os.makedirs(env_model_path, exist_ok=True)

        for algo in algo_names:
            if algo == "OPS-VBQN" and posterior_samples:
                for ps in posterior_samples:
                    os.makedirs(os.path.join(env_model_path, algo, f"post_{ps}"), exist_ok=True)
                    os.makedirs(os.path.join(env_data_path, algo, f"post_{ps}"), exist_ok=True)
            elif algo == "BootstrapDQN" and bootstrap_heads:
                for bh in bootstrap_heads:
                    os.makedirs(os.path.join(env_model_path, algo, f"heads_{bh}"), exist_ok=True)
                    os.makedirs(os.path.join(env_data_path, algo, f"heads_{bh}"), exist_ok=True)
            else:
                os.makedirs(os.path.join(env_model_path, algo), exist_ok=True)
                os.makedirs(os.path.join(env_data_path, algo), exist_ok=True)

    print("Folder structure ready!")


# List of supported environments and algorithms
ENVIRONMENTS = ["CartPole-v1", "LunarLander-v3", "Taxi-v3", "Acrobot-v1"]
ALGORITHMS = ["DQN", "OPS-VBQN", "BootstrapDQN"]
ENV_CONFIG = {
    "CartPole-v1": {
        "upper": 600,
        "lower": 0,
        "smooth_factor": 0.001,
        "max_episodes": 3600,
        "max_steps": 500000,
        "benchmark_episodes": 50,
        "reward_limit": 500
    },
    "LunarLander-v3": {
        "upper": 350,
        "lower": -300,
        "smooth_factor": 0.0005,
        "max_episodes": 5600,
        "max_steps": 1800000,
        "benchmark_episodes": 50,
        "reward_limit": 260
    },
    "Taxi-v3": {
        "upper": 100,
        "lower": -900,
        "smooth_factor": 0.001,
        "max_episodes": 1200,
        "max_steps": 150000,
        "benchmark_episodes": 50,
        "reward_limit": 8   
    },
    "Acrobot-v1": {
        "upper": -50,
        "lower": -500,
        "smooth_factor": 0.001,
        "max_episodes": 1500,
        "max_steps": 500000,
        "benchmark_episodes": 50,
        "reward_limit": -85    
    }
}



# Common base config
COMMON_CONFIG = {
    "alpha": 0.00025,
    "gamma": 0.99,
    "start_e": 1,
    "end_e": 0.05,
    "train_frequency": 10,
    "learning_start": 10000,
    "network_sync_rate": 500,
    "mini_batch_size": 128,
    "hidden_layer_size": 128
}



# Per-environment override
ENV_SPECIFIC = {
    "LunarLander-v3": {
        "replay_memory_size": 50000,
        "exploration_fraction": 0.5,
        "alpha": 0.00075
    },
    "Acrobot-v1": {
        "replay_memory_size": 10000,
        "exploration_fraction": 0.5
    },
    "CartPole-v1": {
        "replay_memory_size": 10000,
        "exploration_fraction": 0.5
    },
    "Taxi-v3": {
        "replay_memory_size": 20000,
        "exploration_fraction": 0.3,
        "end_e": 0.05,
        "learning_start": 1000,
        "train_frequency": 1,
        "network_sync_rate": 100,
        "mini_batch_size": 32,
        "alpha": 0.001,
        "hidden_layer_size": 64
    }
}


def create_agent(env_name: str, model_class: Type[torch.nn.Module], env_instance: gym.Env) -> Agent:
    """
    Instantiates an Agent with environment-specific configuration.

    Combines common and environment-specific hyperparameters, creates the Agent,
    and assigns the provided environment instance to it.

    Args:
        env_name (str): Name of the environment (e.g., "CartPole-v1").
        model_class (Type[torch.nn.Module]): The neural network class to use for the agent's policy.
        env_instance (gym.Env): An initialized Gym environment instance for the agent.

    Returns:
        Agent: A fully initialized agent with the specified model and environment assigned.
    """
    config = {**COMMON_CONFIG, **ENV_SPECIFIC[env_name]}
    agent = Agent(env_name = env_name, model=model_class, **config)
    agent.env = env_instance
    return agent