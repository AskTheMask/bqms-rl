"""
dqn.py

Command-line interface for training and benchmarking DQN agents on Gym environments.

This script allows specifying environment IDs and the number of random seeds to run.
For each environment, it creates a Gym environment instance, instantiates a DQN agent,
and runs benchmarking according to environment-specific configuration settings.

It handles:
- Argument parsing using Tyro
- Validation of selected environments
- Setup of project directories for models and benchmark data
- Agent creation with environment-specific hyperparameters
- Benchmarking and evaluation of the agent, saving episodic returns and metrics

Example:
    python dqn.py --env-ids CartPole-v1 Acrobot-v1 --seeds 3
"""

import tyro
from dataclasses import dataclass
from typing import List
from configs import ENVIRONMENTS, ALGORITHMS, ENV_CONFIG, create_agent, setup_project_dirs
from models import DQN
from utils import make_env, get_random_seeds, validate_args

@dataclass
class Args:
    """Command-line arguments for running DQN agents.

    Attributes:
        env_ids (List[str]): List of environment IDs to run the agent on.
        seeds (int): Number of random seeds to run for each environment. Defaults to 1.
    """
    env_ids: List[str]
    seeds: int = 1 


if __name__ == "__main__":
    # Parse command-line arguments
    args = tyro.cli(Args) 

    # Validate environment IDs
    validate_args(args.env_ids, environments = ENVIRONMENTS)
    
    # Generate selected random seeds
    selected_seeds = get_random_seeds(args.seeds)  

    # Set up project directories for storing results
    setup_project_dirs(ENVIRONMENTS, ALGORITHMS)

    # Run DQN agents for each environment
    for env in args.env_ids:
        print(f"\n Running DQN on {env}")
        env_instance = make_env(env)  # create gym environment
        agent = create_agent(env, DQN, env_instance)  # pass env_instance to agent if your Agent supports it
        agent.benchmarks(
            max_steps=ENV_CONFIG[env]["max_steps"],
            episodes=ENV_CONFIG[env]["benchmark_episodes"],
            reward_limit=ENV_CONFIG[env]["reward_limit"],
            max_episodes=ENV_CONFIG[env]["max_episodes"],
            num_samples=None,
            bootstrap_heads=None,
            seeds=selected_seeds
        )


