"""
bootstrapped_dqn.py

Command-line interface for training and benchmarking BootstrapDQN agents
across one or more Gym environments.

This module uses Tyro for argument parsing and relies on shared configuration
utilities to manage environments, agent creation, and experiment reproducibility.
"""


import tyro
from dataclasses import dataclass
from typing import List
from configs import ENVIRONMENTS, ALGORITHMS, setup_project_dirs
from models import BootstrapDQN
from configs import create_agent, ENV_CONFIG 
from utils import make_env, get_random_seeds, validate_args

@dataclass
class Args:
    """Command-line arguments for running BootstrapDQN agents.

    Attributes:
        env_ids (List[str]): List of environment IDs to run the agent on.
        bootstrap_heads (List[int]): Bootstrap head counts to use for each run.
        seeds (int): Number of random seeds to run for each environment. Defaults to 1.
    """
    env_ids: List[str] 
    bootstrap_heads: List[int] 
    seeds: int = 1 



if __name__ == "__main__":
    # Parse command-line arguments
    args = tyro.cli(Args)
    
    # Validate environment IDs and bootstrap head sizes
    validate_args(args.env_ids, environments = ENVIRONMENTS, bootstrap_heads=args.bootstrap_heads)

    # Generate selected random seeds
    selected_seeds = get_random_seeds(args.seeds)  

    # Set up project directories for storing results
    setup_project_dirs(ENVIRONMENTS, ALGORITHMS, bootstrap_heads = args.bootstrap_heads)

    # Run BootstrapDQN agents for each environment and bootstrap head configuration
    for env in args.env_ids:
        print(f"\n Running Bootstrapped DQN on {env}")
        for bootstrap_heads in args.bootstrap_heads:
            print(f"\n Bootstrap Heads: {bootstrap_heads}")
            env_instance = make_env(env)  
            agent = create_agent(env, BootstrapDQN, env_instance) 
            agent.benchmarks(
                max_steps=ENV_CONFIG[env]["max_steps"],
                episodes=ENV_CONFIG[env]["benchmark_episodes"],
                reward_limit=ENV_CONFIG[env]["reward_limit"],
                max_episodes=ENV_CONFIG[env]["max_episodes"],
                num_samples=None,
                bootstrap_heads=bootstrap_heads,
                seeds=selected_seeds
            )


