"""
ops-vbqn.py

Script to run OPS-VBQN agents on selected environments.

This script allows running OPS-VBQN (Bayesian Q-Network) agents on multiple
environments, with configurable posterior sample sizes and random seeds.
"""

import tyro
from dataclasses import dataclass
from typing import List
from configs import ENVIRONMENTS, ALGORITHMS, setup_project_dirs
from models import OPS_VBQN
from configs import create_agent, ENV_CONFIG 
from utils import make_env, get_random_seeds, validate_args

@dataclass
class Args:
    """Command-line arguments for running OPS-VBQN agents.

    Attributes:
        env_ids (List[str]): List of environment IDs to run the agent on.
        num_samples (List[int]): Posterior sample counts to use for each run.
        seeds (int): Number of random seeds to run for each environment. Defaults to 1.
    """
    env_ids: List[str]
    num_samples: List[int]  
    seeds: int = 1 


if __name__ == "__main__":
    # Parse command-line arguments
    args = tyro.cli(Args) 

    # Validate environment IDs and posterior sample sizes
    validate_args(args.env_ids, environments = ENVIRONMENTS, num_samples=args.num_samples)
    
    # Generate selected random seeds
    selected_seeds = get_random_seeds(args.seeds)  

    # Set up project directories for storing results
    setup_project_dirs(ENVIRONMENTS, ALGORITHMS, posterior_samples = args.num_samples)
    
    # Run OPS-VBQN agents for each environment and posterior sample configuration
    for env in args.env_ids:
        print(f"\n Running OPS-VBQN on {env}")
        for num_samples in args.num_samples:
            print(f"\n Posterior Samples: {num_samples}")
            env_instance = make_env(env)  # create gym environment
            agent = create_agent(env, OPS_VBQN, env_instance) 
            agent.benchmarks(
                max_steps=ENV_CONFIG[env]["max_steps"],
                episodes=ENV_CONFIG[env]["benchmark_episodes"],
                reward_limit=ENV_CONFIG[env]["reward_limit"],
                max_episodes=ENV_CONFIG[env]["max_episodes"],
                num_samples=num_samples,
                bootstrap_heads=None,
                seeds=selected_seeds
            )


