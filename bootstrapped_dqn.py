# dqn.py
import tyro
from dataclasses import dataclass
from typing import List
from configs import ENVIRONMENTS, ALGORITHMS, setup_project_dirs
from models import BootstrapDQN
from configs import create_agent, ENV_CONFIG # assumes you moved make_env to utils.py
from utils import make_env, get_random_seeds, validate_args

@dataclass
class Args:
    env_ids: List[str]
    bootstrap_heads: List[int]  # pick which bootstrap heads to run
    seeds: int = 1  # pick how many seeds to run



if __name__ == "__main__":
    args = tyro.cli(Args)
    
    validate_args(args.env_ids, environments = ENVIRONMENTS, bootstrap_heads=args.bootstrap_heads)
    selected_seeds = get_random_seeds(args.seeds)  

    setup_project_dirs(ENVIRONMENTS, ALGORITHMS, bootstrap_heads = args.bootstrap_heads)


    for env in args.env_ids:
        print(f"\n Running DQN on {env}")
        for bootstrap_heads in args.bootstrap_heads:
            print(f"\n Bootstrap Heads: {bootstrap_heads}")
            env_instance = make_env(env)  # create gym environment
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


