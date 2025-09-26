# dqn.py
import tyro
from dataclasses import dataclass
from typing import List
from configs import ENVIRONMENTS, ALGORITHMS, setup_project_dirs
import random
from models import BQMS
from configs import create_agent, ENV_CONFIG # assumes you moved make_env to utils.py
from utils import make_env, get_random_seeds, validate_args

@dataclass
class Args:
    env_ids: List[str]
    num_samples: List[int]  # pick which posterior samples to run
    seeds: int = 1 # pick how many seeds to run


if __name__ == "__main__":
    args = tyro.cli(Args) 


    validate_args(args.env_ids, environments = ENVIRONMENTS, num_samples=args.num_samples)
    selected_seeds = get_random_seeds(args.seeds)  

    setup_project_dirs(ENVIRONMENTS, ALGORITHMS, posterior_samples = args.num_samples)

    

    

    for env in args.env_ids:
        print(f"\n Running DQN on {env}")
        for num_samples in args.num_samples:
            print(f"\n Posterior Samples: {num_samples}")
            env_instance = make_env(env)  # create gym environment
            agent = create_agent(env, BQMS, env_instance) 
            agent.benchmarks(
                max_steps=ENV_CONFIG[env]["max_steps"],
                episodes=ENV_CONFIG[env]["benchmark_episodes"],
                reward_limit=ENV_CONFIG[env]["reward_limit"],
                max_episodes=ENV_CONFIG[env]["max_episodes"],
                num_samples=num_samples,
                bootstrap_heads=None,
                seeds=selected_seeds
            )


