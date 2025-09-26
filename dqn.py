# dqn.py
import tyro
from dataclasses import dataclass
from typing import List
from configs import ENVIRONMENTS, ALGORITHMS, ENV_CONFIG, create_agent, setup_project_dirs
from models import DQN
from utils import make_env, get_random_seeds, validate_args

@dataclass
class Args:
    env_ids: List[str]
    seeds: int = 1  # pick how many seeds to run


if __name__ == "__main__":
    args = tyro.cli(Args) 


    validate_args(args.env_ids, environments = ENVIRONMENTS)

    selected_seeds = get_random_seeds(args.seeds)  

    setup_project_dirs(ENVIRONMENTS, ALGORITHMS)


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


