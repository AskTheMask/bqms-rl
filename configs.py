import os
from agent import Agent

def setup_project_dirs(env_names, algo_names, posterior_samples=None, bootstrap_heads=None,
                       base_model_dir="models", base_data_dir="data"):
    os.makedirs(base_model_dir, exist_ok=True)
    os.makedirs(base_data_dir, exist_ok=True)

    for env in env_names:
        env_data_path = os.path.join(base_data_dir, env)
        os.makedirs(env_data_path, exist_ok=True)

        env_model_path = os.path.join(base_model_dir, env)
        os.makedirs(env_model_path, exist_ok=True)

        for algo in algo_names:
            if algo == "BQMS" and posterior_samples:
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

    print("✅ Folder structure ready!")


# List of environments and algorithms
ENVIRONMENTS = ["CartPole-v1", "LunarLander-v3", "Taxi-v3", "Acrobot-v1"]
ALGORITHMS = ["DQN", "BQMS", "BootstrapDQN"]
ENV_CONFIG = {
    "CartPole-v1": {
        "upper": 600,
        "lower": 0,
        "smooth_factor": 0.001,
        "max_episodes": 3600,
        "max_steps": 500000,
        "benchmark_episodes": 50,
        "reward_limit": 500,
        "posterior_sample_list": [1, 50, 100, 200, 300, 400]
    },
    "LunarLander-v3": {
        "upper": 350,
        "lower": -300,
        "smooth_factor": 0.0005,
        "max_episodes": 5600,
        "max_steps": 1800000,
        "benchmark_episodes": 50,
        "reward_limit": 260,
        "posterior_sample_list": [1, 200, 400, 600]
    },
    "Taxi-v3": {
        "upper": 100,
        "lower": -900,
        "smooth_factor": 0.001,
        "max_episodes": 1200,
        "max_steps": 150000,
        "benchmark_episodes": 50,
        "reward_limit": 8,
        "posterior_sample_list": [1, 50, 100, 200, 300, 400]
    },
    "Acrobot-v1": {
        "upper": -50,
        "lower": -500,
        "smooth_factor": 0.001,
        "max_episodes": 1500,
        "max_steps": 500000,
        "benchmark_episodes": 50,
        "reward_limit": -85,
        "posterior_sample_list": [1, 200, 400, 600, 800, 1000]
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


def create_agent(env_name, model_class, env_instance):
    config = {**COMMON_CONFIG, **ENV_SPECIFIC[env_name]}
    agent = Agent(env_name = env_name, model=model_class, **config)
    agent.env = env_instance
    return agent