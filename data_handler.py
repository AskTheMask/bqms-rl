import os
import json
import torch
import numpy as np

def _maybe_add_variant_dir(base_path, alg_name, posterior_samples=None, bootstrap_heads=None):
    if alg_name == "BQMS" and posterior_samples is not None:
        return os.path.join(base_path, f"post_{posterior_samples}")
    elif alg_name == "BootstrapDQN" and bootstrap_heads is not None:
        return os.path.join(base_path, f"heads_{bootstrap_heads}")
    return base_path


# === MODEL PATHS ===

def get_model_path(env_name, alg_name, seed, posterior_samples=None, bootstrap_heads=None):
    base_path = os.path.join("models", env_name, alg_name)
    base_path = _maybe_add_variant_dir(base_path, alg_name, posterior_samples, bootstrap_heads)
    return os.path.join(base_path, f"seed{seed}.pt")


def save_model(model, env_name, alg_name, seed, posterior_samples=None, bootstrap_heads=None):
    path = get_model_path(env_name, alg_name, seed, posterior_samples, bootstrap_heads)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model_class, env_name, alg_name, seed, in_features, hidden_layer_size, out_features,
               posterior_samples=None, bootstrap_heads=None):
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

def get_returns_path(env_name, alg_name, seed, posterior_samples=None, bootstrap_heads=None):
    base_path = os.path.join("data", env_name, alg_name)
    base_path = _maybe_add_variant_dir(base_path, alg_name, posterior_samples, bootstrap_heads)
    return os.path.join(base_path, f"seed{seed}_returns.npy")


def get_benchmarks_path(env_name, alg_name, posterior_samples=None, bootstrap_heads=None):
    base_path = os.path.join("data", env_name, alg_name)
    base_path = _maybe_add_variant_dir(base_path, alg_name, posterior_samples, bootstrap_heads)
    return os.path.join(base_path, "benchmarks.json")


def save_seed_data(
    env_name,
    alg_name,
    seed,
    benchmark_score=None,
    returns=None,
    cumulative_regret=None,
    episodes_to_solve=None,
    posterior_samples=None,
    bootstrap_heads=None
):
    folder_path = os.path.dirname(get_returns_path(env_name, alg_name, seed, posterior_samples, bootstrap_heads))
    os.makedirs(folder_path, exist_ok=True)

    # ✅ Only save returns if provided
    if returns is not None:
        returns_path = get_returns_path(env_name, alg_name, seed, posterior_samples, bootstrap_heads)
        np.save(returns_path, returns)

    # ✅ Load existing benchmark JSON or create new
    benchmarks_path = get_benchmarks_path(env_name, alg_name, posterior_samples, bootstrap_heads)
    if os.path.exists(benchmarks_path):
        with open(benchmarks_path, "r") as f:
            benchmarks = json.load(f)
    else:
        benchmarks = {}

    seed_key = f"seed{seed}"

    # ✅ Ensure the benchmark entry exists
    if seed_key not in benchmarks:
        benchmarks[seed_key] = {}

    # ✅ Update metrics individually if provided
    if benchmark_score is not None:
        benchmarks[seed_key]["benchmark"] = benchmark_score

    if cumulative_regret is not None:
        benchmarks[seed_key]["cumulative_regret"] = cumulative_regret

    if episodes_to_solve is not None:
        benchmarks[seed_key]["episodes_to_solve"] = episodes_to_solve

    # ✅ Save updated JSON
    with open(benchmarks_path, "w") as f:
        json.dump(benchmarks, f, indent=2)



# === LOAD ALL BENCHMARKS ===

def load_all_benchmarks(data_root="data"):
    all_data = {}
    for env_name in os.listdir(data_root):
        env_path = os.path.join(data_root, env_name)
        if not os.path.isdir(env_path):
            continue

        all_data[env_name] = {}

        for alg_name in os.listdir(env_path):
            alg_path = os.path.join(env_path, alg_name)

            # === Handle BQMS with posterior sample subfolders ===
            if alg_name == "BQMS":
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

def load_returns_for_env(env_name, seeds=None, data_root="data", posterior_samples=None, bootstrap_heads=None):
    env_path = os.path.join(data_root, env_name)
    if not os.path.isdir(env_path):
        raise ValueError(f"Environment folder '{env_name}' not found.")

    alg_data = {}

    for alg_name in os.listdir(env_path):
        alg_path = os.path.join(env_path, alg_name)
        if not os.path.isdir(alg_path):
            continue

        # === Handle BQMS ===
        if alg_name == "BQMS":
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


def _load_seed_returns(folder_path, seeds):
    """Helper function to load seed returns from a folder."""
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


