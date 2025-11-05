# Reinforcement Learning Benchmark Suite

## Project Overview
The Reinforcement Learning Benchmark Suite is a modular framework for training, evaluating, and visualizing the performance of RL agents across multiple environments. It supports standard Gym environments and provides implementations of DQN, OPS-VBQN, and Bootstrapped DQN. With built-in tools for tracking performance, generating plots, and exporting results, this framework is designed for reproducible, comparative RL research.

## Features
- **Multi Agent Training:** Train DQN, OPS-VBQN, and BootstrapDQN agents seamlessly.
- **Flexible Experiment Configuration:** Easily customize random seeds, posterior samples, and bootstrap heads.
- **Rich Visualization:** Automatically generate learning curves, posterior metrics, and summary tables.
- **Reproducibility Focused:** Save experiment data and plots to ensure consistent, repeatable results.
- **Command-Line Interface:** Run experiments and generate plots with simple CLI commands.
- **Multi-Environment Support:** Evaluate agents across multiple Gym environments in a single run.

## Requirements
Before running the experiments, make sure you have the following installed:

- Python 3.10+  
- PyTorch  (deep learning backend, includes torch.nn and torch.nn.functional)
- Gymnasium  (modern replacement for gym, standard RL environments)
- Tyro  (CLI argument parsing)
- Matplotlib (plotting)
- NumPy (numerical computations)
- Tqdm (progress bars)
- LaTeX (required for PDF generation using pdflatex)

Install all Python dependencies via:

```bash
pip install -r requirements.txt
```

**Note:** LaTeX is only required if you plan to generate PDFs from the benchmark tables. On Linux/macOS, install TeX Live or MacTeX; on Windows, use MiKTeX.

## Project Structure
```
ops-vbqn-rl/
├─ configs.py           # Environment & algorithm configurations
├─ data_handler.py      # Save/load training results and models
├─ memory.py            # Replay buffer implementation
├─ agent.py             # Agent class with full training logic
├─ models.py            # Neural network models for DQN, OPS-VBQN, and BootstrapDQN
├─ utils.py             # Helper functions (env creation, seeding, formatting)
├─ dqn.py               # CLI for training DQN agents
├─ ops-vbqn.py          # CLI for training OPS-VBQN agents
├─ bootstrapped_dqn.py  # CLI for training BootstrapDQN agents
├─ plot_cli.py          # CLI for generating plots and PDF tables
├─ plots.py             # Plotting helper functions
├─ results/             # Stores benchmark results and trained models
└─ README.md            # Project overview and instructions
```


## Training Agents

Each algorithm has its own training script.

### Scripts Overview

| Script                  | Algorithm        | Required Arguments                         |
|-------------------------|------------------|--------------------------------------------|
| `dqn.py`                | DQN              | `--env-ids`, `--seeds`                     |
| `ops-vbqn.py`           | OPS-VBQN         | `--env-ids`, `--seeds`, `--num_samples`    |
| `bootstrapped_dqn.py`   | Bootstrapped DQN | `--env-ids`, `--seeds`, `--bootstrap_heads`|

### Example Commands

```bash
# Train DQN on multiple environments with 5 seeds
python dqn.py --env-ids CartPole-v1 Acrobot-v1 --seeds 5

# Train OPS-VBQN on Taxi-v3 with 3 seeds and  posterior samples 1, 100 and 200
python ops-vbqn.py --env-ids Taxi-v3 --seeds 3 --num_samples 1 100 200

# Train BootstrapDQN on LunarLander-v3 with 4 seeds and bootstrap heads 4 and 8
python bootstrapped_dqn.py --env-ids LunarLander-v3 --seeds 4 --bootstrap_heads 4 8
```


## Plotting & Visualization

All plots and PDF generation are handled by `plot_cli.py`.

### Supported Plot Types

| Type                | Description                                 | Required Arguments                                      |
|--------------------|---------------------------------------------|--------------------------------------------------------|
| `learning_curve`    | Episodic rewards  | `--env-id`, `--seeds`, `--num_sample`, `--bootstrap_head` |
| `posterior_metrics` | Episodes to solve and cumulative regret vs posterior samples      | `--env-id`, `--num_samples_list`, `--seeds`          |
| `results_table`     | Generates a PDF table summarizing result metrics for all available data   | `--seeds`, `--save_path`                              |

### Arguments

| Argument            | Type        | Description |
|--------------------|------------|-------------|
| `--type`            | str        | `"learning_curve"`, `"posterior_metrics"`, or `"results_table"` |
| `--env-id`          | str        | Environment ID (required for plots needing an environment) |
| `--num_sample`      | int        | Number of posterior samples for `OPS-VBQN` |
| `--bootstrap_head`  | int        | Number of bootstrap heads for `Bootstrapped DQN` |
| `--num_samples_list`| list[int]  | List of posterior samples for `posterior_metrics` |
| `--seeds`           | int        | Number of random seeds to run (max 10) |
| `--save_path`       | str        | Path to save plots or PDF |
| `--show`            | bool       | Display plots or PDF interactively |

### Example Commands

```bash
# Plot learning curves for DQN, OPS-VBQN with 800 posterior samples and Bootstrapped DQN with 4 bootstrap heads for the CartPole-v1 environment  using 3 seeds
python plot_cli.py --type learning_curve --env-id CartPole-v1 --num_sample 800 --bootstrap_head 4 --seeds 3 --show

# Plots Episodes_to_solve and cumulative regret boxplot for the CartPole-v1 environment using 1, 100 and 200 posterior samples and 3 seeds
python plot_cli.py --type posterior_metrics --env-id CartPole-v1 --num_samples_list 1 100 200 --seeds 3 --show

# Generates the results table for all metrics and available data using 3 seeds and saves it to path ./results.pdf
python plot_cli.py --type results_table --seeds 3 --save_path ./results.pdf
```

## Reproducibility

This framework ensures experiments are reproducible by carefully controlling random seeds and sources of randomness.

- **Random Seeds:**  
  Run experiments with multiple seeds using the `--seeds` argument. Deterministic seeds are generated for each run to guarantee reproducibility.

- **Seeding Across Libraries:**  
  Seeds are applied to Python, NumPy, PyTorch (CPU and GPU), and the Gym environment. This ensures consistent neural network initialization, stochastic agent behavior, and environment dynamics.

- **Consistent Parameters:**  
  Parameters such as `num_samples` (for OPS-VBQN) or `bootstrap_heads` (for Bootstrapped DQN) are applied consistently across seeds to maintain reproducibility.

- **Seed Limit:**  
  To prevent excessive computation, a maximum number of seeds can be specified (default: 10). Exceeding this limit will raise an error.
