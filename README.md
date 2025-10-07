# Reinforcement Learning Benchmark Suite

## Project Overview
The Reinforcement Learning Benchmark Suite is a modular framework for training, evaluating, and visualizing the performance of RL agents across multiple environments. It supports standard Gym environments and provides implementations of DQN, OPS-VBQN, and BootstrapDQN. With built-in tools for tracking performance, generating plots, and exporting results, this framework is designed for reproducible, comparative RL research.

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
project/
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

| Script                  | Algorithm     | Required Arguments                         |
|-------------------------|--------------|--------------------------------------------|
| `dqn.py`                | DQN          | `--env-ids`, `--seeds`                     |
| `ops-vbqn.py`           | OPS-VBQN     | `--env-ids`, `--seeds`, `--num_samples`   |
| `bootstrapped_dqn.py`   | BootstrapDQN | `--env-ids`, `--seeds`, `--bootstrap_heads` |

### Example Commands

```bash
# Train DQN on multiple environments with 5 seeds
python dqn.py --env-ids CartPole-v1 Acrobot-v1 --seeds 5

# Train OPS-VBQN on Taxi-v3 with 3 seeds and specific posterior samples
python ops-vbqn.py --env-ids Taxi-v3 --seeds 3 --num_samples 1 100 200

# Train BootstrapDQN on LunarLander-v3 with 4 seeds and bootstrap heads
python bootstrapped_dqn.py --env-ids LunarLander-v3 --seeds 4 --bootstrap_heads 4 8
```


## Plotting & Visualization

All plots and PDF generation are handled by `plot_cli.py`.

### Supported Plot Types

| Type                | Description                                 | Required Arguments                                      |
|--------------------|---------------------------------------------|--------------------------------------------------------|
| `learning_curve`    | Cumulative reward or posterior sample trends | `--env-id`, `--seeds`, `--num_sample`, `--bootstrap_head` |
| `posterior_metrics` | Episodes-to-solve vs posterior samples      | `--env-id`, `--num_samples_list`, `--seeds`          |
| `results_table`     | Generates a PDF table summarizing results   | `--seeds`, `--save_path`                              |

### Arguments

| Argument            | Type        | Description |
|--------------------|------------|-------------|
| `--type`            | str        | `"learning_curve"`, `"posterior_metrics"`, or `"results_table"` |
| `--env-id`          | str        | Environment ID (required for plots needing an environment) |
| `--num_sample`      | int        | Number of posterior samples |
| `--bootstrap_head`  | int        | Number of heads for `BootstrapDQN` |
| `--num_samples_list`| list[int]  | List of posterior samples for `posterior_metrics` |
| `--seeds`           | int        | Number of random seeds to run (max 10) |
| `--save_path`       | str        | Path to save plots or PDFs |
| `--show`            | bool       | Display plots or PDFs interactively |

### Example Commands

**Learning curve:**  
```bash
python plot_cli.py --type learning_curve --env-id CartPole-v1 --num_sample 800 --bootstrap_head 4 --seeds 3 --show
```


**Episodes_to_solve and cumulative regret boxplot:**
```bash
python plot_cli.py --type posterior_metrics --env-id CartPole-v1 --num_samples_list 1 100 200 --seeds 3 --show
```

**Results table PDF:**
```bash
python plot_cli.py --type results_table --seeds 3 --save_path ./results.pdf
```

## Reproducibility

This framework ensures experiments are reproducible by carefully controlling random seeds and sources of randomness.

- **Random Seeds:**  
  Run experiments with multiple seeds using the `--seeds` argument. Deterministic seeds are generated for each run to guarantee reproducibility.

- **Seeding Across Libraries:**  
  Seeds are applied to Python, NumPy, PyTorch (CPU and GPU), and the Gym environment. This ensures consistent neural network initialization, stochastic agent behavior, and environment dynamics.

- **Consistent Parameters:**  
  Parameters such as `num_samples` (for OPS-VBQN) or `bootstrap_heads` (for BootstrapDQN) are applied consistently across seeds to maintain reproducibility.

- **Seed Limit:**  
  To prevent excessive computation, a maximum number of seeds can be specified (default: 10). Exceeding this limit will raise an error.
