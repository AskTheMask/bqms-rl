# Reinforcement Learning Benchmark Suite

## Project Overview
This project provides a comprehensive framework for training, evaluating, and visualizing the performance of different reinforcement learning (RL) agents across multiple environments. It supports standard Gym environments and allows experimentation with DQN, BQMS, and BootstrapDQN agents. The framework includes features for tracking performance, generating learning curves, box plots, and exporting results in a reproducible way.

## Features
- **Train Multiple RL Agents:** Supports DQN, BQMS, and BootstrapDQN.
- **Configurable Experiments:** Easily set seeds, posterior samples, and bootstrap heads.
- **Visualize Results:** Generate learning curves, box plots, and summary tables.
- **Reproducibility:** Save plots and benchmark data.
- **Flexible CLI:** Simple command-line interface for running experiments and generating plots.
- **Multi-environment Support:** Run experiments across multiple Gym environments.

## Requirements

- Python 3.10+  
- PyTorch  
- Gym  
- Tyro  
- Matplotlib / Seaborn / Plotly  
- ReportLab or FPDF for PDF generation  

Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

project/
├─ configs.py           # Environment & algorithm configurations
├─ models.py            # Agent implementations: DQN, BQMS, BootstrapDQN
├─ utils.py             # Helper functions (e.g., make_env, seeding)
├─ train_dqn.py         # CLI to train DQN agents
├─ train_bqms.py        # CLI to train BQMS agents
├─ train_bootstrap.py   # CLI to train BootstrapDQN agents
├─ plot_cli.py          # CLI to plot results and generate PDFs
├─ plots.py             # Plotting functions
├─ results/             # Stores benchmark results and generated plots
└─ README.md            # Project overview and instructions



## Training Agents

Each algorithm has its own training script.

### Scripts Overview

| Script | Algorithm | Required Arguments |
|--------|-----------|------------------|
| `train_dqn.py` | DQN | `--env-ids`, `--seeds` |
| `train_bqms.py` | BQMS | `--env-ids`, `--seeds`, `--num_samples` |
| `train_bootstrap.py` | BootstrapDQN | `--env-ids`, `--seeds`, `--bootstrap_heads` |




### Example Commands

```bash
# Train DQN
python train_dqn.py --env-ids CartPole-v1 MountainCar-v0 --seeds 5

# Train BQMS
python train_bqms.py --env-ids CartPole-v1 --seeds 3 --num_samples 50 100 200

# Train BootstrapDQN
python train_bootstrap.py --env-ids MountainCar-v0 --seeds 4 --bootstrap_heads 4 8
```



## Plotting & Visualization

All plots and PDF generation are handled by `plot_cli.py`.

### Supported Plot Types

| Type | Description | Required Arguments |
|------|------------|------------------|
| `learning_curve` | Cumulative reward or posterior sample trends | `--env-id`, `--seeds` (optional: `--num_sample`, `--bootstrap_head`) |
| `box_plot` | Episodes-to-solve vs posterior samples | `--env-id`, `--num_samples_list`, `--seeds` |
| `results_table` | Generates a PDF table summarizing results | `--seeds`, `--save_path` |



### Arguments

| Argument | Type | Description |
|----------|------|------------|
| `--type` | str | `"learning_curve"`, `"box_plot"`, or `"results_table"` |
| `--env-id` | str | Environment ID (required for plots needing an environment) |
| `--num_sample` | int | Number of posterior samples (optional for `learning_curve`) |
| `--bootstrap_head` | int | Number of heads for `BootstrapDQN` (optional for `learning_curve`) |
| `--num_samples_list` | list[int] | List of posterior samples for `box_plot` |
| `--seeds` | int | Number of random seeds to run (max 10) |
| `--save_path` | str | Path to save plots or PDFs |
| `--show` | bool | Display plots interactively |



### Example Commands

**Learning curve:**
```bash
python plot_cli.py --type learning_curve --env-id CartPole-v1 --num_sample 50 --bootstrap_head 4 --seeds 3 --show
```


**Episodes_to_solve and cumulative regret boxplot:**
```bash
python plot_cli.py --type plot_posterior_metrics --env-id CartPole-v1 --num_samples_list 50 100 200 --seeds 3 --show
```

**Results table PDF:**
```bash
python plot_cli.py --type results_table --seeds 3 --save_path ./results.pdf
```

## Reproducibility

This framework ensures experiments are reproducible by carefully controlling random seeds and sources of randomness:

- **Random Seeds:**  
  Experiments can be run with multiple seeds using the `--seeds` argument. The framework generates deterministic seeds for each run to guarantee reproducibility.

- **Seeding Across Libraries:**  
  Seeds are applied to Python, NumPy, PyTorch (CPU and GPU), and the Gym environment. This ensures consistent neural network initialization, stochastic agent behavior, and environment dynamics.

- **Optional Parameters:**  
  Parameters like `num_samples` (for BQMS) or `bootstrap_heads` (for BootstrapDQN) are applied consistently across seeds to maintain reproducibility.

- **Limit on Seeds:**  
  To avoid excessive computation, a maximum number of seeds can be specified (default: 10). Exceeding this will raise an error.
