"""
plotting.py

Script to generate plots and result table from RL experiments.

This script supports three types of operations:

1. `learning_curve`:
    - Plots learning curves for all algorithms on a specified environment.
    - Requires a number of posterior samples (for OPS-VBQN) and a number of bootstrap heads (for Boootstrapped DQN)

2. `posterior_metrics`:
    - Plots boxplots of episodes to solve and cumulative regret across multiple posterior samples.
    - Requires a list of posterior sample sizes to include.

3. `results_table`:
    - Generates a LaTeX table summarizing benchmark results.
    - Can optionally save and display the table as a PDF.
"""


from dataclasses import dataclass
from typing import List, Optional
from utils import validate_args, get_random_seeds
from configs import ENVIRONMENTS, ENV_CONFIG, ALGORITHMS
from plots import plot_learning_curves, plot_posterior_metrics, save_latex_table_pdf
import tyro

@dataclass
class Args:
    """Command-line arguments for plotting RL experiment results.

    Attributes:
        type (str): Type of plot or operation. One of "learning_curve",
            "posterior_metrics", "results_table".

        save_path (Optional[str]): Path to save generated plots or table.
        seeds (int): Number of random seeds to include. Defaults to 1.

        bootstrap_head (Optional[int]): Head index for bootstrapped DQN
            when plotting learning curves.
        num_sample (Optional[int]): Number of posterior samples for OPS-VBQN
            when plotting learning curves.

        num_samples_list (Optional[List[int]]): List of posterior samples
            to include when plotting posterior metrics.

        env_id (Optional[str]): Environment ID for plotting learning curves
            or posterior metrics.
        show (bool): Whether to display the plot/table immediately. Defaults to False.
    """
    type: str
    save_path: Optional[str] = None
    seeds: int = 1
    bootstrap_head: Optional[int] = None
    num_sample: Optional[int] = None
    num_samples_list: Optional[List[int]] = None
    env_id: Optional[str] = None
    show: bool = False


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Validate seeds
    selected_seeds = get_random_seeds(args.seeds)

    # Validate environment if relevant
    if args.type in ["learning_curve", "posterior_metrics"]:
        if args.env_id not in ENVIRONMENTS:
            raise ValueError(f"Invalid environment '{args.env_id}'. Must be one of {ENVIRONMENTS}.")
        
    # Validate posterior sample list if needed
    if args.type == "posterior_metrics":
        if not args.num_samples_list or len(args.num_samples_list) == 0:
            raise ValueError("a list of num_samples must be provided for posterior_metrics plots.")

    # Plot according to type
    if args.type == "learning_curve":
        config = ENV_CONFIG[args.env_id]
        plot_learning_curves(
            env_name = args.env_id,
            alg_names = ALGORITHMS,
            seed_list = selected_seeds,
            upper_reward = config["upper"],
            lower_reward = config["lower"],
            smooth_factor = config["smooth_factor"],
            show = args.show,
            posterior_samples = args.num_sample,
            bootstrap_heads = args.bootstrap_head,
            save_path = args.save_path
        )
    elif args.type == "posterior_metrics":
        plot_posterior_metrics(
            env_name=args.env_id,
            seeds=selected_seeds,
            posterior_samples_list=args.num_samples_list,
            save_path=args.save_path,
            show=args.show
        )
    elif args.type == "results_table":
        save_latex_table_pdf(
            seeds=selected_seeds,
            save_path=args.save_path,
            show=args.show
            )
