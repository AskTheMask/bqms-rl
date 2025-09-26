from dataclasses import dataclass
from typing import List, Optional
from utils import validate_args, get_random_seeds
from configs import ENVIRONMENTS, ENV_CONFIG, ALGORITHMS
from plots import plot_learning_curves, plot_posterior_metrics, save_latex_table_pdf
import tyro

@dataclass
class Args:
    type: str  # "learning_curve", "posterior_metrics", "results_table"
    
    # Common args
    save_path: Optional[str] = None 
    seeds: int = 1


    # For learning_curve
    env_id: Optional[str] = None
    bootstrap_head: Optional[int] = None
    num_sample: Optional[int] = None

    # For box_plot
    num_samples_list: Optional[List[int]] = None  # list of samples to include in the boxplot

    # For boxplot and learning_curve
    show: bool = False



if __name__ == "__main__":
    args = tyro.cli(Args)

    # validate seeds (all plot types)
    selected_seeds = get_random_seeds(args.seeds)

    # validate environment only if needed
    if args.type in ["learning_curve", "posterior_metrics"]:
        if args.env_id not in ENVIRONMENTS:
            raise ValueError(f"Invalid environment '{args.env_id}'. Must be one of {ENVIRONMENTS}.")
    
    if args.type == "posterior_metrics":
        if not args.num_samples_list or len(args.num_samples_list) == 0:
            raise ValueError("a list of num_samples must be provided for posterior_metrics plots.")

  

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
