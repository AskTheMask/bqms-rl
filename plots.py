"""
plots.py

Functions for visualizing Reinforcement Learning experiment results, including:
- Learning curves over episodes
- Posterior distribution metrics (e.g., rewards, uncertainties)
- Boxplots for performance comparison
- Exporting results tables as LaTeX PDFs

This module is intended to support analysis and presentation of OPS-VBQN,
BootstrapDQN, or other RL algorithms' results.

Notes:
    - Many functions use Matplotlib for plotting.
    - Optional `save_path` parameters allow exporting plots or tables.
    - `show=True` can be used to visualize plots interactively.
"""

import numpy as np
from utils import last_nonzero_index, exponential_moving_average
from data_handler import load_all_benchmarks, get_returns_path
from configs import ENV_CONFIG
from typing import List, Optional
from collections import defaultdict
from matplotlib.ticker import ScalarFormatter
import os
import matplotlib.pyplot as plt 
import subprocess
import tempfile
from pathlib import Path
import sys


        
def plot_posterior_metrics(
    env_name: str,
    seeds: List[int],
    posterior_samples_list: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Generate boxplots for OPS-VBQN posterior metrics in a specified environment.

    This function plots two side-by-side boxplots:
        1. Cumulative regret
        2. Episodes to solve

    Data is aggregated across specified random seeds and filtered
    by a list of posterior sample sizes if provided. Useful for
    analyzing the variability and performance of OPS-VBQN across
    different posterior samples.

    Args:
        env_name (str): Name of the environment (must exist in stored benchmark data).
        seeds (List[int]): List of seed numbers to include in the plots.
        posterior_samples_list (Optional[List[int]], optional): 
            List of posterior sample sizes to include. If None, includes all available.
        save_path (Optional[str], optional): File path to save the figure. 
            If None, the figure is not saved.
        show (bool, optional): Whether to display the figure interactively. 
            Defaults to False.

    Raises:
        ValueError: If no OPS-VBQN data exists for the specified environment.

    Notes:
        - Assumes benchmark data is loaded via `load_all_benchmarks()`.
        - Each seed should have `cumulative_regret` and `episodes_to_solve` metrics.
        - Boxplots use consistent styling for publications or presentations.
    """
    all_data = load_all_benchmarks()

    if env_name not in all_data or "OPS-VBQN" not in all_data[env_name]:
        raise ValueError(f"No OPS-VBQN data found for environment '{env_name}'.")

    ops_vbqn_data = all_data[env_name]["OPS-VBQN"]
    grouped_data = defaultdict(lambda: {'cumulative_regret': [], 'episodes_to_solve': []})

    for key, seed_data in ops_vbqn_data.items():
        try:
            sample_size = int(key)  # expects keys like "400", not "posterior_400"
        except ValueError:
            continue

        if posterior_samples_list and sample_size not in posterior_samples_list:
            continue

        for seed in seeds:
            seed_key = f"seed{seed}"
            if seed_key in seed_data:
                metrics = seed_data[seed_key]
                if (
                    "cumulative_regret" in metrics
                    and "episodes_to_solve" in metrics
                    and isinstance(metrics["cumulative_regret"], (int, float))
                    and isinstance(metrics["episodes_to_solve"], (int, float))
                ):
                    grouped_data[sample_size]['cumulative_regret'].append(metrics["cumulative_regret"])
                    grouped_data[sample_size]['episodes_to_solve'].append(metrics["episodes_to_solve"])

    if not grouped_data:
        print("No matching data found.")
        return

    sorted_items = sorted(grouped_data.items())
    sorted_keys = [key for key, _ in sorted_items]

    # Prepare data arrays (no scaling)
    regret_data = [np.array(entry['cumulative_regret']).flatten() for _, entry in sorted_items]
    episode_data = [np.array(entry['episodes_to_solve']).flatten() for _, entry in sorted_items]

    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(max(len(sorted_keys) * 1.5, 10), 8), sharey=False)

    # Plot style properties
    boxprops = dict(linewidth=1.8)
    whiskerprops = dict(linewidth=1.6)
    capprops = dict(linewidth=1.6)
    medianprops = dict(linewidth=2)
    flierprops = dict(marker='o', markersize=3, linestyle='none')

    # --- Cumulative Regret ---
    axes[0].boxplot(
        regret_data, vert=True, patch_artist=False, notch=False,
        boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops,
        medianprops=medianprops, flierprops=flierprops, whis=[0, 100], widths=0.4
    )
    axes[0].set_xticks(range(1, len(sorted_keys) + 1))
    axes[0].set_xticklabels([str(k) for k in sorted_keys], rotation=90)
    axes[0].set_xlabel("Number of Actor Samples", fontsize=20)
    axes[0].set_ylabel("Cumulative Regret", fontsize=20)
    axes[0].tick_params(axis='both', labelsize=18)

    # --- Episodes to Solve ---
    axes[1].boxplot(
        episode_data, vert=True, patch_artist=False, notch=False,
        boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops,
        medianprops=medianprops, flierprops=flierprops, whis=[0, 100], widths=0.4
    )
    axes[1].set_xticks(range(1, len(sorted_keys) + 1))
    axes[1].set_xticklabels([str(k) for k in sorted_keys], rotation=90)
    axes[1].set_xlabel("Number of Actor Samples", fontsize=20)
    axes[1].set_ylabel("Episodes to Solve", fontsize=20)
    axes[1].tick_params(axis='both', labelsize=18)


    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)



def plot_learning_curves(
    env_name: str,
    alg_names: List[str],
    seed_list: List[int],
    upper_reward: int,
    lower_reward: int,
    smooth_factor: float = 0.99,
    save_path: Optional[str] = None,
    show: bool = False,
    posterior_samples: Optional[int] = None,
    bootstrap_heads: Optional[int] = None
) -> plt.Figure:
    """
    Plot smoothed learning curves (mean ± std) for multiple RL algorithms in a single environment.

    This function generates a publication-ready figure showing the learning
    performance of each algorithm across multiple seeds. The curves are
    smoothed using an exponential moving average, and the standard deviation
    is plotted as a shaded region. Optionally, data for OPS-VBQN and 
    BootstrapDQN can include posterior sample size or number of bootstrap heads.

    Args:
        env_name (str): Name of the environment.
        alg_names (List[str]): List of algorithm names to include in the plot.
        seed_list (List[int]): List of seed numbers to aggregate results.
        upper_reward (int): Maximum reward value for clipping the shaded region.
        lower_reward (int): Minimum reward value for clipping the shaded region.
        smooth_factor (float, optional): Smoothing factor for exponential moving average. Defaults to 0.99.
        save_path (Optional[str], optional): File path to save the figure. If None, figure is not saved.
        show (bool, optional): Whether to display the figure interactively. Defaults to False.
        posterior_samples (Optional[int], optional): Number of posterior samples for OPS-VBQN (used for labeling).
        bootstrap_heads (Optional[int], optional): Number of bootstrap heads for BootstrapDQN (used for labeling).

    Returns:
        plt.Figure: Matplotlib Figure object containing the plotted learning curves.

    Notes:
        - Automatically formats the x-axis using scientific notation (10^n) for publication.
        - Curves are smoothed using an exponential moving average.
        - If no data is available for a particular algorithm or seed, it is skipped with a printed message.
        - Uses color-coding and transparency for clarity in multi-algorithm plots.
    """
    fig, ax = plt.subplots(figsize=(8, 5))  # Paper-friendly size

    colors = ['#1b9e77', '#d95f02', '#7570b3'] #["#0072B2", "#E69F00", "#009E73"] for color blind-friendly?
    color_map = {alg: colors[i % len(colors)] for i, alg in enumerate(alg_names)}

    for alg in alg_names:
        results = []

        for seed in seed_list:
            file_path = get_returns_path(env_name, alg, seed, posterior_samples, bootstrap_heads)
            if os.path.exists(file_path):
                arr = np.load(file_path)
                results.append(arr)

        if not results:
            print(f"No data for {alg} in {env_name}. Skipping.")
            continue

        max_len = min(last_nonzero_index(arr) for arr in results)
        trimmed = np.vstack([arr[:max_len] for arr in results])
        mean = np.mean(trimmed, axis=0)
        std = np.std(trimmed, axis=0)

        smoothed_mean = exponential_moving_average(mean, alpha=smooth_factor)
        smoothed_std = exponential_moving_average(std, alpha=smooth_factor)

        x = np.arange(max_len)
        y_upper = np.clip(smoothed_mean + smoothed_std, None, upper_reward)
        y_lower = np.clip(smoothed_mean - smoothed_std, lower_reward, None)

        if alg == "OPS-VBQN" and posterior_samples is not None:
            label_name = f"{alg}, N={posterior_samples}"
        elif alg == "BootstrapDQN" and bootstrap_heads is not None:
            label_name = f"Bootstrapped DQN, K={bootstrap_heads}"
        else:
            label_name = alg

        if alg == "OPS-VBQN":
            ax.plot(x, smoothed_mean, label=label_name, color=color_map[alg], linewidth=1.4)
        else:
            ax.plot(x, smoothed_mean, label=label_name, color=color_map[alg], linewidth=0.8)

        ax.fill_between(x, y_lower, y_upper, color=color_map[alg], alpha=0.15)

    # Axis labels
    ax.set_xlabel("Steps", fontsize=14)
    ax.set_ylabel("Episodic Return", fontsize=14)

    # Automatic scientific notation in 10^n style
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    

    # Grid and spines
    ax.yaxis.grid(True, linestyle='--', linewidth=0.8, alpha=0.7)
    ax.xaxis.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    ax.legend(fontsize=10, loc="best", frameon=False)

    # Tick font size
    ax.tick_params(labelsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig



def generate_latex_table(
    seeds: List[int],
    use_latex: Optional[bool] = True,
    ) -> str:
    """
    Generate a LaTeX-formatted table summarizing benchmark results across 
    multiple environments and algorithms.

    The table includes three metrics:
        1. Benchmark score
        2. Cumulative regret (scaled by 10^3)
        3. Episodes to solve

    Data is aggregated across the specified seeds. For each metric, the mean 
    and standard deviation are calculated and formatted in LaTeX-friendly 
    notation (mean ± std). Missing data for any seed results in "N/A" 
    for that environment/algorithm combination.

    Args:
        seeds (list of int): List of seed numbers to include in the table.
        use_latex (bool, optional): If True, format values for LaTeX with 
            math mode and \pm; otherwise, use plain text. Defaults to True.

    Returns:
        str: A string containing the full LaTeX tabular environment with 
            the aggregated results.

    Notes:
        - Algorithms included are DQN, BootstrapDQN (with all K values found),
          and OPS-VBQN (with all N values found).
        - Table sections are grouped by metric: Benchmark, Cumulative Regret, 
          and Episodes to Solve.
        - Cumulative regret values are scaled by 10^3 for readability.
        - Uses `\toprule`, `\midrule`, and `\bottomrule` for professional 
          table formatting.
    """

    all_data = load_all_benchmarks()
    envs = sorted(all_data.keys())

    # Collect algorithms
    algs = []
    if "DQN" in {alg for env in all_data.values() for alg in env}:
        algs.append(("DQN", None))
    algs += [("BootstrapDQN", k) for k in sorted({
        int(k) for env in all_data.values()
        for alg, runs in env.items() if alg == "BootstrapDQN"
        for k in runs.keys()
    })]
    algs += [("OPS-VBQN", n) for n in sorted({
        int(n) for env in all_data.values()
        for alg, runs in env.items() if alg == "OPS-VBQN"
        for n in runs.keys()
    })]

    def fmt(vals, metric):
        if not vals:
            return "N/A"
        mean, std = np.mean(vals), np.std(vals)
        # Only scale cumulative regret
        if metric == "cumulative_regret":
            mean *= 1e-3
            std *= 1e-3
        if use_latex:
            return f"${mean:.2f} \\!\\pm\\! {{\\scriptstyle {std:.2f}}}$"
        return f"{mean:.2f} ± {std:.2f}"

    def extract(env, alg, param, metric):
        """Extract metric values for all seeds, only if complete."""
        if alg == "DQN":
            seed_scores = all_data[env][alg]
        else:
            seed_scores = all_data[env][alg].get(str(param), {})
        values = []
        for seed in seeds:
            key = f"seed{seed}"
            if key not in seed_scores or metric not in seed_scores[key]:
                return None
            values.append(seed_scores[key][metric])
        return values

    # Table sections
    lines = []
    header = " & " + " & ".join([f"\\textbf{{{env}}}" for env in envs]) + " \\\\"
    lines.append("\\begin{tabular}{l" + "c" * len(envs) + "}")
    lines.append("\\toprule")
    lines.append(header)

    # Benchmark section
    lines.append("\\multicolumn{" + str(len(envs)+1) + "}{c}{\\textbf{Benchmark Scores ($\\downarrow$)}} \\\\")
    lines.append("\\midrule")
    for alg, param in algs:
        row_label = "DQN" if alg == "DQN" else (
            f"Bootstrapped DQN, K={param}" if alg == "BootstrapDQN" else f"OPS-VBQN, N={param}"
        )
        row = [row_label]
        for env in envs:
            values = extract(env, alg, param, "benchmark")
            row.append(fmt(values, "benchmark") if values else "N/A")
        lines.append(" & ".join(row) + " \\\\")

    # Regret section (scaled)
    lines.append("\\midrule")
    lines.append("\\multicolumn{" + str(len(envs)+1) + "}{c}{\\textbf{Cumulative Regret ($\\times 10^3$)}} \\\\")
    lines.append("\\midrule")
    for alg, param in algs:
        row_label = "DQN" if alg == "DQN" else (
            f"Bootstrapped DQN, K={param}" if alg == "BootstrapDQN" else f"OPS-VBQN, N={param}"
        )
        row = [row_label]
        for env in envs:
            values = extract(env, alg, param, "cumulative_regret")
            row.append(fmt(values, "cumulative_regret") if values else "N/A")
        lines.append(" & ".join(row) + " \\\\")

    # Episodes section
    lines.append("\\midrule")
    lines.append("\\multicolumn{" + str(len(envs)+1) + "}{c}{\\textbf{Episodes to Solve ($\\downarrow$)}} \\\\")
    lines.append("\\midrule")
    for alg, param in algs:
        row_label = "DQN" if alg == "DQN" else (
            f"Bootstrapped DQN, K={param}" if alg == "BootstrapDQN" else f"OPS-VBQN, N={param}"
        )
        row = [row_label]
        for env in envs:
            values = extract(env, alg, param, "episodes_to_solve")
            row.append(fmt(values, "episodes_to_solve") if values else "N/A")
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    return "\n".join(lines)



def save_latex_table_pdf(seeds: List[int], save_path: str, show: bool = False) -> None:
    """
    Generate a LaTeX table of RL benchmark results and save it as a compiled PDF.

    This function uses `generate_latex_table` to create a LaTeX tabular 
    environment for the specified seeds, compiles it with `pdflatex`, and 
    saves the resulting PDF to the given path. Optionally, the PDF can be 
    opened automatically after generation.

    Args:
        seeds (List[int]): List of seed numbers to include in the table.
        save_path (str): Path where the compiled PDF should be saved.
        show (bool, optional): If True, automatically open the PDF after 
            generation. Defaults to False.

    Raises:
        RuntimeError: If `pdflatex` fails to compile the LaTeX source.

    Notes:
        - Requires a working LaTeX installation with `pdflatex` available in 
          the system PATH.
        - The generated LaTeX document uses the `article` class and includes 
          packages `booktabs`, `amsmath`, and `amssymb`.
        - The table is formatted with professional spacing and booktabs rules.
    """
    latex_table = generate_latex_table(seeds, use_latex=True)

    tex_template = r"""
    \documentclass{article}
    \usepackage{booktabs}
    \usepackage{amsmath}
    \usepackage{amssymb}
    \usepackage[margin=1in]{geometry}
    \begin{document}
    \section*{RL Benchmark Results}
    %s
    \end{document}
    """ % latex_table

    if save_path != None:
        save_path = Path(save_path).resolve()
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_file = Path(tmpdir) / "table.tex"
            pdf_file = tex_file.with_suffix(".pdf")

            # Write LaTeX source
            tex_file.write_text(tex_template)

            # Run pdflatex
            try:
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", str(tex_file)],
                    cwd=tmpdir,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError:
                raise RuntimeError("pdflatex failed. Please check your LaTeX installation.")

            # Copy compiled PDF to target
            save_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_file.replace(save_path)

        print(f"PDF successfully generated at {save_path}")
    # Open PDF if requested
    if show and save_path != None:
        if sys.platform.startswith("darwin"):  # macOS
            subprocess.run(["open", str(save_path)])
        elif sys.platform.startswith("win"):  # Windows
            subprocess.run(["start", str(save_path)], shell=True)
        else:  # Linux / Unix
            subprocess.run(["xdg-open", str(save_path)])