import numpy as np
from utils import last_nonzero_index, exponential_moving_average
from data_handler import load_all_benchmarks, get_returns_path
from configs import ENV_CONFIG
from typing import List, Optional
from collections import defaultdict
import pandas as pd
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
    Plot boxplots of cumulative regret and episodes to solve for a given environment using BQMS.
    Optionally formats the y-axis with scientific notation for paper-friendly presentation.
    """
    all_data = load_all_benchmarks()

    if env_name not in all_data or "BQMS" not in all_data[env_name]:
        raise ValueError(f"No BQMS data found for environment '{env_name}'.")

    bqms_data = all_data[env_name]["BQMS"]
    grouped_data = defaultdict(lambda: {'cumulative_regret': [], 'episodes_to_solve': []})

    for key, seed_data in bqms_data.items():
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
    Plots smoothed learning curves (mean ± std) for multiple algorithms in a single environment.
    Publication-ready version with automatic 10^n axis formatting.
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

        if alg == "BQMS" and posterior_samples is not None:
            label_name = f"{alg}, N={posterior_samples}"
        elif alg == "BootstrapDQN" and bootstrap_heads is not None:
            label_name = f"Bootstrapped DQN, K={bootstrap_heads}"
        else:
            label_name = alg

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
    seeds,
    use_latex=True,
):
    """
    Generate a single LaTeX table with all 3 metrics
    (Benchmark, Cumulative Regret, Episodes to Solve)
    across all 4 environments and algorithms.

    Cumulative regret will be scaled by 10^3.
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
    algs += [("BQMS", n) for n in sorted({
        int(n) for env in all_data.values()
        for alg, runs in env.items() if alg == "BQMS"
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
            f"Bootstrapped DQN, K={param}" if alg == "BootstrapDQN" else f"BQMS, N={param}"
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
            f"Bootstrapped DQN, K={param}" if alg == "BootstrapDQN" else f"BQMS, N={param}"
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
            f"Bootstrapped DQN, K={param}" if alg == "BootstrapDQN" else f"BQMS, N={param}"
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
    Save a LaTeX table into a compiled PDF using pdflatex.

    Args:
        latex_table (str): LaTeX tabular environment string (e.g., from generate_latex_table).
        save_path (str): Path to save the compiled PDF.
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

    print(f"✅ PDF successfully generated at {save_path}")
    # Open PDF if requested
    if show:
        if sys.platform.startswith("darwin"):  # macOS
            subprocess.run(["open", str(save_path)])
        elif sys.platform.startswith("win"):  # Windows
            subprocess.run(["start", str(save_path)], shell=True)
        else:  # Linux / Unix
            subprocess.run(["xdg-open", str(save_path)])