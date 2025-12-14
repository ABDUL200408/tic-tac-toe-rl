"""
visualisations.py

Generates learning curves, experiment plots, and (optionally) Q-table
visualisations for Tic-Tac-Toe RL.

"""

from __future__ import annotations

from typing import Dict, Any, List

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import safe_mkdir


# ----------------------------------------------------------------------
# Learning curve
# ----------------------------------------------------------------------
def plot_learning_curve(stats: Dict[str, List[Any]], config: Dict[str, Any]) -> None:
    """
    Plot agent learning over time: win/draw/loss counts, epsilon, and (optionally)
    average reward per evaluation batch.

    Parameters
    ----------
    stats : dict
        Should contain lists (all same length):
            - 'wins'
            - 'draws'
            - 'losses'
            - 'epsilons'
            - 'episodes'        (episode numbers at which evaluation occurred)
            - 'avg_reward'      (optional: average reward per evaluation window)
    config : dict
        Experiment/config dictionary (from config.get_config()).
    """
    wins = stats.get("wins", [])
    draws = stats.get("draws", [])
    losses = stats.get("losses", [])
    epsilons = stats.get("epsilons", [])
    episodes = stats.get("episodes", [])
    avg_rewards = stats.get("avg_reward", [])

    if not episodes:
        # Fallback: infer episodes using eval_interval if not explicitly provided
        eval_interval = config.get("eval_interval", 500)
        episodes = np.arange(1, len(wins) + 1) * eval_interval

    plt.figure(figsize=(10, 6))

    # Left y-axis: win/draw/loss counts
    ax1 = plt.gca()
    ax1.plot(episodes, wins, label="Wins", marker="o")
    ax1.plot(episodes, draws, label="Draws", marker="o")
    ax1.plot(episodes, losses, label="Losses", marker="o")
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Games out of evaluation batch")
    ax1.grid(True, alpha=0.3)

    # Right y-axis: epsilon and (optionally) average reward
    ax2 = ax1.twinx()
    ax2.plot(episodes, epsilons, "k--", alpha=0.6, label="Epsilon")
    if avg_rewards:
        ax2.plot(episodes, avg_rewards, alpha=0.8, label="Avg. Reward")
    ax2.set_ylabel("Epsilon / Average Reward")

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.title(config.get("plot_title", "Tic-Tac-Toe RL: Performance Over Time"))

    # Save plot if requested
    if config.get("save_plots", False):
        plots_dir = config.get("plots_dir", "results/")
        safe_mkdir(plots_dir)
        dpi = config.get("dpi", 150)
        filepath = os.path.join(plots_dir, "learning_curve.png")
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Hyperparameter sweep plots
# ----------------------------------------------------------------------
def plot_experiment_results(
    results: List[Dict[str, Any]],
    experiment_name: str,
    config: Dict[str, Any],
) -> None:
    """
    Plot results from hyperparameter sweeps (e.g. learning rate, epsilon decay).

    Parameters
    ----------
    results : list of dict
        Each dict should contain:
            - One hyperparameter key (e.g. 'learning_rate' or 'epsilon_decay')
            - 'wins', 'draws', 'losses'
            - 'avg_reward' (optional)
    experiment_name : str
        Name of the experiment (e.g. 'learning_rate_sweep', 'epsilon_decay_sweep').
    config : dict
        Global configuration (for saving plots etc.).
    """
    if not results:
        print(f"[WARN] No results to plot for experiment '{experiment_name}'.")
        return

    # Infer which hyperparameter was swept:
    # first key that is not a result metric
    metric_candidates = [
        k
        for k in results[0].keys()
        if k not in ("wins", "draws", "losses", "avg_reward")
    ]
    if not metric_candidates:
        print(f"[WARN] Could not infer hyperparameter for experiment '{experiment_name}'.")
        return
    metric = metric_candidates[0]

    x = [res[metric] for res in results]
    wins = [res["wins"] for res in results]
    draws = [res["draws"] for res in results]
    losses = [res["losses"] for res in results]
    avg_rewards = [res.get("avg_reward") for res in results]

    plt.figure(figsize=(8, 5))
    ax1 = plt.gca()

    # Left y-axis: counts
    ax1.plot(x, wins, label="Wins", marker="o")
    ax1.plot(x, draws, label="Draws", marker="o")
    ax1.plot(x, losses, label="Losses", marker="o")
    ax1.set_ylabel("Games out of evaluation batch")
    ax1.grid(True, alpha=0.3)

    x_label = metric.replace("_", " ").title()
    ax1.set_xlabel(x_label)
    plt.title(f"Effect of {x_label} on RL Agent Performance")

    # Right y-axis: average reward if available
    if any(v is not None for v in avg_rewards):
        ax2 = ax1.twinx()
        ax2.plot(x, avg_rewards, color="tab:green", marker="s", label="Avg. Reward")
        ax2.set_ylabel("Average Reward")

        # Merge legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
    else:
        ax1.legend(loc="best")

    # Save plot if requested
    if config.get("save_plots", False):
        plots_dir = config.get("plots_dir", "results/")
        safe_mkdir(plots_dir)
        dpi = config.get("dpi", 150)
        fname = f"{experiment_name}.png"
        filepath = os.path.join(plots_dir, fname)
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Board visualisation: board_size.png
# ----------------------------------------------------------------------
def plot_board_template(board_size: int, config: Dict[str, Any]) -> None:
    """
    Draw an empty Tic-Tac-Toe board with the configured size and save it
    as 'board_size.png' in the results directory.

    - Uses coordinates 0..(n-1) on both axes.
    - Works for any n (3x3, 4x4, 5x5, ...).
    """
    plots_dir = config.get("plots_dir", "results/")
    safe_mkdir(plots_dir)
    dpi = config.get("dpi", 150)

    fig, ax = plt.subplots(figsize=(5, 5))

    # Draw grid lines
    for i in range(board_size + 1):
        ax.plot([0, board_size], [i, i], color="black", linewidth=1)
        ax.plot([i, i], [0, board_size], color="black", linewidth=1)

    # Set limits and aspect
    ax.set_xlim(0, board_size)
    ax.set_ylim(0, board_size)
    ax.set_aspect("equal")

    # Tick positions at cell centres
    ax.set_xticks(np.arange(0.5, board_size, 1.0))
    ax.set_yticks(np.arange(0.5, board_size, 1.0))
    ax.set_xticklabels(range(board_size))
    ax.set_yticklabels(range(board_size))

    # Origin at top-left like the terminal render
    ax.invert_yaxis()

    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")
    ax.set_title(f"Tic-Tac-Toe Board ({board_size} x {board_size})")

    filepath = os.path.join(plots_dir, "board_size.png")
    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Q-table visualisation (optional / illustrative)
# ----------------------------------------------------------------------
def visualise_q_table(agent, filename: str = "q_table_heatmap.png") -> None:
    """
    Visualise the Q-table as a heatmap for qualitative analysis.

    WARNING:
        - For a 5x5 or 7x7 Tic-Tac-Toe with many states, this can become large.
        - Intended mainly for small subsets or illustrative screenshots
          in the report, not as a routine diagnostic.

    Parameters
    ----------
    agent :
        Trained QLearningAgent instance (must have 'q_table' attribute).
    filename : str, optional
        Path where the heatmap PNG will be saved.
    """
    import pandas as pd  # Lazy import

    if not agent.q_table:
        print("[WARN] Q-table is empty; nothing to visualise.")
        return

    # Convert Q-table into a tabular DataFrame
    keys = list(agent.q_table.keys())
    q_values = list(agent.q_table.values())

    data = []
    for (state, action), q in zip(keys, q_values):
        flat_state = "".join(str(x) for x in state)
        data.append(
            {
                "state": flat_state,
                "action": f"{action}",
                "Q": q,
            }
        )

    df = pd.DataFrame(data)

    # Pivot: states as rows, actions as columns
    heatmap_data = df.pivot_table(
        index="state", columns="action", values="Q", fill_value=0.0
    )

    # Limit size for readability
    max_rows = 50
    if heatmap_data.shape[0] > max_rows:
        heatmap_data = heatmap_data.head(max_rows)

    plt.figure(figsize=(12, max(4, heatmap_data.shape[0] // 4)))
    sns.heatmap(heatmap_data, cmap="viridis", annot=False)
    plt.title("Learned Q-values for State–Action Pairs")
    plt.xlabel("Action (row, col)")
    plt.ylabel("Encoded State")
    plt.tight_layout()

    # Ensure directory exists
    out_dir = os.path.dirname(filename)
    if out_dir:
        safe_mkdir(out_dir)

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
