"""
main.py

Entry point for training + evaluating a Q-Learning Tic-Tac-Toe agent
and running structured experiments.
"""

from __future__ import annotations

import random
import numpy as np
import pandas as pd

from environment import TicTacToeEnv
from agent import QLearningAgent
from evaluator import Evaluator
from config import get_config
from experiments import run_experiments
from visualisations import (
    plot_learning_curve,
    plot_experiment_results,
    plot_board_template,
)
from utils import set_random_seed


# ---------------------------------------------------------------------------
# Opponent action helper (same logic as evaluator/experiments)
# ---------------------------------------------------------------------------
def choose_opponent_action(env: TicTacToeEnv, policy: str = "heuristic"):
    """Select an opponent action based on desired policy."""
    available = env.get_action_space()
    if not available:
        return None

    if policy == "heuristic":
        centre = (env.n // 2, env.n // 2)
        if centre in available:
            return centre

    return random.choice(available)


# ---------------------------------------------------------------------------
# Demonstration: show a single game in the terminal after training
# ---------------------------------------------------------------------------
def demo_agent_vs_opponent(
    env: TicTacToeEnv,
    agent: QLearningAgent,
    opponent_policy: str = "random",
) -> None:
    """
    Play one full game in the terminal: trained agent (X, player 1)
    vs opponent (O, player -1).

    - Uses epsilon=0.0 (purely greedy) during the demo.
    - Works for any board size configured in config.py.
    """
    print("\n===== Demo Game: Agent (X) vs Opponent (O) =====")
    print(f"Board size: {env.n}x{env.n}, win condition: {env.k} in a row")
    print(f"Opponent policy: {opponent_policy}\n")

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # greedy behaviour for demonstration

    try:
        state = env.reset()
        done = False
        player = 1  # agent starts as player 1 (X)
        move_no = 1

        print("Initial board:")
        env.render()
        print("-" * (2 * env.n - 1))

        while not done:
            if player == 1:
                action = agent.choose_action(state)
                if action is None:
                    print("\n[Demo] No legal move for agent (board full).")
                    break
                print(f"\nMove {move_no}: Agent (X) plays {action}")
            else:
                action = choose_opponent_action(env, policy=opponent_policy)
                if action is None:
                    print("\n[Demo] No legal move for opponent (board full).")
                    break
                print(f"\nMove {move_no}: Opponent (O) plays {action}")

            state, reward, done, info = env.step(action)
            env.render()
            print("-" * (2 * env.n - 1))

            move_no += 1
            player *= -1

        winner = info.get("winner")
        print("\n===== Demo Game Over =====")
        if winner == 1:
            print("Result: Agent (X) wins.")
        elif winner == -1:
            print("Result: Opponent (O) wins.")
        else:
            print("Result: Draw.")
        print("============================================\n")

    finally:
        # Restore original exploration rate
        agent.epsilon = original_epsilon


# ---------------------------------------------------------------------------
# Main training + evaluation pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    # Load configuration
    config = get_config()

    episodes = config["episodes"]
    show_progress = config.get("show_progress", True)

    # Random seeds for reproducibility
    seed = config.get("random_seed", 42)
    set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Unpack environment settings + reward structure
    board_size = config["board_size"]
    win_condition = config["win_condition"]

    reward_win = config.get("reward_win", 1.0)
    reward_loss = config.get("reward_loss", -1.0)
    reward_draw = config.get("reward_draw", 0.0)

    # Define action space for n x n board
    all_actions = [(r, c) for r in range(board_size) for c in range(board_size)]

    # Create environment and agent (with reward parameters)
    env = TicTacToeEnv(
        board_size=board_size,
        win_condition=win_condition,
        reward_win=reward_win,
        reward_loss=reward_loss,
        reward_draw=reward_draw,
    )

    agent = QLearningAgent(
        actions=all_actions,
        learning_rate=config["learning_rate"],
        discount_factor=config["discount_factor"],
        initial_epsilon=config["epsilon"],
        epsilon_decay=config["epsilon_decay"],
        epsilon_min=config["epsilon_min"],
        board_size=board_size,
    )

    evaluator = Evaluator(env, agent)

    opponent_policy = config.get("opponent_policy", "heuristic")
    eval_opponent = config.get("evaluation_opponent", "random")
    eval_interval = config.get("eval_interval", 500)
    eval_games = config.get("eval_episodes", 200)   # evaluation batch size
    demo_after_training = config.get("demo_after_training", True)

    # --- Save a static board image matching config board_size ---
    # This will create results/board_size.png (3x3, 4x4, 5x5, ...)
    plot_board_template(board_size, config)

    # Stats for plotting learning curve
    stats = {
        "wins": [],
        "draws": [],
        "losses": [],
        "avg_reward": [],
        "epsilons": [],
        "episodes": [],
    }

    # ----------------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------------
    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            # Agent's turn
            if env.current_player == 1:
                action = agent.choose_action(state)
                if action is None:
                    break

                next_state, reward, done, _ = env.step(action)
                agent.learn(state, action, reward, next_state, done)
                state = next_state

            else:
                # Opponent turn
                opp_action = choose_opponent_action(env, opponent_policy)
                if opp_action is None:
                    break
                next_state, _, done, _ = env.step(opp_action)
                state = next_state

        # ----------------------------------------------------------------------
        # Periodic evaluation
        # ----------------------------------------------------------------------
        if (episode + 1) % eval_interval == 0 or episode == episodes - 1:
            wins, draws, losses, avg_reward = evaluator.evaluate(
                n_games=eval_games,
                opponent_policy=eval_opponent,
                greedy=True
            )

            stats["wins"].append(wins)
            stats["draws"].append(draws)
            stats["losses"].append(losses)
            stats["avg_reward"].append(avg_reward)
            stats["epsilons"].append(agent.epsilon)
            stats["episodes"].append(episode + 1)

            if show_progress:
                print(
                    f"[Episode {episode + 1}] "
                    f"Wins={wins}, Draws={draws}, Losses={losses}, "
                    f"AvgReward={avg_reward:.3f}, Epsilon={agent.epsilon:.3f}"
                )

    # ----------------------------------------------------------------------
    # Save results + Q-table + learning curve
    # ----------------------------------------------------------------------
    agent.save(config["q_table_save_path"])
    plot_learning_curve(stats, config)
    print("Learning curve saved to:", config.get("plots_dir", "results/"))

    # Optional human vs AI demo (interactive)
    if config.get("play_interactive", False):
        evaluator.human_vs_agent()

    # Automatic demo game (printed board, no user input)
    if demo_after_training:
        demo_agent_vs_opponent(env, agent, opponent_policy=eval_opponent)

    # ----------------------------------------------------------------------
    # Structured hyperparameter experiments
    # ----------------------------------------------------------------------
    experiment_results = run_experiments(config)

    plots_dir = config.get("plots_dir", "results/")
    for name, data in experiment_results.items():
        print(f"\n===== Experiment results: {name} =====")
        df = pd.DataFrame(data)

        # Use to_string instead of to_markdown to avoid 'tabulate' dependency
        print(df.to_string(index=False))

        plot_experiment_results(data, name, config)

        df.to_csv(f"{plots_dir.rstrip('/')}/{name}.csv", index=False)


if __name__ == "__main__":
    main()
