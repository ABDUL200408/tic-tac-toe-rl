"""
experiments.py

Facilitates structured experiments (hyperparameter sweeps, ablation studies)
for Q-Learning in Tic-Tac-Toe (n x n board).
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple

import random
import numpy as np

from agent import QLearningAgent
from environment import TicTacToeEnv
from evaluator import Evaluator


# ---------------------------------------------------------------------------
# Helper: opponent policy (mirrors behaviour in Evaluator)
# ---------------------------------------------------------------------------
def _select_opponent_action(env: TicTacToeEnv, policy: str = "random"):
    """
    Choose an action for the opponent based on the specified policy.

    Parameters
    ----------
    env : TicTacToeEnv
        Current environment instance (board + action space).
    policy : str, optional
        "random"   : uniform random choice among legal moves.
        "heuristic": simple rule-based policy, e.g. prefer centre if available.

    Returns
    -------
    (row, col) action, or None if no legal moves exist.
    """
    available = env.get_action_space()
    if not available:
        return None

    if policy == "heuristic":
        n = env.n
        centre = (n // 2, n // 2)
        if centre in available:
            return centre

    return random.choice(available)


# ---------------------------------------------------------------------------
# Helper: training loop for one agent under given config
# ---------------------------------------------------------------------------
def _train_agent(
    config: Dict[str, Any],
    learning_rate: float,
    epsilon_decay: float,
) -> Tuple[QLearningAgent, TicTacToeEnv]:
    """
    Train a Q-learning agent under a given (learning_rate, epsilon_decay)
    configuration, using the opponent policy specified in `config`.

    Returns the trained agent and its environment.
    """
    board_size = config["board_size"]
    win_condition = config["win_condition"]
    episodes = config["episodes"]

    # Reward configuration (passed into the environment)
    reward_win = config.get("reward_win", 1.0)
    reward_loss = config.get("reward_loss", -1.0)
    reward_draw = config.get("reward_draw", 0.0)

    # Define full action space for the n x n board
    actions = [(r, c) for r in range(board_size) for c in range(board_size)]

    # Environment and agent
    env = TicTacToeEnv(
        board_size=board_size,
        win_condition=win_condition,
        reward_win=reward_win,
        reward_loss=reward_loss,
        reward_draw=reward_draw,
    )

    agent = QLearningAgent(
        actions=actions,
        learning_rate=learning_rate,
        discount_factor=config["discount_factor"],
        initial_epsilon=config["epsilon"],
        epsilon_decay=epsilon_decay,
        epsilon_min=config["epsilon_min"],
        board_size=board_size,
    )

    opponent_policy = config.get("opponent_policy", "random")

    # Main training loop
    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            # Environment tracks current_player internally:
            #   1  -> agent's turn
            #  -1  -> opponent's turn
            if env.current_player == 1:
                # Agent's move
                action = agent.choose_action(state)
                if action is None:
                    # No legal move (should only happen on a full board)
                    break
                next_state, reward, done, _ = env.step(action)
                # Update Q from the agent's perspective
                agent.learn(state, action, reward, next_state, done)
                state = next_state
            else:
                # Opponent's move (no learning, just environment transition)
                opp_action = _select_opponent_action(env, opponent_policy)
                if opp_action is None:
                    break
                next_state, reward, done, _ = env.step(opp_action)
                state = next_state  # agent just observes the new state

    return agent, env


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------
def run_experiments(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs key experiments to analyse RL agent performance under varied
    hyperparameters.

    Experiment 1: Learning rate sweep (α).
    Experiment 2: Epsilon decay sweep (ε schedule).

    For each setting:
        - Train a fresh agent from scratch.
        - Evaluate using the Evaluator (wins / draws / losses / avg reward)
          against the evaluation opponent specified in the config.

    Parameters
    ----------
    config : dict
        Baseline hyperparameter dictionary (see config.get_config()).

    Returns
    -------
    results : dict
        {
            'learning_rate_sweep': [
                {
                    'learning_rate': α,
                    'wins': ...,
                    'draws': ...,
                    'losses': ...,
                    'avg_reward': ...
                },
                ...
            ],
            'epsilon_decay_sweep': [
                {
                    'epsilon_decay': d,
                    'wins': ...,
                    'draws': ...,
                    'losses': ...,
                    'avg_reward': ...
                },
                ...
            ]
        }
    """
    # Set seeds for reproducibility
    seed = config.get("random_seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    eval_games = config.get("eval_episodes", 200)
    eval_opponent = config.get("evaluation_opponent", "random")

    results: Dict[str, Any] = {
        "learning_rate_sweep": [],
        "epsilon_decay_sweep": [],
    }

    # ----------------------------------------------------------------------
    # Experiment 1: Learning Rate Sweep
    # ----------------------------------------------------------------------
    learning_rates: List[float] = [0.05, 0.1, 0.3, 0.5]
    for lr in learning_rates:
        agent, env = _train_agent(
            config=config,
            learning_rate=lr,
            epsilon_decay=config["epsilon_decay"],
        )
        evaluator = Evaluator(env, agent)

        wins, draws, losses, avg_reward = evaluator.evaluate(
            n_games=eval_games,
            opponent_policy=eval_opponent,
            greedy=True,  # evaluate learned policy without exploration
        )

        results["learning_rate_sweep"].append(
            {
                "learning_rate": lr,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "avg_reward": avg_reward,
            }
        )

    # ----------------------------------------------------------------------
    # Experiment 2: Epsilon Decay Sweep
    # ----------------------------------------------------------------------
    epsilon_decays: List[float] = [0.99, 0.995, 0.9995]
    for decay in epsilon_decays:
        agent, env = _train_agent(
            config=config,
            learning_rate=config["learning_rate"],
            epsilon_decay=decay,
        )
        evaluator = Evaluator(env, agent)

        wins, draws, losses, avg_reward = evaluator.evaluate(
            n_games=eval_games,
            opponent_policy=eval_opponent,
            greedy=True,
        )

        results["epsilon_decay_sweep"].append(
            {
                "epsilon_decay": decay,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "avg_reward": avg_reward,
            }
        )

    # Extendable:
    # - Different opponents
    # - Discount factor sweep
    # - Ablation: no exploration, different reward shaping, etc.

    return results
