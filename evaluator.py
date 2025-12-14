"""
evaluator.py

Provides evaluation and analysis of RL agent performance in Tic-Tac-Toe.


"""

from __future__ import annotations
from typing import Tuple, Literal, Optional, Dict, Any
import random


class Evaluator:
    """
    Evaluation helper for a Q-learning Tic-Tac-Toe agent.

    Parameters
    ----------
    env : TicTacToeEnv
        Environment instance.
    agent : QLearningAgent
        Trained or partially trained agent.
    """

    def __init__(self, env, agent) -> None:
        self.env = env
        self.agent = agent

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        n_games: int = 100,
        opponent_policy: Literal["random", "heuristic"] = "random",
        greedy: bool = True,
    ) -> Tuple[int, int, int, float]:
        """
        Evaluate the agent over multiple games.

        Parameters
        ----------
        n_games : int
            Number of evaluation games.
        opponent_policy : {"random", "heuristic"}
            Opponent strategy.
        greedy : bool
            If True, sets epsilon = 0 during evaluation.

        Returns
        -------
        wins : int
        draws : int
        losses : int
        avg_reward : float
            Average agent reward across games.
        """
        wins, draws, losses = 0, 0, 0
        total_reward = 0.0

        # Backup epsilon for restoration later
        original_epsilon = self.agent.epsilon
        if greedy:
            self.agent.epsilon = 0.0

        try:
            for _ in range(n_games):
                state = self.env.reset()
                done = False
                player = 1  # agent always considered player 1
                cumulative_game_reward = 0.0

                while not done:
                    if player == 1:
                        # Agent move
                        action = self.agent.choose_action(state, evaluation=True)
                        if action is None:
                            break
                    else:
                        # Opponent move
                        action = self._select_opponent_action(opponent_policy)

                    next_state, reward, done, info = self.env.step(action)

                    # Only count reward from agent perspective
                    if player == 1:
                        cumulative_game_reward += reward

                    state = next_state
                    player *= -1

                # Final result
                winner = info.get("winner")
                if winner == 1:
                    wins += 1
                elif winner == -1:
                    losses += 1
                else:
                    draws += 1

                total_reward += cumulative_game_reward

        finally:
            # Restore epsilon after evaluation
            self.agent.epsilon = original_epsilon

        avg_reward = total_reward / n_games
        return wins, draws, losses, avg_reward

    # ------------------------------------------------------------------
    # Interactive demonstration (human vs agent)
    # ------------------------------------------------------------------
    def human_vs_agent(self) -> None:
        """
        Play a human vs agent game in the console.
        Human = O (-1), Agent = X (+1).
        Agent is forced into greedy mode (epsilon = 0).
        """
        print("Play Tic-Tac-Toe against RL Agent (X). You are O (-1).")
        print("Enter moves as: row,col   (e.g. 0,0)")

        state = self.env.reset()
        done = False
        player = 1

        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0

        try:
            while not done:
                self.env.render()

                if player == 1:
                    action = self.agent.choose_action(state, evaluation=True)
                    print(f"\nAgent (X) chooses: {action}")
                else:
                    action = self._read_human_move()

                state, reward, done, info = self.env.step(action)
                player *= -1

            self.env.render()
            winner = info.get("winner")
            print("\nGame Over!")
            if winner == 1:
                print("Agent (X) wins!")
            elif winner == -1:
                print("You (O) win!")
            else:
                print("It's a draw!")

        finally:
            self.agent.epsilon = original_epsilon

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _select_opponent_action(
        self,
        opponent_policy: Literal["random", "heuristic"]
    ):
        """Select opponent action according to chosen policy."""
        available = self.env.get_action_space()
        if not available:
            return None

        if opponent_policy == "heuristic":
            # Simple stronger baseline: pick centre if available
            centre = (self.env.n // 2, self.env.n // 2)
            if centre in available:
                return centre

        return random.choice(available)

    def _read_human_move(self):
        """Read human input and validate."""
        while True:
            try:
                move = input("Your move (row,col): ")
                r, c = map(int, move.split(","))
                if (r, c) in self.env.get_action_space():
                    return (r, c)
                else:
                    print("Illegal move. Try again.")
            except Exception:
                print("Invalid input. Use: row,col")
