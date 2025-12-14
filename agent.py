# agent.py
# --------------------------------------------------------------
# Q-learning agent for Tic-Tac-Toe (n x n)
# Improvements added:
# - epsilon freeze/unfreeze for evaluation
# - action legality validation
# - optional state initialization for faster learning
# - safer Q-table key handling
# --------------------------------------------------------------

from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import random

State = Tuple[int, ...]      # flat board
Action = Tuple[int, int]     # (row, col)
QKey = Tuple[State, Action]


class QLearningAgent:
    """
    Enhanced Q-learning agent for Tic-Tac-Toe.

    Supports:
        - epsilon-greedy exploration
        - Q-table persistence
        - evaluation mode (epsilon suppressed)
        - works with arbitrary n x n boards
    """

    def __init__(
        self,
        actions: Iterable[Action],
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.05,
        board_size: int = 5,
    ) -> None:

        self.actions: List[Action] = list(actions)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table: Dict[QKey, float] = {}

        self.board_size = board_size

        # Store a reference epsilon for evaluation
        self._epsilon_backup = initial_epsilon

    # --------------------------------------------------------------
    # Policy Functions
    # --------------------------------------------------------------
    def choose_action(self, state: State, evaluation: bool = False) -> Optional[Action]:
        """
        Choose an action using epsilon-greedy (training) or greedy-only (evaluation).
        """
        available = self._get_available_actions(state)
        if not available:
            return None

        if evaluation:
            # pure exploitation for evaluation
            q_values = [self.get_q(state, a) for a in available]
            max_q = max(q_values)
            return random.choice([a for a, q in zip(available, q_values) if q == max_q])

        # --- Training (epsilon-greedy) ---
        if random.random() < self.epsilon:
            return random.choice(available)

        q_values = [self.get_q(state, a) for a in available]
        max_q = max(q_values)
        best = [a for a, q in zip(available, q_values) if q == max_q]
        return random.choice(best)

    # --------------------------------------------------------------
    # Learning Rule
    # --------------------------------------------------------------
    def learn(
        self,
        state: State,
        action: Optional[Action],
        reward: float,
        next_state: Optional[State],
        done: bool,
    ) -> None:

        if action is None:
            self._decay_epsilon()
            return

        # Safety: ensure action is legal
        if action not in self._get_available_actions(state):
            # Do not update Q if environment passed illegal action
            return

        old_q = self.get_q(state, action)

        if done or next_state is None:
            target = reward
        else:
            next_available = self._get_available_actions(next_state)
            if next_available:
                max_next_q = max(self.get_q(next_state, a) for a in next_available)
            else:
                max_next_q = 0.0

            target = reward + self.discount_factor * max_next_q

        new_q = old_q + self.learning_rate * (target - old_q)
        self.q_table[(tuple(state), action)] = new_q

        self._decay_epsilon()

    # --------------------------------------------------------------
    # Q-table access
    # --------------------------------------------------------------
    def get_q(self, state: State, action: Action) -> float:
        key = (tuple(state), action)
        return self.q_table.get(key, 0.0)

    def instantiate_state(self, state: State) -> None:
        """
        Optional: initialise Q-values for all available actions in a new state.
        Helps faster convergence.
        """
        for a in self._get_available_actions(state):
            key = (tuple(state), a)
            if key not in self.q_table:
                self.q_table[key] = 0.0

    # --------------------------------------------------------------
    # Saving/Loading
    # --------------------------------------------------------------
    def save(self, filepath: str) -> None:
        np.save(filepath, self.q_table, allow_pickle=True)

    def load(self, filepath: str) -> None:
        loaded = np.load(filepath, allow_pickle=True).item()
        if isinstance(loaded, dict):
            self.q_table = loaded
        else:
            raise ValueError("Loaded Q-table is not a dict.")

    # --------------------------------------------------------------
    # Evaluation Mode Control
    # --------------------------------------------------------------
    def enable_evaluation_mode(self):
        """Freeze epsilon temporarily."""
        self._epsilon_backup = self.epsilon
        self.epsilon = 0.0

    def disable_evaluation_mode(self):
        """Restore epsilon after evaluation."""
        self.epsilon = self._epsilon_backup

    # --------------------------------------------------------------
    # Internal Helpers
    # --------------------------------------------------------------
    def _get_available_actions(self, state: State) -> List[Action]:
        n = self.board_size
        return [(r, c) for (r, c) in self.actions if state[r * n + c] == 0]

    def _decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
