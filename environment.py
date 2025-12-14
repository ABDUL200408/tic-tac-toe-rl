"""
environment.py

Encapsulates the Tic-Tac-Toe environment for RL training.

Conventions:
    Board values:
        0  : empty
        1  : RL agent (X)
       -1  : opponent (O)

    current_player:
        1  : agent's move
       -1  : opponent's move
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


State = Tuple[int, ...]   # flat board: length n*n, values in {-1, 0, 1}
Action = Tuple[int, int]  # (row, col)


class TicTacToeEnv:
    """
    Tic-Tac-Toe environment for an n x n board with configurable win condition.

    Parameters
    ----------
    board_size : int, optional (default=5)
        Dimension n of the board (n x n).
    win_condition : int, optional
        Number of marks in a line required to win (k).
        If None, defaults to board_size (i.e. full row/column/diagonal).
    reward_win : float, optional (default=1.0)
        Reward given to the player who wins.
    reward_loss : float, optional (default=-1.0)
        Reward given to the player when the *other* player is considered
        the winner in this transition (kept for completeness).
    reward_draw : float, optional (default=0.0)
        Reward assigned on a draw (board full, no winner).

    Notes
    -----
    Examples:
        board_size=5, win_condition=5  -> classic 5-in-a-row on 5x5
        board_size=5, win_condition=4  -> 4-in-a-row variant on 5x5
        board_size=3, win_condition=3  -> standard Tic-Tac-Toe
    """

    def __init__(
        self,
        board_size: int = 5,
        win_condition: Optional[int] = None,
        reward_win: float = 1.0,
        reward_loss: float = -1.0,
        reward_draw: float = 0.0,
    ) -> None:
        self.n: int = int(board_size)
        self.k: int = int(win_condition) if win_condition is not None else self.n

        if self.k < 3 or self.k > self.n:
            raise ValueError("win_condition must be between 3 and board_size (inclusive).")

        # 0 = empty, 1 = agent (X), -1 = opponent (O)
        self.board: np.ndarray = np.zeros((self.n, self.n), dtype=int)
        self.current_player: int = 1  # 1: RL agent, -1: opponent

        # Reward configuration
        self.reward_win: float = reward_win
        self.reward_loss: float = reward_loss
        self.reward_draw: float = reward_draw

    # ------------------------------------------------------------------
    # Core environment API
    # ------------------------------------------------------------------
    def reset(self) -> State:
        """
        Reset the board for a new episode.

        Returns
        -------
        State
            Initial state as a flat, hashable tuple.
        """
        self.board[:] = 0
        self.current_player = 1
        return self.get_state()

    def get_state(self) -> State:
        """
        Encodes the current board as a tuple (hashable, RL-friendly).

        Returns
        -------
        State
            Flattened board as a tuple of ints.
        """
        return tuple(self.board.flatten())

    def get_action_space(self) -> List[Action]:
        """
        Returns the list of currently available actions.

        Each action is a (row, col) pair where the cell is empty.

        Returns
        -------
        List[Action]
            List of legal moves on the current board.
        """
        return [
            (r, c)
            for r in range(self.n)
            for c in range(self.n)
            if self.board[r, c] == 0
        ]

    def step(self, action: Action) -> Tuple[State, float, bool, Dict[str, Any]]:
        """
        Apply the given action for the current player.

        Parameters
        ----------
        action : Action
            (row, col) position where the current player will place their mark.

        Returns
        -------
        next_state : State
            Encoded board after the move.
        reward : float
            Reward from the perspective of the player who just moved:
                reward_win   if they win,
                reward_draw  if the game ends in a draw,
                0.0          for non-terminal moves,
                reward_loss  if somehow the other player is deemed winner.
        done : bool
            True if the game has ended (win or draw), False otherwise.
        info : dict
            Additional diagnostic information, including:
                - 'winner': 1 (agent), -1 (opponent), or None
                - 'board_full': bool
        """
        r, c = action

        # Validity checks
        if r < 0 or r >= self.n or c < 0 or c >= self.n:
            raise ValueError(f"Invalid action {action}: out of bounds for board size {self.n}.")
        if self.board[r, c] != 0:
            raise ValueError(f"Invalid action {action}: cell already taken.")

        # Place the mark for the current player
        self.board[r, c] = self.current_player

        # Terminal checks
        winner = self.check_winner()
        board_full = not self.get_action_space()
        done = winner is not None or board_full

        # Reward from perspective of the player who just moved
        if not done:
            reward = 0.0
        else:
            if winner is None:
                # Draw
                reward = self.reward_draw
            elif winner == self.current_player:
                reward = self.reward_win
            else:
                reward = self.reward_loss

        info: Dict[str, Any] = {
            "winner": winner,
            "board_full": board_full,
        }

        # Only switch players if the game continues
        if not done:
            self.current_player *= -1

        return self.get_state(), reward, done, info

    # ------------------------------------------------------------------
    # Winner detection
    # ------------------------------------------------------------------
    def check_winner(self) -> Optional[int]:
        """
        Determine if the current board is in a winning state for any player.

        Returns
        -------
        int or None
            1   if player 1 (agent) has k-in-a-row,
           -1   if player -1 (opponent) has k-in-a-row,
            None if there is no winner yet.
        """
        for player in (1, -1):
            if self._has_k_in_a_row(player):
                return player
        return None

    def _has_k_in_a_row(self, player: int) -> bool:
        """
        Check whether the given player has `k` consecutive marks in any direction.

        Directions checked:
            - Horizontal (right)
            - Vertical (down)
            - Main diagonal (down-right)
            - Anti-diagonal (down-left)
        """
        n, k = self.n, self.k
        board = self.board

        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # main diagonal
            (1, -1),  # anti-diagonal
        ]

        for r in range(n):
            for c in range(n):
                if board[r, c] != player:
                    continue
                for dr, dc in directions:
                    end_r = r + (k - 1) * dr
                    end_c = c + (k - 1) * dc
                    # Check that the full segment fits on the board
                    if 0 <= end_r < n and 0 <= end_c < n:
                        if all(board[r + i * dr, c + i * dc] == player for i in range(k)):
                            return True
        return False

    # ------------------------------------------------------------------
    # Rendering (for debugging / demos)
    # ------------------------------------------------------------------
    def render(self) -> None:
        """
        Pretty-print the board for debugging or human play.

        Representation:
            'X' for player 1 (agent)
            'O' for player -1 (opponent)
            ' ' for empty cells
        """
        symbols = {0: " ", 1: "X", -1: "O"}
        for r in range(self.n):
            row_str = "|".join(symbols[self.board[r, c]] for c in range(self.n))
            print(row_str)
            if r < self.n - 1:
                print("-" * (2 * self.n - 1))
