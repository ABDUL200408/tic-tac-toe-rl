"""
utils.py

Utility functions for Tic-Tac-Toe Reinforcement Learning.


"""

from __future__ import annotations
import numpy as np
import random
import os


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed across numpy, random, and OS-level hashing.

    Required for:
        - Reproducibility in Q-learning experiments
        - Fair comparisons in hyperparameter sweeps
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ----------------------------------------------------------------------
# State representation utilities
# ----------------------------------------------------------------------
def encode_state(board: np.ndarray) -> tuple:
    """
    Encode an n x n board as a flat tuple (hashable).

    Parameters
    ----------
    board : np.ndarray
        Board containing values {0, 1, -1}.

    Returns
    -------
    tuple
        Flattened tuple used as a dictionary key in Q-learning.
    """
    return tuple(board.flatten())


def decode_state(state_tuple: tuple, board_size: int) -> np.ndarray:
    """
    Decode a flattened tuple back into an n x n board.

    Parameters
    ----------
    state_tuple : tuple
        Flattened tuple produced by encode_state().
    board_size : int
        Size of the board (e.g., 3, 5, 7). Must be provided explicitly.

    Returns
    -------
    np.ndarray
        Reconstructed (board_size x board_size) board.
    """
    arr = np.array(state_tuple, dtype=int)
    return arr.reshape((board_size, board_size))


# Optional convenience aliases
flatten = encode_state
unflatten = decode_state


# ----------------------------------------------------------------------
# Logging utility
# ----------------------------------------------------------------------
def log(msg: str, prefix: str = "[INFO]") -> None:
    """
    Log a message with timestamp and optional prefix.

    Useful for:
        - Debugging
        - Tracing experiment progress
        - Writing transparent, auditable RL logs
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{prefix} {timestamp}: {msg}")


# ----------------------------------------------------------------------
# Filesystem helpers
# ----------------------------------------------------------------------
def safe_mkdir(path: str) -> None:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path : str
        Directory path to create.

    Notes
    -----
    Used for saving:
        - Plots
        - Experiment CSV files
        - Q-tables
    """
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as ex:
        log(f"Failed to create directory '{path}': {ex}", prefix="[ERROR]")
