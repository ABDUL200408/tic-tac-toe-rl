"""
tests.py

Unit and integration tests for Tic-Tac-Toe RL components.

"""

from __future__ import annotations

import numpy as np

from environment import TicTacToeEnv
from agent import QLearningAgent
from utils import encode_state, decode_state


def test_environment_reset():
    """
    Environment.reset() should return an empty board encoded as state.
    """
    env = TicTacToeEnv()  # defaults to 5x5 with k = 5
    state = env.reset()

    n = env.n
    expected_board = np.zeros((n, n), dtype=int)
    expected_state = encode_state(expected_board)

    assert state == expected_state, (
        "Environment reset does not produce an encoded empty board."
    )


def test_valid_action_space_initial():
    """
    At the start of a new game, the action space should contain
    all cells on the n x n board.
    """
    env = TicTacToeEnv()
    env.reset()
    actions = env.get_action_space()

    n = env.n
    assert len(actions) == n * n, (
        f"Initial action space should contain {n * n} moves "
        f"for an empty {n}x{n} board, got {len(actions)}."
    )


def test_step_and_win_detection():
    """
    Simulate a horizontal win for the agent (player 1) and verify:
        - The game is marked as done.
        - The correct winner is reported.
        - Reward is +1 for the winning move.
    """
    env = TicTacToeEnv()  # default: 5x5, win_condition = 5
    env.reset()

    # Construct a 5-in-a-row on row 0 for the agent (X: player 1)
    # Interleave with opponent moves to keep the turn order valid.
    env.step((0, 0))  # X
    env.step((1, 0))  # O
    env.step((0, 1))  # X
    env.step((1, 1))  # O
    env.step((0, 2))  # X
    env.step((1, 2))  # O
    env.step((0, 3))  # X
    env.step((1, 3))  # O

    state, reward, done, info = env.step((0, 4))  # X completes five in a row

    assert done, "Game should be 'done' after the winning move."
    assert info.get("winner") == 1, "Agent (player 1) should be recorded as winner."
    assert reward == 1.0, "Winning move should provide reward +1 for the current player."


def test_agent_learn_and_choose_action():
    """
    Basic agent sanity check:
        - Agent chooses a valid action on an empty board.
        - With learning_rate=1.0 and terminal reward=1, the Q-table
          should store Q(s, a) = 1 for that state-action pair.
    """
    n = 5
    actions = [(r, c) for r in range(n) for c in range(n)]

    # Use learning_rate=1.0 so immediate reward fully overwrites old Q-value
    agent = QLearningAgent(actions=actions, learning_rate=1.0, board_size=n)
    env = TicTacToeEnv(board_size=n)

    state = env.reset()
    action = agent.choose_action(state)

    assert action in actions, "Agent should pick a valid action from the action space."

    # Terminal update: reward=1, done=True, next_state can be anything (ignored in target)
    agent.learn(state, action, reward=1.0, next_state=state, done=True)

    q_value = agent.get_q(state, action)
    assert q_value == 1.0, (
        "Q-table should update Q(s, a) to 1.0 when learning_rate=1.0, "
        "reward=1 and done=True."
    )


def test_state_encoding_and_decoding():
    """
    encode_state and decode_state should be inverse operations:
        decode_state(encode_state(board)) == board
    """
    n = 5
    board = np.zeros((n, n), dtype=int)
    board[0, 0] = 1
    board[0, 2] = -1
    board[1, 1] = 1

    state = encode_state(board)
    recovered = decode_state(state,board_size=n)

    assert isinstance(state, tuple), "Encoded state should be a tuple (hashable)."
    assert recovered.shape == board.shape, "Decoded board should have the same shape."
    assert np.array_equal(board, recovered), (
        "State encoding/decoding is inconsistent: "
        "decode_state(encode_state(board)) != board."
    )


def run_all_tests():
    """
    Simple test runner for manual execution.

    In an assessment setting, this demonstrates:
        - That the environment and agent behave as expected.
        - That state representation utilities are consistent.
    """
    print("Running Tic-Tac-Toe RL tests...")

    test_environment_reset()
    print("  ✔ Environment reset test passed.")

    test_valid_action_space_initial()
    print("  ✔ Initial action space test passed.")

    test_step_and_win_detection()
    print("  ✔ Step and win detection test passed.")

    test_agent_learn_and_choose_action()
    print("  ✔ Agent learn & action choice test passed.")

    test_state_encoding_and_decoding()
    print("  ✔ State encode/decode test passed.")

    print("ALL TESTS PASSED ✅")


if __name__ == "__main__":
    run_all_tests()
