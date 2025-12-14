"""
config.py
"""
def get_config():
    config = {
        # ---------------------------------------------------------------------
        # Environment settings
        # ---------------------------------------------------------------------
        'board_size': 5,          # n for an n x n board
        'win_condition': 5,       # number in a row needed to win (<= board_size)

        # Reward scheme (kept here so we can tweak it in experiments/report)
        'reward_win': 1.0,
        'reward_loss': -1.0,
        'reward_draw': 0.0,       # can be set slightly negative to encourage wins

        # ---------------------------------------------------------------------
        # Core RL hyperparameters (Q-learning)
        # ---------------------------------------------------------------------
        'episodes': 10000,        # total number of training games
        'learning_rate': 0.1,     # alpha: step size for updates
        'discount_factor': 0.95,  # gamma: future reward weight

        # Epsilon-greedy exploration parameters
        'epsilon': 1.0,           # initial exploration probability (start fully random)
        'epsilon_decay': 0.9995,  # per-episode decay (discourages over-exploration)
        'epsilon_min': 0.05,      # minimum epsilon (ensures ongoing exploration)

        # ---------------------------------------------------------------------
        # Evaluation settings
        # ---------------------------------------------------------------------
        'eval_interval': 500,     # how often to evaluate agent (in episodes)
        'eval_episodes': 200,     # number of games per evaluation run (averaging)
        'show_progress': True,    # print progress to console
        'play_interactive': False,# enable human vs agent game after training

        # Opponent policy used during training: 'random' or 'heuristic'
        'opponent_policy': 'heuristic',

        # Which opponent to use during evaluation: 'random' or 'heuristic'
        'evaluation_opponent': 'random',

        # ---------------------------------------------------------------------
        # Experiment controls
        # ---------------------------------------------------------------------
        'random_seed': 42,             # ensures reproducibility (important in analysis)
        'q_table_save_path': 'q_table.npy',  # where to save/load Q-table

        # ---------------------------------------------------------------------
        # Visualisation
        # ---------------------------------------------------------------------
        'plot_title': "Q-Learning in Tic-Tac-Toe: Learning Curve",
        'save_plots': True,            # save plots as PNG
        'plots_dir': 'results/',       # output directory for plots
    }

    return config
