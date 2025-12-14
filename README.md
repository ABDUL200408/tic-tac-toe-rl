
## Module Alignment

- **Knowledge Representation**  
  Game states are encoded as hashable tuples for efficient RL indexing, matching the module’s approach to discrete state modelling.

- **Reasoning Under Uncertainty**  
  The agent learns through repeated interaction, using an epsilon-greedy exploration strategy to handle incomplete information and unpredictable opponents.

- **Critical Evaluation**  
  The system includes learning curves, win/draw/loss statistics, experiment CSVs, and hyperparameter sweeps for analytical comparison.

- **Assessment Brief Compliance**  
  The implementation mirrors the lectures and labs (Markov processes, MDPs, Q-Learning, exploration–exploitation) and generates evidence required for the final report.

---

## How to Use

1. **Install Requirements**
    ```
    pip install -r requirements.txt
    ```
2. **Run Tests** (validates environment, agent, utilities)
    ```
    python tests.py
    ```
3. **Train and Evaluate Agent**
    ```
    python main.py
    ```
    - Modify hyperparameters in `config.py` as required for experiments.
    - After training, learning curves and metrics will be displayed/saved.
    - Optional: play against the trained agent interactively.

4. **Run Experiments**
    - For hyperparameter sweeps and advanced analysis, import and use `run_experiments()` in a Python notebook or from `main.py`.

---

## Key Files Explained

- `main.py`: Training loop, evaluation pipeline, plot generation, and experiment execution.
- `environment.py`: Defines Tic-Tac-Toe rules, state representation, transitions, rewards, and win detection.
- `agent.py`: Q-Learning agent with epsilon-greedy action selection, Q-value updates, saving/loading Q-tables.
- `evaluator.py`: Runs evaluation episodes, tracks win/draw/loss outcomes, and supports human-vs-agent games.
- `experiments.py`: Hyperparameter sweeps (learning rate, epsilon decay) for analytical comparison.
- `visualisations.py`:Generates learning curves, experiment plots, and board visualisation images.
- `utils.py`: Reproducibility (random seeds), state encoding/decoding, filesystem helpers.
- `tests.py`: Unit tests confirming correctness of the environment, agent, and utilities.
- `q_table.npy`: Automatically saved Q-table after training.


---

## After running main.py, the results/ folder will contain:

- learning_curve.png – visual performance curve

- board_size.png – generated board visualisation

- learning_rate_sweep.png / csv

- epsilon_decay_sweep.png / csv

- Saved Q-table: q_table.npy