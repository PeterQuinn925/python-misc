# Connect 4 with Minimax and Q-learning

This project implements a Connect 4 game in Python with a Pygame GUI. It supports two computer AI modes:
- Look-ahead Minimax with alpha-beta pruning
- Q-learning agent that starts from scratch and learns by playing

There is also a training script to train the Q-learning agent against the Minimax AI over many self-play games.

## Requirements
- Python 3.9+
- Windows, macOS, or Linux

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the game (GUI)

- Play vs Minimax (depth 5):

```bash
python main.py --ai minimax --depth 5 --human-first
```

- Play vs Q-learning (load a trained model):

```bash
python main.py --ai qlearn --q-path q_agent.pkl --human-first
```

Controls: Click a column to drop your piece. Press R to restart after a game ends.

## Train the Q-learning agent

Run a headless training session where the Q-learning agent plays vs Minimax:

```bash
python train.py --episodes 5000 --depth 4 --epsilon 0.2 --alpha 0.2 --gamma 0.99 --save q_agent.pkl
```

- episodes: number of self-play games
- depth: minimax depth for the opponent
- epsilon: exploration rate
- alpha: learning rate
- gamma: discount factor

After training, start the GUI using the saved q_agent.pkl:

```bash
python main.py --ai qlearn --q-path q_agent.pkl
```

## Notes
- The board is 6x7 with classic Connect 4 rules.
- Minimax evaluation weights prioritize center control and threats.
- The Q-table is keyed by a compact byte encoding of board state and turn.
- Saved Q-tables are pickled with extension .pkl.

