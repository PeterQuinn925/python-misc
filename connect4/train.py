import argparse
import time
from typing import Optional

import numpy as np

from c4.board import Board
from c4.ai_minimax import MinimaxAI
from c4.ai_qlearning import QLearningAI


def play_game(agent_first: bool, q_agent: QLearningAI, mm_ai: MinimaxAI, train: bool = True) -> int:
    # returns winner: 1 if first player wins, -1 if second wins, 0 draw
    board = Board()
    players = [q_agent, mm_ai] if agent_first else [mm_ai, q_agent]
    player_ids = [1, -1]
    trajectory = []  # (state_key, action)

    while True:
        cur_idx = 0 if board.turn == 1 else 1
        cur_player = players[cur_idx]
        if isinstance(cur_player, QLearningAI):
            state_key = board.encode()
            action = cur_player.select_action(board, training=train)
            trajectory.append((state_key, action))
            board.drop(action)
        else:
            action = cur_player.best_move(board)
            board.drop(action)
        terminal, winner = board.terminal()
        if terminal:
            # rewards from perspective of Q agent only
            if train:
                # backward update with final reward
                final_reward = 0.0
                if winner == 1:
                    final_reward = 1.0 if agent_first else -1.0
                elif winner == -1:
                    final_reward = -1.0 if agent_first else 1.0
                else:
                    final_reward = 0.0
                next_board = None
                for idx in range(len(trajectory) - 1, -1, -1):
                    s_key, a = trajectory[idx]
                    q_agent.update(s_key, a, final_reward, next_board)
                    # small step penalty to encourage faster wins
                    final_reward *= 0.99
                    next_board = None  # terminal backup
            return winner


def train_loop(episodes: int, depth: int, epsilon: float, alpha: float, gamma: float,
               save_path: Optional[str] = None, seed: Optional[int] = 42):
    q_agent = QLearningAI(alpha=alpha, gamma=gamma, epsilon=epsilon, seed=seed)
    mm_ai = MinimaxAI(depth=depth, seed=seed)
    results = []
    start = time.time()
    for ep in range(1, episodes + 1):
        agent_first = (ep % 2 == 1)
        winner = play_game(agent_first, q_agent, mm_ai, train=True)
        results.append(winner)
        if ep % 100 == 0:
            rate = ep / (time.time() - start + 1e-9)
            last100 = results[-100:]
            q_wins = last100.count(1 if agent_first else -1)
            draws = last100.count(0)
            print(f"Episode {ep}/{episodes} | {rate:.1f} eps/s | last100: wins={q_wins}, draws={draws}")
    if save_path:
        q_agent.save(save_path)
    # Simple evaluation without exploration
    q_agent.epsilon = 0.0
    eval_wins = 0
    eval_draws = 0
    N = 200
    for i in range(N):
        agent_first = (i % 2 == 0)
        w = play_game(agent_first, q_agent, mm_ai, train=False)
        if (agent_first and w == 1) or ((not agent_first) and w == -1):
            eval_wins += 1
        elif w == 0:
            eval_draws += 1
    print(f"Evaluation vs minimax depth {depth}: wins={eval_wins}/{N}, draws={eval_draws}/{N}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--save', type=str, default='q_agent.pkl')
    args = parser.parse_args()

    train_loop(args.episodes, args.depth, args.epsilon, args.alpha, args.gamma, save_path=args.save)

