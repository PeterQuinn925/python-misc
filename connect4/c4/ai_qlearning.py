import os
import pickle
import random
from typing import Dict, Tuple, Optional

import numpy as np

from .board import Board

# Simple tabular Q-learning agent for Connect 4
class QLearningAI:
    def __init__(self,
                 alpha: float = 0.2,
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 seed: Optional[int] = None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q: Dict[bytes, np.ndarray] = {}
        self.rng = random.Random(seed)

    def _get_state_key(self, board: Board) -> bytes:
        return board.encode()

    def _ensure_state(self, key: bytes, valid_actions) -> np.ndarray:
        if key not in self.q:
            qvals = np.full(7, -1e6, dtype=np.float32)
            for a in valid_actions:
                qvals[a] = 0.0
            self.q[key] = qvals
        return self.q[key]

    def select_action(self, board: Board, training: bool = False) -> int:
        key = self._get_state_key(board)
        valid = board.valid_moves()
        qvals = self._ensure_state(key, valid)
        if training and self.rng.random() < self.epsilon:
            return self.rng.choice(valid)
        # Greedy among valid actions
        valid_q = [(a, qvals[a]) for a in valid]
        max_q = max(v for _, v in valid_q)
        best_actions = [a for a, v in valid_q if v == max_q]
        return self.rng.choice(best_actions)

    def update(self, state_key: bytes, action: int, reward: float, next_board: Optional[Board]):
        qvals = self.q[state_key]
        target = reward
        if next_board is not None:
            next_key = self._get_state_key(next_board)
            next_valid = next_board.valid_moves()
            next_q = self._ensure_state(next_key, next_valid)
            target = reward + self.gamma * np.max(next_q)
        qvals[action] += self.alpha * (target - qvals[action])

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(dict(q=self.q,
                             alpha=self.alpha,
                             gamma=self.gamma,
                             epsilon=self.epsilon), f)

    @classmethod
    def load(cls, path: str) -> 'QLearningAI':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        agent = cls(alpha=data.get('alpha', 0.2),
                    gamma=data.get('gamma', 0.99),
                    epsilon=data.get('epsilon', 0.1))
        agent.q = data['q']
        return agent

