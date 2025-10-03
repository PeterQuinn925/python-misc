from __future__ import annotations
from typing import Optional, Tuple
import math
import random
import numpy as np

from .board import Board, ROWS, COLS, CONNECT_N

# Minimax AI with alpha-beta pruning
class MinimaxAI:
    def __init__(self, depth: int = 5, seed: Optional[int] = None):
        self.depth = depth
        self.rng = random.Random(seed)

    def evaluate(self, board: Board, player: int) -> int:
        # Heuristic: count windows of 2/3, penalize opponent; prioritize center
        g = board.grid
        score = 0
        center_col = COLS // 2
        center_count = int(np.sum(g[:, center_col] == player))
        score += 3 * center_count

        def score_window(window):
            cnt_p = int(np.sum(window == player))
            cnt_o = int(np.sum(window == -player))
            cnt_e = int(np.sum(window == 0))
            s = 0
            if cnt_p == 4:
                s += 10000
            elif cnt_p == 3 and cnt_e == 1:
                s += 50
            elif cnt_p == 2 and cnt_e == 2:
                s += 10
            if cnt_o == 3 and cnt_e == 1:
                s -= 80
            elif cnt_o == 2 and cnt_e == 2:
                s -= 8
            return s

        # Horizontal
        for r in range(ROWS):
            row = g[r, :]
            for c in range(COLS - 3):
                window = row[c:c+4]
                score += score_window(window)
        # Vertical
        for c in range(COLS):
            col = g[:, c]
            for r in range(ROWS - 3):
                window = col[r:r+4]
                score += score_window(window)
        # Diagonals
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                window = np.array([g[r+i, c+i] for i in range(4)])
                score += score_window(window)
        for r in range(3, ROWS):
            for c in range(COLS - 3):
                window = np.array([g[r-i, c+i] for i in range(4)])
                score += score_window(window)
        return score

    def best_move(self, board: Board) -> int:
        player = board.turn
        valid = board.valid_moves()
        self.rng.shuffle(valid)
        best_score = -math.inf
        best_col = valid[0]
        alpha, beta = -math.inf, math.inf
        for col in valid:
            b2 = board.clone()
            b2.drop(col)
            score = self._minimax(b2, self.depth - 1, alpha, beta, False, player)
            if score > best_score:
                best_score = score
                best_col = col
            alpha = max(alpha, score)
        return best_col

    def _minimax(self, board: Board, depth: int, alpha: float, beta: float, maximizing: bool, player: int) -> float:
        terminal, winner = board.terminal()
        if terminal:
            if winner == player:
                return 1e9  # winning terminal
            elif winner == -player:
                return -1e9  # losing terminal
            else:
                return 0
        if depth == 0:
            return self.evaluate(board, player)
        valid = board.valid_moves()
        if maximizing:
            value = -math.inf
            for col in valid:
                board.drop(col)
                value = max(value, self._minimax(board, depth - 1, alpha, beta, False, player))
                board.undo(col)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = math.inf
            for col in valid:
                board.drop(col)
                value = min(value, self._minimax(board, depth - 1, alpha, beta, True, player))
                board.undo(col)
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

