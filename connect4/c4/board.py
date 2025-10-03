import numpy as np
from typing import List, Optional, Tuple

ROWS = 6
COLS = 7
CONNECT_N = 4

# Board representation: 0 empty, 1 player 1, -1 player 2
class Board:
    def __init__(self, grid: Optional[np.ndarray] = None, turn: int = 1):
        if grid is None:
            self.grid = np.zeros((ROWS, COLS), dtype=np.int8)
        else:
            self.grid = grid.astype(np.int8)
        self.turn = int(turn)  # 1 or -1

    def clone(self) -> 'Board':
        return Board(self.grid.copy(), self.turn)

    def valid_moves(self) -> List[int]:
        return [c for c in range(COLS) if self.grid[0, c] == 0]

    def drop(self, col: int) -> bool:
        if col < 0 or col >= COLS or self.grid[0, col] != 0:
            return False
        # place in lowest empty row of column
        for r in range(ROWS - 1, -1, -1):
            if self.grid[r, col] == 0:
                self.grid[r, col] = self.turn
                self.turn *= -1
                return True
        return False

    def undo(self, col: int) -> bool:
        # remove topmost piece in col
        for r in range(ROWS):
            if self.grid[r, col] != 0:
                self.grid[r, col] = 0
                self.turn *= -1
                return True
        return False

    def is_full(self) -> bool:
        return not any(self.grid[0, c] == 0 for c in range(COLS))

    def check_winner(self) -> int:
        # returns 1 if player 1 wins, -1 if player -1 wins, 0 otherwise
        g = self.grid
        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - CONNECT_N + 1):
                s = int(np.sum(g[r, c:c+CONNECT_N]))
                if abs(s) == CONNECT_N and len(set(g[r, c:c+CONNECT_N])) == 1:
                    return int(np.sign(s))
        # Vertical
        for c in range(COLS):
            col = g[:, c]
            for r in range(ROWS - CONNECT_N + 1):
                s = int(np.sum(col[r:r+CONNECT_N]))
                if abs(s) == CONNECT_N and len(set(col[r:r+CONNECT_N])) == 1:
                    return int(np.sign(s))
        # Diagonals
        for r in range(ROWS - CONNECT_N + 1):
            for c in range(COLS - CONNECT_N + 1):
                diag = [g[r+i, c+i] for i in range(CONNECT_N)]
                s = int(sum(diag))
                if abs(s) == CONNECT_N and len(set(diag)) == 1:
                    return int(np.sign(s))
        for r in range(CONNECT_N - 1, ROWS):
            for c in range(COLS - CONNECT_N + 1):
                diag = [g[r-i, c+i] for i in range(CONNECT_N)]
                s = int(sum(diag))
                if abs(s) == CONNECT_N and len(set(diag)) == 1:
                    return int(np.sign(s))
        return 0

    def terminal(self) -> Tuple[bool, int]:
        w = self.check_winner()
        if w != 0:
            return True, w
        if self.is_full():
            return True, 0
        return False, 0

    def encode(self) -> bytes:
        # compact encoding for Q-table key
        return self.grid.tobytes() + bytes([self.turn & 0xFF])

    def render_text(self) -> str:
        # For debugging/logging
        chars = {0: '.', 1: 'X', -1: 'O'}
        lines = []
        for r in range(ROWS):
            lines.append(' '.join(chars[int(x)] for x in self.grid[r]))
        lines.append('0 1 2 3 4 5 6')
        return '\n'.join(lines)

