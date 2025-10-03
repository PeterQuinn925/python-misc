import pygame
import sys
from typing import Optional

from .board import Board, ROWS, COLS
from .ai_minimax import MinimaxAI
from .ai_qlearning import QLearningAI

CELL_SIZE = 100
PADDING = 20
RADIUS = CELL_SIZE // 2 - 8
WIDTH = COLS * CELL_SIZE
HEIGHT = (ROWS + 1) * CELL_SIZE  # extra row for dropping indicator

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (50, 50, 200)
RED = (200, 50, 50)
YELLOW = (230, 200, 50)
GRAY = (180, 180, 180)

class GameGUI:
    def __init__(self, ai_type: str = 'minimax', depth: int = 5, q_path: Optional[str] = None, human_first: bool = True):
        pygame.init()
        pygame.display.set_caption('Connect 4')
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 28)
        self.small_font = pygame.font.SysFont('Arial', 18)

        self.board = Board()
        self.human_player = 1 if human_first else -1
        self.ai_player = -self.human_player
        self.ai_type = ai_type
        # Stats: track wins/losses/draws for human, minimax, and qlearn
        self.stats = {
            'human': {'W': 0, 'L': 0, 'D': 0},
            'minimax': {'W': 0, 'L': 0, 'D': 0},
            'qlearn': {'W': 0, 'L': 0, 'D': 0},
        }
        self._stats_updated = False

        if ai_type == 'minimax':
            self.ai = MinimaxAI(depth=depth)
            self.q_agent = None
        elif ai_type == 'qlearn':
            if q_path:
                try:
                    self.q_agent = QLearningAI.load(q_path)
                except Exception:
                    self.q_agent = QLearningAI()
            else:
                self.q_agent = QLearningAI()
            self.ai = None
        else:
            raise ValueError('Unknown ai_type')

    def draw_board(self, hover_col: Optional[int] = None):
        self.screen.fill(BLUE)
        # draw board holes
        for r in range(ROWS):
            for c in range(COLS):
                cx = c * CELL_SIZE + CELL_SIZE // 2
                cy = (r + 1) * CELL_SIZE + CELL_SIZE // 2
                pygame.draw.circle(self.screen, BLACK, (cx, cy), RADIUS + 4)
                color = WHITE
                if self.board.grid[r, c] == 1:
                    color = RED
                elif self.board.grid[r, c] == -1:
                    color = YELLOW
                pygame.draw.circle(self.screen, color, (cx, cy), RADIUS)
        # top hover row
        if hover_col is not None and 0 <= hover_col < COLS:
            cx = hover_col * CELL_SIZE + CELL_SIZE // 2
            cy = CELL_SIZE // 2
            color = RED if self.board.turn == 1 else YELLOW
            pygame.draw.circle(self.screen, color, (cx, cy), RADIUS)
        # scoreboard overlay
        self.draw_scoreboard()

    def draw_scoreboard(self):
        # Render simple W-L-D counters for each participant
        h = self.stats['human']
        mm = self.stats['minimax']
        ql = self.stats['qlearn']
        txt_h = self.small_font.render(f"Human W-L-D: {h['W']}-{h['L']}-{h['D']}", True, WHITE)
        txt_m = self.small_font.render(f"Minimax W-L-D: {mm['W']}-{mm['L']}-{mm['D']}", True, WHITE)
        txt_q = self.small_font.render(f"Q-Learn W-L-D: {ql['W']}-{ql['L']}-{ql['D']}", True, WHITE)
        # Place across the top bar to minimize overlap with hover disc
        self.screen.blit(txt_h, (10, 6))
        self.screen.blit(txt_m, (250, 6))
        self.screen.blit(txt_q, (500, 6))

    def _apply_result_to_stats(self, winner: int):
        ai_key = 'minimax' if self.ai_type == 'minimax' else 'qlearn'
        if winner == 0:
            self.stats['human']['D'] += 1
            self.stats[ai_key]['D'] += 1
        elif winner == self.human_player:
            self.stats['human']['W'] += 1
            self.stats[ai_key]['L'] += 1
        elif winner == self.ai_player:
            self.stats['human']['L'] += 1
            self.stats[ai_key]['W'] += 1

    def get_col_from_mouse(self, x):
        col = x // CELL_SIZE
        return int(col)

    def make_ai_move(self):
        if self.board.turn != self.ai_player:
            return
        if self.ai is not None:
            col = self.ai.best_move(self.board)
        elif self.q_agent is not None:
            col = self.q_agent.select_action(self.board, training=False)
        else:
            return
        self.board.drop(col)

    def run(self):
        hover_col = None
        game_over = False
        winner = 0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if game_over:
                    continue
                if event.type == pygame.MOUSEMOTION:
                    hover_col = self.get_col_from_mouse(event.pos[0])
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.board.turn == self.human_player:
                        col = self.get_col_from_mouse(event.pos[0])
                        if col in self.board.valid_moves():
                            self.board.drop(col)
            if not game_over and self.board.turn == self.ai_player:
                self.make_ai_move()
            term, w = self.board.terminal()
            if term:
                game_over = True
                winner = w
                if not self._stats_updated:
                    self._apply_result_to_stats(winner)
                    self._stats_updated = True
            self.draw_board(hover_col)
            if game_over:
                if winner == 0:
                    msg = 'Draw!'
                elif winner == self.human_player:
                    msg = 'You win!'
                else:
                    msg = 'Computer wins!'
                text = self.font.render(msg + ' Press R to restart.', True, WHITE)
                self.screen.blit(text, (20, 40))
                keys = pygame.key.get_pressed()
                if keys[pygame.K_r]:
                    self.board = Board()
                    game_over = False
                    winner = 0
                    self._stats_updated = False
            pygame.display.flip()
            self.clock.tick(60)

