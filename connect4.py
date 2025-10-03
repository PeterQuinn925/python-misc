import pygame
import numpy as np
import random
import pickle
import os
from typing import Dict, List, Tuple, Optional

class Connect4AI:
    """Improved Q-Learning AI for Connect 4 with better state representation"""

    # Board constants to avoid magic numbers
    ROWS: int = 6
    COLS: int = 7
    DIRECTIONS: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1))
    
    def __init__(self, learning_rate=0.3, discount_factor=0.95, epsilon=0.9):
        self.q_table: Dict[str, float] = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.9995  # Slower decay
        self.min_epsilon = 0.1  # Higher minimum exploration
        self.game_history = []
        
    def get_state_key(self, board: np.ndarray, player: int) -> str:
        """Convert board state to a more compact string key"""
        # Use a more efficient state representation
        state_str = ''.join(str(cell) for row in board for cell in row)
        return f"{state_str}_{player}"
    
    def get_board_features(self, board: np.ndarray, player: int) -> str:
        """Extract important board features for better learning"""
        features = []
        
        # Check for immediate wins/blocks for both players
        for col in range(self.COLS):
            if board[0][col] == 0:  # Column not full
                # Simulate move for current player
                temp_board = board.copy()
                row = self.get_drop_row(temp_board, col)
                if row != -1:
                    temp_board[row][col] = player
                    if self.check_win_fast(temp_board, row, col, player):
                        features.append(f"win_{col}")
                    
                    # Check if opponent can win next turn
                    temp_board[row][col] = 3 - player
                    if self.check_win_fast(temp_board, row, col, 3 - player):
                        features.append(f"block_{col}")
                    
                    temp_board[row][col] = 0  # Reset
        
        # Add center column preference
        center_count = np.sum(board[:, 3] == player)
        features.append(f"center_{center_count}")
        
        # Add connectivity features (pieces next to each other)
        connectivity = self.count_connectivity(board, player)
        features.append(f"conn_{connectivity}")
        
        return '_'.join(features) if features else "empty"
    
    def get_drop_row(self, board: np.ndarray, col: int) -> int:
        """Get the row where a piece would drop in the given column"""
        for row in range(self.ROWS - 1, -1, -1):
            if board[row][col] == 0:
                return row
        return -1
    
    def check_win_fast(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        """Fast win check for a specific position"""
        for dx, dy in self.DIRECTIONS:
            count = 1
            
            # Check positive direction
            r, c = row + dx, col + dy
            while 0 <= r < self.ROWS and 0 <= c < self.COLS and board[r][c] == player:
                count += 1
                r, c = r + dx, c + dy
            
            # Check negative direction
            r, c = row - dx, col - dy
            while 0 <= r < self.ROWS and 0 <= c < self.COLS and board[r][c] == player:
                count += 1
                r, c = r - dx, c - dy
            
            if count >= 4:
                return True
        
        return False
    
    def count_connectivity(self, board: np.ndarray, player: int) -> int:
        """Count how many pieces are connected (adjacent) for the player"""
        count = 0
        for row in range(self.ROWS):
            for col in range(self.COLS):
                if board[row][col] == player:
                    for dx, dy in self.DIRECTIONS:
                        new_row, new_col = row + dx, col + dy
                        if (0 <= new_row < self.ROWS and 0 <= new_col < self.COLS and 
                            board[new_row][new_col] == player):
                            count += 1
        
        return count // 2  # Divide by 2 since we count each connection twice
    
    def get_q_value(self, state: str, action: int) -> float:
        """Get Q-value for state-action pair"""
        key = f"{state}_{action}"
        return self.q_table.get(key, 0.0)
    
    def set_q_value(self, state: str, action: int, value: float):
        """Set Q-value for state-action pair"""
        key = f"{state}_{action}"
        self.q_table[key] = value
    
    def get_valid_moves(self, board: np.ndarray) -> List[int]:
        """Get list of valid column moves"""
        return [col for col in range(self.COLS) if board[0][col] == 0]
    
    def evaluate_move(self, board: np.ndarray, col: int, player: int) -> float:
        """Evaluate a move with hand-crafted heuristics"""
        if board[0][col] != 0:  # Invalid move
            return -1000
        
        row = self.get_drop_row(board, col)
        if row == -1:
            return -1000
        
        score = 0
        temp_board = board.copy()
        temp_board[row][col] = player
        
        # Check for immediate win
        if self.check_win_fast(temp_board, row, col, player):
            return 1000
        
        # Check for blocking opponent win
        temp_board[row][col] = 3 - player
        if self.check_win_fast(temp_board, row, col, 3 - player):
            score += 500
        
        temp_board[row][col] = player  # Reset to our move
        
        # Prefer center columns
        center_distance = abs(col - 3)
        score += (4 - center_distance) * 10
        
        # Prefer moves that create threats
        threats = self.count_threats(temp_board, row, col, player)
        score += threats * 50
        
        # Avoid moves that give opponent good opportunities
        temp_board[row][col] = 0
        if row > 0:  # If there's a space above this move
            temp_board[row-1][col] = 3 - player
            if self.check_win_fast(temp_board, row-1, col, 3 - player):
                score -= 200  # Penalty for giving opponent a win opportunity
        
        return score
    
    def count_threats(self, board: np.ndarray, row: int, col: int, player: int) -> int:
        """Count potential winning threats created by this move"""
        threats = 0
        for dx, dy in self.DIRECTIONS:
            count = 1
            empty_ends = 0
            
            # Check positive direction
            r, c = row + dx, col + dy
            while 0 <= r < self.ROWS and 0 <= c < self.COLS:
                if board[r][c] == player:
                    count += 1
                elif board[r][c] == 0:
                    empty_ends += 1
                    break
                else:
                    break
                r, c = r + dx, c + dy
            
            # Check negative direction
            r, c = row - dx, col - dy
            while 0 <= r < self.ROWS and 0 <= c < self.COLS:
                if board[r][c] == player:
                    count += 1
                elif board[r][c] == 0:
                    empty_ends += 1
                    break
                else:
                    break
                r, c = r - dx, c - dy
            
            # Count as threat if we have 2-3 in a row with open ends
            if 2 <= count < 4 and empty_ends > 0:
                threats += count - 1
        
        return threats
    
    def get_best_action(self, board: np.ndarray, player: int) -> int:
        """Get best action using improved epsilon-greedy policy with heuristics"""
        valid_moves = self.get_valid_moves(board)
        if not valid_moves:
            return -1
        
        # Use heuristics for initial guidance and fallback
        move_scores = [(col, self.evaluate_move(board, col, player)) for col in valid_moves]
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Check for immediate win or necessary block
        best_heuristic_move = move_scores[0][0]
        best_heuristic_score = move_scores[0][1]
        
        if best_heuristic_score >= 500:  # Win or critical block
            return best_heuristic_move
        
        # Use Q-learning with exploration
        if random.random() < self.epsilon:
            # Explore: but bias towards better heuristic moves
            weights = [max(0.1, score + 100) for _, score in move_scores]
            total_weight = sum(weights)
            rand_val = random.uniform(0, total_weight)
            
            cumulative = 0
            for i, (col, _) in enumerate(move_scores):
                cumulative += weights[i]
                if rand_val <= cumulative:
                    return col
            return move_scores[0][0]
        
        # Exploit: use Q-values but consider heuristics
        state = self.get_board_features(board, player)
        best_move = valid_moves[0]
        best_value = self.get_q_value(state, best_move) + self.evaluate_move(board, best_move, player) * 0.1
        
        for move in valid_moves[1:]:
            q_value = self.get_q_value(state, move)
            heuristic_bonus = self.evaluate_move(board, move, player) * 0.1
            total_value = q_value + heuristic_bonus
            
            if total_value > best_value:
                best_value = total_value
                best_move = move
        
        return best_move
    
    def update_q_value(self, state: str, action: int, reward: float, 
                      next_state: str, done: bool):
        """Update Q-value using Q-learning update rule"""
        current_q = self.get_q_value(state, action)
        
        if done:
            max_next_q = 0
        else:
            # Find max Q-value for next state (simplified)
            max_next_q = max([self.get_q_value(next_state, a) for a in range(self.COLS)], default=0)
        
        # Q-learning update with higher learning rate
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.set_q_value(state, action, new_q)
    
    def decay_epsilon(self):
        """Decay exploration rate more slowly"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filename: str):
        """Save Q-table to file"""
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    
    def load_model(self, filename: str):
        """Load Q-table from file"""
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                    self.q_table = data.get('q_table', {})
                    self.epsilon = float(data.get('epsilon', self.epsilon))
                    self.learning_rate = float(data.get('learning_rate', self.learning_rate))
                    self.discount_factor = float(data.get('discount_factor', self.discount_factor))
                return True
            except Exception:
                return False
        return False


class Connect4Game:
    """Connect 4 game with improved ML AI opponent"""
    
    def __init__(self):
        pygame.init()
        
        # Game constants
        self.ROWS = 6
        self.COLS = 7
        self.CELL_SIZE = 80
        self.MARGIN = 10
        
        # Colors
        self.BLUE = (0, 100, 200)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 245, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128)
        self.GREEN = (0, 255, 0)
        
        # Screen setup
        self.BOARD_WIDTH = self.COLS * self.CELL_SIZE + (self.COLS + 1) * self.MARGIN
        self.BOARD_HEIGHT = self.ROWS * self.CELL_SIZE + (self.ROWS + 1) * self.MARGIN
        self.PANEL_WIDTH = 300
        self.SCREEN_WIDTH = self.BOARD_WIDTH + self.PANEL_WIDTH
        self.SCREEN_HEIGHT = max(self.BOARD_HEIGHT, 600)
        
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Connect 4 ML Game - Improved")
        
        # Fonts
        self.font_large = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)
        
        # Game state
        self.board = np.zeros((self.ROWS, self.COLS), dtype=int)
        self.current_player = 1  # 1 = human/red, 2 = AI/yellow
        self.game_over = False
        self.winner = None
        self.game_mode = 'human'  # 'human' or 'ai_vs_ai'
        
        # AI and statistics
        self.ai = Connect4AI()
        self.ai.load_model('connect4_improved_model.pkl')
        
        self.stats = {
            'games_played': 0,
            'human_wins': 0,
            'ai_wins': 0,
            'draws': 0
        }
        
        # Training
        self.training = False
        self.training_progress = 0
        self.training_total = 0
        
        # Game history for learning
        self.game_states = []
        
        self.clock = pygame.time.Clock()
        self.running = True
    
    def reset_game(self):
        """Reset game to initial state"""
        # Save game history for AI learning
        if self.game_states and self.winner is not None:
            self.update_ai_from_game()
        
        self.board = np.zeros((self.ROWS, self.COLS), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.game_states = []
    
    def update_ai_from_game(self):
        """Update AI Q-values based on game outcome with improved rewards"""
        if not self.game_states:
            return
        
        # Improved reward system
        if self.winner == 2:  # AI won
            final_reward = 100
        elif self.winner == 1:  # AI lost
            final_reward = -100
        else:  # Draw
            final_reward = 10  # Small positive reward for draws
        
        # Update Q-values for AI moves (player 2) with temporal difference
        ai_states = [(state, action) for state, action, player in self.game_states if player == 2]
        
        for i, (state, action) in enumerate(ai_states):
            # Give higher rewards to later moves (closer to outcome)
            move_reward = final_reward * (0.95 ** (len(ai_states) - 1 - i))
            
            # Add small positive reward for good moves during game
            if i < len(ai_states) - 1:
                next_state = ai_states[i + 1][0]
                self.ai.update_q_value(state, action, move_reward, next_state, False)
            else:
                self.ai.update_q_value(state, action, move_reward, state, True)
        
        self.ai.decay_epsilon()
    
    def make_move(self, col: int) -> bool:
        """Make a move in the specified column"""
        if self.game_over or col < 0 or col >= self.COLS:
            return False
        
        # Find the lowest empty row
        for row in range(self.ROWS - 1, -1, -1):
            if self.board[row][col] == 0:
                # Save state for AI learning
                if self.current_player == 2:
                    state = self.ai.get_board_features(self.board.copy(), self.current_player)
                    self.game_states.append((state, col, self.current_player))
                
                self.board[row][col] = self.current_player
                
                # Check for win
                if self.check_win(row, col):
                    self.game_over = True
                    self.winner = self.current_player
                    self.update_stats()
                # Check for draw
                elif self.check_draw():
                    self.game_over = True
                    self.winner = None
                    self.update_stats()
                
                # Switch players
                self.current_player = 3 - self.current_player
                return True
        
        return False
    
    def check_win(self, row: int, col: int) -> bool:
        """Check if the last move resulted in a win"""
        player = self.board[row][col]
        
        # Check all four directions
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1
            
            # Check positive direction
            r, c = row + dx, col + dy
            while 0 <= r < self.ROWS and 0 <= c < self.COLS and self.board[r][c] == player:
                count += 1
                r, c = r + dx, c + dy
            
            # Check negative direction
            r, c = row - dx, col - dy
            while 0 <= r < self.ROWS and 0 <= c < self.COLS and self.board[r][c] == player:
                count += 1
                r, c = r - dx, c - dy
            
            if count >= 4:
                return True
        
        return False
    
    def check_draw(self) -> bool:
        """Check if the game is a draw"""
        return np.all(self.board[0] != 0)
    
    def update_stats(self):
        """Update game statistics"""
        self.stats['games_played'] += 1
        if self.winner == 1:
            if self.game_mode == 'human':
                self.stats['human_wins'] += 1
            else:
                self.stats['ai_wins'] += 1  # AI 1 in ai_vs_ai mode
        elif self.winner == 2:
            self.stats['ai_wins'] += 1
        else:
            self.stats['draws'] += 1
    
    def make_ai_move(self):
        """Make AI move"""
        if self.game_over:
            return
        
        action = self.ai.get_best_action(self.board, self.current_player)
        if action != -1:
            self.make_move(action)
    
    def draw_board(self):
        """Draw the game board"""
        # Draw board background
        board_rect = pygame.Rect(0, 0, self.BOARD_WIDTH, self.BOARD_HEIGHT)
        pygame.draw.rect(self.screen, self.BLUE, board_rect)
        
        # Draw cells
        for row in range(self.ROWS):
            for col in range(self.COLS):
                x = col * self.CELL_SIZE + (col + 1) * self.MARGIN
                y = row * self.CELL_SIZE + (row + 1) * self.MARGIN
                
                # Draw cell background
                cell_rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
                
                if self.board[row][col] == 0:
                    pygame.draw.ellipse(self.screen, self.BLACK, cell_rect)
                elif self.board[row][col] == 1:
                    pygame.draw.ellipse(self.screen, self.RED, cell_rect)
                else:
                    pygame.draw.ellipse(self.screen, self.YELLOW, cell_rect)
    
    def draw_panel(self):
        """Draw the control panel"""
        panel_x = self.BOARD_WIDTH
        panel_rect = pygame.Rect(panel_x, 0, self.PANEL_WIDTH, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.WHITE, panel_rect)
        
        y_offset = 20
        
        # Title
        title = self.font_large.render("Connect 4 ML+", True, self.BLACK)
        self.screen.blit(title, (panel_x + 10, y_offset))
        y_offset += 50
        
        # Game status
        if self.game_over:
            if self.winner == 1:
                status = "You Win!" if self.game_mode == 'human' else "AI 1 Wins!"
                color = self.RED
            elif self.winner == 2:
                status = "AI Wins!" if self.game_mode == 'human' else "AI 2 Wins!"
                color = self.YELLOW
            else:
                status = "Draw!"
                color = self.GRAY
        else:
            if self.game_mode == 'human':
                status = "Your Turn" if self.current_player == 1 else "AI Thinking..."
            else:
                status = f"AI {self.current_player} Turn"
            color = self.BLACK
        
        status_text = self.font_medium.render(status, True, color)
        self.screen.blit(status_text, (panel_x + 10, y_offset))
        y_offset += 40
        
        # Buttons
        buttons = [
            ("New Game", self.reset_game),
            ("Toggle Mode", self.toggle_mode),
            ("Quick Train (100)", lambda: self.start_training(100)),
            ("Train (1K)", lambda: self.start_training(1000)),
            ("Deep Train (100K)", lambda: self.start_training(100000)),
            ("Save Model", lambda: self.ai.save_model('connect4_improved_model.pkl'))
        ]
        
        self.button_rects = []
        for i, (text, callback) in enumerate(buttons):
            button_rect = pygame.Rect(panel_x + 10, y_offset, 180, 30)
            self.button_rects.append((button_rect, callback))
            
            color = self.GRAY if self.training else self.GREEN
            pygame.draw.rect(self.screen, color, button_rect)
            pygame.draw.rect(self.screen, self.BLACK, button_rect, 2)
            
            button_text = self.font_small.render(text, True, self.BLACK)
            text_rect = button_text.get_rect(center=button_rect.center)
            self.screen.blit(button_text, text_rect)
            
            y_offset += 35
        
        # Statistics
        y_offset += 15
        stats_title = self.font_medium.render("Statistics", True, self.BLACK)
        self.screen.blit(stats_title, (panel_x + 10, y_offset))
        y_offset += 30
        
        stats_items = [
            f"Games: {self.stats['games_played']}",
            f"Human Wins: {self.stats['human_wins']}",
            f"AI Wins: {self.stats['ai_wins']}",
            f"Draws: {self.stats['draws']}",
            f"Win Rate: {self.stats['ai_wins']/(max(1, self.stats['games_played'] - self.stats['draws'])):.1%}",
            f"AI Exploration: {self.ai.epsilon:.3f}",
            f"Q-Table Size: {len(self.ai.q_table)}",
            f"Mode: {self.game_mode.title()}"
        ]
        
        for item in stats_items:
            stat_text = self.font_small.render(item, True, self.BLACK)
            self.screen.blit(stat_text, (panel_x + 10, y_offset))
            y_offset += 20
        
        # Training progress
        if self.training:
            y_offset += 10
            progress_title = self.font_medium.render("Training", True, self.BLACK)
            self.screen.blit(progress_title, (panel_x + 10, y_offset))
            y_offset += 25
            
            progress_text = f"{self.training_progress}/{self.training_total}"
            progress_render = self.font_small.render(progress_text, True, self.BLACK)
            self.screen.blit(progress_render, (panel_x + 10, y_offset))
            y_offset += 25
            
            # Progress bar
            bar_rect = pygame.Rect(panel_x + 10, y_offset, 180, 20)
            pygame.draw.rect(self.screen, self.GRAY, bar_rect)
            
            if self.training_total > 0:
                progress_width = int(180 * self.training_progress / self.training_total)
                progress_rect = pygame.Rect(panel_x + 10, y_offset, progress_width, 20)
                pygame.draw.rect(self.screen, self.GREEN, progress_rect)
            
            pygame.draw.rect(self.screen, self.BLACK, bar_rect, 2)
            
            # Training speed info
            y_offset += 30
            speed_text = f"Speed: {1000/max(1, self.training_total//100):.0f} games/sec"
            speed_render = self.font_small.render(speed_text, True, self.BLACK)
            self.screen.blit(speed_render, (panel_x + 10, y_offset))
    
    def toggle_mode(self):
        """Toggle between human vs AI and AI vs AI"""
        if not self.training:
            self.game_mode = 'ai_vs_ai' if self.game_mode == 'human' else 'human'
            self.reset_game()
    
    def start_training(self, num_games: int):
        """Start AI training"""
        if not self.training:
            self.training = True
            self.training_progress = 0
            self.training_total = num_games
            self.train_ai(num_games)
    
    def train_ai(self, num_games: int):
        """Train AI by playing games against itself with improved training"""
        original_mode = self.game_mode
        original_board = self.board.copy()
        original_player = self.current_player
        original_game_over = self.game_over
        original_winner = self.winner
        
        self.game_mode = 'ai_vs_ai'
        
        # Create a second AI for more diverse training
        ai2 = Connect4AI(learning_rate=0.2, epsilon=0.7)
        
        for i in range(num_games):
            self.reset_game()
            
            # Play a complete game (no visual updates during training)
            move_count = 0
            while not self.game_over and move_count < 42:  # Prevent infinite games
                if self.current_player == 1:
                    # Use main AI
                    action = self.ai.get_best_action(self.board, self.current_player)
                else:
                    # Use second AI for more diverse opponent
                    action = ai2.get_best_action(self.board, self.current_player)
                
                if action == -1 or not self.make_move(action):
                    break
                
                move_count += 1
            
            # Also train the second AI
            if self.game_states:
                ai2_reward = 50 if self.winner == 1 else (-50 if self.winner == 2 else 5)
                # Simple update for AI2 (player 1 moves)
                for state, action, player in self.game_states:
                    if player == 1:
                        ai2.update_q_value(state, action, ai2_reward, state, True)
                ai2.decay_epsilon()
            
            self.training_progress = i + 1
            
            # Update display only to show progress (not the game)
            if i % max(1, num_games // 50) == 0 or i == num_games - 1:
                # Handle pygame events to prevent freezing
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        return
                
                self.screen.fill(self.WHITE)
                # Draw the original board state, not the training games
                temp_board = self.board.copy()
                temp_player = self.current_player
                temp_game_over = self.game_over
                temp_winner = self.winner
                
                self.board = original_board
                self.current_player = original_player
                self.game_over = original_game_over
                self.winner = original_winner
                
                self.draw_board()
                self.draw_panel()
                pygame.display.flip()
                
                # Restore training state
                self.board = temp_board
                self.current_player = temp_player
                self.game_over = temp_game_over
                self.winner = temp_winner
        
        # Restore original game state
        self.training = False
        self.game_mode = original_mode
        self.board = original_board
        self.current_player = original_player
        self.game_over = original_game_over
        self.winner = original_winner
        
        # Save the trained model
        self.ai.save_model('connect4_improved_model.pkl')
        print(f"Training complete! Q-table now has {len(self.ai.q_table)} entries.")
        print(f"AI exploration rate: {self.ai.epsilon:.3f}")
    
    def handle_click(self, pos: Tuple[int, int]):
        """Handle mouse click"""
        x, y = pos
        
        # Check if click is on the board
        if x < self.BOARD_WIDTH and not self.game_over and not self.training:
            if self.game_mode == 'human' and self.current_player == 1:
                # Calculate column accounting for the left margin
                if x >= self.MARGIN:
                    col = (x - self.MARGIN) // (self.CELL_SIZE + self.MARGIN)
                else:
                    col = -1
                if 0 <= col < self.COLS:
                    if self.make_move(col):
                        # AI move after human move
                        if not self.game_over and self.current_player == 2:
                            pygame.time.wait(500)  # Brief pause before AI move
                            self.make_ai_move()
        
        # Check button clicks
        if not self.training:
            for button_rect, callback in self.button_rects:
                if button_rect.collidepoint(pos):
                    callback()
                    break
    
    def run(self):
        """Main game loop"""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
            
            # AI vs AI mode - automatic moves
            if (self.game_mode == 'ai_vs_ai' and not self.game_over and 
                not self.training):
                pygame.time.wait(1000)  # Pause between moves
                self.make_ai_move()
            
            # Draw everything
            self.screen.fill(self.WHITE)
            self.draw_board()
            self.draw_panel()
            pygame.display.flip()
            
            self.clock.tick(60)
        
        # Save model before quitting
        self.ai.save_model('connect4_improved_model.pkl')
        pygame.quit()


if __name__ == "__main__":
    game = Connect4Game()
    game.run()