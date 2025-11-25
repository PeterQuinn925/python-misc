#!/usr/bin/env python3
"""
Mazewar Bot for PDP-10 ITS System

TODO:
Enable more than one bot
Enable shooting the humans
Automate everything - kiosk mode?

"""

import telnetlib
import time
import random

class ANSITerminal:
    """Simple ANSI/VT100 terminal emulator to track screen state"""
    def __init__(self, width=80, height=24):
        self.width = width
        self.height = height
        self.screen = [[' ' for _ in range(width)] for _ in range(height)]
        self.cursor_x = 0
        self.cursor_y = 0
        self.escape_mode = False
        self.escape_buffer = ""
        
    def process_char(self, char):
        """Process a single character, handling ANSI escape sequences"""
        if self.escape_mode:
            self.escape_buffer += char
            
            if char.isalpha() or char == '@':
                self.handle_escape_sequence(self.escape_buffer)
                self.escape_mode = False
                self.escape_buffer = ""
                
        elif char == '\x1b':  # ESC character
            self.escape_mode = True
            self.escape_buffer = ""
        elif char == '\r':  # Carriage return
            self.cursor_x = 0
        elif char == '\n':  # Line feed
            self.cursor_y = min(self.cursor_y + 1, self.height - 1)
        elif char == '\b':  # Backspace
            self.cursor_x = max(0, self.cursor_x - 1)
        elif ord(char) >= 32:  # Printable character
            if self.cursor_y < self.height and self.cursor_x < self.width:
                self.screen[self.cursor_y][self.cursor_x] = char
                self.cursor_x += 1
                if self.cursor_x >= self.width:
                    self.cursor_x = 0
                    self.cursor_y = min(self.cursor_y + 1, self.height - 1)
    
    def handle_escape_sequence(self, seq):
        """Handle ANSI escape sequences"""
        if not seq:
            return
            
        if seq[0] == '[':  # CSI sequence
            seq = seq[1:]
            if not seq:
                return
                
            cmd = seq[-1]
            params = seq[:-1]
            
            if params:
                try:
                    nums = [int(x) if x else 0 for x in params.split(';')]
                except:
                    nums = []
            else:
                nums = []
            
            if cmd == 'H' or cmd == 'f':  # Cursor position
                row = nums[0] - 1 if len(nums) > 0 and nums[0] > 0 else 0
                col = nums[1] - 1 if len(nums) > 1 and nums[1] > 0 else 0
                self.cursor_y = max(0, min(row, self.height - 1))
                self.cursor_x = max(0, min(col, self.width - 1))
            elif cmd == 'J':  # Clear screen
                if not nums or nums[0] == 2:
                    self.screen = [[' ' for _ in range(self.width)] for _ in range(self.height)]
            elif cmd == 'K':  # Clear line
                if not nums or nums[0] == 0:
                    for x in range(self.cursor_x, self.width):
                        self.screen[self.cursor_y][x] = ' '
    
    def process_string(self, data):
        """Process a string of data"""
        for char in data:
            self.process_char(char)
    
    def get_screen_text(self):
        """Get current screen as text"""
        return '\n'.join(''.join(row) for row in self.screen)


class MazewarBot:
    # Deterministic spawn positions by player join order
    # Note: Positions are in PROTOCOL coordinates (1-indexed)
    # Maze coordinates are 0-indexed, so subtract 1
    SPAWN_POSITIONS = {
        1: (10, 3, 3),   # Second player: protocol (10,3) = maze (9,2), facing 3=West
        2: None,
        3: None,
    }
    
    def __init__(self, host, port, username="BOT", game_dir="GAMES", verbose=True, player_number=1):
        self.host = host
        self.port = port
        self.username = username
        self.game_dir = game_dir
        self.tn = None
        self.terminal = ANSITerminal(80, 24)
        self.verbose = verbose
        self.player_number = player_number  # Which player are we? (0=first, 1=second, etc.)
        self.player_id = player_number + 1  # Player ID (1-based)
        self.game_data = "" # responses back from the PDP-10 during game play
        
        # Maze state - actual maze is 16 columns x 32 rows (per assembly code)
        # Assembly: AND [17'] for X = 0-15, AND [37'] for Y = 0-31
        self.maze_map = None  # 2D array of the maze
        self.maze_width = 16   # Actual playable maze width (assembly uses 0-15)
        self.maze_height = 32
        
        # Player state - will be set to known spawn position
        self.pos_x = 0
        self.pos_y = 0
        self.facing = 0  # 0=North, 1=East, 2=South, 3=West
        
        # Navigation
        self.visited = set()
        self.move_history = []
        self.loop_counter = 0 #used to determine if we're in a loop
        
    def log(self, message, level="INFO"):
        """Print log message if verbose mode enabled"""
        if self.verbose:
            print(f"[{level}] {message}")
        
    def connect(self):
        """Establish telnet connection to PDP-10"""
        self.log(f"Connecting to {self.host}:{self.port}...")
        self.tn = telnetlib.Telnet(self.host, self.port)
        
        time.sleep(1.5)
        response = self.wait_for_response(2.0)
        self.log(f"Connection established", "DEBUG")
        
        # Send Control-Z to get ITS attention
        self.tn.write(b'\x1a')
        time.sleep(1.0)
        
        response = self.wait_for_response(3.0)
        self.log(f"Connected successfully")
        
    def login(self):
        """Login to ITS system"""
        self.log(f"Logging in as {self.username}...")
        
        self.send_command(f":LOGIN {self.username}")
        response = self.wait_for_response(3.0)
        if response:
            self.terminal.process_string(response)
        
        time.sleep(0.5)
        response = self.wait_for_response(1.0)
        if response:
            self.terminal.process_string(response)
        
        self.log("Login complete")
        
    def send_command(self, command):
        """Send a command followed by carriage return"""
        self.tn.write(command.encode('latin-1') + b'\r')
        if self.verbose:
            self.log(f"Sent command: {command}", "DEBUG")
        
    def send_key(self, key):
        """Send a single keypress"""
        if isinstance(key, str):
            self.tn.write(key.encode('latin-1'))
        else:
            self.tn.write(bytes([key]))
        
    def read_and_update_screen(self):
        """Read from connection and update terminal emulator"""
        try:
            data = self.tn.read_very_eager().decode('latin-1', errors='ignore')
            if data:
                #self.terminal.process_string(data)
                self.game_data = data
#                print(self.game_data)
                return True
        except Exception as e:
            self.log(f"Error reading screen: {e}", "ERROR")
            return False
    
    def wait_for_response(self, timeout=2.0):
        """Wait for a response with timeout"""
        time.sleep(0.2)
        start = time.time()
        response = ""
        last_data_time = time.time()
        
        while time.time() - start < timeout:
            try:
                chunk = self.tn.read_very_eager().decode('latin-1', errors='ignore')
                if chunk:
                    response += chunk
                    last_data_time = time.time()
                else:
                    idle_time = time.time() - last_data_time
                    if response and idle_time > 0.3:
                        break
                time.sleep(0.05)
            except:
                break
        return response
    
    def parse_maze_from_binary(self, data):
        """Parse the maze from the binary loader data
        
        The maze is 32 octal words (6 digits each = 18 bits).
        Each word represents one row of the maze.
        However, only the first 16 bits are used for the playable area (columns 0-15).
        The last 2 bits (columns 16-17) are borders.
        1 = wall, 0 = passage
        """
        self.log("Parsing maze from binary data...", "DEBUG")
        
        # The default maze from the assembly listing  
        # These are 32 octal words from the assembly code
        default_maze_octal = [
            0o177777, 0o106401, 0o124675, 0o121205, 0o132055, 0o122741, 0o106415, 0o124161,
            0o121405, 0o135775, 0o101005, 0o135365, 0o121205, 0o127261, 0o120205, 0o106765,
            0o124405, 0o166575, 0o122005, 0o107735, 0o120001, 0o135575, 0o105005, 0o125365,
            0o125225, 0o121265, 0o105005, 0o135375, 0o100201, 0o135675, 0o110041, 0o177777
        ]
        
# Use the default maze
        self.log("Using default maze from assembly listing", "INFO")
        maze_words = default_maze_octal
        
        # Convert the maze words to a 2D map
        # Only use the first 16 bits (columns 0-15) for the playable maze
        self.maze_map = []

        for row_idx, word in enumerate(maze_words):
            str_word = bin(word)[2:] #strings are easier to manipulate than bits
            row = []
            for bit_pos in range(16):
                if str_word[bit_pos]=="1": # Store True = passable, False = wall
                    row.append(False)
                else:
                    row.append(True)
            self.maze_map.append(row)

        # Print ASCII representation
        self.log(f"\n=== MAZE MAP (16x32) ===", "INFO")
        for y, row in enumerate(self.maze_map):
            line = f"{y:2d}: "
            for x, passable in enumerate(row):
                if passable:
                    line += " "
                else:
                    line += "#"
            self.log(line, "INFO")
        self.log("=== END MAZE MAP ===\n", "INFO")
        
        
        # Set spawn position based on player number
        if self.player_number in self.SPAWN_POSITIONS and self.SPAWN_POSITIONS[self.player_number] is not None:
            spawn_info = self.SPAWN_POSITIONS[self.player_number]
            protocol_x, protocol_y, self.facing = spawn_info
            # Convert from protocol coordinates (1-indexed) to maze coordinates (0-indexed)
            #self.pos_x = protocol_x - 1
            #self.pos_y = protocol_y - 1
            self.pos_x = protocol_x
            self.pos_y = protocol_y
            self.log(f"Using known spawn position for player {self.player_number}: protocol({protocol_x}, {protocol_y}) = maze({self.pos_x}, {self.pos_y}) facing {['N','E','S','W'][self.facing]}", "INFO")
        else:
            # Find a starting position
            self.log(f"Unknown spawn position for player {self.player_number}, searching...", "WARN")
            for y in range(1, 31):
                for x in range(1, 15):
                    if self.maze_map[y][x]:
                        self.pos_x = x
                        self.pos_y = y
                        self.facing = 0
                        self.log(f"Using fallback starting position: ({x}, {y})", "INFO")
                        break
                else:
                    continue
                break
        
        # Verify the spawn position is passable
        if self.pos_x >= 16 or self.pos_y >= 32:
            self.log(f"ERROR: Spawn position ({self.pos_x}, {self.pos_y}) is out of bounds!", "ERROR")
            return False
        
        if not self.maze_map[self.pos_y][self.pos_x]:
            self.log(f"ERROR: Spawn position ({self.pos_x}, {self.pos_y}) is a wall!", "ERROR")
            return False
        
        self.log(f"Starting position: ({self.pos_x}, {self.pos_y}) facing {['North','East','South','West'][self.facing]}", "INFO")
        return True
    
    def start_mazewar(self):
        """Start the Mazewar game and parse the maze"""
        self.log(f"Starting Mazewar from {self.game_dir}...")
        
        self.send_command(":TCTYPE OIMLAC")
        time.sleep(0.5)
        self.wait_for_response(1.0)
        
        # Change directory
        self.send_command(f":CWD {self.game_dir}")
        time.sleep(0.5)
        self.wait_for_response(1.0)
        
        # Launch game
        self.log("Launching Mazewar...", "DEBUG")
        self.send_command(":RUN MAZE C")
        
        # Wait for game to load and capture all data
        all_response = ""
        for attempt in range(15):
            time.sleep(1.0)
            chunk = self.wait_for_response(1.0)
            if chunk:
                all_response += chunk
                self.terminal.process_string(chunk)
                
                if "blk ldr" in chunk:
                    self.log("Game loaded!", "DEBUG")
                    time.sleep(1.0)
                    final = self.wait_for_response(2.0)
                    if final:
                        all_response += final
                        self.terminal.process_string(final)
                    break
        
        # Parse the maze (will use default if can't extract from data)
        if not self.parse_maze_from_binary(all_response):
            self.log("Failed to parse maze!", "ERROR")
            return False
        
        # Handle join prompts if needed
        if "Hang on" in all_response or "joining" in all_response.lower():
            self.log("Joining game...", "DEBUG")
            time.sleep(1)
            #self.send_key('\r')
            time.sleep(2.0)
            self.wait_for_response(3.0)
        
        # Send protocol messages to announce position
        # This makes the bot appear in the game at the spawn position
        self.log("Sending spawn position protocol messages...", "DEBUG")
        self.send_spawn_message()
        
        time.sleep(0.5)
        self.wait_for_response(1.0)
        
        self.log("Mazewar startup complete")
        return True
    
    def send_spawn_message(self):
        """Send Type 2 (PLAYER MOVED) message to announce spawn position
        
        """
        # Type 2: Player Moved
        self.tn.write(b'\x02')
        time.sleep(1)
        #self.tn.write(b'\x02')
        time.sleep(1)
        # Player ID

        self.tn.write(bytes([self.player_id]))
        time.sleep(1)
        
        # Direction with 0x40 added
        direction_byte = self.facing + 0x40
        self.tn.write(bytes([direction_byte]))
        time.sleep(1)
        
        # X location: convert from maze coords (0-indexed) to protocol coords (1-indexed)
        protocol_x = self.pos_x
        x_byte = protocol_x + 0x40
        self.tn.write(bytes([x_byte]))
        time.sleep(1)
        
        # Y location: convert from maze coords (0-indexed) to protocol coords (1-indexed)
        protocol_y = self.pos_y
        y_byte = protocol_y + 0x40
        self.tn.write(bytes([y_byte]))
        time.sleep(1)
        
        # End marker
        self.tn.write(b'\x11')
        self.tn.write(b'\x19')
        time.sleep(10)
        
        self.log(f"Sent spawn message: ID={self.player_id}, Dir={self.facing}, Protocol=({protocol_x},{protocol_y}), Maze=({self.pos_x},{self.pos_y})", "DEBUG")
    
    def can_move_forward(self):
        """Check if we can move forward from current position"""
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][self.facing]
        new_x = self.pos_x + dx
        new_y = self.pos_y + dy
        
        # Check bounds (16x32 maze)
        if new_x < 0 or new_x >= 16 or new_y < 0 or new_y >= 32:
            return False
        
        return self.maze_map[new_y][new_x]
    
    def move_forward(self):
        """Move forward (update internal state)"""
        # Send move command FIRST: up arrow (206 octal = 0x86 = 134 decimal)
        # I don't think that's right. wireshark indicates \x69 or ASCII i
        #self.send_key(0o206)  # Up arrow - octal 206
        self.send_key("i")
        time.sleep(0.3)
        
        # Then update our internal state
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][self.facing]
        self.pos_x += dx
        self.pos_y += dy
        if (self.pos_x, self.pos_y) in self.visited:
            #already been here. Add to the consectutive visited spaces counter
            self.loop_counter = self.loop_counter +1
        else:
            #haven't been here before, reset loop counter to 0
            self.loop_counter = 0
        self.visited.add((self.pos_x, self.pos_y))
        self.move_history.append('F')
        self.log(f"Moved to ({self.pos_x}, {self.pos_y})", "DEBUG")
    
    def turn_left(self):
        """Turn left (update internal state)"""
        # Send left turn FIRST: left arrow (210 octal = 0x88 = 136 decimal)
        #wireshark indicates 19 or 11 hex. guessing 19
        #self.send_key(0o210)  # Left arrow - octal 210
        self.send_key("\x19")
        time.sleep(0.3)
        
        # Then update internal state
        self.facing = (self.facing - 1) % 4
        self.move_history.append('L')
        direction = ['North', 'East', 'South', 'West'][self.facing]
        self.log(f"Turned left, now facing {direction}", "DEBUG")
    
    def turn_right(self):
        """Turn right (update internal state)"""
        # Send right turn FIRST: right arrow (205 octal = 0x85 = 133 decimal)
        #self.send_key(0o205)  # Right arrow - octal 205
        self.send_key("\x11")
        time.sleep(0.3)
        
        # Then update internal state
        self.facing = (self.facing + 1) % 4
        self.move_history.append('R')
        direction = ['North', 'East', 'South', 'West'][self.facing]
        self.log(f"Turned right, now facing {direction}", "DEBUG")
    
    def explore_maze(self):
        """Simple right-hand wall following exploration"""
        # Try to move according to right-hand rule:
        # 1. Try to turn right and move
        # 2. If can't, try to move forward
        # 3. If can't, try to turn left
        # 4. If can't, turn around
        # except if the loop_counter says we've gone over the same cells already n times
        if self.loop_counter < 5:
            # First check right
            self.turn_right()
            if self.can_move_forward():
                self.move_forward()
                return "turned right and moved forward"
              
            # Can't go right, try straight
            self.turn_left()  # Undo the right turn
            if self.can_move_forward():
                self.move_forward()
                return "moved forward"
                
            # Can't go straight, try left
            self.turn_left()
            if self.can_move_forward():
                self.move_forward()
                return "turned left and moved forward"
                
            # Can't go left, turn around
            self.turn_left()
            if self.can_move_forward():
                self.move_forward()
                return "turned around and moved forward"
                
            # Completely stuck (shouldn't happen in valid maze)
            self.log("WARNING: Stuck in all directions!", "WARN")
            return "stuck"
        else:
            self.loop_counter = 0 # reset the loop counter
            # this time we're preferentially going forward or turning left
            # try straight
            if self.can_move_forward():
                self.move_forward()
                return "moved forward"
            # Can't go straight, try left
            self.turn_left()
            if self.can_move_forward():
                self.move_forward()
                return "turned left and moved forward"
            # check right
            self.turn_right() # undo left
            self.turn_right()
            if self.can_move_forward():
                self.move_forward()
                return "turned right and moved forward"
            # Can't go left, turn around
            self.turn_right() # second right turn to make a U
            if self.can_move_forward():
                self.move_forward()
                return "turned around and moved forward"
                
            # Completely stuck (shouldn't happen in valid maze)
            self.log("WARNING: Stuck in all directions!", "WARN")
            return "stuck"
            
    def play(self, duration=300):
        """Main game loop with navigation"""
        self.log(f"Starting autonomous play for {duration} seconds...")
        
        if self.maze_map is None:
            self.log("ERROR: No maze map loaded!", "ERROR")
            return
        
        start_time = time.time()
        move_count = 0
        
        # Mark starting position as visited
        self.visited.add((self.pos_x, self.pos_y))
        
        try:
            while time.time() - start_time < duration:
                # Read any data from server
                self.read_and_update_screen()
                # Have I been shot?
                if len(self.game_data) > 1 and self.game_data[0] == "\x03": #should be shot
                    self.log("Oh no! I've been shot!","DEBUG")
                    #go back to the starting point and restart everything
                    spawn_info = self.SPAWN_POSITIONS[self.player_number]
                    protocol_x, protocol_y, self.facing = spawn_info
                    self.pos_x = protocol_x
                    self.pos_y = protocol_y
                    self.game_data = ""
                    self.visited.clear # reset the visited spaces
                    self.send_spawn_message() #go back to the original position
                else:
                    # Make a move
                    action = self.explore_maze()
                    move_count += 1
                
                    self.log(f"Move {move_count}: {action} at ({self.pos_x},{self.pos_y}) facing {['N','E','S','W'][self.facing]}")
                
                    # Small delay between moves
                    time.sleep(0.5)
                    
        except KeyboardInterrupt:
            self.log("\nBot stopped by user")
        except Exception as e:
            self.log(f"\nBot crashed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
        
        # Show stats
        self.log(f"\n=== Navigation Stats ===")
        self.log(f"Total moves: {move_count}")
        self.log(f"Cells visited: {len(self.visited)}")
        self.log(f"Current position: ({self.pos_x}, {self.pos_y})")
        self.log(f"Facing: {['North', 'East', 'South', 'West'][self.facing]}")
            
        # Graceful exit
        self.graceful_exit()
        
        self.log(f"\nBot finished.")
    
    def graceful_exit(self):
        """Gracefully exit the game and ITS system"""
        self.log("Performing graceful exit...", "DEBUG")
        try:
            # Send Control-Z and Bell as in original
            self.tn.write(b'\x1a')  # Control-Z
            time.sleep(1)
            self.tn.write(b'\x07')  # Bell/beep (Control-G)
            time.sleep(1)
            self.send_command(":KILL")
            time.sleep(0.5)
            self.send_command(":LOGOUT")
            time.sleep(0.5)
        except Exception as e:
            self.log(f"Error during graceful exit: {e}", "WARN")
        
    def disconnect(self):
        """Close telnet connection"""
        if self.tn:
            self.tn.close()
            self.log("Disconnected")


def main():
    """Example usage"""
    botnum = random.randint(1, 100)
    HOST = "localhost"
    PORT = 10003
    USERNAME = "BOT" + str(botnum)
    PLAYER_NUMBER = 1  # Bot is typically the second player (index 1)
    
    print("=== Mazewar PDP-10 Bot ===\n")
    
    bot = MazewarBot(HOST, PORT, USERNAME, player_number=PLAYER_NUMBER)
    
    try:
        bot.connect()
        bot.login()
        if bot.start_mazewar():
            bot.play(duration=600)
        else:
            print("Failed to start Mazewar!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        bot.disconnect()


if __name__ == "__main__":
    main()
