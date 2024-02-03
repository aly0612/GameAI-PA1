"""
@Author: 	Austin Ly
@Date:		2024-02-02
@Description: CS 4900 GameAI PA1, Running RL using stable-baselines3 and WandB on Connect4 Game.
Connect4Env is a custom environment for the Connect4 game. It is a subclass of the gym.Env class.
Connect4Env comes from https://www.askpython.com/python/examples/connect-four-game
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Connect4Env(gym.Env):
    
    metadata = {'render.modes': ['console']}
    def __init__(self):
        super(Connect4Env, self).__init__()
        self.row_count = 6
        self.column_count = 7
        self.action_space = spaces.Discrete(self.column_count)
        self.observation_space = spaces.Box(low=0, high=2,
                                             shape=(self.row_count, self.column_count), dtype=np.int)
        self.board = None
        self.current_player = 1
        self.done = False
        self.reset()
    
    def reset(self, **kwargs):  # Accept arbitrary keyword arguments, not sure why??
        self.board = np.zeros((self.row_count, self.column_count), dtype=np.int)
        self.current_player = 1
        self.done = False
        return self.board, {}  # Return the board and an empty info dict

    
    def step(self, action):
        if not self.is_valid_location(action):
            return self.board, -1, True, False, {"reason": "invalid move"}
    
        row = self.get_next_open_row(action)
        self.drop_piece(row, action, self.current_player)

        if self.winning_move(self.current_player):
            reward = 1  # Win
            self.done = True
        elif np.all(self.board != 0):  # Draw
            reward = 0.5  
            self.done = True
        else:
            reward = 0  # Game continues
            self.current_player = 3 - self.current_player
            self.done = False

    # Ensure to include 'truncated' flag in the return, set to False
        return self.board, reward, self.done, False, {}


    def render(self, mode='console'):
        if mode == 'console':
            print(np.flip(self.board, 0))

    def close(self):
        pass

    # Adapted from game script
    def is_valid_location(self, col):
        return self.board[self.row_count-1][col] == 0

    def get_next_open_row(self, col):
        for r in range(self.row_count):
            if self.board[r][col] == 0:
                return r

    def drop_piece(self, row, col, piece):
        self.board[row][col] = piece

    def winning_move(self, piece):
        # Horizontal
        for c in range(self.column_count-3):
            for r in range(self.row_count):
                if self.board[r][c] == piece and all(self.board[r][c+i] == piece for i in range(1, 4)):
                    return True
        # Vertical
        for c in range(self.column_count):
            for r in range(self.row_count-3):
                if self.board[r][c] == piece and all(self.board[r+i][c] == piece for i in range(1, 4)):
                    return True
        # Positive diagonal
        for c in range(self.column_count-3):
            for r in range(self.row_count-3):
                if self.board[r][c] == piece and all(self.board[r+i][c+i] == piece for i in range(1, 4)):
                    return True
        # Negative diagonal
        for c in range(self.column_count-3):
            for r in range(3, self.row_count):
                if self.board[r][c] == piece and all(self.board[r-i][c+i] == piece for i in range(1, 4)):
                    return True
        return False
