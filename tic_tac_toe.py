# tic_tac_toe.py
import numpy as np

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the game board."""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_winner = None
        return self.board.flatten()

    def available_moves(self):
        """Return a list of available moves (empty spots)."""
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def make_move(self, position, player):
        """Place a marker on the board."""
        if self.board[position] == 0:
            self.board[position] = player
            if self.check_win(player):
                self.current_winner = player
            return True
        return False

    def check_win(self, player):
        """Check if the player has won."""
        # Check rows, columns, and diagonals
        for i in range(3):
            if all(self.board[i, :] == player):
                return True
            if all(self.board[:, i] == player):
                return True
        if all([self.board[i, i] == player for i in range(3)]) or \
           all([self.board[i, 2 - i] == player for i in range(3)]):
            return True
        return False

    def is_draw(self):
        """Check if the game is a draw."""
        return len(self.available_moves()) == 0 and self.current_winner is None

    def game_over(self):
        """Check if the game is over."""
        return self.current_winner is not None or self.is_draw()

    def get_state(self):
        """Get the current state of the board."""
        return tuple(self.board.flatten())

    def render(self):
        """Print the board."""
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        print('\n' + '\n-----\n'.join(['|'.join([symbols[self.board[i, j]] for j in range(3)]) for i in range(3)]) + '\n')
