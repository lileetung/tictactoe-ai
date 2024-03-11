import numpy as np

class TicTacToeGame:
    def __init__(self):
        self.n = 3
        self.n_in_row = 3

    def get_init_board(self):
        return np.zeros((self.n, self.n), dtype=int)

    def get_board_size(self):
        return (self.n, self.n)

    def get_action_size(self):
        return self.n * self.n

    def get_next_state(self, board, player, action):
        if not isinstance(action, int):
            raise ValueError("Action must be an integer")
        new_board = np.copy(board)
        x, y = divmod(action, self.n)
        if new_board[x, y] == 0:
            new_board[x, y] = player
        return new_board, -player

    def get_valid_moves(self, board):
        return (board == 0).astype(int).flatten()
    
    def is_win(self, board, player):
        for i in range(self.n):
            if np.all(board[i, :] == player) or np.all(board[:, i] == player):
                return True
        if np.all(np.diag(board) == player) or np.all(np.diag(np.fliplr(board)) == player):
            return True
        return False

    def get_reward_for_player(self, board, player):
        """Get the reward for the player based on the current board state."""
        if self.is_win(board, player):
            return 1
        if self.is_win(board, -player):
            return -1
        if not np.any(board == 0):
            return 0
        return None

    def get_canonical_board(self, board, player):
        """Return the board from the perspective of the current player."""
        return player * board

if __name__ == "__main__":
    game = TicTacToeGame()
    board = game.get_init_board()

    print("Initial Board:")
    print(board)

    print("\nBoard Size:", game.get_board_size())

    print("\nAction Size:", game.get_action_size())

    action = 4  # center position
    new_board, next_player = game.get_next_state(board, 1, action)
    print("\nBoard After Action 4 by Player 1:")
    print(new_board)

    print("\nValid Moves after one action:")
    print(game.get_valid_moves(new_board))

    # test is_win
    winning_board = np.array([[1, -1, 0], [-1, 1, 0], [0, 0, 1]])
    print("\nWinning Board for Player 1:")
    print(winning_board)
    print("Is Player 1 Win:", game.is_win(winning_board, 1))

    print("\nReward for Player 1 on winning board:", game.get_reward_for_player(winning_board, 1))
    print("Reward for Player -1 on winning board:", game.get_reward_for_player(winning_board, -1))

    print("\nCanonical Board from Player 1's perspective:")
    print(game.get_canonical_board(new_board, 1))
    print("\nCanonical Board from Player -1's perspective:")
    print(game.get_canonical_board(new_board, -1))
