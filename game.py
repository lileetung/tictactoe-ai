import numpy as np
import unittest

class Board():
    def __init__(self, n):
        self.n = n
        self.board = np.zeros((n, n), dtype=int)

    def __getitem__(self, index):
        return self.board[index]

    def get_legal_moves(self):
        return np.argwhere(self.board == 0).tolist()

    def has_legal_moves(self):
        return np.any(self.board == 0)

    def execute_move(self, move, player):
        (x, y) = move
        assert self.board[x, y] == 0
        self.board[x, y] = player

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
        x, y = divmod(action, self.n) # 将一个一维的动作索引action转换成二维的棋盘坐标x和y，(1, 2) = divmod(5, 3)
        if board[x, y] == 0:
            board[x, y] = player
        return board, -player

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
    # 定義遊戲和棋盤
    game = TicTacToeGame()
    board = game.get_init_board()

    # 測試 get_init_board
    print("Initial Board:")
    print(board)

    # 測試 get_board_size
    print("\nBoard Size:", game.get_board_size())

    # 測試 get_action_size
    print("\nAction Size:", game.get_action_size())

    # 測試執行動作
    action = 4  # 選擇中心位置
    new_board, next_player = game.get_next_state(board, 1, action)
    print("\nBoard After Action 4 by Player 1:")
    print(new_board)

    # 測試 get_valid_moves
    print("\nValid Moves after one action:")
    print(game.get_valid_moves(new_board))

    # 測試 is_win
    # 為了測試勝利條件，我們需要設定一個勝利的棋盤狀態
    winning_board = np.array([[1, -1, 0], [-1, 1, 0], [0, 0, 1]])
    print("\nWinning Board for Player 1:")
    print(winning_board)
    print("Is Player 1 Win:", game.is_win(winning_board, 1))

    # 測試 get_reward_for_player
    print("\nReward for Player 1 on winning board:", game.get_reward_for_player(winning_board, 1))
    print("Reward for Player -1 on winning board:", game.get_reward_for_player(winning_board, -1))

    # 測試 get_canonical_board
    print("\nCanonical Board from Player 1's perspective:")
    print(game.get_canonical_board(new_board, 1))
    print("\nCanonical Board from Player -1's perspective:")
    print(game.get_canonical_board(new_board, -1))

