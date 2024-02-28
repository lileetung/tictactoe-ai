import numpy as np
import pygame as pg
import sys

class Environment():
    def __init__(self):
        """
        Initializes the game environment with default settings.
        Sets up the game board, window size, and other necessary initial configurations.
        """
        self.width = 300
        self.height = 300
        self.boardSize = 3
        self.rows = self.columns = self.boardSize
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        self.screen = pg.display.set_mode((self.width, self.height))
        self.gameOver = False
        self.cellSize = self.width // self.boardSize
        self.X = 1
        self.O = -1
        self.currentPlayer = self.X
        pg.init()
        self.font = pg.font.SysFont('arial', 24, bold=True)

    def drawScreen(self):
        """
        Draws the game board, the X and O markers, and the game result if there is one.
        Updates the display with any changes made to the game state.
        """
        # Define color constants
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        BLUE = (0, 0, 255)
        RED = (255, 0, 0)

        self.screen.fill(WHITE)
        for x in range(1, self.boardSize):
            pg.draw.line(self.screen, BLACK, (x * self.cellSize, 0), (x * self.cellSize, self.height), 2)
            pg.draw.line(self.screen, BLACK, (0, x * self.cellSize), (self.width, x * self.cellSize), 2)
        for row in range(self.rows):
            for col in range(self.columns):
                if self.board[row][col] == self.X:
                    pg.draw.line(self.screen, BLUE, (col * self.cellSize + 10, row * self.cellSize + 10), ((col + 1) * self.cellSize - 10, (row + 1) * self.cellSize - 10), 3)
                    pg.draw.line(self.screen, BLUE, ((col + 1) * self.cellSize - 10, row * self.cellSize + 10), (col * self.cellSize + 10, (row + 1) * self.cellSize - 10), 3)
                elif self.board[row][col] == self.O:
                    pg.draw.circle(self.screen, BLACK, (int(col * self.cellSize + self.cellSize // 2), int(row * self.cellSize + self.cellSize // 2)), self.cellSize // 2 - 5, 3)
        winner = self.check_winner()
        if winner is not None:
            self.gameOver = True
            message_background = pg.Surface((self.width, 50))
            message_background.set_alpha(128)
            message_background.fill(BLACK)
            background_rect = message_background.get_rect(center=(self.width / 2, self.height / 2))
            winner_text = "Draw!" if winner == 0 else "X Wins!" if winner == self.X else "O Wins!"
            text_surface = self.font.render(winner_text, True, WHITE)
            text_rect = text_surface.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(message_background, background_rect)
            self.screen.blit(text_surface, text_rect)
        pg.display.update()

    def check_winner(self):
        """
        Checks the current state of the board for a winner or a draw.
        
        Returns:
            self.X (1) if player X wins,
            self.O (-1) if player O wins,
            0 if the game is a draw,
            None if the game is ongoing.
        """
        for i in range(self.boardSize):
            if np.all(self.board[i, :] == self.X) or np.all(self.board[:, i] == self.X):
                return self.X
            if np.all(self.board[i, :] == self.O) or np.all(self.board[:, i] == self.O):
                return self.O
        if np.all(np.diag(self.board) == self.X) or np.all(np.diag(np.fliplr(self.board)) == self.X):
            return self.X
        if np.all(np.diag(self.board) == self.O) or np.all(np.diag(np.fliplr(self.board)) == self.O):
            return self.O
        if not np.any(self.board == 0):
            return 0
        return None

    def get_user_action(self):
        """
        Handles user input for placing a marker on the board.
        Updates the board based on user clicks.
        
        Returns:
            A tuple (clicked_row, clicked_col) indicating the row and column where the player placed their marker,
            None if no valid action was taken (e.g., clicking outside the board or on an already filled cell).
        """
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if event.type == pg.MOUSEBUTTONDOWN and not self.gameOver:
                mouseX, mouseY = pg.mouse.get_pos()
                clicked_row = mouseY // self.cellSize
                clicked_col = mouseX // self.cellSize
                if self.board[clicked_row, clicked_col] == 0:
                    self.board[clicked_row, clicked_col] = self.currentPlayer
                    self.gameOver = self.check_winner() is not None
                    self.currentPlayer *= -1
                    return clicked_row, clicked_col
        return None

    def step(self, action):
        """
        Processes a player's action and updates the game state accordingly.
        
        Parameters:
            action: A tuple containing the row and column where the player wants to place their marker.
        
        Returns:
            self.board: The current state of the game board after the action is taken.
            winner: The result of the check_winner function after the action is taken.
            True if the game is over (either a win or a draw), False otherwise.
        """
        row, col = action
        if self.board[row, col] == 0 and not self.gameOver:
            self.board[row, col] = self.currentPlayer
            winner = self.check_winner()
            self.gameOver = winner is not None
            if self.gameOver:
                return self.board, winner, True
            self.currentPlayer *= -1
            return self.board, 0, False
        return self.board, -1, False

    def reset(self):
        """
        Resets the game to its initial state for a new game.
        
        Returns:
            self.board: The reset game board with all cells set to 0.
        """
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        self.currentPlayer = 1
        self.gameOver = False
        return self.board

if __name__ == '__main__':
    game = Environment()
    while True:
        game.drawScreen()
        if game.gameOver:
            pg.time.wait(2000)
            game.reset()
            continue
        action = game.get_user_action()
        if action:
            game.step(action)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
