import sys
import numpy as np
import pygame as pg
import torch
from model import TicTacToeModel

def AI(env):
    model = TicTacToeModel()
    try:
        model.load_state_dict(torch.load("model.pth"))
    except:
        pass
    model.eval()
    action_probs, value = model.predict(env.board)
    valid_moves = (env.board == 0).astype(int).flatten()
    action_probs = action_probs * valid_moves
    action_probs_sum = np.sum(action_probs)

    if action_probs_sum > 0:
        action_probs /= action_probs_sum
    else:
        action_probs = np.ones(len(action_probs)) / len(action_probs)

    max_index = np.argmax(action_probs)
    row, col = divmod(max_index, 3)
    action = (row, col)
    return action

class Environment():
    def __init__(self):
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
        self.currentPlayer = np.random.choice([self.O, self.X])
        pg.init()
        self.font = pg.font.SysFont('arial', 24, bold=True)
        self.mode = None  # 'Human' or 'AI'

    def draw_buttons(self):
        """
        Draws redesigned buttons for selecting between Human and AI mode with lower contrast,
        simpler lines, and positions them evenly across the screen for balance.
        """
        self.screen.fill((240, 240, 240)) 
        title = self.font.render('Tic-Tac-Toe', True, (0, 0, 0))
        self.screen.blit(title, (self.width / 2 - title.get_width() / 2, self.height / 4))
        
        # Define button colors with lower contrast
        button_color_human = (180, 180, 255)
        button_color_ai = (255, 180, 180)
        text_color = (50, 50, 50)

        # Calculate positions for an evenly distributed layout
        button_width = 100
        button_height = 50
        gap = (self.width - 2 * button_width) / 3 
        human_btn_x = gap
        ai_btn_x = gap * 2 + button_width 

        human_btn = pg.Rect(human_btn_x, 150, button_width, button_height)
        ai_btn = pg.Rect(ai_btn_x, 150, button_width, button_height)

        # Draw buttons
        pg.draw.rect(self.screen, button_color_human, human_btn, border_radius=5)
        pg.draw.rect(self.screen, button_color_ai, ai_btn, border_radius=5)

        # Render and place text on buttons
        human_text = self.font.render('Human', True, text_color)
        ai_text = self.font.render('AI', True, text_color)
        self.screen.blit(human_text, human_text.get_rect(center=human_btn.center))
        self.screen.blit(ai_text, ai_text.get_rect(center=ai_btn.center))

        pg.display.flip()
        return human_btn, ai_btn

    def select_mode(self):
        """
        Handles the mode selection by the user.
        """
        human_btn, ai_btn = self.draw_buttons()
        while self.mode is None:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()
                if event.type == pg.MOUSEBUTTONDOWN:
                    if human_btn.collidepoint(pg.mouse.get_pos()):
                        self.mode = 'Human'
                    elif ai_btn.collidepoint(pg.mouse.get_pos()):
                        self.mode = 'AI'
        self.screen.fill((255, 255, 255))  # Clear the screen after selection

    def drawScreen(self):
        """
        Draws the game board, the X and O markers, and the game result if there is one.
        Updates the display with any changes made to the game state.
        """
        # Define color constants
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        BLUE = (0, 0, 255)

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
            winner_text = "O Wins!" if winner == self.O else "X Wins!" if winner == self.X else "Draw!" 
            text_surface = self.font.render(winner_text, True, WHITE)
            text_rect = text_surface.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(message_background, background_rect)
            self.screen.blit(text_surface, text_rect)
        pg.display.update()

    def step(self, action):
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
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        self.currentPlayer = np.random.choice([self.O, self.X])
        self.gameOver = False
        return self.board

    def check_winner(self):
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

    
if __name__ == '__main__':
    env = Environment()
    env.select_mode()
    while True:
        env.drawScreen()
        if env.gameOver:
            pg.time.wait(2000)
            env.reset()
            env.select_mode()
            continue
        if env.mode == 'AI' and env.currentPlayer == env.O:  # Assuming AI plays as 'O'
            action = AI(env)
            if action:
                env.step(action)
        else: # env.mode == 'Human' 
            action = env.get_user_action()
            if action:
                env.step(action)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()