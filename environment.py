import pygame
import sys
import time

# 初始化Pygame
pygame.init()

# 定义一个函数来显示文本
def display_message(message):
    # 消息背景设置
    message_background = pygame.Surface((WIDTH, 50))  # 创建一个表面来作为消息的背景
    message_background.set_alpha(128)  # 设置透明度
    message_background.fill((0, 0, 0))  # 设置背景颜色为黑色
    background_rect = message_background.get_rect(center=(WIDTH / 2, HEIGHT / 2))

    # 文本设置
    FONT = pygame.font.SysFont('arial', 24, bold=True)  # 使用更专业的字体和大小
    text = FONT.render(message, True, WHITE)  # 使用白色字体增强可读性
    text_rect = text.get_rect(center=(WIDTH / 2, HEIGHT / 2))

    # 先绘制背景再绘制文本
    WIN.blit(message_background, background_rect)  # 绘制消息背景到屏幕
    WIN.blit(text, text_rect)  # 绘制文本到屏幕上
    pygame.display.update()  # 更新屏幕显示

# 设置游戏窗口大小
WIDTH, HEIGHT = 300, 300
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic Tac Toe")

# 定义颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

# 定义游戏状态
PLAYING = True
DRAW = False
X_WIN = "X"
O_WIN = "O"

# 定义游戏棋盘
BOARD_SIZE = 3
SQUARE_SIZE = WIDTH // BOARD_SIZE
board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

# 绘制棋盘网格
def draw_grid():
    for i in range(1, BOARD_SIZE):
        pygame.draw.line(WIN, BLACK, (0, i * SQUARE_SIZE), (WIDTH, i * SQUARE_SIZE))
        pygame.draw.line(WIN, BLACK, (i * SQUARE_SIZE, 0), (i * SQUARE_SIZE, HEIGHT))

# 绘制棋子
def draw_pieces():
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == 'X':
                pygame.draw.line(WIN, BLUE, (col * SQUARE_SIZE + 10, row * SQUARE_SIZE + 10),
                                 ((col + 1) * SQUARE_SIZE - 10, (row + 1) * SQUARE_SIZE - 10), 3)
                pygame.draw.line(WIN, BLUE, ((col + 1) * SQUARE_SIZE - 10, row * SQUARE_SIZE + 10),
                                 (col * SQUARE_SIZE + 10, (row + 1) * SQUARE_SIZE - 10), 3)
            elif board[row][col] == 'O':
                pygame.draw.circle(WIN, BLACK, (col * SQUARE_SIZE + SQUARE_SIZE // 2,
                                                row * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 2 - 5, 3)


# 检查游戏是否结束
def check_winner():
    # 检查行
    for row in range(BOARD_SIZE):
        if board[row][0] == board[row][1] == board[row][2] != ' ':
            return board[row][0]

    # 检查列
    for col in range(BOARD_SIZE):
        if board[0][col] == board[1][col] == board[2][col] != ' ':
            return board[0][col]

    # 检查对角线
    if board[0][0] == board[1][1] == board[2][2] != ' ':
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != ' ':
        return board[0][2]

    # 检查平局
    if all(board[row][col] != ' ' for row in range(BOARD_SIZE) for col in range(BOARD_SIZE)):
        return DRAW

    return None

# 重置游戏
def reset_game():
    global board
    board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

# 主游戏循环
def main():
    global PLAYING
    currentPlayer = 'X'
    game_over = False

    while PLAYING:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not game_over:
                mouseX, mouseY = pygame.mouse.get_pos()
                clickedRow = mouseY // SQUARE_SIZE
                clickedCol = mouseX // SQUARE_SIZE

                if board[clickedRow][clickedCol] == ' ':
                    board[clickedRow][clickedCol] = currentPlayer
                    winner = check_winner()
                    if winner:
                        print("Winner:", winner)
                        game_over = True
                    elif winner == DRAW:
                        print("It's a draw!")
                        game_over = True
                    else:
                        currentPlayer = 'O' if currentPlayer == 'X' else 'X'

        WIN.fill(WHITE)
        draw_grid()
        draw_pieces()
        pygame.display.update()

        if game_over:
            if winner == X_WIN:
                display_message("X Wins!")
            elif winner == O_WIN:
                display_message("O Wins!")
            elif winner == DRAW:
                display_message("It's a Draw!")
            
            time.sleep(3)  # 等待3秒
            reset_game()
            game_over = False
# 启动游戏
if __name__ == "__main__":
    main()
