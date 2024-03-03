import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TicTacToeModel(nn.Module):
    def __init__(self):
        super(TicTacToeModel, self).__init__()
        self.board_x, self.board_y = 3, 3
        self.board_size = self.board_x * self.board_y
        self.action_size = 9
        
        # Simple neural network
        self.fc1 = nn.Linear(self.board_size, 64)  # First fully connected layer
        self.fc2 = nn.Linear(64, 128)  # Second fully connected layer
        self.fc_pi = nn.Linear(128, self.action_size)  # Output layer for policy (action probabilities)
        self.fc_v = nn.Linear(128, 1)  # Output layer for value (game outcome prediction)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        pi = F.softmax(self.fc_pi(x), dim=1)  # Use softmax to get probabilities
        v = torch.tanh(self.fc_v(x))  # Value in range [-1, 1]
        return pi, v

    def predict(self, board):
        # Convert numpy board to PyTorch tensor and flatten
        board = torch.FloatTensor(board.flatten()).unsqueeze(0)  # Add batch dimension
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            pi, v = self.forward(board)
        return pi.numpy()[0], v.item()  # Return as numpy array and scalar
    

if __name__ == "__main__":
    # Initialize the model with the mock game environment
    tic_tac_toe_model = TicTacToeModel()

    # Create some test board states as fake data
    test_boards = [
        np.array([[1, 1, 0], [0, 1, -1], [0, 0, 1]]),  # Mid-game state
        np.array([[1, -1, 1], [-1, -1, 1], [1, 1, -1]]),  # Full board, draw
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])  # Empty board, start of the game
    ]

    # Test the model predictions on the fake data
    for board in test_boards:
        pi, v = tic_tac_toe_model.predict(board)
        print("pi: ", pi)
        print("v: ", v)
        pi /= np.sum(pi)
        max_index = np.argmax(pi)
        # Convert this index to 2D coordinates
        row = max_index // 3  # Integer division to find the row
        col = max_index % 3   # Modulo to find the column
        action = (row, col)
        print("action: ", action)
