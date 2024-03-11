import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TicTacToeCNN(nn.Module):
    def __init__(self):
        super(TicTacToeCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc1_pi = nn.Linear(64 * 2 * 2, 64)  # Adjusted input size to match conv output
        self.dropout_pi = nn.Dropout(0.3)  # Adding dropout for regularization
        self.pi = nn.Linear(64, 9)  # 9 possible actions in Tic-Tac-Toe
        
        self.fc1_v = nn.Linear(64 * 2 * 2, 64)  # Similarly adjusted for the value head
        self.dropout_v = nn.Dropout(0.3)  # Adding dropout for regularization
        self.v = nn.Linear(64, 1)  # Scalar value prediction

    def forward(self, x):
        x = x.view(-1, 1, 3, 3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten

        # Policy head
        x_pi = F.relu(self.fc1_pi(x))
        x_pi = self.dropout_pi(x_pi)  # Apply dropout
        pi = F.softmax(self.pi(x_pi), dim=1)

        # Value head
        x_v = F.relu(self.fc1_v(x))
        x_v = self.dropout_v(x_v)  # Apply dropout
        v = torch.tanh(self.v(x_v))

        return pi, v
    
    def predict(self, board):
        board = torch.FloatTensor(board.flatten()).unsqueeze(0)  # Add batch dimension
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            pi, v = self.forward(board)
        return pi.numpy()[0], v.item()  # Return as numpy array and scalar


if __name__ == "__main__":
    # Test the model with a single 3x3 board input
    model = TicTacToeCNN()
    board = np.zeros((3, 3), dtype=int)
    board_tensor = torch.FloatTensor(board).unsqueeze(0)  # Adding batch dimension
    pi, v = model.predict(board_tensor)
    print(pi, v)