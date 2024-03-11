from game import TicTacToeGame
from model import TicTacToeCNN
from trainer import Trainer
import torch
import numpy as np
args = {
    'batch_size': 256,
    'epochs': 20,               # Number of epochs of training per iteration
    'numIters': 100,            # Total number of training iterations for learning function
    'numEps': 100,              # Number of full games (episodes) to run during each iteration
    'num_simulations': 100,   # for MCTS simulations to run when deciding on a move to play
    'start_temperature': 10e5,
    'decay_rate': 0.9
    }

if __name__ == "__main__":
    game = TicTacToeGame()
    model = TicTacToeCNN()
    trainer = Trainer(game, model, args)
    trainer.learn()