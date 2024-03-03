from game import TicTacToeGame
from model import TicTacToeModel
from trainer import Trainer

args = {
    'batch_size': 128,
    'numIters': 1000,
    'num_simulations': 128,
    'numEps': 100,
    'epochs': 100,
}

game = TicTacToeGame()
model = TicTacToeModel()




trainer = Trainer(game, model, args)
trainer.learn()