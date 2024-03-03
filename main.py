from game import TicTacToeGame
from model import TicTacToeModel
from trainer import Trainer

args = {
    'batch_size': 128,
    'numIters': 3,
    'num_simulations': 256,
    'numEps': 100,
    'epochs': 10,
}

game = TicTacToeGame()
model = TicTacToeModel()

trainer = Trainer(game, model, args)
trainer.learn()