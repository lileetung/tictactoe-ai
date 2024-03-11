import numpy as np
from random import shuffle
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from mcts import MCTS

class Trainer():
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.mcts = MCTS(self.game, self.model, self.args)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    @staticmethod
    def augment_board(board, action_probs, game):
        n = game.n
        action_probs_matrix = np.reshape(action_probs, (n, n))
        
        augmented_boards = []
        augmented_probs = []
        
        for flip in [False, True]:
            if flip:
                augmented_board = np.fliplr(board)
                augmented_prob = np.fliplr(action_probs_matrix)
            else:
                augmented_board = board
                augmented_prob = action_probs_matrix
            for rotation in range(4):
                rotated_board = np.rot90(augmented_board, rotation)
                rotated_probs = np.rot90(augmented_prob, rotation)
                augmented_boards.append(rotated_board)
                augmented_probs.append(rotated_probs.flatten())
        return augmented_boards, augmented_probs
    
    @staticmethod
    def remove_duplicates(examples):
        unique_examples = []
        unique_boards = []  
        for ex in examples:
            board_tuple = tuple(map(tuple, ex[0]))
            
            if board_tuple not in unique_boards:
                unique_boards.append(board_tuple)
                unique_examples.append(ex)
        return unique_examples

    def execute_episode(self, temperature):
        aug = False
        train_examples = []
        current_player = 1
        state = self.game.get_init_board()
        while True:
            canonical_board = self.game.get_canonical_board(state, current_player)
            self.mcts = MCTS(self.game, self.model, self.args)
            root = self.mcts.run(canonical_board, to_play=1)
            action_probs = [0 for _ in range(self.game.get_action_size())]
            for k, v in root.children.items():
                action_probs[k] = v.visit_count
            action_probs /= np.sum(action_probs)

            if not np.all(canonical_board == 0):
                if aug:
                    augmented_boards, augmented_probs = Trainer.augment_board(canonical_board, action_probs, self.game)
                    for aug_board, aug_prob in zip(augmented_boards, augmented_probs):
                        train_examples.append((aug_board, current_player, aug_prob))
                else:
                    train_examples.append((canonical_board, current_player, action_probs))

            train_examples.append((canonical_board, current_player, action_probs))
            action = root.select_action(temperature)
            state, current_player = self.game.get_next_state(state, current_player, action)
            reward = self.game.get_reward_for_player(state, current_player)
            if reward is not None:
                # same rewards for each augmented sample
                augmented_rewards = [reward * ((-1) ** (hist_current_player != current_player)) for _, hist_current_player, _ in train_examples]
                train_examples = [(ex[0], ex[2], aug_reward) for ex, aug_reward in zip(train_examples, augmented_rewards)]
                return Trainer.remove_duplicates(train_examples)

    def train(self, examples):
        self.model.train()  # Set the model to training mode
        input_boards, target_pis, target_vs = list(zip(*examples))

        # Convert to numpy.ndarray then to PyTorch tensors
        input_boards = np.array(input_boards)
        target_pis = np.array(target_pis)
        target_vs = np.array(target_vs)
        input_boards = torch.FloatTensor(input_boards)
        target_pis = torch.FloatTensor(target_pis)
        target_vs = torch.FloatTensor(target_vs)

        # Create a dataset and dataloader for batch processing
        dataset = TensorDataset(input_boards, target_pis, target_vs)
        dataloader = DataLoader(dataset, batch_size=self.args["batch_size"], shuffle=True)

        for epoch in range(self.args["epochs"]):
            epoch_loss_pi = 0
            epoch_loss_v = 0
            epoch_total_loss = 0
            for boards, pis, vs in dataloader:
                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                out_pis, out_vs = self.model(boards)

                # Calculate loss
                loss_pi = F.cross_entropy(out_pis, pis)
                loss_v = F.mse_loss(out_vs.squeeze(-1), vs)
                total_loss = loss_pi + loss_v

                # Backward pass and optimize
                total_loss.backward()
                self.optimizer.step()

                # Accumulate losses for reporting
                epoch_loss_pi += loss_pi.item()
                epoch_loss_v += loss_v.item()
                epoch_total_loss += total_loss.item()

            num_batches = len(dataloader)
            avg_loss_pi = epoch_loss_pi / num_batches
            avg_loss_v = epoch_loss_v / num_batches
            print(f'Epoch {epoch+1}/{self.args["epochs"]}, Policy Loss: {avg_loss_pi:.4f}, Value Loss: {avg_loss_v:.4f}')

        torch.save(self.model.state_dict(), f"model.pth")


    def learn(self):
        print(self.args)
        start_temperature = self.args["start_temperature"]
        end_temperature = 10
        decay_rate = self.args["decay_rate"]
        temperature = start_temperature
        for i in range(1, self.args['numIters'] + 1):
            print("Iter {}/{} with temperature {}".format(i, self.args['numIters'], temperature))
            train_examples = []
            for eps in tqdm(range(self.args['numEps']), desc="Simulating Game"):
                iteration_train_examples = self.execute_episode(temperature)
                train_examples.extend(iteration_train_examples)
            print(len(train_examples))
            shuffle(train_examples)
            self.train(train_examples)
            temperature = max(temperature * decay_rate, end_temperature)


if __name__ == "__main__":
    from game import TicTacToeGame
    from model import TicTacToeCNN

    args = {
    'batch_size': 128,
    'epochs': 10,               # Number of epochs of training per iteration
    'numIters': 1,            # Total number of training iterations for learning function
    'numEps': 2,              # Number of full games (episodes) to run during each iteration
    'num_simulations': 1,   # for MCTS simulations to run when deciding on a move to play
    'start_temperature': 1000.0,
    'decay_rate': 0.1
    }

    game = TicTacToeGame()
    model = TicTacToeCNN()
    trainer = Trainer(game, model, args)
    train_examples_with_rewards = trainer.execute_episode(1)
    print(len(train_examples_with_rewards))
    print(train_examples_with_rewards)
    trainer.learn()



