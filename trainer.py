import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from mcts import MCTS

class Trainer():
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.mcts = MCTS(self.game, self.model, self.args)

    def exceute_episode(self):
        """
        - 这个方法执行一次游戏的模拟，从游戏的初始状态开始，直到游戏结束（即有一方胜利或者游戏平局）。
        - 在每一步，它使用MCTS来决定下一步的动作，并更新游戏状态。
        - 对于每个游戏状态，它记录下规范化的棋盘(canonical_board)、当前玩家(current_player)和根据MCTS结果得到的动作概率(action_probs)。
        - 一旦游戏结束，它会计算最终奖励，并将每一步的状态、动作概率和最终奖励（调整为基于当前玩家的视角）保存下来，用于训练。
        """
        train_examples = []
        current_player = 1
        state = self.game.get_init_board()
        while True:
            canonical_board = self.game.get_canonical_board(state, current_player)
            self.mcts = MCTS(self.game, self.model, self.args)
            root = self.mcts.run(self.model, canonical_board, player=1)
            action_probs = [0 for _ in range(self.game.get_action_size())]
            for k, v in root.children.items():
                action_probs[k] = v.visit_count
            action_probs = action_probs / np.sum(action_probs)
            train_examples.append((canonical_board, current_player, action_probs))
            action = root.select_action(temperature=0)
            state, current_player = self.game.get_next_state(state, current_player, action)
            reward = self.game.get_reward_for_player(state, current_player)
            if reward is not None:
                ret = []
                for hist_state, hist_current_player, hist_action_probs in train_examples:
                    ret.append((hist_state, hist_action_probs, reward * ((-1) ** (hist_current_player != current_player))))
                return ret

    def learn(self):
        """
        这个方法控制整个训练过程，包括多次迭代的模拟游戏和学习。
        在每次迭代中，它会执行多个模拟游戏（episode），收集训练样本，并在所有模拟结束后使用这些样本来训练模型。
        在每次迭代结束后，它会保存当前模型的状态，以便于后续的使用或评估。
        """
        for i in range(1, self.args['numIters'] + 1):
            print("{}/{}".format(i, self.args['numIters']))
            train_examples = []
            for eps in range(self.args['numEps']):
                iteration_train_examples = self.exceute_episode()
                train_examples.extend(iteration_train_examples)
            np.random.shuffle(train_examples)  # 使用 numpy.random.shuffle 替代 shuffle
            self.train(train_examples)

        torch.save(self.model.state_dict(), "model_weight.pt")

    def train(self, examples):
        input_boards, target_pis, target_vs = list(zip(*examples))
        
        # 转换为Tensor
        input_boards = torch.FloatTensor(np.asarray(input_boards)).view(-1, self.game.get_board_size()[0] * self.game.get_board_size()[1])
        target_pis = torch.FloatTensor(np.asarray(target_pis))
        target_vs = torch.FloatTensor(np.asarray(target_vs)).view(-1, 1)

        self.model.train()  # 设置模型为训练模式

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # 使用Adam优化器

        for epoch in range(self.args["epochs"]):
            optimizer.zero_grad()  # 清空梯度
            # 前向传播
            output_pis, output_vs = self.model(input_boards)
            # 计算损失
            loss_pi = -torch.sum(target_pis * torch.log(output_pis + 1e-6)) / target_pis.size()[0]  # 避免log(0)
            loss_v = F.mse_loss(output_vs, target_vs)
            total_loss = loss_pi + loss_v
            # 反向传播和优化
            total_loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print('Epoch {}/{}，Loss PI: {:.4f}, Loss V: {:.4f}'.format(
                    epoch + 1, self.args["epochs"], loss_pi.item(), loss_v.item()))

        # 保存模型，注意PyTorch的保存方式
        torch.save(self.model.state_dict(), "model_{:05d}_iters.pt".format(self.args['numIters']))
