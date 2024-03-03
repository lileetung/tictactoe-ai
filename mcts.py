import numpy as np

class Node:
    def __init__(self, prior, player):
        self.visit_count = 0    # 訪問次數
        self.player = player    # 玩家
        self.prior = prior      # 先驗概率
        self.value_sum = 0      # 值的總和
        self.children = {}      # 子節點
        self.state = None       # 遊戲狀態

    def expanded(self):
        """判斷節點是否已擴展（即是否有子節點）"""
        return len(self.children) > 0
    
    def value(self):
        """計算節點的平均值，用於評估該節點的性能"""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def select_action(self, temperature):
        """據訪問次數和溫度參數選擇動作。溫度參數控制探索和利用的平衡"""
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)
        return action
    
    def select_child(self):
        """使用UCB（Upper Confidence Bound）公式選擇最佳子節點進行遊戲樹的下一步探索。"""
        best_score = -np.inf
        best_action = -1
        best_child = None
        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child
    
    def expand(self, state, player, action_probs):
        """使用動作概率擴展當前節點，為每個可能的動作創建一個新的子節點。"""
        self.player = player
        self.state = state
        for i, prob in enumerate(action_probs):
            if prob != 0:
                self.children[i] = Node(prior=prob, player=self.player * -1)

    def __repr__(self):
        """定義節點的字符串表示，方便調試和日誌記錄。"""
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), prior, self.visit_count, self.value())


def ucb_score(parent, child):
    """根據父節點和子節點的訪問次數以及子節點的先驗概率，計算UCB分數。這個分數幫助在探索和利用之間取得平衡。"""
    prior_score = child.prior * np.sqrt(parent.visit_count) / (child.visit_count + 1)

    if child.visit_count > 0:
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + prior_score

class MCTS:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model 
        self.args = args

    def backpropagate(self, search_path, value, player):
        """從葉子節點反向傳播更新節點的值和訪問次數。"""
        for node in reversed(search_path):
            node.value_sum += value if node.player == player else -value
            node.visit_count += 1

    def run(self, model, state, player):
        """執行MCTS模擬，從根節點開始，重複選擇動作和擴展節點，直到達到預定的模擬次數。每次迭代會對遊戲樹進行一次探索和擴展，然後根據遊戲結果反向更新節點統計。"""
        root = Node(0, player)
        action_probs, value = model.predict(state)
        valid_moves = self.game.get_valid_moves(state)
        action_probs = action_probs * valid_moves
        # action_probs的总和为零，就将其设置为一个均匀分布，
        action_probs_sum = np.sum(action_probs)
        if action_probs_sum > 0:
            action_probs /= action_probs_sum
        else:
            action_probs = np.ones_like(action_probs) / len(action_probs)
        root.expand(state, player, action_probs)

        for _ in range(self.args['num_simulations']):
            node = root
            search_path = [node]
            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)

            parent = search_path[-2]
            state = parent.state
            next_state, _ = self.game.get_next_state(state, player=1, action=action)
            next_state = self.game.get_canonical_board(next_state, player=-1)
            value = self.game.get_reward_for_player(next_state, player=1)

            # Rollout (輕量級模擬 or 滾動預測)
            if value is None:
                action_probs, value = model.predict(next_state)
                valid_moves = self.game.get_valid_moves(next_state)
                action_probs = action_probs * valid_moves
                # action_probs的总和为零，就将其设置为一个均匀分布，
                action_probs_sum = np.sum(action_probs)
                if action_probs_sum > 0:
                    action_probs /= action_probs_sum
                else:
                    action_probs = np.ones_like(action_probs) / len(action_probs)
                node.expand(next_state, parent.player*-1, action_probs)

            self.backpropagate(search_path, value, parent.player*-1)

        return root



        