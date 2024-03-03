import numpy as np
import graphviz

def ucb_score(parent, child):
    prior_score = child.prior * np.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        value_score = -child.value()
    else:
        value_score = 0
    return value_score + prior_score
class Node:
    def __init__(self, prior, to_play):
        self.visit_count = 0
        self.to_play = to_play
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None
    def expanded(self):
        return len(self.children) > 0
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    def select_action(self, temperature):
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
    def expand(self, state, to_play, action_probs):
            self.to_play = to_play
            self.state = np.copy(state)  # Make sure to store a copy of the state
            for a, prob in enumerate(action_probs):
                if prob != 0:
                    new_state = np.copy(state)  # Make a copy for the new child node
                    new_state[a // 3, a % 3] = to_play  # Assuming the action is a move in the Tic-Tac-Toe board
                    self.children[a] = Node(prior=prob, to_play=-to_play)
                    self.children[a].state = new_state  # Assign the new state to the child node
    def __repr__(self):
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), prior, self.visit_count, self.value())

class MCTS:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
    def run(self, model, state, to_play):
        root = Node(0, to_play)
        action_probs, value = model.predict(state)
        valid_moves = self.game.get_valid_moves(state)
        action_probs = action_probs * valid_moves
        action_probs /= np.sum(action_probs)
        root.expand(state, to_play, action_probs)
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
            if value is None:
                action_probs, value = model.predict(next_state)
                valid_moves = self.game.get_valid_moves(next_state)
                action_probs = action_probs * valid_moves
                action_probs /= np.sum(action_probs)
                node.expand(next_state, parent.to_play * -1, action_probs)
            self.backpropagate(search_path, value, parent.to_play * -1)
        return root
    def backpropagate(self, search_path, value, to_play):
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1

def visualize_tree(node, graph=None, parent_name=None, edge_label=None, perspective_player=1):
    if graph is None:
        graph = graphviz.Digraph(comment='MCTS Tree', format='png')
    
    node_name = str(id(node))  # Unique identifier for the node
    
    # Normalize the state for visualization from the perspective of player=1
    state_str = np.array2string(perspective_player * node.state, separator=',').replace('\n', '') if node.state is not None else 'None'
    node_label = f'Player: {node.to_play}\nState:\n{state_str}\nVisits: {node.visit_count}\nValue: {node.value():.2f}\nPrior: {node.prior:.2f}'
    graph.node(node_name, label=node_label)
    
    if parent_name:
        graph.edge(parent_name, node_name, label=str(edge_label))
    
    for action, child in node.children.items():
        visualize_tree(child, graph, node_name, action, perspective_player)
    
    return graph

if __name__ == "__main__":
    from model import TicTacToeModel
    from game import TicTacToeGame

    model = TicTacToeModel()
    game = TicTacToeGame()
    args = {'num_simulations': 1}
    mcts = MCTS(game, model, args)
    root = mcts.run(model, game.get_init_board(), 1)
    graph = visualize_tree(root, perspective_player=1)
    graph.render(filename='mcts_tree', view=True)