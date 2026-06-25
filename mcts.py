import math
import torch
import reversi_env

class MCTSNode:
    def __init__(self, game, parent=None, action=None, prior_prob=0.0):
        self.game = game
        self.parent = parent
        self.action = action
        self.children = {}
        
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = prior_prob
        
        self.is_expanded = False
        
        # terminal status
        self.is_terminal = game.is_game_over()
        if self.is_terminal:
            winner = game.get_winner()
            current_player = game.get_current_player()
            if winner == reversi_env.Color.NONE:
                self.reward = 0.0
            elif winner == current_player:
                self.reward = 1.0
            else:
                self.reward = -1.0
        else:
            self.reward = 0.0
            
    def expand(self, action_probs):
        """action_probs is a dict of {action: probability}"""
        self.is_expanded = True
        for action, prob in action_probs.items():
            next_game = self.game.clone()
            row, col = action // 8, action % 8
            next_game.apply_move_fast(row, col)
            self.children[action] = MCTSNode(next_game, parent=self, action=action, prior_prob=prob)
            
    def backup(self, value):
        self.N += 1
        self.W += value
        self.Q = self.W / self.N
        if self.parent is not None:
            self.parent.backup(-value)
            
    def best_child(self, c_puct=1.0):
        best_score = -float('inf')
        best_node = None
        
        for action, child in self.children.items():
            # puct formula: q + c * p * sqrt(n_parent) / (1 + n_child)
            u = c_puct * child.P * math.sqrt(self.N) / (1 + child.N)
            score = child.Q + u
            
            if score > best_score:
                best_score = score
                best_node = child
                
        if best_node is None:
            raise ValueError("best_child called on an expanded node with no children!")
                
        return best_node

class MCTS:
    def __init__(self, model, num_simulations=100, c_puct=1.0, device='cpu'):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device
        
    def get_action_probs(self, game, temperature=1.0):
        root = MCTSNode(game)
        
        for _ in range(self.num_simulations):
            node = root
            
            # 1. selection
            while node.is_expanded and not node.is_terminal:
                node = node.best_child(self.c_puct)
                
            # 2. expansion & evaluation
            if not node.is_terminal:
                policy, value = self.evaluate(node.game)
                
                current_player = node.game.get_current_player()
                board = node.game.get_board()
                valid_mask = board.get_valid_moves_mask(current_player)
                
                action_probs = {}
                sum_probs = 0.0
                
                for i in range(64):
                    if (valid_mask & (1 << i)) != 0:
                        prob = policy[i].item()
                        action_probs[i] = prob
                        sum_probs += prob
                        
                # normalize
                if sum_probs > 0:
                    for a in action_probs:
                        action_probs[a] /= sum_probs
                else:
                    for a in action_probs:
                        action_probs[a] = 1.0 / len(action_probs)
                        
                node.expand(action_probs)
                
                # 3. backup
                # value is from node.game's current player
                # pass -value because parent's turn is opponent
                node.backup(-value)
            else:
                # node is terminal
                node.backup(-node.reward)
                
        # calculate final action probabilities based on visit counts
        action_probs = [0.0] * 64
        valid_actions = list(root.children.keys())
        
        if temperature == 0:
            # greedy play (e.g. during actual testing/benchmarking)
            best_action = max(root.children.items(), key=lambda x: x[1].N)[0]
            action_probs[best_action] = 1.0
            return action_probs
            
        # apply temperature for exploration
        visits = [root.children[a].N ** (1.0 / temperature) for a in valid_actions]
        sum_visits = sum(visits)
        
        for a, v in zip(valid_actions, visits):
            action_probs[a] = v / sum_visits
            
        return action_probs

    def evaluate(self, game):
        board = game.get_board()
        current_player = game.get_current_player()
        
        def to_signed(val):
            return val - (1 << 64) if val >= (1 << 63) else val
            
        if current_player == reversi_env.Color.BLACK:
            p0 = to_signed(board.get_black_pieces())
            p1 = to_signed(board.get_white_pieces())
        else:
            p0 = to_signed(board.get_white_pieces())
            p1 = to_signed(board.get_black_pieces())
            
        shifts = 1 << torch.arange(64)
        p0_tensor = torch.tensor([p0], dtype=torch.int64).unsqueeze(1)
        p1_tensor = torch.tensor([p1], dtype=torch.int64).unsqueeze(1)
        
        state_tensor = torch.stack([
            (p0_tensor & shifts) != 0,
            (p1_tensor & shifts) != 0
        ], dim=1).view(-1, 2, 8, 8).float().to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits, value = self.model(state_tensor)
            probs = torch.softmax(logits, dim=1)
            
        return probs.squeeze(0).cpu(), value.item()
