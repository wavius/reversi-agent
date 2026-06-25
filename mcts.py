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
            
    def backup(self, value, player):
        self.N += 1
        if self.parent is not None:
            parent_player = self.parent.game.get_current_player()
            if parent_player == player:
                self.W += value
            else:
                self.W += -value
            self.Q = self.W / self.N
            self.parent.backup(value, player)
        else:
            root_player = self.game.get_current_player()
            if root_player == player:
                self.W += value
            else:
                self.W += -value
            self.Q = self.W / self.N
            
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
            
            # selection
            while node.is_expanded and not node.is_terminal:
                node = node.best_child(self.c_puct)
                
            # expansion and evaluation
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
                # policy pruning to drop moves with < 2% probability
                if sum_probs > 0:
                    for a in action_probs:
                        action_probs[a] /= sum_probs
                else:
                    for a in action_probs:
                        action_probs[a] = 1.0 / len(action_probs)
                        
                node.expand(action_probs)
                
                # backup
                # value is from node.game's current player
                node.backup(value, node.game.get_current_player())
            else:
                # node is terminal
                node.backup(node.reward, node.game.get_current_player())
                
        # calculate final action probabilities based on visit counts
        action_probs = [0.0] * 64
        valid_actions = list(root.children.keys())
        
        if temperature == 0:
            # greedy play
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

class BatchedMCTS:
    def __init__(self, model, num_simulations=25, c_puct=1.0, device='cpu'):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device
        
    def get_action_probs_batch(self, games, temperature=1.0):
        roots = [MCTSNode(game) for game in games]
        
        if not hasattr(self, 'cache'):
            self.cache = {}
            
        for _ in range(self.num_simulations):
            nodes = [root for root in roots]
            
            # selection
            for i in range(len(nodes)):
                while nodes[i].is_expanded and not nodes[i].is_terminal:
                    nodes[i] = nodes[i].best_child(self.c_puct)
                    
            # expansion and evaluation
            evaluate_data = []
            state_tensors = []
            
            def to_signed(val):
                return val - (1 << 64) if val >= (1 << 63) else val
            shifts = 1 << torch.arange(64)
            
            for i, node in enumerate(nodes):
                if node.is_terminal:
                    node.backup(node.reward, node.game.get_current_player())
                else:
                    board = node.game.get_board()
                    current_player = node.game.get_current_player()
                    if current_player == reversi_env.Color.BLACK:
                        p0 = to_signed(board.get_black_pieces())
                        p1 = to_signed(board.get_white_pieces())
                    else:
                        p0 = to_signed(board.get_white_pieces())
                        p1 = to_signed(board.get_black_pieces())
                    
                    state_key = (p0, p1)
                    valid_mask = board.get_valid_moves_mask(current_player)
                    
                    if state_key in self.cache:
                        policy, value = self.cache[state_key]
                        
                        action_probs = {}
                        sum_probs = 0.0
                        for a in range(64):
                            if (valid_mask & (1 << a)) != 0:
                                prob = policy[a].item()
                                action_probs[a] = prob
                                sum_probs += prob
                        
                        if sum_probs > 0:
                            for a in action_probs:
                                action_probs[a] /= sum_probs
                        else:
                            for a in action_probs:
                                action_probs[a] = 1.0 / len(action_probs)
                                
                        node.expand(action_probs)
                        node.backup(value, current_player)
                    else:
                        p0_tensor = torch.tensor([p0], dtype=torch.int64).unsqueeze(1)
                        p1_tensor = torch.tensor([p1], dtype=torch.int64).unsqueeze(1)
                        state = torch.stack([
                            (p0_tensor & shifts) != 0,
                            (p1_tensor & shifts) != 0
                        ], dim=1).view(-1, 2, 8, 8).float()
                        state_tensors.append(state)
                        evaluate_data.append((node, current_player, valid_mask, state_key))
            
            if state_tensors:
                batch_states = torch.cat(state_tensors, dim=0).to(self.device)
                self.model.eval()
                with torch.no_grad():
                    logits, values = self.model(batch_states)
                    probs = torch.softmax(logits, dim=1).cpu()
                    values = values.cpu().squeeze(1)
                
                for eval_idx, (node, current_player, valid_mask, state_key) in enumerate(evaluate_data):
                    policy = probs[eval_idx]
                    value = values[eval_idx].item()
                    
                    self.cache[state_key] = (policy, value)
                    
                    action_probs = {}
                    sum_probs = 0.0
                    for a in range(64):
                        if (valid_mask & (1 << a)) != 0:
                            prob = policy[a].item()
                            action_probs[a] = prob
                            sum_probs += prob
                    
                    if sum_probs > 0:
                        for a in action_probs:
                            action_probs[a] /= sum_probs
                    else:
                        for a in action_probs:
                            action_probs[a] = 1.0 / len(action_probs)
                            
                    node.expand(action_probs)
                    node.backup(value, current_player)
                    
        # calculate final action probabilities based on visit counts for all games
        batch_action_probs = []
        for root in roots:
            action_probs = [0.0] * 64
            valid_actions = list(root.children.keys())
            
            if temperature == 0:
                best_action = max(root.children.items(), key=lambda x: x[1].N)[0]
                action_probs[best_action] = 1.0
            else:
                visits = [root.children[a].N ** (1.0 / temperature) for a in valid_actions]
                sum_visits = sum(visits)
                for a, v in zip(valid_actions, visits):
                    action_probs[a] = v / sum_visits
                    
            batch_action_probs.append(action_probs)
            
        return batch_action_probs

class CppBatchedMCTS:
    def __init__(self, model, num_simulations=25, c_puct=1.0, device='cpu'):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device
        self.engine = reversi_env.BatchedMCTSEngine(c_puct)
        
    def get_action_probs_batch(self, games, temperature=1.0):
        self.engine.initialize(games)
        
        shifts = 1 << torch.arange(64, dtype=torch.int64)
        
        for _ in range(self.num_simulations):
            p0_list, p1_list, valid_masks, players = self.engine.prepare_evaluation()
            
            if len(p0_list) > 0:
                p0_tensor = torch.tensor(p0_list, dtype=torch.int64).unsqueeze(1)
                p1_tensor = torch.tensor(p1_list, dtype=torch.int64).unsqueeze(1)
                
                state_tensors = torch.stack([
                    (p0_tensor & shifts) != 0,
                    (p1_tensor & shifts) != 0
                ], dim=1).view(-1, 2, 8, 8).float()
                
                batch_states = state_tensors.to(self.device)
                self.model.eval()
                with torch.no_grad():
                    logits, values = self.model(batch_states)
                    probs = torch.softmax(logits, dim=1).cpu().tolist()
                    values = values.cpu().squeeze(1).tolist()
                    
                self.engine.process_evaluation(probs, values)
            else:
                self.engine.process_evaluation([], [])
                
        return self.engine.get_action_probs(temperature)
