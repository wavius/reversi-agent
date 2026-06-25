import os
import torch
import reversi_env
from agent import ReversiNet
from mcts import MCTS, BatchedMCTS

EPISODES = 1000 * 1
NUM_ENVS = 50
LEARNING_RATE = 1e-4 
MCTS_SIMULATIONS = 50

if __name__ == "__main__":
    # use nvidia gpu if available, else cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ReversiNet().to(device)

    # load existing model
    model_path = "reversi_model.pth"
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("No existing model found.")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for episode_batch in range(EPISODES // NUM_ENVS):
        current_episode = episode_batch * NUM_ENVS
        
        games = [reversi_env.Game() for _ in range(NUM_ENVS)]

        memory_states = [[] for _ in range(NUM_ENVS)]
        memory_mcts_probs = [[] for _ in range(NUM_ENVS)]
        memory_players = [[] for _ in range(NUM_ENVS)]
        memory_valid_masks = [[] for _ in range(NUM_ENVS)]

        active_games = set(range(NUM_ENVS))
        shifts = 1 << torch.arange(64)

        def to_signed(val):
            return val - (1 << 64) if val >= (1 << 63) else val

        model.eval() # turn off batch norm tracking during gameplay
        while active_games:
            active_list = list(active_games)
            
            p0_list, p1_list, valid_list = [], [], []

            # extract integer bitboards for all active games
            for batch_idx, game_idx in enumerate(active_list):
                game = games[game_idx]
                board = game.get_board()
                current_player = game.get_current_player()

                if current_player == reversi_env.Color.BLACK:
                    p0_list.append(board.get_black_pieces())
                    p1_list.append(board.get_white_pieces())
                else:
                    p0_list.append(board.get_white_pieces())
                    p1_list.append(board.get_black_pieces())

                valid_list.append(board.get_valid_moves_mask(current_player))

            # vectorized extraction of bits into (B, 64) boolean masks, then reshape
            p0_tensor = torch.tensor([to_signed(v) for v in p0_list], dtype=torch.int64).unsqueeze(1)
            p1_tensor = torch.tensor([to_signed(v) for v in p1_list], dtype=torch.int64).unsqueeze(1)
            v_tensor = torch.tensor([to_signed(v) for v in valid_list], dtype=torch.int64).unsqueeze(1)

            state_tensor = torch.stack([
                (p0_tensor & shifts) != 0,
                (p1_tensor & shifts) != 0
            ], dim=1).view(-1, 2, 8, 8).float().to(device)

            valid_tensor = ((v_tensor & shifts) != 0).float().to(device)

            for batch_idx, game_idx in enumerate(active_list):
                current_player = games[game_idx].get_current_player()
                # store single game state to memory (keeping it on GPU is fine here)
                memory_states[game_idx].append(state_tensor[batch_idx].unsqueeze(0))
                memory_players[game_idx].append(current_player)
                memory_valid_masks[game_idx].append(valid_tensor[batch_idx].unsqueeze(0))

            # use batched mcts for action selection
            mcts_engine = BatchedMCTS(model, num_simulations=MCTS_SIMULATIONS, device=device)
            active_game_objs = [games[i] for i in active_list]
            
            # get mcts policy for all active games simultaneously!
            batch_probs = mcts_engine.get_action_probs_batch(active_game_objs, temperature=1.0)
            
            sampled_actions = []
            for batch_idx, game_idx in enumerate(active_list):
                probs = batch_probs[batch_idx]
                memory_mcts_probs[game_idx].append(probs)
                
                # sample action based on mcts probabilities
                action = torch.multinomial(torch.tensor(probs), 1).item()
                sampled_actions.append(action)

            # apply actions and update games
            for batch_idx, game_idx in enumerate(active_list):
                game = games[game_idx]
                
                action = int(sampled_actions[batch_idx])
                row, col = action // 8, action % 8
                game.apply_move_fast(row, col)

                if game.is_game_over():
                    active_games.remove(game_idx)

        # collect memory from all games
        all_states = []
        all_mcts_probs = []
        all_rewards = []
        all_valid_masks = []

        for game_idx in range(NUM_ENVS):
            game = games[game_idx]
            winner = game.get_winner()

            for player in memory_players[game_idx]:
                if winner == reversi_env.Color.NONE:
                    all_rewards.append(0.0)
                elif player == winner:
                    all_rewards.append(1.0)
                else:
                    all_rewards.append(-1.0)

            all_states.extend(memory_states[game_idx])
            all_mcts_probs.extend(memory_mcts_probs[game_idx])
            all_valid_masks.extend(memory_valid_masks[game_idx])

        batch_states = torch.cat(all_states, dim=0) # tensors already on gpu
        batch_mcts_probs = torch.tensor(all_mcts_probs, dtype=torch.float32, device=device)
        batch_rewards = torch.tensor(all_rewards, dtype=torch.float32, device=device)
        batch_valid_masks = torch.cat(all_valid_masks, dim=0)

        # training (policy gradient)
        model.train() # turn batch norm tracking back on for the update
        
        # feed batch of boards to network to get logits and values with gradients
        batch_logits, batch_values = model(batch_states)
        batch_values = batch_values.squeeze(-1) # make it 1D to match batch_rewards shape
        
        # apply action mask to training logits so invalid actions don't explode the loss
        # using -1e9 instead of -inf is numerically safer for pytorch log_softmax
        batch_logits[~batch_valid_masks.bool()] = -1e9

        # convert logits to log-probabilities
        log_probs = torch.log_softmax(batch_logits, dim=1)

        # alpha zero policy + value loss
        policy_loss = -(batch_mcts_probs * log_probs).sum(dim=1).mean()
        value_loss = torch.nn.functional.mse_loss(batch_values, batch_rewards)

        # total loss = sum of policy + value losses
        loss = policy_loss + value_loss

        # backpropagation 
        optimizer.zero_grad()  # clear old math
        loss.backward()        # calculate weight adjustments
        
        # clip gradients to prevent them from exploding to NaN
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()       # apply adjustments

        print(f"Games {current_episode + 1} to {current_episode + NUM_ENVS}: Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")

    torch.save(model.state_dict(), "reversi_model.pth")
