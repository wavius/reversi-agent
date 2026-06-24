import reversi_env
import torch
import torch.nn as nn
import random

BOARD_SIZE = 8

EPISODES = 10000
NUM_ENVS = 128
LEARNING_RATE = 1e-3

INITIAL_EPSILON = 0.20
FINAL_EPSILON = 0.01
DECAY_STEPS = 2000


class ReversiNet(nn.Module):
    def __init__(self):
        super().__init__()

        # feature extractor: 2d convolutional neural network
        self.conv = nn.Sequential(
            # in/out_channels: depth of data
            # kernel_size: size of sliding filter window
            # padding: zeroed border around input grid
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1),
            # rectified linear unit: f(x) = max(0, x)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # flattens 3d grid into 1d list
            nn.Flatten(),
        )

        # decision maker: linear neural layer
        # in_features: channels * rows * columns (64 * 8 * 8)
        # out_features: 64 squares (probabilities for next move)
        # todo: add a value_head to predict winning/losing
        self.policy_head = nn.Linear(
            in_features=64 * BOARD_SIZE * BOARD_SIZE, out_features=64
        )

    def forward(self, in_tensor):
        # extract features from the board tensor
        features = self.conv(in_tensor)

        # get un-normalized probabilities for 64 squares
        logits = self.policy_head(features)

        return logits


def epsilon_decay_schedule(initial_epsilon, final_epsilon, decay_steps, current_step):
    # calculate progress (capped at 1.0)
    progress = min(1.0, current_step / decay_steps)
    # linearly interpolate between initial and final
    return initial_epsilon - (progress * (initial_epsilon - final_epsilon))


import os

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
    # update epsilon for this batch of episodes
    epsilon = epsilon_decay_schedule(INITIAL_EPSILON, FINAL_EPSILON, DECAY_STEPS, current_episode)
    
    games = [reversi_env.Game() for _ in range(NUM_ENVS)]

    memory_states = [[] for _ in range(NUM_ENVS)]
    memory_actions = [[] for _ in range(NUM_ENVS)]
    memory_players = [[] for _ in range(NUM_ENVS)]

    active_games = set(range(NUM_ENVS))

    while active_games:
        active_list = list(active_games)
        
        # create batched tensors
        state_tensor = torch.zeros((len(active_list), 2, 8, 8), device=device)
        valid_tensor = torch.zeros((len(active_list), 64), device=device)

        # populate tensors for all active games
        for batch_idx, game_idx in enumerate(active_list):
            game = games[game_idx]
            board = game.get_board()
            current_player = game.get_current_player()

            white_pieces = board.get_white_pieces()
            black_pieces = board.get_black_pieces()
            valid_mask = board.get_valid_moves_mask(current_player)

            for r in range(8):
                for c in range(8):
                    index = (r * 8) + c
                    is_white = (white_pieces >> index) & 1
                    is_black = (black_pieces >> index) & 1
                    
                    if current_player == reversi_env.Color.BLACK:
                        state_tensor[batch_idx, 0, r, c] = is_black
                        state_tensor[batch_idx, 1, r, c] = is_white
                    else:
                        state_tensor[batch_idx, 0, r, c] = is_white
                        state_tensor[batch_idx, 1, r, c] = is_black

                    valid_tensor[batch_idx, index] = (valid_mask >> index) & 1
            
            # store single game state to memory
            memory_states[game_idx].append(state_tensor[batch_idx].unsqueeze(0))
            memory_players[game_idx].append(current_player)

        # get network predictions for the whole batch
        logits = model(state_tensor)

        # apply action mask
        logits[~valid_tensor.bool()] = -float("inf")

        # epsilon-greedy exploration for the batch
        is_random = torch.rand(len(active_list)) < epsilon
        best_actions = torch.argmax(logits, dim=1)

        # apply actions and update games
        for batch_idx, game_idx in enumerate(active_list):
            game = games[game_idx]
            
            if is_random[batch_idx].item():
                valid_indices = valid_tensor[batch_idx].nonzero()[:, 0].tolist()
                action = random.choice(valid_indices)
            else:
                action = best_actions[batch_idx].item()

            memory_actions[game_idx].append(action)

            row, col = action // 8, action % 8
            game.apply_move_fast(row, col)

            if game.is_game_over():
                active_games.remove(game_idx)

    # collect memory from all games
    all_states = []
    all_actions = []
    all_rewards = []

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
        all_actions.extend(memory_actions[game_idx])

    batch_states = torch.cat(all_states, dim=0) # tensors already on gpu
    batch_actions = torch.tensor(all_actions, dtype=torch.long, device=device)
    batch_rewards = torch.tensor(all_rewards, dtype=torch.float32, device=device)

    # training (policy gradient)
    # feed batch of boards to network to get logits with gradients
    batch_logits = model(batch_states)

    # convert logits to log-probabilities
    log_probs = torch.log_softmax(batch_logits, dim=1)

    # extract the log-probabilities of actions played
    action_log_probs = log_probs[range(len(batch_actions)), batch_actions]

    # calculate loss: multiply log-probabilities by rewards
    loss = -(action_log_probs * batch_rewards).mean()

    # backpropagation 
    optimizer.zero_grad()  # clear old math
    loss.backward()        # calculate weight adjustments
    optimizer.step()       # apply adjustments

    print(f"Games {current_episode + 1} to {current_episode + NUM_ENVS}: Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "reversi_model.pth")
