import reversi_env
import torch
import torch.nn as nn
import random

BOARD_SIZE = 8

EPISODES = 1000000
NUM_ENVS = 100
LEARNING_RATE = 1e-3

INITIAL_EPSILON = 0.20
FINAL_EPSILON = 0.01
DECAY_STEPS = 10000

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        # feature extractor: 2d convolutional neural network
        # in/out_channels: depth of data
        # kernel_size: size of sliding filter window
        # padding: zeroed border around input grid
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()

    # save for Residual connection
    def forward(self, x):
        identity = x

        # pass through two conv layers
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # add input to the output
        out += identity
        out = self.relu2(out)

        return out


class ReversiNet(nn.Module):
    def __init__(self, channels=128):
        super().__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            # rectified linear unit: f(x) = max(0, x)
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            ResBlock(channels),
            ResBlock(channels),
            ResBlock(channels),
            ResBlock(channels),
            ResBlock(channels),
            ResBlock(channels),
            ResBlock(channels),
            ResBlock(channels),
        )

        self.policy_conv = nn.Conv2d(in_channels=channels, out_channels=2, kernel_size=1)
        self.policy_flatten = nn.Flatten()

        # decision maker: linear neural layer
        # in_features: channels * rows * columns (64 * 8 * 8)
        # out_features: 64 squares (probabilities for next move)
        # todo: add a value_head to predict winning/losing
        self.policy_head = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, 64)

    def forward(self, x):
        # extract features from the board tensor
        x = self.initial_conv(x)
        x = self.res_blocks(x)

        x = self.policy_conv(x)
        x = self.policy_flatten(x)
        x = self.policy_head(x)

        return x


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
    memory_valid_masks = [[] for _ in range(NUM_ENVS)]

    active_games = set(range(NUM_ENVS))
    shifts = 1 << torch.arange(64)

    def to_signed(val):
        return val - (1 << 64) if val >= (1 << 63) else val

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

            action = int(action)
            row, col = action // 8, action % 8
            game.apply_move_fast(row, col)

            if game.is_game_over():
                active_games.remove(game_idx)

    # collect memory from all games
    all_states = []
    all_actions = []
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
        all_actions.extend(memory_actions[game_idx])
        all_valid_masks.extend(memory_valid_masks[game_idx])

    batch_states = torch.cat(all_states, dim=0) # tensors already on gpu
    batch_actions = torch.tensor(all_actions, dtype=torch.long, device=device)
    batch_rewards = torch.tensor(all_rewards, dtype=torch.float32, device=device)
    batch_valid_masks = torch.cat(all_valid_masks, dim=0)

    # training (policy gradient)
    # feed batch of boards to network to get logits with gradients
    batch_logits = model(batch_states)
    
    # apply action mask to training logits so invalid actions don't explode the loss
    batch_logits[~batch_valid_masks.bool()] = -float("inf")

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
