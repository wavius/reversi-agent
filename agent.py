import reversi_env
import torch
import torch.nn as nn

BOARD_SIZE = 8


class ReversiNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature extractor: 2D convolutional neural network
        self.conv = nn.Sequential(
            # in/out_channels: depth of data
            # kernel_size: size of sliding filter window
            # padding: zeroed border around input grid
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1),
            # Rectified Linear Unit: f(x) = max(0, x)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Flattens 3D grid into 1D list
            nn.Flatten(),
        )

        # Decision maker: linear neural layer
        # in_features: Channels * rows * columns (64 * 8 * 8)
        # out_features: 64 squares (probabilities for next move)
        # TODO: Add a value_head to predict winning/losing
        self.policy_head = nn.Linear(
            in_features=64 * BOARD_SIZE * BOARD_SIZE, out_features=64
        )

    def forward(self, in_tensor):
        # Extract features from the board tensor
        features = self.conv(in_tensor)

        # Get un-normalized probabilities for 64 squares
        logits = self.policy_head(features)

        return logits


game = reversi_env.Game()
model = ReversiNet()

while not game.is_game_over():
    board = game.get_board()
    current_player = game.get_current_player()

    # Board state tensor
    state_tensor = torch.zeros((1, 2, 8, 8))
    white_pieces = board.get_white_pieces()
    black_pieces = board.get_black_pieces()

    # Valid move tensor
    valid_mask = board.get_valid_moves_mask(current_player)
    valid_tensor = torch.zeros((1, 64))

    # Populate tensors
    for r in range(8):
        for c in range(8):
            index = (r * 8) + c

            # Extract bit at index
            is_white = (white_pieces >> index) & 1
            is_black = (black_pieces >> index) & 1
            
            # Channel 0: player pieces
            # Channel 1: opponent pieces
            if current_player == reversi_env.Color.BLACK:
                state_tensor[0, 0, r, c] = is_black
                state_tensor[0, 1, r, c] = is_white
            else:
                state_tensor[0, 0, r, c] = is_white
                state_tensor[0, 1, r, c] = is_black

            # valid moves mask
            valid_tensor[0, index] = (valid_mask >> index) & 1

    # Get network prediction
    logits = model(state_tensor)

    # Apply action mask
    logits[~valid_tensor.bool()] = -float("inf")

    action = int(torch.argmax(logits).item())
    row, col = action // 8, action % 8

    game.apply_move_fast(row, col)

winner = game.get_winner()
