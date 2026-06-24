import reversi_env
import torch
import torch.nn as nn

BOARD_SIZE = 8

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

def make_move(model, board, current_player, device):
    """
    returns the best action according to the residual neural network
    """
    def to_signed(val):
        return val - (1 << 64) if val >= (1 << 63) else val

    if current_player == reversi_env.Color.BLACK:
        p0 = to_signed(board.get_black_pieces())
        p1 = to_signed(board.get_white_pieces())
    else:
        p0 = to_signed(board.get_white_pieces())
        p1 = to_signed(board.get_black_pieces())
        
    v = to_signed(board.get_valid_moves_mask(current_player))
    
    shifts = 1 << torch.arange(64)
    p0_tensor = torch.tensor([p0], dtype=torch.int64).unsqueeze(1)
    p1_tensor = torch.tensor([p1], dtype=torch.int64).unsqueeze(1)
    v_tensor = torch.tensor([v], dtype=torch.int64).unsqueeze(1)

    state_tensor = torch.stack([
        (p0_tensor & shifts) != 0,
        (p1_tensor & shifts) != 0
    ], dim=1).view(-1, 2, 8, 8).float().to(device)
    
    valid_tensor = ((v_tensor & shifts) != 0).float().to(device)

    with torch.no_grad():
        logits = model(state_tensor)
        logits[~valid_tensor.bool()] = -float("inf")
        action = torch.argmax(logits, dim=1).item()
    
    return int(action)
