# Building a Reversi RL Agent

Now that you have a blazing-fast C++ engine exposed to Python, you have the perfect foundation for a Reinforcement Learning agent. For a board game like Reversi, the most common and successful approaches are **Deep Q-Networks (DQN)** or an **AlphaZero-style** architecture (Policy/Value networks with MCTS).

Here is a roadmap for how to get started, along with a small PyTorch skeleton.

## 1. State Representation (The Observation)

Neural networks can't read C++ objects directly. You need to convert the game board into a PyTorch Tensor.
A good representation is a 3D tensor of shape `(2, 8, 8)`:

- **Channel 0:** A binary grid (1s and 0s) of the current player's pieces.
- **Channel 1:** A binary grid of the opponent's pieces.

*Tip: You can get these by extracting the 64 bits from `board.count_black_pieces()` (if you modify the C++ to return the raw `uint64_t` bitboards, which would be even faster! Right now you can just iterate, or we can expose the raw bitboards).*

## 2. Action Masking

Your neural network will output 64 values (one for each square). However, most of these are illegal moves.
You will use `board.get_valid_moves_mask()` to generate a mask. Before passing your network's output to a softmax or selecting a move, you set the value of all illegal moves to `-infinity`. This forces the network to only choose valid actions.

## 3. The Neural Network

A Convolutional Neural Network (CNN) is highly recommended for Reversi because it understands 2D spatial relationships (like lines and edges).

## 4. The Self-Play Loop

You will write a loop where the agent plays games against itself (or random moves at first) using the `apply_move_fast()` method. You'll store the `(state, action, reward)` tuples and periodically use them to train the network.

---

## Basic PyTorch Skeleton

Here is a conceptual skeleton of how your python code will interface with the C++ environment.

```python
import torch
import torch.nn as nn
import reversi_env

class ReversiNet(nn.Module):
    def __init__(self):
        super().__init__()
        # A simple CNN taking 2 channels (my pieces, opponent pieces)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.policy_head = nn.Linear(64 * 8 * 8, 64) # Outputs probabilities for 64 squares

    def forward(self, x):
        features = self.conv(x)
        logits = self.policy_head(features)
        return logits

# Example Loop
game = reversi_env.Game()
model = ReversiNet()

while not game.is_game_over():
    board = game.get_board()
    current_player = game.get_current_player()
    
    # 1. TODO: Convert `board` to a [1, 2, 8, 8] float tensor
    state_tensor = torch.zeros((1, 2, 8, 8)) 
    
    # 2. Get network predictions
    logits = model(state_tensor)
    
    # 3. Apply Action Mask
    valid_mask_int = board.get_valid_moves_mask(current_player)
    # TODO: Convert the 64-bit integer mask into a 64-element boolean tensor
    
    # (Set invalid moves to -inf here)
    
    # 4. Pick a move (e.g., argmax)
    action = torch.argmax(logits).item()
    row, col = action // 8, action % 8
    
    # 5. Apply the move incredibly fast in C++!
    game.apply_move_fast(row, col)

# 6. Check winner and calculate rewards
winner = game.get_winner()
```

## Next Steps

If you want to maximize performance, you might want me to add a quick method to the C++ side that returns the raw `uint64_t` bitboards directly (like `board.get_black_pieces()`), or even one that directly writes the board state into a Numpy array! Let me know how you want to proceed.
