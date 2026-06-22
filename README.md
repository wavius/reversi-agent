# Reversi RL Agent

This project combines a high-performance C++ Reversi engine with a Python-based Reinforcement Learning (RL) agent, designed to beat a minimax algorithm.

## Architecture

1.  **C++ Game Engine (`engine/`)**
    *   **Core Logic:** Implemented using **bitboards** (two 64-bit integers representing the black and white pieces). This provides extremely fast move generation and state updates, essential for RL and Minimax.
    *   **Python Bindings:** Uses **pybind11** to expose the C++ game state and logic directly to Python as a module. This avoids the overhead of inter-process communication (pipes/sockets) and is much cleaner.

2.  **Python RL Agent (`agent/`)**
    *   **Environment:** A custom Gym-like environment (`env.py`) that wraps the C++ pybind11 module.
    *   **Agent:** The RL algorithm (e.g., DQN, PPO, or AlphaZero-style) implemented using PyTorch or similar libraries.
    *   **Training Loop:** Scripts to play games, collect experiences, and train the neural network.

## Proposed Directory Structure

```text
reversi-agent/
├── CMakeLists.txt         # Build config for pybind11 module
├── engine/                # C++ Source
│   ├── board.h            # Bitboard representation & macros
│   ├── game.h/.cpp        # Game rules, move generation
│   └── bindings.cpp       # Pybind11 wrapper code
├── agent/                 # Python Source
│   ├── env.py             # Gym-style wrapper
│   ├── dqn_agent.py       # RL Agent implementation
│   └── train.py           # Training script
├── requirements.txt       # Python dependencies
└── README.md
```

## Implementation Phases

1.  **Phase 1: The C++ Engine**
    *   Implement bitboard logic for move generation (checking valid flips).
    *   Implement game state management (applying moves, detecting terminal states, calculating scores).
2.  **Phase 2: Python Bindings**
    *   Set up CMake and `pybind11`.
    *   Write `bindings.cpp` to expose the game state and mechanics to Python.
    *   Compile the C++ code into a Python extension module.
3.  **Phase 3: Python RL Environment**
    *   Create `agent/env.py` conforming to the standard reinforcement learning interface (e.g., OpenAI Gym interface with reset, step, render).
4.  **Phase 4: RL Agent Implementation**
    *   Implement the chosen RL agent.
    *   Create the training loop to play self-play games or play against a random/minimax agent to bootstrap learning.
