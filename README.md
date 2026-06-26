# Reversi RL Agent

**Play the agent online here:** [wavius.github.io/reversi-agent/web/](https://wavius.github.io/reversi-agent/web/)

A PyTorch Reinforcement Learning (RL) agent running on a C++ Reversi engine, with a web interface.

## How the Agent Works

* **Neural Network:** A deep Convolutional Residual Network evaluates the board state. It outputs a policy (probabilities for each valid move) and a value (who is currently winning).
* **MCTS (Monte Carlo Tree Search):** It searches the game tree to choose the move that maximizes the value.
* **Training:** It learns by playing many games against itself and updating its weights to minimize the difference between its predictions and the actual game outcomes. 

## Architecture

1. **C++ Game Engine (`game/`)**
    * **Bitboards:** The board is represented as two 64-bit integers so the game logic can use bitwise operations.
    * **Python Bindings:** Uses `pybind11` to expose the C++ engine directly to Python for fast RL training.
    * **WebAssembly:** Uses `emscripten` to compile the C++ engine into Wasm so the MCTS and game rules run natively in the browser.

2. **Python Training Pipeline (`agent/`)**
    * Trains the ResNet model via self-play and batched MCTS.
    * Exports the final `.pth` weights to `.onnx` for web deployment.

3. **Web Interface (`web/`)**
    * A static site built with vanilla JS and CSS.
    * Runs the ONNX neural network entirely client-side using `onnxruntime-web` with WebGL hardware acceleration.
    * Allows playing against the Agent, Minimax, or Greedy algorithms, and includes an auto-play feature to watch the Agent play against the Minimax algorithm.

## Credits

* The Minimax algorithm implementation was adapted from [ShengwenChang/reversi](https://github.com/ShengwenChang/reversi).
