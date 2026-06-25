import torch
import random
import reversi_env
import agent
from agent import ReversiNet
from mcts import BatchedMCTS

MCTS_SIMULATIONS = 25

# test agent against greedy alg
def benchmark(num_games=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReversiNet().to(device)
    
    try:
        model.load_state_dict(torch.load("reversi_model.pth", map_location=device, weights_only=True))
        print("Loaded reversi_model.pth successfully.")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        
    model.eval()

    print(f"Starting batched benchmark of {num_games} games")

    games = [reversi_env.Game() for _ in range(num_games)]
    agent_colors = [reversi_env.Color.BLACK if i % 2 == 0 else reversi_env.Color.WHITE for i in range(num_games)]
    active_games = set(range(num_games))

    # randomize the first 4 moves to prevent deterministic games
    for game in games:
        for _ in range(4):
            if not game.is_game_over():
                current_player = game.get_current_player()
                board = game.get_board()
                valid_mask = board.get_valid_moves_mask(current_player)
                
                valid_moves = []
                for move in range(64):
                    if (valid_mask >> move) & 1:
                        valid_moves.append(move)
                        
                if valid_moves:
                    random_move = random.choice(valid_moves)
                    row, col = random_move // 8, random_move % 8
                    game.apply_move_fast(row, col)

    shifts = 1 << torch.arange(64)
    def to_signed(val):
        return val - (1 << 64) if val >= (1 << 63) else val

    while active_games:
        active_list = list(active_games)
        
        agent_game_indices = []
        p0_list, p1_list, valid_list = [], [], []

        for game_idx in active_list:
            game = games[game_idx]
            current_player = game.get_current_player()
            
            if current_player == agent_colors[game_idx]:
                agent_game_indices.append(game_idx)
                board = game.get_board()
                if current_player == reversi_env.Color.BLACK:
                    p0_list.append(board.get_black_pieces())
                    p1_list.append(board.get_white_pieces())
                else:
                    p0_list.append(board.get_white_pieces())
                    p1_list.append(board.get_black_pieces())
                valid_list.append(board.get_valid_moves_mask(current_player))
            else:
                # Minimax move
                action = reversi_env.minimax_move(game.get_board(), current_player)
                row, col = action // 8, action % 8
                game.apply_move_fast(row, col)

        # Batch evaluate all agent moves using BatchedMCTS
        if agent_game_indices:
            mcts_engine = BatchedMCTS(model, num_simulations=MCTS_SIMULATIONS, device=device)
            active_game_objs = [games[i] for i in agent_game_indices]
            
            # get MCTS policy (temperature=0 for greedy best play)
            batch_probs = mcts_engine.get_action_probs_batch(active_game_objs, temperature=0.0)

            for i, game_idx in enumerate(agent_game_indices):
                probs = batch_probs[i]
                action = probs.index(max(probs)) # pick the move with the highest probability
                row, col = action // 8, action % 8
                games[game_idx].apply_move_fast(row, col)

        for game_idx in active_list:
            if games[game_idx].is_game_over():
                active_games.remove(game_idx)

    agent_wins = 0
    greedy_wins = 0
    draws = 0

    for i, game in enumerate(games):
        winner = game.get_winner()
        agent_color = agent_colors[i]
        
        if winner == agent_color:
            agent_wins += 1
        elif winner == reversi_env.Color.NONE:
            draws += 1
        else:
            greedy_wins += 1

    print("-" * 30)
    print(f"Agent Wins:  {agent_wins}")
    print(f"Alg Wins: {greedy_wins}")
    print(f"Draws:       {draws}")
    winrate = agent_wins / num_games * 100
    print(f"Agent Winrate: {winrate:.2f}%")

if __name__ == "__main__":
    benchmark(100)
