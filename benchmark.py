import torch
import reversi_env
import agent
from agent import ReversiNet

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

    agent_wins = 0
    greedy_wins = 0
    draws = 0

    print(f"Starting benchmark of {num_games} games")

    # agent plays half the games as black, half as white
    import random

    for i in range(num_games):
        game = reversi_env.Game()
        agent_color = reversi_env.Color.BLACK if i % 2 == 0 else reversi_env.Color.WHITE
        
        # randomize the first 4 moves to prevent deterministic games
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

        while not game.is_game_over():
            current_player = game.get_current_player()
            
            if current_player == agent_color:
                # Agent move
                action = agent.make_move(model, game.get_board(), current_player, device)
                
                action = int(action)
                row, col = action // 8, action % 8
                game.apply_move_fast(row, col)
            else:
                # Greedy move
                action = reversi_env.greedy_move(game.get_board(), current_player)
                row, col = action // 8, action % 8
                game.apply_move_fast(row, col)

        winner = game.get_winner()
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
