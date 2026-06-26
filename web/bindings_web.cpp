#include <emscripten/bind.h>
#include "../game/game.hpp"
#include "../game/alg.hpp"
#include "../game/mcts.hpp"

using namespace emscripten;
using namespace reversi;

// stl type wrappers for js interop
EMSCRIPTEN_BINDINGS(stl_wrappers) {
    register_vector<int>("VectorInt");
    register_vector<float>("VectorFloat");
    register_vector<uint64_t>("VectorUInt64");
    register_vector<int64_t>("VectorInt64");
    
    // mcts eval request
    value_object<BatchedMCTSEngine::EvalRequest>("EvalRequest")
        .field("p0", &BatchedMCTSEngine::EvalRequest::p0)
        .field("p1", &BatchedMCTSEngine::EvalRequest::p1)
        .field("valid_masks", &BatchedMCTSEngine::EvalRequest::valid_masks)
        .field("players", &BatchedMCTSEngine::EvalRequest::players);
        
    register_vector<std::vector<float>>("VectorVectorFloat");
    
    register_vector<Game>("VectorGame");
}

EMSCRIPTEN_BINDINGS(reversi_module) {
    enum_<Color>("Color")
        .value("NONE", Color::NONE)
        .value("BLACK", Color::BLACK)
        .value("WHITE", Color::WHITE);

    class_<Board>("Board")
        .constructor<>()
        .function("get_piece", &Board::getPiece)
        .function("get_black_pieces", &Board::getBlackPieces)
        .function("get_white_pieces", &Board::getWhitePieces)
        .function("count_black_pieces", &Board::countBlackPieces)
        .function("count_white_pieces", &Board::countWhitePieces)
        .function("count_pieces", &Board::countPieces)
        .function("get_valid_moves_mask", &Board::getValidMovesMask)
        .function("get_flips", &Board::getFlips)
        .function("apply_move_with_flips", &Board::applyMoveWithFlips)
        .function("make_move", &Board::makeMove)
        .function("has_any_valid_move", &Board::hasAnyValidMove)
        .function("is_valid_move", &Board::isValidMove);

    class_<Game>("Game")
        .constructor<>()
        .function("get_board", &Game::getBoard)
        .function("get_current_player", &Game::getCurrentPlayer)
        .function("is_game_over", &Game::isGameOver)
        .function("get_winner", &Game::getWinner)
        .function("make_move", &Game::makeMove)
        .function("apply_move_fast", &Game::applyMoveFast);

    function("greedy_move", &Algorithms::greedyMove);
    function("minimax_move", &Algorithms::minimaxMove);

    class_<BatchedMCTSEngine>("BatchedMCTSEngine")
        .constructor<float>()
        .function("initialize", &BatchedMCTSEngine::initialize)
        .function("clear_cache", &BatchedMCTSEngine::clear_cache)
        .function("prepare_evaluation", &BatchedMCTSEngine::prepare_evaluation)
        .function("process_evaluation", &BatchedMCTSEngine::process_evaluation)
        .function("get_action_probs", &BatchedMCTSEngine::get_action_probs);
}
