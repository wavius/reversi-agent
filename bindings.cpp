#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "game/game.hpp"
#include "game/alg.hpp"
#include "game/mcts.hpp"

namespace py = pybind11;
using namespace reversi;

PYBIND11_MODULE(reversi_env, m) {
    m.doc() = "Reversi C++ engine bindings for Python";

    py::enum_<Color>(m, "Color")
        .value("NONE", Color::NONE)
        .value("BLACK", Color::BLACK)
        .value("WHITE", Color::WHITE)
        .export_values();

    m.def("greedy_move", &Algorithms::greedyMove, "Get greedy move");
    m.def("minimax_move", &Algorithms::minimaxMove, "Get minimax move");

    py::class_<Board>(m, "Board")
        .def(py::init<>())
        .def("clone", [](const Board &b) { return Board(b); })
        .def("get_piece", &Board::getPiece)
        .def("get_black_pieces", &Board::getBlackPieces)
        .def("get_white_pieces", &Board::getWhitePieces)
        .def("count_black_pieces", &Board::countBlackPieces)
        .def("count_white_pieces", &Board::countWhitePieces)
        .def("count_pieces", &Board::countPieces)
        .def("get_valid_moves_mask", &Board::getValidMovesMask)
        .def("get_flips", &Board::getFlips)
        .def("apply_move_with_flips", &Board::applyMoveWithFlips)
        .def("make_move", &Board::makeMove)
        .def("has_any_valid_move", &Board::hasAnyValidMove)
        .def("is_valid_move", &Board::isValidMove);

    py::class_<Game>(m, "Game")
        .def(py::init<>())
        .def("clone", [](const Game &g) { return Game(g); })
        .def("get_board", &Game::getBoard)
        .def("get_current_player", &Game::getCurrentPlayer)
        .def("is_game_over", &Game::isGameOver)
        .def("get_winner", &Game::getWinner)
        .def("make_move", &Game::makeMove)
        .def("apply_move_fast", &Game::applyMoveFast);

    py::class_<BatchedMCTSEngine>(m, "BatchedMCTSEngine")
        .def(py::init<float>())
        .def("initialize", &BatchedMCTSEngine::initialize)
        .def("clear_cache", &BatchedMCTSEngine::clear_cache)
        .def("prepare_evaluation", [](BatchedMCTSEngine& self) {
            auto req = self.prepare_evaluation();
            return py::make_tuple(req.p0, req.p1, req.valid_masks, req.players);
        })
        .def("process_evaluation", &BatchedMCTSEngine::process_evaluation)
        .def("get_action_probs", &BatchedMCTSEngine::get_action_probs);
}
