#include <pybind11/pybind11.h>
#include "game/game.hpp"
#include "game/alg.hpp"

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

    py::class_<Board>(m, "Board")
        .def(py::init<>())
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
        .def("get_board", &Game::getBoard)
        .def("get_current_player", &Game::getCurrentPlayer)
        .def("is_game_over", &Game::isGameOver)
        .def("get_winner", &Game::getWinner)
        .def("make_move", &Game::makeMove)
        .def("apply_move_fast", &Game::applyMoveFast);
}
