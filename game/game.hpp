#pragma once

#include "board.hpp"

namespace reversi {

class Game {
private:
  Board board;
  Color currentPlayer;

public:
  // Lifecycle
  Game() : currentPlayer(Color::BLACK) {}
  ~Game() = default;

  // Accessors
  Board getBoard() const { return board; }
  Color getCurrentPlayer() const { return currentPlayer; }

  // Game Logic

  // For UI: does not assume move is valid
  bool makeMove(std::size_t row, std::size_t col) {
    uint64_t flips = board.getFlips(row, col, currentPlayer);
    if (flips == 0) return false;

    board.applyMoveWithFlips(row, col, currentPlayer, flips);
    switchPlayer();
    return true;
  }

  // For agent: assumes move is valid
  void applyMoveFast(std::size_t row, std::size_t col) {
    board.makeMove(row, col, currentPlayer);
    switchPlayer();
  }

private:
  void switchPlayer() {
    Color nextPlayer = (currentPlayer == Color::BLACK) ? Color::WHITE : Color::BLACK;
    if (board.hasAnyValidMove(nextPlayer)) {
      currentPlayer = nextPlayer;
    }
  }
public:

  bool isGameOver() const { return !board.hasAnyValidMove(Color::BLACK) && !board.hasAnyValidMove(Color::WHITE); }

  // Returns winner, Color::NONE for draw
  Color getWinner() const {
    if (!isGameOver()) return Color::NONE;

    int blackScore = board.countBlackPieces();
    int whiteScore = board.countWhitePieces();

    if (blackScore > whiteScore) return Color::BLACK;
    if (whiteScore > blackScore) return Color::WHITE;
    return Color::NONE; // Draw
  }

};

} // namespace reversi