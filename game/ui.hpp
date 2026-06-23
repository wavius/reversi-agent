#pragma once

#include "game.hpp"
#include <iostream>
#include <string>

namespace reversi {

class TerminalUI {
public:
  TerminalUI() = default;

  void drawBoard(const Board& board) const {
    std::cout << "\n  0 1 2 3 4 5 6 7\n";
    for (std::size_t r = 0; r < 8; ++r) {
      std::cout << r << " ";
      for (std::size_t c = 0; c < 8; ++c) {
        Color p = board.getPiece(r, c);
        if (p == Color::BLACK) {
          std::cout << "B ";
        } else if (p == Color::WHITE) {
          std::cout << "W ";
        } else {
          std::cout << ". ";
        }
      }
      std::cout << "\n";
    }
    std::cout << "Score - Black (B): " << board.countBlackPieces() 
              << " White (W): " << board.countWhitePieces() << "\n\n";
  }

  void play(Game& game) {
    std::string input;
    while (!game.isGameOver()) {
      drawBoard(game.getBoard());
      
      Color curr = game.getCurrentPlayer();
      std::cout << "Player " << (curr == Color::BLACK ? "Black (B)" : "White (W)") << "'s turn.\n";
      std::cout << "Enter move (row col): ";
      
      std::size_t r, c;
      if (!(std::cin >> r >> c)) {
        std::cout << "Invalid input.\n";
        std::cin.clear();
        std::cin.ignore(10000, '\n');
        continue;
      }
      
      if (r >= 8 || c >= 8) {
        std::cout << "Invalid coordinates. Must be 0-7.\n";
        continue;
      }

      if (!game.makeMove(r, c)) {
        std::cout << "Invalid move.\n";
      }
    }

    drawBoard(game.getBoard());
    Color winner = game.getWinner();
    if (winner == Color::BLACK) {
      std::cout << "Black wins!\n";
    } else if (winner == Color::WHITE) {
      std::cout << "White wins!\n";
    } else {
      std::cout << "It's a draw!\n";
    }
  }
};

} // namespace reversi
