#pragma once

#include <cstdint>
#include <cassert>

namespace reversi {

#define FILLED 1
#define EMPTY 0

// Square/Player color
enum class Color { NONE, BLACK, WHITE };

class Board {
private:
  uint64_t black_pieces;
  uint64_t white_pieces;

  // Helpers
  std::size_t getIndex(std::size_t row, std::size_t col) const {
    assert(row < 8);
    assert(col < 8);
    return (row * 8) + col;
  }

  bool isFilled(std::size_t row, std::size_t col) const {
    return isFilledBlack(row, col) || isFilledWhite(row, col);
  }

  bool isFilledBlack(std::size_t row, std::size_t col) const {
    return ((black_pieces >> getIndex(row, col)) & 1ULL) == FILLED;
  }

  bool isFilledWhite(std::size_t row, std::size_t col) const {
    return ((white_pieces >> getIndex(row, col)) & 1ULL) == FILLED;
  }

public:
  // Lifecycle
  Board() : black_pieces(0x0000000810000000ULL), white_pieces(0x0000001008000000ULL) {}

  ~Board() = default;

  // Accessors
  Color getPiece(std::size_t row, std::size_t col) const {
    if (isFilledBlack(row, col)) {
      return Color::BLACK;
    } else if (isFilledWhite(row, col)) {
      return Color::WHITE;
    } else {
      return Color::NONE;
    }
  }

  // Setters
  void setPiece(std::size_t row, std::size_t col, Color color) {
    if (color == Color::BLACK) black_pieces |= (1ULL << getIndex(row, col));
    else if (color == Color::WHITE) white_pieces |= (1ULL << getIndex(row, col));
  }

  // Returns a bitboard mask of all pieces that would be flipped, 0 if invalid
  uint64_t getFlips(std::size_t row, std::size_t col, Color color) const {
    if (color == Color::NONE) return 0;

    uint64_t newPiece = 1ULL << getIndex(row, col);
    if (((black_pieces | white_pieces) & newPiece) != 0) return 0; // Square is already occupied

    uint64_t myPieces = (color == Color::BLACK) ? black_pieces : white_pieces;
    uint64_t oppPieces = (color == Color::BLACK) ? white_pieces : black_pieces;

    uint64_t allFlips = 0;

    // 8 directions: (dRow, dCol)
    int dR[8] = {-1, -1, -1,  0,  0,  1,  1,  1};
    int dC[8] = {-1,  0,  1, -1,  1, -1,  0,  1};

    for (int i = 0; i < 8; ++i) {
      uint64_t flips = 0;
      int r = row + dR[i];
      int c = col + dC[i];

      while (r >= 0 && r < 8 && c >= 0 && c < 8) {
        uint64_t mask = 1ULL << ((r * 8) + c);
        if (oppPieces & mask) {
          flips |= mask; 
        } else if (myPieces & mask) {
          allFlips |= flips; 
          break;
        } else {
          break; 
        }
        r += dR[i];
        c += dC[i];
      }
    }
    return allFlips;
  }

  // Checks if a move is valid
  bool isValidMove(std::size_t row, std::size_t col, Color color) const {
    return getFlips(row, col, color) != 0;
  }

  // Checks if a player has any valid moves
  bool hasAnyValidMove(Color color) const {
    for (std::size_t r = 0; r < 8; ++r) {
      for (std::size_t c = 0; c < 8; ++c) {
        if (isValidMove(r, c, color)) return true; 
      }
    }
    return false;
  }

  // Returns a bitboard where every 1 is a valid move 
  uint64_t getValidMovesMask(Color color) const {
    uint64_t validMoves = 0;
    for (std::size_t r = 0; r < 8; ++r) {
      for (std::size_t c = 0; c < 8; ++c) {
        if (isValidMove(r, c, color)) {
          validMoves |= (1ULL << getIndex(r, c));
        }
      }
    }
    return validMoves;
  }

  // Applies a move using pre-calculated flips
  void applyMoveWithFlips(std::size_t row, std::size_t col, Color color, uint64_t flips) {
    uint64_t newPiece = 1ULL << getIndex(row, col);

    if (color == Color::BLACK) {
      black_pieces ^= flips; 
      white_pieces ^= flips; 
      black_pieces |= newPiece; 
    } else {
      white_pieces ^= flips; 
      black_pieces ^= flips; 
      white_pieces |= newPiece; 
    }
  }

  // Applies a move and flips the pieces
  // Panics if move is invalid
  void makeMove(std::size_t row, std::size_t col, Color color) {
    uint64_t allFlips = getFlips(row, col, color);
    assert(allFlips != 0 && "makeMove called with an invalid move!");
    applyMoveWithFlips(row, col, color, allFlips);
  }

  // Board state
  std::size_t countBlackPieces() const { return __builtin_popcountll(black_pieces); }
  std::size_t countWhitePieces() const { return __builtin_popcountll(white_pieces); }
  std::size_t countPieces() const { return countBlackPieces() + countWhitePieces(); }
};

} // namespace reversi
