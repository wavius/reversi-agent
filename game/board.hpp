#include <cstdint>
#include <cassert>

namespace reversi {

#define FILLED 1
#define EMPTY 0

// Square state
enum class State { NONE, BLACK, WHITE };

class Board {
private:
  uint64_t black_pieces;
  uint64_t white_pieces;

  // Helpers
  std::size_t getIndex(std::size_t row, std::size_t col) const {
    assert(row >= 0 && row < 8);
    assert(col >= 0 && col < 8);
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
  Board();
  ~Board();

  // Accessors
  State getPiece(std::size_t row, std::size_t col) const {
    if (isFilledBlack(row, col)) {
      return State::BLACK;
    } else if (isFilledWhite(row, col)) {
      return State::WHITE;
    } else {
      return State::NONE;
    }
  }

  // Settors
  void setPiece(std::size_t row, std::size_t col, State color) {
  }
};

} // namespace reversi
