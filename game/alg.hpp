#pragma once

#include "game.hpp"
#include <cstdint>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace reversi {

class Algorithms {
public:
    // Returns move that flips most pieces
    static int greedyMove(const Board& board, Color color) {
        int bestMove = -1;
        int maxFlips = -1;

        for (std::size_t r = 0; r < 8; ++r) {
            for (std::size_t c = 0; c < 8; ++c) {
                uint64_t flips = board.getFlips(r, c, color);
                if (flips > 0) {
                    int numFlips = 0;
#ifdef _MSC_VER
                    numFlips = __popcnt64(flips);
#else
                    numFlips = __builtin_popcountll(flips);
#endif
                    if (numFlips > maxFlips) {
                        maxFlips = numFlips;
                        bestMove = (r * 8) + c;
                    }
                }
            }
        }
        return bestMove;
    }
};

} // namespace reversi
