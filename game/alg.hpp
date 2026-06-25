#pragma once

#include "game.hpp"
#include <cstdint>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace reversi {

class Algorithms {
private:
    static inline int popcnt(uint64_t val) {
#ifdef _MSC_VER
        return __popcnt64(val);
#else
        return __builtin_popcountll(val);
#endif
    }

    static int getCoinParity(const Board& board, Color currentPlayer) {
        int current = (currentPlayer == Color::BLACK) ? board.countBlackPieces() : board.countWhitePieces();
        int opp = (currentPlayer == Color::BLACK) ? board.countWhitePieces() : board.countBlackPieces();
        return current - opp;
    }

    static int getCornerOccupancy(const Board& board, Color currentPlayer) {
        uint64_t corners = 0x8100000000000081ULL;
        uint64_t myPieces = (currentPlayer == Color::BLACK) ? board.getBlackPieces() : board.getWhitePieces();
        uint64_t oppPieces = (currentPlayer == Color::BLACK) ? board.getWhitePieces() : board.getBlackPieces();
        return (popcnt(myPieces & corners) - popcnt(oppPieces & corners)) * 5;
    }

    static int getMobility(const Board& board, Color currentPlayer) {
        Color opp = (currentPlayer == Color::BLACK) ? Color::WHITE : Color::BLACK;
        return popcnt(board.getValidMovesMask(currentPlayer)) - popcnt(board.getValidMovesMask(opp));
    }

    static int getEdgeOccupancy(const Board& board, Color currentPlayer) {
        uint64_t edgeMask = 0x3C0081818181003CULL;
        uint64_t myPieces = (currentPlayer == Color::BLACK) ? board.getBlackPieces() : board.getWhitePieces();
        uint64_t oppPieces = (currentPlayer == Color::BLACK) ? board.getWhitePieces() : board.getBlackPieces();
        return popcnt(myPieces & edgeMask) - popcnt(oppPieces & edgeMask);
    }

    static float evaluate(const Board& board, Color evalPlayer) {
        return 4.0f * getCoinParity(board, evalPlayer) +
               11.0f * getCornerOccupancy(board, evalPlayer) +
               6.0f * getMobility(board, evalPlayer) +
               5.0f * getEdgeOccupancy(board, evalPlayer);
    }

    static float alphaBeta(const Board& board, float alpha, float beta, Color currentPlayer, Color evalPlayer, int depth, int& bestMove) {
        uint64_t movesMask = board.getValidMovesMask(currentPlayer);
        if (depth == 0 || movesMask == 0) {
            return evaluate(board, evalPlayer);
        }

        Color opp = (currentPlayer == Color::BLACK) ? Color::WHITE : Color::BLACK;

        if (currentPlayer == evalPlayer) {
            float value = -1e9f;
            for (int i = 0; i < 64; ++i) {
                if ((movesMask >> i) & 1) {
                    Board child = board;
                    child.makeMove(i / 8, i % 8, currentPlayer);
                    int dummy;
                    float score = alphaBeta(child, alpha, beta, opp, evalPlayer, depth - 1, dummy);
                    if (score > value) {
                        value = score;
                        if (depth == 5) bestMove = i;
                    }
                    alpha = std::max(alpha, value);
                    if (alpha >= beta) return value;
                }
            }
            return value;
        } else {
            float value = 1e9f;
            for (int i = 0; i < 64; ++i) {
                if ((movesMask >> i) & 1) {
                    Board child = board;
                    child.makeMove(i / 8, i % 8, currentPlayer);
                    int dummy;
                    float score = alphaBeta(child, alpha, beta, opp, evalPlayer, depth - 1, dummy);
                    if (score < value) {
                        value = score;
                        if (depth == 5) bestMove = i;
                    }
                    beta = std::min(beta, value);
                    if (alpha >= beta) return value;
                }
            }
            return value;
        }
    }

public:
    static int greedyMove(const Board& board, Color color) {
        int bestMove = -1;
        int maxFlips = -1;

        for (std::size_t r = 0; r < 8; ++r) {
            for (std::size_t c = 0; c < 8; ++c) {
                uint64_t flips = board.getFlips(r, c, color);
                if (flips > 0) {
                    int numFlips = popcnt(flips);
                    if (numFlips > maxFlips) {
                        maxFlips = numFlips;
                        bestMove = (r * 8) + c;
                    }
                }
            }
        }
        return bestMove;
    }

    static int minimaxMove(const Board& board, Color color) {
        int bestMove = -1;
        alphaBeta(board, -1e9f, 1e9f, color, color, 5, bestMove);
        return bestMove;
    }
};

} // namespace reversi
