#pragma once
#include "game.hpp"
#include <vector>
#include <unordered_map>
#include <map>
#include <cmath>
#include <memory>
#include <algorithm>

namespace reversi {

struct MCTSNode {
    Game game;
    MCTSNode* parent;
    int action;
    float P;
    int N;
    float W;
    float Q;
    bool is_expanded;
    bool is_terminal;
    float reward;
    std::unordered_map<int, std::unique_ptr<MCTSNode>> children;

    MCTSNode(const Game& g, MCTSNode* p = nullptr, int a = -1, float prob = 0.0)
        : game(g), parent(p), action(a), P(prob), N(0), W(0.0), Q(0.0),
          is_expanded(false), is_terminal(g.isGameOver()), reward(0.0) {
        if (is_terminal) {
            Color winner = game.getWinner();
            Color current = game.getCurrentPlayer();
            if (winner == Color::NONE) reward = 0.0;
            else if (winner == current) reward = 1.0;
            else reward = -1.0;
        }
    }

    void expand(const std::unordered_map<int, float>& action_probs) {
        is_expanded = true;
        for (const auto& pair : action_probs) {
            int a = pair.first;
            float prob = pair.second;
            Game next_game = game;
            next_game.applyMoveFast(a / 8, a % 8);
            children[a] = std::make_unique<MCTSNode>(next_game, this, a, prob);
        }
    }

    void backup(float value, Color player) {
        N += 1;
        if (parent != nullptr) {
            Color parent_player = parent->game.getCurrentPlayer();
            if (parent_player == player) W += value;
            else W -= value;
            Q = W / N;
            parent->backup(value, player);
        } else {
            Color root_player = game.getCurrentPlayer();
            if (root_player == player) W += value;
            else W -= value;
            Q = W / N;
        }
    }

    MCTSNode* best_child(float c_puct) {
        float best_score = -1e9;
        MCTSNode* best_node = nullptr;
        for (auto& pair : children) {
            MCTSNode* child = pair.second.get();
            float u = c_puct * child->P * std::sqrt(N) / (1 + child->N);
            float score = child->Q + u;
            if (score > best_score) {
                best_score = score;
                best_node = child;
            }
        }
        return best_node;
    }
};

class BatchedMCTSEngine {
public:
    std::vector<std::unique_ptr<MCTSNode>> roots;
    float c_puct;
    std::vector<MCTSNode*> current_eval_nodes;
    std::map<std::pair<uint64_t, uint64_t>, std::pair<std::vector<float>, float>> cache;

    BatchedMCTSEngine(float c_puct) : c_puct(c_puct) {}

    void initialize(const std::vector<Game>& games) {
        roots.clear();
        for (const auto& g : games) {
            roots.push_back(std::make_unique<MCTSNode>(g));
        }
    }
    
    void clear_cache() {
        cache.clear();
    }

    struct EvalRequest {
        std::vector<int64_t> p0;
        std::vector<int64_t> p1;
        std::vector<uint64_t> valid_masks;
        std::vector<int> players;
    };

    EvalRequest prepare_evaluation() {
        current_eval_nodes.clear();
        EvalRequest req;

        for (auto& root : roots) {
            MCTSNode* node = root.get();
            while (node->is_expanded && !node->is_terminal) {
                node = node->best_child(c_puct);
            }
            if (node->is_terminal) {
                node->backup(node->reward, node->game.getCurrentPlayer());
            } else {
                Board b = node->game.getBoard();
                Color c = node->game.getCurrentPlayer();
                uint64_t p0, p1;
                if (c == Color::BLACK) {
                    p0 = b.getBlackPieces(); p1 = b.getWhitePieces();
                } else {
                    p0 = b.getWhitePieces(); p1 = b.getBlackPieces();
                }
                
                std::pair<uint64_t, uint64_t> state_key = {p0, p1};
                uint64_t valid_mask = b.getValidMovesMask(c);
                
                if (cache.find(state_key) != cache.end()) {
                    // Cache hit! Instant evaluate.
                    const auto& cached_data = cache[state_key];
                    const std::vector<float>& policy = cached_data.first;
                    float value = cached_data.second;
                    
                    std::unordered_map<int, float> action_probs;
                    float sum_probs = 0.0;
                    for (int a = 0; a < 64; ++a) {
                        if ((valid_mask & (1ULL << a)) != 0) {
                            float p = policy[a];
                            action_probs[a] = p;
                            sum_probs += p;
                        }
                    }

                    if (sum_probs > 0) {
                        for (auto& pair : action_probs) pair.second /= sum_probs;
                    } else {
                        float uniform = 1.0f / action_probs.size();
                        for (auto& pair : action_probs) pair.second = uniform;
                    }

                    node->expand(action_probs);
                    node->backup(value, c);
                } else {
                    current_eval_nodes.push_back(node);
                    req.p0.push_back(static_cast<int64_t>(p0));
                    req.p1.push_back(static_cast<int64_t>(p1));
                    req.valid_masks.push_back(valid_mask);
                    req.players.push_back(static_cast<int>(c));
                }
            }
        }
        return req;
    }

    void process_evaluation(const std::vector<std::vector<float>>& policies, const std::vector<float>& values) {
        for (size_t i = 0; i < current_eval_nodes.size(); ++i) {
            MCTSNode* node = current_eval_nodes[i];
            float value = values[i];
            const auto& policy = policies[i];
            Color c = node->game.getCurrentPlayer();
            
            Board b = node->game.getBoard();
            uint64_t p0, p1;
            if (c == Color::BLACK) {
                p0 = b.getBlackPieces(); p1 = b.getWhitePieces();
            } else {
                p0 = b.getWhitePieces(); p1 = b.getBlackPieces();
            }
            cache[{p0, p1}] = {policy, value};

            uint64_t valid_mask = b.getValidMovesMask(c);

            std::unordered_map<int, float> action_probs;
            float sum_probs = 0.0;
            for (int a = 0; a < 64; ++a) {
                if ((valid_mask & (1ULL << a)) != 0) {
                    float p = policy[a];
                    action_probs[a] = p;
                    sum_probs += p;
                }
            }

            if (sum_probs > 0) {
                for (auto& pair : action_probs) {
                    pair.second /= sum_probs;
                }
            } else {
                float uniform = 1.0f / action_probs.size();
                for (auto& pair : action_probs) {
                    pair.second = uniform;
                }
            }

            node->expand(action_probs);
            node->backup(value, c);
        }
        current_eval_nodes.clear();
    }

    std::vector<std::vector<float>> get_action_probs(float temperature) {
        std::vector<std::vector<float>> batch_probs;
        for (auto& root : roots) {
            std::vector<float> probs(64, 0.0f);
            if (temperature == 0.0f) {
                int best_a = -1;
                int max_n = -1;
                for (const auto& pair : root->children) {
                    if (pair.second->N > max_n) {
                        max_n = pair.second->N;
                        best_a = pair.first;
                    }
                }
                if (best_a != -1) probs[best_a] = 1.0f;
            } else {
                float sum_visits = 0.0f;
                std::unordered_map<int, float> visits;
                for (const auto& pair : root->children) {
                    float v = std::pow(pair.second->N, 1.0f / temperature);
                    visits[pair.first] = v;
                    sum_visits += v;
                }
                for (const auto& pair : visits) {
                    probs[pair.first] = pair.second / sum_visits;
                }
            }
            batch_probs.push_back(probs);
        }
        return batch_probs;
    }
};

}
