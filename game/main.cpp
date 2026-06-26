#include "game/ui.hpp"

int main() {
  reversi::Game game;
  reversi::TerminalUI ui;
  ui.play(game);
  return 0;
}
