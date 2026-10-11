#include "../game/gamelogic.h"

/*
 * gamelogic.cpp
 * Logics of game rules
 * Some other game logics are in board.h/cpp
 */

using namespace std;

Color GameLogic::checkWinner(const Board& board, const Rules& rules) {
  //RULES.md 5: a player who owns no cells has lost
  int whiteArea = board.getArea(C_WHITE);
  int blackArea = board.getArea(C_BLACK);
  if(whiteArea == 0 && blackArea == 0)
    return C_EMPTY;
  if(whiteArea == 0)
    return C_BLACK;
  if(blackArea == 0)
    return C_WHITE;
  //Technical move limit, counted in full player turns
  if(rules.maxMoves > 0 && board.turnNumber >= rules.maxMoves)
    return C_EMPTY;
  return C_WALL;
}

Color GameLogic::checkWinnerAfterPlayed(
  const Board& board,
  const BoardHistory& hist,
  Player pla,
  Loc loc) {
  (void)pla;
  (void)loc;
  return checkWinner(board, hist.rules);
}
