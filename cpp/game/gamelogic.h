/*
 * gamelogic.h
 * Logics of game rules
 * Some other game logics are in board.h/cpp
 *
 * Influence
 */

#ifndef GAME_GAMELOGIC_H_
#define GAME_GAMELOGIC_H_

#include "../game/boardhistory.h"

namespace GameLogic {

  //Result of the position: C_EMPTY = draw, C_WALL = not finished, else the winner.
  Color checkWinner(const Board& board, const Rules& rules);

  //C_EMPTY = draw, C_WALL = not finished
  Color checkWinnerAfterPlayed( const Board& board, const BoardHistory& hist, Player pla, Loc loc);

}

#endif // GAME_GAMELOGIC_H_
