#include "../game/randomopening.h"

#include <vector>

using namespace std;

Board RandomOpening::initStartPosition(
  int xSize,
  int ySize,
  Rand& rand,
  double randomStartProb,
  double randomWallsProb,
  double maxWallDensity) {
  Board defaultBoard = Board::getDefaultStart(xSize, ySize);
  Board board(xSize, ySize);

  if(randomWallsProb > 0 && rand.nextBool(randomWallsProb)) {
    double density = rand.nextDouble() * maxWallDensity;
    for(int y = 0; y < ySize; y++)
      for(int x = 0; x < xSize; x++)
        if(rand.nextBool(density))
          board.setWall(Location::getLoc(x, y, xSize), true);
  }

  vector<Loc> cells;
  for(int y = 0; y < ySize; y++) {
    for(int x = 0; x < xSize; x++) {
      Loc loc = Location::getLoc(x, y, xSize);
      if(board.isOnBoard(loc))
        cells.push_back(loc);
    }
  }

  if(randomStartProb > 0 && rand.nextBool(randomStartProb) && cells.size() >= 2) {
    //Random starting cells, at hex distance at least 2 when possible so that they don't start in contact
    Loc wLoc = cells[rand.nextUInt((uint32_t)cells.size())];
    Loc bLoc = Board::NULL_LOC;
    for(int tries = 0; tries < 100; tries++) {
      Loc candidate = cells[rand.nextUInt((uint32_t)cells.size())];
      if(candidate == wLoc)
        continue;
      bLoc = candidate;
      if(Location::distance(wLoc, candidate, xSize) >= 2)
        break;
    }
    if(bLoc == Board::NULL_LOC)
      return initStartPosition(xSize, ySize, rand, 0.0, 0.0, 0.0);
    int wStrength = 2 + (int)rand.nextUInt(Board::SMALL_CAP - 1);
    int bStrength = 2 + (int)rand.nextUInt(Board::SMALL_CAP - 1);
    board.setCell(wLoc, C_WHITE, wStrength);
    board.setCell(bLoc, C_BLACK, bStrength);
  } else {
    for(int y = 0; y < ySize; y++) {
      for(int x = 0; x < xSize; x++) {
        Loc loc = Location::getLoc(x, y, xSize);
        if(defaultBoard.colors[loc] == C_BLACK || defaultBoard.colors[loc] == C_WHITE) {
          board.setWall(loc, false);
          board.setCell(loc, defaultBoard.colors[loc], defaultBoard.strength[loc]);
        }
      }
    }
  }
  return board;
}
