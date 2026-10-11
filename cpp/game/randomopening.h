#pragma once
#include "../core/rand.h"
#include "board.h"

namespace RandomOpening {

  //Create a starting position for selfplay on an xSize*ySize box. White moves first.
  //- With probability randomWallsProb, cells become walls independently with a random density in [0, maxWallDensity].
  //- With probability randomStartProb, the two starting cells and their strengths are random,
  //  otherwise the default setup of RULES.md 3 is used.
  //All cells are small for now.
  Board initStartPosition(
    int xSize,
    int ySize,
    Rand& rand,
    double randomStartProb,
    double randomWallsProb,
    double maxWallDensity);

}  // namespace RandomOpening
