//Unit tests for the Influence game logic, NN input features and symmetries.
//Run with: katago runinfluencetests

#include <algorithm>
#include <sstream>
#include <vector>

#include "../core/global.h"
#include "../core/rand.h"
#include "../core/test.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../game/gamelogic.h"
#include "../game/graphhash.h"
#include "../game/randomopening.h"
#include "../neuralnet/nninputs.h"
#include "../main.h"

using namespace std;

static Loc L(const Board& b, int x, int y) {
  return Location::getLoc(x, y, b.x_size);
}

static vector<Loc> legalMoves(const Board& board, const BoardHistory& hist) {
  vector<Loc> moves;
  if(hist.isLegal(board, Board::PASS_LOC, board.nextPla))
    moves.push_back(Board::PASS_LOC);
  for(int y = 0; y < board.y_size; y++)
    for(int x = 0; x < board.x_size; x++)
      if(hist.isLegal(board, L(board, x, y), board.nextPla))
        moves.push_back(L(board, x, y));
  return moves;
}

static void play(Board& board, BoardHistory& hist, Loc loc) {
  Player pla = board.nextPla;
  testAssert(hist.isLegal(board, loc, pla));
  hist.makeBoardMoveAssumeLegal(board, loc, pla);
  board.checkConsistency();
}

static void testAdjacency() {
  Board board(5, 5);
  //Neighbours of the center (2,2) are (1,2),(3,2),(2,1),(2,3),(3,3),(1,1) in (x,y)
  Loc c = L(board, 2, 2);
  vector<Loc> expected = {L(board, 1, 2), L(board, 3, 2), L(board, 2, 1), L(board, 2, 3), L(board, 3, 3), L(board, 1, 1)};
  for(Loc n: expected)
    testAssert(Location::isAdjacent(c, n, board.x_size));
  testAssert(!Location::isAdjacent(c, L(board, 3, 1), board.x_size));
  testAssert(!Location::isAdjacent(c, L(board, 1, 3), board.x_size));
  int count = 0;
  for(int i = 0; i < 6; i++)
    if(board.isOnBoard(c + board.adj_offsets[i])) {
      count++;
      testAssert(Location::isAdjacent(c, c + board.adj_offsets[i], board.x_size));
    }
  testAssert(count == 6);
  //Corner (0,0) has neighbours (1,0),(0,1),(1,1); corner (4,0) has (3,0),(4,1) only
  auto countNeighbours = [&](Loc loc) {
    int n = 0;
    for(int i = 0; i < 6; i++)
      if(board.isOnBoard(loc + board.adj_offsets[i]))
        n++;
    return n;
  };
  testAssert(countNeighbours(L(board, 0, 0)) == 3);
  testAssert(countNeighbours(L(board, 4, 0)) == 2);
  testAssert(countNeighbours(L(board, 0, 4)) == 2);
  testAssert(countNeighbours(L(board, 4, 4)) == 3);
  testAssert(Location::distance(L(board, 0, 0), L(board, 4, 4), 5) == 4);
  testAssert(Location::distance(L(board, 4, 0), L(board, 0, 4), 5) == 8);
  cout << "adjacency ok" << endl;
}

static void testDefaultStartAndAttacks() {
  Board board = Board::getDefaultStart(11, 11);
  //White +5 at row 9 col 1, Black -7 at row 1 col 9
  testAssert(board.colors[L(board, 1, 9)] == C_WHITE && board.strength[L(board, 1, 9)] == 5);
  testAssert(board.colors[L(board, 9, 1)] == C_BLACK && board.strength[L(board, 9, 1)] == 7);
  testAssert(board.nextPla == P_WHITE);
  testAssert(board.stage == Board::STAGE_ATTACK_SELECT);
  Rules rules(100);
  BoardHistory hist(board, P_WHITE, rules);
  testAssert(!hist.isGameFinished);

  //White attacks empty (2,9) from (1,9): target gets 4, source 1
  testAssert(!hist.isLegal(board, L(board, 2, 9), P_BLACK));
  testAssert(!hist.isLegal(board, L(board, 2, 9), P_WHITE)); //not own
  play(board, hist, L(board, 1, 9));
  testAssert(board.stage == Board::STAGE_ATTACK_TARGET);
  testAssert(!hist.isLegal(board, Board::PASS_LOC, P_WHITE));
  testAssert(!hist.isLegal(board, L(board, 3, 9), P_WHITE)); //not adjacent
  testAssert(!hist.isLegal(board, L(board, 2, 8), P_WHITE)); //(dx,dy)=(1,-1) is not a hex neighbour
  testAssert(hist.isLegal(board, L(board, 2, 10), P_WHITE)); //(1,1) is a neighbour
  play(board, hist, L(board, 2, 9));
  testAssert(board.strength[L(board, 1, 9)] == 1 && board.colors[L(board, 2, 9)] == C_WHITE && board.strength[L(board, 2, 9)] == 4);
  testAssert(board.stage == Board::STAGE_ATTACK_SELECT && board.nextPla == P_WHITE);
  //Source at strength 1 cannot attack again
  testAssert(!hist.isLegal(board, L(board, 1, 9), P_WHITE));
  //Chain from the captured cell
  play(board, hist, L(board, 2, 9));
  play(board, hist, L(board, 3, 9));
  testAssert(board.strength[L(board, 3, 9)] == 3);
  play(board, hist, L(board, 3, 9));
  play(board, hist, L(board, 4, 9));
  testAssert(board.strength[L(board, 4, 9)] == 2);
  play(board, hist, L(board, 4, 9));
  play(board, hist, L(board, 5, 9));
  testAssert(board.strength[L(board, 5, 9)] == 1);
  //No attack is left: only pass is legal
  vector<Loc> moves = legalMoves(board, hist);
  testAssert(moves.size() == 1 && moves[0] == Board::PASS_LOC);
  //Pass: 5 cells, all strength 1, room 7 each = 35 > 5, so reinforcement is a choice
  play(board, hist, Board::PASS_LOC);
  testAssert(board.stage == Board::STAGE_REINFORCE && board.reinforceLeft == 5 && board.nextPla == P_WHITE);
  testAssert(!hist.isLegal(board, Board::PASS_LOC, P_WHITE));
  testAssert(!hist.isLegal(board, L(board, 6, 9), P_WHITE)); //empty
  for(int i = 0; i < 4; i++)
    play(board, hist, L(board, 5, 9));
  testAssert(board.strength[L(board, 5, 9)] == 5 && board.reinforceLeft == 1);
  play(board, hist, L(board, 1, 9));
  //Turn passed to Black
  testAssert(board.nextPla == P_BLACK && board.stage == Board::STAGE_ATTACK_SELECT && board.turnNumber == 1);
  testAssert(hist.presumedNextMovePla == P_BLACK);
  testAssert(board.getArea(C_WHITE) == 5 && board.getValue(C_WHITE) == 2 + 1 + 1 + 1 + 5);

  //Black: pass immediately, 1 cell at 7 with room 1 >= points 1 -> forced fill and the turn ends at once
  play(board, hist, Board::PASS_LOC);
  testAssert(board.nextPla == P_WHITE && board.stage == Board::STAGE_ATTACK_SELECT && board.turnNumber == 2);
  testAssert(board.strength[L(board, 9, 1)] == 8);
  cout << "default start and basic attacks ok" << endl;
}

static void testCombatCases() {
  Rules rules(0);
  //s > e, s == e, s < e
  Board board = Board::parseBoard(4, 2, R"(
+6 -2 . .
-6 +3 -5 .
)");
  testAssert(board.nextPla == P_WHITE);
  BoardHistory hist(board, P_WHITE, rules);
  play(board, hist, L(board, 0, 0));
  play(board, hist, L(board, 1, 0)); // 6 vs 2 -> White 4
  testAssert(board.colors[L(board, 1, 0)] == C_WHITE && board.strength[L(board, 1, 0)] == 4);
  testAssert(board.strength[L(board, 0, 0)] == 1);
  play(board, hist, L(board, 1, 1));
  play(board, hist, L(board, 2, 1)); // 3 vs 5 -> Black keeps 2
  testAssert(board.colors[L(board, 2, 1)] == C_BLACK && board.strength[L(board, 2, 1)] == 2);
  testAssert(board.colors[L(board, 1, 1)] == C_WHITE && board.strength[L(board, 1, 1)] == 1);
  //(1,0) at 4 attacks (1,1)? own. Attack (2,1) which is (dx,dy)=(1,1): Black 2 -> White 2
  play(board, hist, L(board, 1, 0));
  play(board, hist, L(board, 2, 1));
  testAssert(board.colors[L(board, 2, 1)] == C_WHITE && board.strength[L(board, 2, 1)] == 2);
  //(2,1) at 2 attacks (3,1) empty -> 1
  play(board, hist, L(board, 2, 1));
  play(board, hist, L(board, 3, 1));
  testAssert(board.colors[L(board, 3, 1)] == C_WHITE && board.strength[L(board, 3, 1)] == 1);
  testAssert(!hist.isGameFinished);
  testAssert(board.getArea(C_BLACK) == 1);

  //Equal strength: target becomes empty
  Board b2 = Board::parseBoard(3, 1, "+4 -4 -1");
  BoardHistory h2(b2, P_WHITE, rules);
  play(b2, h2, L(b2, 0, 0));
  play(b2, h2, L(b2, 1, 0));
  testAssert(b2.colors[L(b2, 1, 0)] == C_EMPTY && b2.strength[L(b2, 1, 0)] == 0);
  testAssert(!h2.isGameFinished);

  //Capturing the last enemy cell wins immediately
  Board b3 = Board::parseBoard(3, 1, "+5 -2 .");
  BoardHistory h3(b3, P_WHITE, rules);
  play(b3, h3, L(b3, 0, 0));
  play(b3, h3, L(b3, 1, 0));
  testAssert(h3.isGameFinished && h3.winner == P_WHITE);

  //Overflow: big 12 captures empty small -> target 8, source keeps 1 + 3 = 4 and can attack again
  Board b4 = Board::parseBoard(2, 2, R"(
+12* .
.* -1
)");
  testAssert(b4.isBig(L(b4, 0, 0)) && !b4.isBig(L(b4, 1, 0)) && b4.isBig(L(b4, 0, 1)));
  BoardHistory h4(b4, P_WHITE, rules);
  play(b4, h4, L(b4, 0, 0));
  play(b4, h4, L(b4, 1, 0));
  testAssert(b4.strength[L(b4, 1, 0)] == 8 && b4.strength[L(b4, 0, 0)] == 4);
  testAssert(b4.getValue(C_WHITE) == 12);
  //Source at 4 attacks the empty big cell below: 3 fits in a big cell, no overflow
  play(b4, h4, L(b4, 0, 0));
  play(b4, h4, L(b4, 0, 1));
  testAssert(b4.strength[L(b4, 0, 1)] == 3 && b4.strength[L(b4, 0, 0)] == 1);
  //Big 12 captures an enemy small cell at 2: 10 -> target 8, source 1 + 2 = 3
  Board b5 = Board::parseBoard(3, 1, "+12* -2 -1");
  BoardHistory h5(b5, P_WHITE, rules);
  play(b5, h5, L(b5, 0, 0));
  play(b5, h5, L(b5, 1, 0));
  testAssert(b5.colors[L(b5, 1, 0)] == C_WHITE && b5.strength[L(b5, 1, 0)] == 8 && b5.strength[L(b5, 0, 0)] == 3);
  //Starting values above the cap are rejected
  bool threw = false;
  try {
    Board::parseBoard(2, 1, "+9 -1");
  } catch(const StringError&) {
    threw = true;
  }
  testAssert(threw);
  cout << "combat cases ok" << endl;
}

static void testWallsAndReinforce() {
  Rules rules(0);
  //Wall blocks the attack, reinforcement auto-fill when only one cell has room
  Board board = Board::parseBoard(3, 2, R"(
+3 # -2
8 . .
)");
  BoardHistory hist(board, P_WHITE, rules);
  testAssert(!board.isOnBoard(L(board, 1, 0)));
  play(board, hist, L(board, 0, 0));
  testAssert(!hist.isLegal(board, L(board, 1, 0), P_WHITE));
  testAssert(hist.isLegal(board, L(board, 1, 1), P_WHITE)); //(1,1) diagonal neighbour
  play(board, hist, L(board, 1, 1));
  //Now (0,0)=1, (0,1)=8, (1,1)=2. Pass: 3 points. Room: 7 + 0 + 6 = 13, two cells with room -> choice
  play(board, hist, Board::PASS_LOC);
  testAssert(board.stage == Board::STAGE_REINFORCE && board.reinforceLeft == 3);
  play(board, hist, L(board, 0, 0));
  play(board, hist, L(board, 0, 0));
  play(board, hist, L(board, 0, 0));
  testAssert(board.nextPla == P_BLACK);

  //Leftover points are lost when every cell is capped
  Board b2 = Board::parseBoard(3, 1, "+8 +8 -1");
  BoardHistory h2(b2, P_WHITE, rules);
  play(b2, h2, Board::PASS_LOC);
  testAssert(b2.nextPla == P_BLACK && b2.turnNumber == 1 && b2.getValue(C_WHITE) == 16);

  //One cell with room takes all points
  Board b3 = Board::parseBoard(4, 1, "+8 +8 +1 -1");
  BoardHistory h3(b3, P_WHITE, rules);
  play(b3, h3, Board::PASS_LOC);
  testAssert(b3.nextPla == P_BLACK && b3.strength[L(b3, 2, 0)] == 4);
  cout << "walls and reinforcement ok" << endl;
}

static void testMoveLimit() {
  Rules rules(3);
  Board board = Board::parseBoard(3, 1, "+8 . -8");
  BoardHistory hist(board, P_WHITE, rules);
  play(board, hist, Board::PASS_LOC);
  play(board, hist, Board::PASS_LOC);
  testAssert(!hist.isGameFinished);
  play(board, hist, Board::PASS_LOC);
  testAssert(hist.isGameFinished && hist.winner == C_EMPTY);
  cout << "move limit ok" << endl;
}

static void testParseRoundTrip(const Board& board) {
  string s = Board::toStringSimple(board, '|');
  Board b2 = Board::parseBoard(board.x_size, board.y_size, s, '|');
  testAssert(b2.pos_hash == board.pos_hash);
  b2.movenum = board.movenum;
  testAssert(b2.isEqualForTesting(board));
}

//Play random games, check consistency, hashing, parse round trip, and that every symmetry commutes with moves.
static void testRandomGames(Rand& rand, bool print) {
  int numSyms = SymmetryHelpers::NUM_SYMMETRIES;
  int totalMoves = 0;
  for(int game = 0; game < 60; game++) {
    int xSize = 3 + rand.nextUInt(9);
    int ySize = (game % 3 == 0) ? 3 + rand.nextUInt(9) : xSize;
    Board board = RandomOpening::initStartPosition(xSize, ySize, rand, 0.5, 0.5, 0.3);
    if(game % 4 == 0) {
      for(int i = 0; i < 3; i++) {
        Loc loc = L(board, rand.nextUInt(xSize), rand.nextUInt(ySize));
        if(board.isOnBoard(loc) && board.colors[loc] == C_EMPTY)
          board.setBig(loc, true);
      }
    }
    Rules rules(40);
    BoardHistory hist(board, board.nextPla, rules);
    vector<Board> symBoards;
    vector<BoardHistory> symHists;
    for(int s = 0; s < numSyms; s++) {
      symBoards.push_back(SymmetryHelpers::getSymBoard(board, s));
      symHists.push_back(BoardHistory(symBoards[s], symBoards[s].nextPla, rules));
    }

    for(int moveIdx = 0; moveIdx < 50000 && !hist.isGameFinished; moveIdx++) {
      vector<Loc> moves = legalMoves(board, hist);
      testAssert(moves.size() > 0);
      //Bias towards attacking and away from passing, so games progress
      Loc loc = moves[rand.nextUInt((uint32_t)moves.size())];
      if(loc == Board::PASS_LOC && moves.size() > 1 && rand.nextBool(0.8))
        loc = moves[1 + rand.nextUInt((uint32_t)moves.size() - 1)];

      for(int s = 0; s < numSyms; s++) {
        Loc symLoc = SymmetryHelpers::getSymLoc(loc, board, s);
        //Legal move sets must correspond
        vector<Loc> symMoves = legalMoves(symBoards[s], symHists[s]);
        testAssert(symMoves.size() == moves.size());
        testAssert(symHists[s].isLegal(symBoards[s], symLoc, symBoards[s].nextPla));
        symHists[s].makeBoardMoveAssumeLegal(symBoards[s], symLoc, symBoards[s].nextPla);
      }
      play(board, hist, loc);
      totalMoves++;
      for(int s = 0; s < numSyms; s++) {
        Board expected = SymmetryHelpers::getSymBoard(board, s);
        testAssert(expected.pos_hash == symBoards[s].pos_hash);
        testAssert(symHists[s].isGameFinished == hist.isGameFinished);
        testAssert(symHists[s].winner == hist.winner);
      }
      if(moveIdx % 7 == 0)
        testParseRoundTrip(board);
    }
    if(print && game < 2)
      Board::printBoard(cout, board, Board::NULL_LOC, &hist.moveHistory);
    testAssert(hist.isGameFinished);
  }
  cout << "random games ok, " << totalMoves << " moves" << endl;
}

//The NN symmetry applied to the input tensor must equal the input features of the symmetric board.
static void testNNInputSymmetry(Rand& rand) {
  const int nSpatial = NNInputs::NUM_FEATURES_SPATIAL_V102;
  const int nGlobal = NNInputs::NUM_FEATURES_GLOBAL_V102;
  for(int game = 0; game < 20; game++) {
    int size = 4 + rand.nextUInt(7);
    int nnLen = size + (game % 2 == 0 ? 0 : 3);
    Board board = RandomOpening::initStartPosition(size, size, rand, 0.7, 0.5, 0.3);
    Rules rules(50);
    BoardHistory hist(board, board.nextPla, rules);
    for(int moveIdx = 0; moveIdx < 400 && !hist.isGameFinished; moveIdx++) {
      vector<Loc> moves = legalMoves(board, hist);
      Loc loc = moves[rand.nextUInt((uint32_t)moves.size())];
      play(board, hist, loc);
      if(hist.isGameFinished || moveIdx % 5 != 0)
        continue;
      for(int useNHWC = 0; useNHWC < 2; useNHWC++) {
        vector<float> bin(nSpatial * nnLen * nnLen);
        vector<float> glob(nGlobal);
        MiscNNInputParams params;
        NNInputs::fillRowV102(board, hist, board.nextPla, params, nnLen, nnLen, useNHWC, bin.data(), glob.data());
        for(int s = 0; s < SymmetryHelpers::NUM_SYMMETRIES; s++) {
          if(nnLen != size && s != 0)
            continue; //Rotating a padded tensor moves the board away from the corner, can't compare directly
          Board symBoard = SymmetryHelpers::getSymBoard(board, s);
          BoardHistory symHist(symBoard, symBoard.nextPla, rules);
          vector<float> symBin(nSpatial * nnLen * nnLen);
          vector<float> symGlob(nGlobal);
          NNInputs::fillRowV102(symBoard, symHist, symBoard.nextPla, params, nnLen, nnLen, useNHWC, symBin.data(), symGlob.data());
          vector<float> transformed(nSpatial * nnLen * nnLen);
          SymmetryHelpers::copyInputsWithSymmetry(bin.data(), transformed.data(), 1, nnLen, nnLen, nSpatial, useNHWC, s);
          for(size_t i = 0; i < transformed.size(); i++)
            testAssert(transformed[i] == symBin[i]);
          for(int i = 0; i < nGlobal; i++)
            testAssert(glob[i] == symGlob[i]);
          for(float v: bin)
            testAssert(v == 0.0f || v == 1.0f);

          //Policy outputs: un-transforming a policy laid out on the symmetric board must give back the original locs
          vector<float> symPolicy(nnLen * nnLen, 0.0f);
          vector<float> policy(nnLen * nnLen, 0.0f);
          for(int y = 0; y < size; y++) {
            for(int x = 0; x < size; x++) {
              Loc loc2 = L(board, x, y);
              Loc symLoc = SymmetryHelpers::getSymLoc(loc2, board, s);
              symPolicy[NNPos::locToPos(symLoc, symBoard.x_size, nnLen, nnLen)] = (float)(1 + NNPos::locToPos(loc2, size, nnLen, nnLen));
            }
          }
          SymmetryHelpers::copyOutputsWithSymmetry(symPolicy.data(), policy.data(), 1, nnLen, nnLen, s);
          for(int y = 0; y < size; y++)
            for(int x = 0; x < size; x++)
              testAssert(policy[NNPos::xyToPos(x, y, nnLen)] == (float)(1 + NNPos::xyToPos(x, y, nnLen)));
        }
      }
    }
  }
  cout << "nn input symmetry ok" << endl;
}

static void testGraphHashTranspositions() {
  //Reinforcing two cells in different orders reaches the same graph hash
  Rules rules(0);
  Board base = Board::parseBoard(4, 1, "+1 +1 +1 -8");
  BoardHistory baseHist(base, P_WHITE, rules);
  Board b1 = base, b2 = base;
  BoardHistory h1 = baseHist, h2 = baseHist;
  play(b1, h1, Board::PASS_LOC);
  play(b2, h2, Board::PASS_LOC);
  play(b1, h1, L(b1, 0, 0));
  play(b1, h1, L(b1, 1, 0));
  play(b2, h2, L(b2, 1, 0));
  play(b2, h2, L(b2, 0, 0));
  testAssert(GraphHash::getGraphHash(h1, b1.nextPla) == GraphHash::getGraphHash(h2, b2.nextPla));
  //Different turn numbers give different hashes even for identical cells
  Board c1 = Board::parseBoard(3, 1, "+8 . -8");
  BoardHistory hc(c1, P_WHITE, rules);
  Hash128 hashBefore = c1.pos_hash;
  play(c1, hc, Board::PASS_LOC);
  play(c1, hc, Board::PASS_LOC);
  testAssert(c1.nextPla == P_WHITE && c1.pos_hash != hashBefore);
  cout << "graph hash ok" << endl;
}

int MainCmds::runinfluencetests(const vector<string>& args) {
  Board::initHash();
  bool print = std::find(args.begin(), args.end(), string("-print")) != args.end();
  Rand rand("influence tests");
  testAdjacency();
  testDefaultStartAndAttacks();
  testCombatCases();
  testWallsAndReinforce();
  testMoveLimit();
  testGraphHashTranspositions();
  testRandomGames(rand, print);
  testNNInputSymmetry(rand);
  cout << "All influence tests passed" << endl;
  return 0;
}
