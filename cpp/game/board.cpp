#include "../game/board.h"

/*
 * board.cpp
 * Originally from an unreleased project back in 2010, modified since.
 * Authors: brettharrison (original), David Wu (original and later modificationss).
 *
 * Influence game logic, see RULES.md and board.h.
 */

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "../core/rand.h"

using namespace std;

// STATIC VARS-----------------------------------------------------------------------------
bool Board::IS_ZOBRIST_INITALIZED = false;
Hash128 Board::ZOBRIST_SIZE_X_HASH[MAX_LEN + 1];
Hash128 Board::ZOBRIST_SIZE_Y_HASH[MAX_LEN + 1];
Hash128 Board::ZOBRIST_CELL_HASH[MAX_ARR_SIZE][3][MAX_CAP + 1];
Hash128 Board::ZOBRIST_WALL_HASH[MAX_ARR_SIZE];
Hash128 Board::ZOBRIST_BIG_HASH[MAX_ARR_SIZE];
Hash128 Board::ZOBRIST_STAGE_HASH[NUM_STAGES];
Hash128 Board::ZOBRIST_SELECTED_HASH[MAX_ARR_SIZE];
Hash128 Board::ZOBRIST_REINFORCE_HASH_BASE;
Hash128 Board::ZOBRIST_TURNNUM_HASH_BASE;
Hash128 Board::ZOBRIST_BOARD_HASH2[MAX_ARR_SIZE][4];
Hash128 Board::ZOBRIST_PLAYER_HASH[6];
const Hash128 Board::ZOBRIST_GAME_IS_OVER =  // Based on sha256 hash of Board::ZOBRIST_GAME_IS_OVER
  Hash128(0xb6f9e465597a77eeULL, 0xf1d583d960a4ce7fULL);

// LOCATION--------------------------------------------------------------------------------
Loc Location::getLoc(int x, int y, int x_size) {
  return (x + 1) + (y + 1) * (x_size + 1);
}
int Location::getX(Loc loc, int x_size) {
  return (loc % (x_size + 1)) - 1;
}
int Location::getY(Loc loc, int x_size) {
  return (loc / (x_size + 1)) - 1;
}
void Location::getAdjacentOffsets(short adj_offsets[8], int x_size) {
  // (dy,dx) = (-1,0), (0,-1), (0,1), (1,0), (-1,-1), (1,1)
  adj_offsets[0] = -(x_size + 1);
  adj_offsets[1] = -1;
  adj_offsets[2] = 1;
  adj_offsets[3] = (x_size + 1);
  adj_offsets[4] = -(x_size + 1) - 1;
  adj_offsets[5] = (x_size + 1) + 1;
  adj_offsets[6] = 0;
  adj_offsets[7] = 0;
}

bool Location::isAdjacent(Loc loc0, Loc loc1, int x_size) {
  int d = loc1 - loc0;
  return d == -(x_size + 1) || d == -1 || d == 1 || d == (x_size + 1) || d == -(x_size + 2) || d == (x_size + 2);
}

Loc Location::getCenterLoc(int x_size, int y_size) {
  if(x_size % 2 == 0 || y_size % 2 == 0)
    return Board::NULL_LOC;
  return getLoc(x_size / 2, y_size / 2, x_size);
}

Loc Location::getCenterLoc(const Board& b) {
  return getCenterLoc(b.x_size, b.y_size);
}

bool Location::isCentral(Loc loc, int x_size, int y_size) {
  int x = getX(loc, x_size);
  int y = getY(loc, x_size);
  return x >= (x_size - 1) / 2 && x <= x_size / 2 && y >= (y_size - 1) / 2 && y <= y_size / 2;
}

bool Location::isNearCentral(Loc loc, int x_size, int y_size) {
  int x = getX(loc, x_size);
  int y = getY(loc, x_size);
  return x >= (x_size - 1) / 2 - 1 && x <= x_size / 2 + 1 && y >= (y_size - 1) / 2 - 1 && y <= y_size / 2 + 1;
}

int Location::distance(Loc loc0, Loc loc1, int x_size) {
  int dx = getX(loc1, x_size) - getX(loc0, x_size);
  int dy = getY(loc1, x_size) - getY(loc0, x_size);
  //Hex distance with neighbour vectors (0,+-1),(+-1,0),(1,1),(-1,-1)
  if((dx >= 0) == (dy >= 0))
    return std::max(std::abs(dx), std::abs(dy));
  return std::abs(dx) + std::abs(dy);
}

int Location::euclideanDistanceSquared(Loc loc0, Loc loc1, int x_size) {
  //In units of hex rows: drawn x = col - row/2, drawn y = row * sqrt(3)/2. Returns 4 * squared drawn distance.
  int dx = getX(loc1, x_size) - getX(loc0, x_size);
  int dy = getY(loc1, x_size) - getY(loc0, x_size);
  int ddx = 2 * dx - dy;
  return ddx * ddx + 3 * dy * dy;
}

// CONSTRUCTORS AND INITIALIZATION----------------------------------------------------------

Board::Board() {
  init(DEFAULT_LEN, DEFAULT_LEN);
}

Board::Board(int x, int y) {
  init(x, y);
}

Board::Board(const Board& other) = default;

void Board::init(int xS, int yS) {
  assert(IS_ZOBRIST_INITALIZED);
  if(xS < 1 || yS < 1 || xS > MAX_LEN || yS > MAX_LEN)
    throw StringError("Board::init - invalid board size");

  x_size = xS;
  y_size = yS;

  for(int i = 0; i < MAX_ARR_SIZE; i++) {
    colors[i] = C_WALL;
    strength[i] = 0;
    cap[i] = 0;
  }
  for(int y = 0; y < y_size; y++) {
    for(int x = 0; x < x_size; x++) {
      Loc loc = Location::getLoc(x, y, x_size);
      colors[loc] = C_EMPTY;
      cap[loc] = SMALL_CAP;
    }
  }
  for(int i = 0; i < 4; i++) {
    area[i] = 0;
    value[i] = 0;
  }
  area[C_EMPTY] = x_size * y_size;

  nextPla = P_WHITE;
  stage = STAGE_ATTACK_SELECT;
  selectedSrc = NULL_LOC;
  reinforceLeft = 0;
  turnNumber = 0;
  movenum = 0;

  cells_hash = ZOBRIST_SIZE_X_HASH[x_size] ^ ZOBRIST_SIZE_Y_HASH[y_size];
  pos_hash = cells_hash ^ turnStateHash();

  Location::getAdjacentOffsets(adj_offsets, x_size);
}

Board Board::getDefaultStart(int x, int y) {
  Board board(x, y);
  //RULES.md 3: offset d=1, White +5 at (row N-1-d, col d), Black -7 at (row d, col N-1-d)
  //On tiny boards use the corners (d=0) so the two cells don't coincide.
  const int d = (x >= 4 && y >= 4) ? 1 : 0;
  int wx = std::min(d, x - 1);
  int wy = std::max(y - 1 - d, 0);
  int bx = std::max(x - 1 - d, 0);
  int by = std::min(d, y - 1);
  Loc wLoc = Location::getLoc(wx, wy, x);
  Loc bLoc = Location::getLoc(bx, by, x);
  if(wLoc == bLoc)
    throw StringError("Board::getDefaultStart - board too small");
  board.setCell(wLoc, C_WHITE, 5);
  board.setCell(bLoc, C_BLACK, 7);
  return board;
}

void Board::initHash() {
  if(IS_ZOBRIST_INITALIZED)
    return;
  Rand rand("Board::initHash() influence");

  auto nextHash = [&rand]() {
    uint64_t h0 = rand.nextUInt64();
    uint64_t h1 = rand.nextUInt64();
    return Hash128(h0, h1);
  };

  for(int i = 0; i < 6; i++)
    ZOBRIST_PLAYER_HASH[i] = nextHash();
  for(int i = 0; i < NUM_STAGES; i++)
    ZOBRIST_STAGE_HASH[i] = nextHash();
  ZOBRIST_REINFORCE_HASH_BASE = nextHash();
  ZOBRIST_TURNNUM_HASH_BASE = nextHash();

  for(int i = 0; i < MAX_ARR_SIZE; i++) {
    for(int c = 0; c < 3; c++) {
      for(int s = 0; s <= MAX_CAP; s++) {
        if(c == C_EMPTY)
          ZOBRIST_CELL_HASH[i][c][s] = Hash128();
        else
          ZOBRIST_CELL_HASH[i][c][s] = nextHash();
      }
    }
    ZOBRIST_WALL_HASH[i] = nextHash();
    ZOBRIST_BIG_HASH[i] = nextHash();
    ZOBRIST_SELECTED_HASH[i] = nextHash();
  }
  ZOBRIST_SELECTED_HASH[NULL_LOC] = Hash128();

  // Reseed the random number generator so that these size hashes are also
  // not affected by the size of the board we compile with
  rand.init("Board::initHash() for ZOBRIST_SIZE hashes");
  for(int i = 0; i < MAX_LEN + 1; i++) {
    ZOBRIST_SIZE_X_HASH[i] = nextHash();
    ZOBRIST_SIZE_Y_HASH[i] = nextHash();
  }

  // Reseed and compute one more set of zobrist hashes, mixed a bit differently
  rand.init("Board::initHash() for second set of ZOBRIST hashes");
  for(int i = 0; i < MAX_ARR_SIZE; i++) {
    for(Color j = 0; j < 4; j++) {
      ZOBRIST_BOARD_HASH2[i][j] = nextHash();
      ZOBRIST_BOARD_HASH2[i][j].hash0 = Hash::murmurMix(ZOBRIST_BOARD_HASH2[i][j].hash0);
      ZOBRIST_BOARD_HASH2[i][j].hash1 = Hash::splitMix64(ZOBRIST_BOARD_HASH2[i][j].hash1);
    }
  }

  IS_ZOBRIST_INITALIZED = true;
}

Hash128 Board::turnStateHash() const {
  Hash128 h = ZOBRIST_STAGE_HASH[stage];
  h ^= ZOBRIST_SELECTED_HASH[selectedSrc];
  if(reinforceLeft != 0)
    h ^= Hash128::mixInt(ZOBRIST_REINFORCE_HASH_BASE, reinforceLeft);
  //The turn number is part of the hash so that graph search never sees cycles
  //(positions can repeat exactly across turns, e.g. both players fully capped and passing).
  h ^= Hash128::mixInt(ZOBRIST_TURNNUM_HASH_BASE, turnNumber);
  return h;
}

// CELL QUERIES ----------------------------------------------------------------------------

bool Board::isOnBoard(Loc loc) const {
  return loc >= 0 && loc < MAX_ARR_SIZE && colors[loc] != C_WALL;
}

bool Board::isEmpty() const {
  return area[C_BLACK] == 0 && area[C_WHITE] == 0;
}

int Board::numStonesOnBoard() const {
  return area[C_BLACK] + area[C_WHITE];
}

int Board::numPlaStonesOnBoard(Player pla) const {
  if(pla != P_BLACK && pla != P_WHITE)
    return 0;
  return area[pla];
}

int Board::numCells() const {
  int n = 0;
  for(int y = 0; y < y_size; y++)
    for(int x = 0; x < x_size; x++)
      if(colors[Location::getLoc(x, y, x_size)] != C_WALL)
        n++;
  return n;
}

int Board::getCapacity(Player pla) const {
  int c = 0;
  for(int y = 0; y < y_size; y++) {
    for(int x = 0; x < x_size; x++) {
      Loc loc = Location::getLoc(x, y, x_size);
      if(colors[loc] == pla && strength[loc] < cap[loc])
        c += cap[loc] - strength[loc];
    }
  }
  return c;
}

bool Board::canAttackFrom(Loc loc, Player pla) const {
  if(loc < 0 || loc >= MAX_ARR_SIZE || colors[loc] != pla || strength[loc] < 2)
    return false;
  FOREACH_HEX_ADJ(loc, adj, {
    Color c = colors[adj];
    if(c != C_WALL && c != pla)
      return true;
  });
  return false;
}

bool Board::anyAttackAvailable(Player pla) const {
  for(int y = 0; y < y_size; y++)
    for(int x = 0; x < x_size; x++)
      if(canAttackFrom(Location::getLoc(x, y, x_size), pla))
        return true;
  return false;
}

// MOVES -----------------------------------------------------------------------------------

bool Board::isLegal(Loc loc, Player pla) const {
  if(pla != nextPla)
    return false;
  if(pla != P_BLACK && pla != P_WHITE)
    return false;
  if(stage == STAGE_ATTACK_SELECT) {
    if(loc == PASS_LOC)
      return true;
    return canAttackFrom(loc, pla);
  }
  if(loc == PASS_LOC || loc < 0 || loc >= MAX_ARR_SIZE)
    return false;
  if(stage == STAGE_ATTACK_TARGET) {
    Color c = colors[loc];
    return c != C_WALL && c != pla && Location::isAdjacent(selectedSrc, loc, x_size);
  }
  if(stage == STAGE_REINFORCE) {
    return colors[loc] == pla && strength[loc] < cap[loc];
  }
  return false;
}

void Board::setCellRaw(Loc loc, Color color, int str) {
  Color oldColor = colors[loc];
  int oldStr = strength[loc];
  assert(oldColor != C_WALL);
  cells_hash ^= ZOBRIST_CELL_HASH[loc][oldColor][oldStr];
  area[oldColor] -= 1;
  value[oldColor] -= oldStr;
  if(str <= 0) {
    color = C_EMPTY;
    str = 0;
  }
  colors[loc] = color;
  strength[loc] = (uint8_t)str;
  cells_hash ^= ZOBRIST_CELL_HASH[loc][color][str];
  area[color] += 1;
  value[color] += str;
}

void Board::endTurn() {
  nextPla = getOpp(nextPla);
  stage = STAGE_ATTACK_SELECT;
  selectedSrc = NULL_LOC;
  reinforceLeft = 0;
  turnNumber += 1;
}

//If spending the remaining points involves no choice, spend them and end the turn.
void Board::maybeAutoReinforce() {
  assert(stage == STAGE_REINFORCE);
  Player pla = nextPla;
  int capacity = 0;
  int numCellsWithRoom = 0;
  for(int y = 0; y < y_size; y++) {
    for(int x = 0; x < x_size; x++) {
      Loc loc = Location::getLoc(x, y, x_size);
      if(colors[loc] == pla && strength[loc] < cap[loc]) {
        capacity += cap[loc] - strength[loc];
        numCellsWithRoom++;
      }
    }
  }
  if(reinforceLeft > 0 && reinforceLeft < capacity && numCellsWithRoom > 1)
    return;
  //Forced: either every cell can be filled to its cap (leftover points are lost), or only one cell can take points.
  for(int y = 0; y < y_size && reinforceLeft > 0; y++) {
    for(int x = 0; x < x_size && reinforceLeft > 0; x++) {
      Loc loc = Location::getLoc(x, y, x_size);
      if(colors[loc] == pla && strength[loc] < cap[loc]) {
        int add = std::min(reinforceLeft, cap[loc] - (int)strength[loc]);
        setCellRaw(loc, pla, strength[loc] + add);
        reinforceLeft -= add;
      }
    }
  }
  endTurn();
}

void Board::beginReinforce() {
  stage = STAGE_REINFORCE;
  selectedSrc = NULL_LOC;
  reinforceLeft = area[nextPla];
  maybeAutoReinforce();
}

void Board::playMoveAssumeLegal(Loc loc, Player pla) {
  assert(pla == nextPla);
  (void)pla;
  movenum++;

  if(stage == STAGE_ATTACK_SELECT) {
    if(loc == PASS_LOC) {
      beginReinforce();
    } else {
      stage = STAGE_ATTACK_TARGET;
      selectedSrc = loc;
    }
  } else if(stage == STAGE_ATTACK_TARGET) {
    Loc src = selectedSrc;
    int s = strength[src];
    Color tColor = colors[loc];
    int e = strength[loc];
    //Overflow (RULES.md 4.2): a captured cell gets at most its cap, the excess stays in the source.
    //Only possible when a big cell captures a small one.
    int overflow = 0;
    if(tColor == C_EMPTY || s > e) {
      int captured = tColor == C_EMPTY ? s - 1 : s - e;
      overflow = std::max(0, captured - (int)cap[loc]);
      setCellRaw(loc, nextPla, captured - overflow);
    } else if(s == e)
      setCellRaw(loc, C_EMPTY, 0);
    else
      setCellRaw(loc, tColor, e - s);
    assert(1 + overflow <= cap[src]);
    setCellRaw(src, nextPla, 1 + overflow);
    stage = STAGE_ATTACK_SELECT;
    selectedSrc = NULL_LOC;
  } else {
    assert(stage == STAGE_REINFORCE);
    setCellRaw(loc, nextPla, strength[loc] + 1);
    reinforceLeft -= 1;
    maybeAutoReinforce();
  }

  pos_hash = cells_hash ^ turnStateHash();
}

// SETUP -----------------------------------------------------------------------------------

bool Board::setCell(Loc loc, Color color, int str) {
  if(!isOnBoard(loc))
    return false;
  if(color != C_EMPTY && color != C_BLACK && color != C_WHITE)
    return false;
  if(color == C_EMPTY)
    str = 0;
  else if(str < 1 || str > cap[loc])
    return false;
  setCellRaw(loc, color, str);
  pos_hash = cells_hash ^ turnStateHash();
  return true;
}

bool Board::setWall(Loc loc, bool isWall) {
  if(loc < 0 || loc >= MAX_ARR_SIZE)
    return false;
  int x = Location::getX(loc, x_size);
  int y = Location::getY(loc, x_size);
  if(x < 0 || x >= x_size || y < 0 || y >= y_size)
    return false;
  bool wasWall = colors[loc] == C_WALL;
  if(wasWall == isWall)
    return true;
  if(isWall) {
    setCellRaw(loc, C_EMPTY, 0);
    if(cap[loc] == BIG_CAP)
      cells_hash ^= ZOBRIST_BIG_HASH[loc];
    area[C_EMPTY] -= 1;
    colors[loc] = C_WALL;
    cap[loc] = 0;
  } else {
    colors[loc] = C_EMPTY;
    cap[loc] = SMALL_CAP;
    area[C_EMPTY] += 1;
  }
  cells_hash ^= ZOBRIST_WALL_HASH[loc];
  if(selectedSrc == loc)
    selectedSrc = NULL_LOC;
  pos_hash = cells_hash ^ turnStateHash();
  return true;
}

bool Board::setBig(Loc loc, bool big) {
  if(!isOnBoard(loc))
    return false;
  int newCap = big ? BIG_CAP : SMALL_CAP;
  if(strength[loc] > newCap)
    return false;
  if(cap[loc] != newCap) {
    cap[loc] = (uint8_t)newCap;
    cells_hash ^= ZOBRIST_BIG_HASH[loc];
    pos_hash = cells_hash ^ turnStateHash();
  }
  return true;
}

bool Board::setTurnState(Player pla, int stg, Loc src, int rLeft, int tNum) {
  if(pla != P_BLACK && pla != P_WHITE)
    return false;
  if(stg < 0 || stg >= NUM_STAGES || tNum < 0 || rLeft < 0)
    return false;
  if(stg == STAGE_ATTACK_TARGET) {
    if(!canAttackFrom(src, pla))
      return false;
  } else if(src != NULL_LOC)
    return false;
  if(stg != STAGE_REINFORCE && rLeft != 0)
    return false;
  if(stg == STAGE_REINFORCE && rLeft == 0)
    return false;
  nextPla = pla;
  stage = stg;
  selectedSrc = src;
  reinforceLeft = rLeft;
  turnNumber = tNum;
  pos_hash = cells_hash ^ turnStateHash();
  return true;
}

Hash128 Board::getSitHash(Player pla) const {
  Hash128 h = pos_hash;
  h ^= Board::ZOBRIST_PLAYER_HASH[pla];
  return h;
}

// CONSISTENCY -----------------------------------------------------------------------------

void Board::checkConsistency() const {
  const string errLabel = string("Board::checkConsistency(): ");

  Hash128 tmpHash = ZOBRIST_SIZE_X_HASH[x_size] ^ ZOBRIST_SIZE_Y_HASH[y_size];
  int tmpArea[4] = {0, 0, 0, 0};
  int tmpValue[4] = {0, 0, 0, 0};
  for(Loc loc = 0; loc < MAX_ARR_SIZE; loc++) {
    int x = Location::getX(loc, x_size);
    int y = Location::getY(loc, x_size);
    bool inBox = loc >= 0 && x >= 0 && x < x_size && y >= 0 && y < y_size && loc == Location::getLoc(x, y, x_size);
    if(!inBox) {
      if(colors[loc] != C_WALL)
        throw StringError(errLabel + "Non-WALL value outside of board legal area");
      continue;
    }
    if(colors[loc] == C_WALL) {
      if(strength[loc] != 0 || cap[loc] != 0)
        throw StringError(errLabel + "Wall with strength or cap");
      tmpHash ^= ZOBRIST_WALL_HASH[loc];
      continue;
    }
    if(cap[loc] != SMALL_CAP && cap[loc] != BIG_CAP)
      throw StringError(errLabel + "Invalid cap");
    if(cap[loc] == BIG_CAP)
      tmpHash ^= ZOBRIST_BIG_HASH[loc];
    if(colors[loc] == C_EMPTY) {
      if(strength[loc] != 0)
        throw StringError(errLabel + "Empty cell with strength");
    } else if(colors[loc] == C_BLACK || colors[loc] == C_WHITE) {
      if(strength[loc] < 1 || strength[loc] > cap[loc])
        throw StringError(errLabel + "Owned cell with invalid strength");
    } else
      throw StringError(errLabel + "Invalid color");
    tmpHash ^= ZOBRIST_CELL_HASH[loc][colors[loc]][strength[loc]];
    tmpArea[colors[loc]] += 1;
    tmpValue[colors[loc]] += strength[loc];
  }

  if(cells_hash != tmpHash)
    throw StringError(errLabel + "Cells hash does not match expected");
  if(pos_hash != (tmpHash ^ turnStateHash()))
    throw StringError(errLabel + "Pos hash does not match expected");
  for(int c = 0; c < 3; c++) {
    if(tmpArea[c] != area[c] || tmpValue[c] != value[c])
      throw StringError(errLabel + "area/value does not match expected");
  }
  if(nextPla != P_BLACK && nextPla != P_WHITE)
    throw StringError(errLabel + "Invalid nextPla");
  if(stage < 0 || stage >= NUM_STAGES)
    throw StringError(errLabel + "Invalid stage");
  if((stage == STAGE_ATTACK_TARGET) != (selectedSrc != NULL_LOC))
    throw StringError(errLabel + "selectedSrc inconsistent with stage");
  if(stage == STAGE_ATTACK_TARGET && !canAttackFrom(selectedSrc, nextPla))
    throw StringError(errLabel + "selectedSrc cannot attack");
  if((stage == STAGE_REINFORCE) != (reinforceLeft > 0))
    throw StringError(errLabel + "reinforceLeft inconsistent with stage");

  short tmpAdjOffsets[8];
  Location::getAdjacentOffsets(tmpAdjOffsets, x_size);
  for(int i = 0; i < 8; i++)
    if(tmpAdjOffsets[i] != adj_offsets[i])
      throw StringError(errLabel + "Corrupted adj_offsets array");
}

bool Board::isEqualForTesting(const Board& other) const {
  checkConsistency();
  other.checkConsistency();
  if(x_size != other.x_size)
    return false;
  if(y_size != other.y_size)
    return false;
  if(pos_hash != other.pos_hash)
    return false;
  if(nextPla != other.nextPla || stage != other.stage || selectedSrc != other.selectedSrc ||
     reinforceLeft != other.reinforceLeft || turnNumber != other.turnNumber)
    return false;
  for(int i = 0; i < MAX_ARR_SIZE; i++) {
    if(colors[i] != other.colors[i] || strength[i] != other.strength[i] || cap[i] != other.cap[i])
      return false;
  }
  return true;
}

// IO FUNCS------------------------------------------------------------------------------------------

char PlayerIO::colorToChar(Color c) {
  switch(c) {
    case C_BLACK:
      return 'X';
    case C_WHITE:
      return 'O';
    case C_EMPTY:
      return '.';
    case C_BLACK_CAPTURED:
      return 'x';
    case C_WHITE_CAPTURED:
      return 'o';
    default:
      return '#';
  }
}

string PlayerIO::playerToString(Color c) {
  switch(c) {
    case C_BLACK:
      return "Black";
    case C_WHITE:
      return "White";
    case C_EMPTY:
      return "Empty";
    default:
      return "Wall";
  }
}

string PlayerIO::playerToStringShort(Color c) {
  switch(c) {
    case C_BLACK:
      return "B";
    case C_WHITE:
      return "W";
    case C_EMPTY:
      return "E";
    default:
      return "";
  }
}

bool PlayerIO::tryParsePlayer(const string& s, Player& pla) {
  string str = Global::toLower(s);
  if(str == "black" || str == "b") {
    pla = P_BLACK;
    return true;
  } else if(str == "white" || str == "w") {
    pla = P_WHITE;
    return true;
  }
  return false;
}

Player PlayerIO::parsePlayer(const string& s) {
  Player pla = C_EMPTY;
  bool suc = tryParsePlayer(s, pla);
  if(!suc)
    throw StringError("Could not parse player: " + s);
  return pla;
}

string Location::toStringMach(Loc loc, int x_size) {
  if(loc == Board::PASS_LOC)
    return string("pass");
  if(loc == Board::NULL_LOC)
    return string("null");
  char buf[128];
  sprintf(buf, "(%d,%d)", getX(loc, x_size), getY(loc, x_size));
  return string(buf);
}

string Location::toString(Loc loc, int x_size, int y_size) {
  if(x_size > 25 * 25)
    return toStringMach(loc, x_size);
  if(loc == Board::PASS_LOC)
    return string("pass");
  if(loc == Board::NULL_LOC)
    return string("null");
  const char* xChar = "ABCDEFGHJKLMNOPQRSTUVWXYZ";
  int x = getX(loc, x_size);
  int y = getY(loc, x_size);
  if(x >= x_size || x < 0 || y < 0 || y >= y_size)
    return toStringMach(loc, x_size);

  char buf[128];
  if(x <= 24)
    sprintf(buf, "%c%d", xChar[x], y_size - y);
  else
    sprintf(buf, "%c%c%d", xChar[x / 25 - 1], xChar[x % 25], y_size - y);
  return string(buf);
}

string Location::toString(Loc loc, const Board& b) {
  return toString(loc, b.x_size, b.y_size);
}

string Location::toStringMach(Loc loc, const Board& b) {
  return toStringMach(loc, b.x_size);
}

static bool tryParseLetterCoordinate(char c, int& x) {
  if(c >= 'A' && c <= 'H')
    x = c - 'A';
  else if(c >= 'a' && c <= 'h')
    x = c - 'a';
  else if(c >= 'J' && c <= 'Z')
    x = c - 'A' - 1;
  else if(c >= 'j' && c <= 'z')
    x = c - 'a' - 1;
  else
    return false;
  return true;
}

bool Location::tryOfString(const string& str, int x_size, int y_size, Loc& result) {
  string s = Global::trim(str);
  if(s.length() < 2)
    return false;
  if(Global::isEqualCaseInsensitive(s, string("pass")) || Global::isEqualCaseInsensitive(s, string("pss"))) {
    result = Board::PASS_LOC;
    return true;
  }
  if(s[0] == '(') {
    if(s[s.length() - 1] != ')')
      return false;
    s = s.substr(1, s.length() - 2);
    vector<string> pieces = Global::split(s, ',');
    if(pieces.size() != 2)
      return false;
    int x;
    int y;
    bool sucX = Global::tryStringToInt(pieces[0], x);
    bool sucY = Global::tryStringToInt(pieces[1], y);
    if(!sucX || !sucY)
      return false;
    result = Location::getLoc(x, y, x_size);
    return true;
  } else {
    int x;
    if(!tryParseLetterCoordinate(s[0], x))
      return false;

    // Extended format
    if((s[1] >= 'A' && s[1] <= 'Z') || (s[1] >= 'a' && s[1] <= 'z')) {
      int x1;
      if(!tryParseLetterCoordinate(s[1], x1))
        return false;
      x = (x + 1) * 25 + x1;
      s = s.substr(2, s.length() - 2);
    } else {
      s = s.substr(1, s.length() - 1);
    }

    int y;
    bool sucY = Global::tryStringToInt(s, y);
    if(!sucY)
      return false;
    y = y_size - y;
    if(x < 0 || y < 0 || x >= x_size || y >= y_size)
      return false;
    result = Location::getLoc(x, y, x_size);
    return true;
  }
}

bool Location::tryOfStringAllowNull(const string& str, int x_size, int y_size, Loc& result) {
  if(str == "null") {
    result = Board::NULL_LOC;
    return true;
  }
  return tryOfString(str, x_size, y_size, result);
}

bool Location::tryOfString(const string& str, const Board& b, Loc& result) {
  return tryOfString(str, b.x_size, b.y_size, result);
}

bool Location::tryOfStringAllowNull(const string& str, const Board& b, Loc& result) {
  return tryOfStringAllowNull(str, b.x_size, b.y_size, result);
}

Loc Location::ofString(const string& str, int x_size, int y_size) {
  Loc result;
  if(tryOfString(str, x_size, y_size, result))
    return result;
  throw StringError("Could not parse board location: " + str);
}

Loc Location::ofStringAllowNull(const string& str, int x_size, int y_size) {
  Loc result;
  if(tryOfStringAllowNull(str, x_size, y_size, result))
    return result;
  throw StringError("Could not parse board location: " + str);
}

Loc Location::ofString(const string& str, const Board& b) {
  return ofString(str, b.x_size, b.y_size);
}

Loc Location::ofStringAllowNull(const string& str, const Board& b) {
  return ofStringAllowNull(str, b.x_size, b.y_size);
}

vector<Loc> Location::parseSequence(const string& str, const Board& board) {
  vector<string> pieces = Global::split(Global::trim(str), ' ');
  vector<Loc> locs;
  for(size_t i = 0; i < pieces.size(); i++) {
    string piece = Global::trim(pieces[i]);
    if(piece.length() <= 0)
      continue;
    locs.push_back(Location::ofString(piece, board));
  }
  return locs;
}
vector<Loc> Location::parseSequenceGom(string str, const Board& b) {
  str = Global::toLower(Global::trim(str));
  vector<Loc> result;
  int i = 0;
  int n = str.length();
  while(i < n) {
    // Check for "pass"
    if(n - i >= 4 && str.substr(i, 4) == "pass") {
      result.push_back(Board::PASS_LOC);
      i += 4;
    }
    // Check for "null"
    else if(n - i >= 4 && str.substr(i, 4) == "null") {
      result.push_back(Board::NULL_LOC);
      i += 4;
    }
    // Check for letter(s) followed by digits
    else if(isalpha(str[i])) {
      int x = 0;
      // Check if next character is also a letter
      if(i + 1 < n && isalpha(str[i + 1])) {
        // Two letters for x
        x = (str[i] - 'a') * 26 + (str[i + 1] - 'a');
        i += 2;
      } else {
        // One letter for x
        x = str[i] - 'a';
        i += 1;
      }
      // Now read digits for y
      int y = 0;
      while(i < n && isdigit(str[i])) {
        y = y * 10 + (str[i] - '0');
        i += 1;
      }
      if(y == 0) {
        // No digits found for y
        throw StringError("Failed to parse loc sequence");
      }
      // Create Loc object
      y = b.y_size - y;
      Loc loc = getLoc(x, y, b.x_size);
      result.push_back(loc);
    } else {
      // Invalid character encountered
      throw StringError("Failed to parse loc sequence");
    }
  }
  return result;
}

string Board::stageToString(int stage) {
  if(stage == STAGE_ATTACK_SELECT)
    return "attack-select";
  if(stage == STAGE_ATTACK_TARGET)
    return "attack-target";
  if(stage == STAGE_REINFORCE)
    return "reinforce";
  return "unknown";
}

static string cellToken(const Board& board, Loc loc) {
  if(board.colors[loc] == C_WALL)
    return "#";
  string s;
  if(board.colors[loc] == C_EMPTY)
    s = ".";
  else if(board.colors[loc] == C_WHITE)
    s = "+" + Global::intToString(board.strength[loc]);
  else
    s = "-" + Global::intToString(board.strength[loc]);
  if(board.cap[loc] == Board::BIG_CAP)
    s += "*";
  return s;
}

void Board::printBoard(ostream& out, const Board& board, Loc markLoc, const vector<Move>* hist) {
  if(hist != NULL)
    out << "MoveNum: " << hist->size() << " ";
  out << "HASH: " << board.pos_hash << "\n";
  out << "Turn " << board.turnNumber << ", " << PlayerIO::playerToString(board.nextPla) << " to move, stage "
      << stageToString(board.stage);
  if(board.stage == STAGE_ATTACK_TARGET)
    out << ", source " << Location::toString(board.selectedSrc, board);
  if(board.stage == STAGE_REINFORCE)
    out << ", points left " << board.reinforceLeft;
  out << "\n";
  out << "White(+) area " << board.area[C_WHITE] << " value " << board.value[C_WHITE] << ", Black(-) area "
      << board.area[C_BLACK] << " value " << board.value[C_BLACK] << "\n";

  //Hex layout: cell (x,y) is drawn at x - y/2, so every row is shifted half a cell left of the previous one.
  //Each cell is 4 chars wide, a half shift is 2 chars.
  const char* xChar = "ABCDEFGHJKLMNOPQRSTUVWXYZ";
  bool showCoords = board.x_size <= 25;
  for(int y = 0; y < board.y_size; y++) {
    if(showCoords) {
      char buf[16];
      sprintf(buf, "%2d ", board.y_size - y);
      out << buf;
    }
    out << string(2 * (board.y_size - 1 - y), ' ');
    for(int x = 0; x < board.x_size; x++) {
      Loc loc = Location::getLoc(x, y, board.x_size);
      string tok = cellToken(board, loc);
      char mark = ' ';
      if(loc == markLoc)
        mark = '@';
      else if(loc == board.selectedSrc)
        mark = '>';
      else if(hist != NULL && hist->size() > 0 && (*hist)[hist->size() - 1].loc == loc)
        mark = '\'';
      while(tok.size() < 3)
        tok = " " + tok;
      out << tok << mark;
    }
    out << "\n";
  }
  if(showCoords) {
    //Column letters, aligned to the last row
    out << "   ";
    for(int x = 0; x < board.x_size; x++)
      out << "  " << xChar[x] << " ";
    out << "\n";
  }
  out << "\n";
}

ostream& operator<<(ostream& out, const Board& board) {
  Board::printBoard(out, board, Board::NULL_LOC, NULL);
  return out;
}

string Board::toStringSimple(const Board& board, char lineDelimiter) {
  string s;
  for(int y = 0; y < board.y_size; y++) {
    for(int x = 0; x < board.x_size; x++) {
      Loc loc = Location::getLoc(x, y, board.x_size);
      if(x > 0)
        s += ' ';
      s += cellToken(board, loc);
    }
    s += lineDelimiter;
  }
  s += "state " + PlayerIO::playerToStringShort(board.nextPla) + " " + Global::intToString(board.stage) + " " +
       Location::toStringMach(board.selectedSrc, board.x_size) + " " + Global::intToString(board.reinforceLeft) + " " +
       Global::intToString(board.turnNumber);
  s += lineDelimiter;
  return s;
}

Board Board::parseBoard(int xSize, int ySize, const string& s) {
  return parseBoard(xSize, ySize, s, '\n');
}

Board Board::parseBoard(int xSize, int ySize, const string& s, char lineDelimiter) {
  Board board(xSize, ySize);
  vector<string> rawLines = Global::split(Global::trim(s), lineDelimiter);
  vector<string> lines;
  string stateLine;
  for(const string& raw: rawLines) {
    string line = Global::trim(raw);
    if(line.length() == 0)
      continue;
    if(Global::isPrefix(line, "state"))
      stateLine = line;
    else
      lines.push_back(line);
  }

  if(lines.size() != ySize)
    throw StringError("Board::parseBoard - string has different number of board rows than ySize");

  for(int y = 0; y < ySize; y++) {
    vector<string> toks;
    for(const string& t: Global::split(lines[y], ' ')) {
      if(Global::trim(t).length() > 0)
        toks.push_back(Global::trim(t));
    }
    if(toks.size() != xSize)
      throw StringError("Board::parseBoard - row " + Global::intToString(y) + " does not have xSize cells");
    for(int x = 0; x < xSize; x++) {
      Loc loc = Location::getLoc(x, y, xSize);
      string tok = toks[x];
      if(tok == "#") {
        board.setWall(loc, true);
        continue;
      }
      bool big = false;
      if(tok.size() > 0 && tok[tok.size() - 1] == '*') {
        big = true;
        tok = tok.substr(0, tok.size() - 1);
      }
      if(big)
        board.setBig(loc, true);
      int v = 0;
      if(tok == "." || tok == "")
        v = 0;
      else if(!Global::tryStringToInt(tok, v))
        throw StringError("Board::parseBoard - could not parse cell token: " + toks[x]);
      bool suc;
      if(v > 0)
        suc = board.setCell(loc, C_WHITE, v);
      else if(v < 0)
        suc = board.setCell(loc, C_BLACK, -v);
      else
        suc = board.setCell(loc, C_EMPTY, 0);
      if(!suc)
        throw StringError("Board::parseBoard - invalid cell value (above cap?): " + toks[x]);
    }
  }

  if(stateLine.length() > 0) {
    vector<string> pieces;
    for(const string& t: Global::split(stateLine, ' ')) {
      if(Global::trim(t).length() > 0)
        pieces.push_back(Global::trim(t));
    }
    if(pieces.size() != 6)
      throw StringError("Board::parseBoard - could not parse state line: " + stateLine);
    Player pla = PlayerIO::parsePlayer(pieces[1]);
    int stg = Global::stringToInt(pieces[2]);
    Loc src = Location::ofStringAllowNull(pieces[3], xSize, ySize);
    int rLeft = Global::stringToInt(pieces[4]);
    int tNum = Global::stringToInt(pieces[5]);
    if(!board.setTurnState(pla, stg, src, rLeft, tNum))
      throw StringError("Board::parseBoard - inconsistent state line: " + stateLine);
  }
  return board;
}

nlohmann::json Board::toJson(const Board& board) {
  nlohmann::json data;
  data["xSize"] = board.x_size;
  data["ySize"] = board.y_size;
  data["movenum"] = board.movenum;
  data["stones"] = Board::toStringSimple(board, '|');
  return data;
}

Board Board::ofJson(const nlohmann::json& data) {
  int xSize = data["xSize"];
  int ySize = data["ySize"].get<int>();
  Board board = Board::parseBoard(xSize, ySize, data["stones"].get<string>(), '|');
  board.movenum = data["movenum"].get<int>();
  return board;
}
