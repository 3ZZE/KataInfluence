/*
 * board.h
 * Originally from an unreleased project back in 2010, modified since.
 * Authors: brettharrison (original), David Wu (original and later modifications).
 *
 * Influence: two-player territory game on a hexagonal board (see RULES.md).
 * A player's turn is a variable-length sequence of single-Loc micro-moves:
 *   STAGE_ATTACK_SELECT: choose an attack source, or PASS to end the attack phase
 *   STAGE_ATTACK_TARGET: choose the target of the selected source
 *   STAGE_REINFORCE:     add +1 strength to one own cell, repeated until the points run out
 * The player to move is board.nextPla, which does NOT simply alternate after each move.
 */

#ifndef GAME_BOARD_H_
#define GAME_BOARD_H_

#include <cstdint>
#include "../core/global.h"
#include "../core/hash.h"
#include "../external/nlohmann_json/json.hpp"
#include "../game/rules.h"

#ifndef COMPILE_MAX_BOARD_LEN
static_assert(false, "COMPILE_MAX_BOARD_LEN should be defined in core/config.h");
#endif

//TYPES AND CONSTANTS-----------------------------------------------------------------

struct Board;


static inline Color getOpp(Color c)
{return c ^ 3;}

//Conversions for players and colors
namespace PlayerIO {
  char colorToChar(Color c);
  std::string playerToStringShort(Player p);
  std::string playerToString(Player p);
  bool tryParsePlayer(const std::string& s, Player& pla);
  Player parsePlayer(const std::string& s);
}

//Location of a point on the board
//(x,y) is represented as (x+1) + (y+1)*(x_size+1)
//x is the hex column, y is the hex row. Hex neighbours of (x,y) are
//(x+-1,y), (x,y+-1), (x+1,y+1), (x-1,y-1).
typedef short Loc;
namespace Location
{
  Loc getLoc(int x, int y, int x_size);
  int getX(Loc loc, int x_size);
  int getY(Loc loc, int x_size);

  //Offsets for the 6 hex neighbours, entries 6 and 7 are unused and set to 0
  void getAdjacentOffsets(short adj_offsets[8], int x_size);
  bool isAdjacent(Loc loc0, Loc loc1, int x_size);
  Loc getCenterLoc(int x_size, int y_size);
  Loc getCenterLoc(const Board& b);
  bool isCentral(Loc loc, int x_size, int y_size);
  bool isNearCentral(Loc loc, int x_size, int y_size);
  //Hex distance
  int distance(Loc loc0, Loc loc1, int x_size);
  int euclideanDistanceSquared(Loc loc0, Loc loc1, int x_size);

  std::string toString(Loc loc, int x_size, int y_size);
  std::string toString(Loc loc, const Board& b);
  std::string toStringMach(Loc loc, int x_size);
  std::string toStringMach(Loc loc, const Board& b);

  bool tryOfString(const std::string& str, int x_size, int y_size, Loc& result);
  bool tryOfString(const std::string& str, const Board& b, Loc& result);
  Loc ofString(const std::string& str, int x_size, int y_size);
  Loc ofString(const std::string& str, const Board& b);

  //Same, but will parse "null" as Board::NULL_LOC
  bool tryOfStringAllowNull(const std::string& str, int x_size, int y_size, Loc& result);
  bool tryOfStringAllowNull(const std::string& str, const Board& b, Loc& result);
  Loc ofStringAllowNull(const std::string& str, int x_size, int y_size);
  Loc ofStringAllowNull(const std::string& str, const Board& b);

  std::vector<Loc> parseSequence(const std::string& str, const Board& b);
  std::vector<Loc> parseSequenceGom(std::string str, const Board& b);
}

//Simple structure for storing moves. Not used below, but this is a convenient place to define it.
STRUCT_NAMED_PAIR(Loc,loc,Player,pla,Move);

struct Board
{
  //Initialization------------------------------
  //Initialize the zobrist hash.
  //MUST BE CALLED AT PROGRAM START!
  static void initHash();

  //Board parameters and Constants----------------------------------------

  static constexpr int MAX_LEN = COMPILE_MAX_BOARD_LEN;  //Maximum edge length allowed for the board
  static constexpr int DEFAULT_LEN = 11; //Default edge length for board if unspecified
  static constexpr int MAX_PLAY_SIZE = MAX_LEN * MAX_LEN;  //Maximum number of playable spaces
  static constexpr int MAX_ARR_SIZE = (MAX_LEN+1)*(MAX_LEN+2)+1; //Maximum size of arrays needed

  //Location used to indicate an invalid spot on the board.
  static constexpr Loc NULL_LOC = 0;
  //Location used to indicate a pass move is desired. In Influence, PASS ends the attack phase.
  static constexpr Loc PASS_LOC = 1;

  //Cell caps (RULES.md 1.1.1)
  static constexpr int SMALL_CAP = 8;
  static constexpr int BIG_CAP = 12;
  static constexpr int MAX_CAP = BIG_CAP;

  //Turn stages
  static constexpr int STAGE_ATTACK_SELECT = 0;
  static constexpr int STAGE_ATTACK_TARGET = 1;
  static constexpr int STAGE_REINFORCE = 2;
  static constexpr int NUM_STAGES = 3;

  //Zobrist Hashing------------------------------
  static bool IS_ZOBRIST_INITALIZED;
  static Hash128 ZOBRIST_SIZE_X_HASH[MAX_LEN+1];
  static Hash128 ZOBRIST_SIZE_Y_HASH[MAX_LEN+1];
  //[loc][owner color: C_BLACK or C_WHITE][strength]
  static Hash128 ZOBRIST_CELL_HASH[MAX_ARR_SIZE][3][MAX_CAP+1];
  //Interior walls (walls inside the x_size*y_size box)
  static Hash128 ZOBRIST_WALL_HASH[MAX_ARR_SIZE];
  static Hash128 ZOBRIST_BIG_HASH[MAX_ARR_SIZE];
  static Hash128 ZOBRIST_STAGE_HASH[NUM_STAGES];
  static Hash128 ZOBRIST_SELECTED_HASH[MAX_ARR_SIZE];
  static Hash128 ZOBRIST_REINFORCE_HASH_BASE;
  static Hash128 ZOBRIST_TURNNUM_HASH_BASE;
  //Second set of cell hashes, mixed differently, used by the book
  static Hash128 ZOBRIST_BOARD_HASH2[MAX_ARR_SIZE][4];
  static Hash128 ZOBRIST_PLAYER_HASH[6];
  static const Hash128 ZOBRIST_GAME_IS_OVER;

  //Constructors---------------------------------
  Board();  //Create empty Board of size (DEFAULT_LEN,DEFAULT_LEN)
  Board(int x, int y); //Create empty Board of size (x,y): all cells small and empty, White to move
  Board(const Board& other);

  Board& operator=(const Board&) = default;

  //Default starting position of RULES.md 3 on a full x*y rhombus of small cells
  static Board getDefaultStart(int x, int y);

  //Functions------------------------------------

  //Check if this location is a cell (on the board and not a wall)
  bool isOnBoard(Loc loc) const;
  //Is no cell owned?
  bool isEmpty() const;
  //Number of owned cells, of both players
  int numStonesOnBoard() const;
  //Number of cells owned by pla (area)
  int numPlaStonesOnBoard(Player pla) const;
  int getArea(Player pla) const { return area[pla]; }
  int getValue(Player pla) const { return value[pla]; }
  //Number of cells (non-wall locations)
  int numCells() const;
  //Total reinforcement room: sum over own cells of (cap - strength)
  int getCapacity(Player pla) const;
  int getCap(Loc loc) const { return cap[loc]; }
  bool isBig(Loc loc) const { return cap[loc] == BIG_CAP; }

  //Can pla attack from this loc right now (ignoring stage)?
  bool canAttackFrom(Loc loc, Player pla) const;
  bool anyAttackAvailable(Player pla) const;

  //Legality of a micro-move for the current stage. pla must be nextPla.
  bool isLegal(Loc loc, Player pla) const;
  void playMoveAssumeLegal(Loc loc, Player pla);

  //Board setup. These do not touch turn state.
  //Sets owner and strength of a cell. color is C_EMPTY/C_BLACK/C_WHITE. Fails if not a cell or strength out of range.
  bool setCell(Loc loc, Color color, int strength);
  //Sets a location in the x_size*y_size box to be a wall or a (small, empty) cell.
  bool setWall(Loc loc, bool isWall);
  //Sets cell size. Fails if not a cell or the current strength would exceed the new cap.
  bool setBig(Loc loc, bool big);
  //Set turn state directly (for setup/parsing). Returns false if inconsistent.
  bool setTurnState(Player pla, int stage, Loc selectedSrc, int reinforceLeft, int turnNumber);

  Hash128 getSitHash(Player pla) const;

  //Run some basic sanity checks on the board state, throws an exception if not consistent, for testing/debugging
  void checkConsistency() const;
  bool isEqualForTesting(const Board& other) const;

  //Text format: rows separated by lineDelimiter, cells separated by spaces.
  //Cell tokens: "#" wall, "." or "0" empty, "+5"/"5" White strength 5, "-7" Black strength 7.
  //A trailing "*" marks a big cell, e.g. "+10*" or ".*".
  //An optional final row "state <pla> <stage> <selectedSrc> <reinforceLeft> <turnNumber>" sets the turn state.
  static Board parseBoard(int xSize, int ySize, const std::string& s);
  static Board parseBoard(int xSize, int ySize, const std::string& s, char lineDelimiter);
  static void printBoard(std::ostream& out, const Board& board, Loc markLoc, const std::vector<Move>* hist);
  static std::string toStringSimple(const Board& board, char lineDelimiter);
  static nlohmann::json toJson(const Board& board);
  static Board ofJson(const nlohmann::json& data);
  static std::string stageToString(int stage);

  //Data--------------------------------------------

  int x_size;                  //Horizontal size of board
  int y_size;                  //Vertical size of board
  Color colors[MAX_ARR_SIZE];  //Owner of each location (C_EMPTY/C_BLACK/C_WHITE), or C_WALL
  uint8_t strength[MAX_ARR_SIZE]; //Strength of each cell, 0 iff empty
  uint8_t cap[MAX_ARR_SIZE];   //Cap of each cell (SMALL_CAP or BIG_CAP), 0 for walls

  Player nextPla;     //Player to move
  int stage;          //One of STAGE_*
  Loc selectedSrc;    //Selected attack source during STAGE_ATTACK_TARGET, else NULL_LOC
  int reinforceLeft;  //Reinforcement points still to spend during STAGE_REINFORCE, else 0
  int turnNumber;     //Number of completed player turns
  int movenum;        //Number of micro-moves played

  int area[4];        //Indexed by color, number of cells owned
  int value[4];       //Indexed by color, total strength owned

  //A zobrist hash of the current position, including cells, walls, cell sizes and the turn state
  //except for the player to move (added by getSitHash / getSituationRulesHash).
  Hash128 pos_hash;

  short adj_offsets[8]; //Indices 0-5: offsets of the 6 hex neighbours

  private:
  void init(int xS, int yS);
  void beginReinforce();
  void maybeAutoReinforce();
  void endTurn();
  void setCellRaw(Loc loc, Color color, int str);
  Hash128 turnStateHash() const;

  Hash128 cells_hash; //Hash of the cells, walls and sizes only

  friend std::ostream& operator<<(std::ostream& out, const Board& board);
};

#define FOREACH_HEX_ADJ(LOC, ADJLOC, BLOCK) \
  for(int _adjI = 0; _adjI < 6; _adjI++) { \
    Loc ADJLOC = (LOC) + adj_offsets[_adjI]; \
    BLOCK \
  }

#endif // GAME_BOARD_H_
