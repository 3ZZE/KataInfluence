#ifndef GAME_RULES_H_
#define GAME_RULES_H_

#include "../core/global.h"
#include "../core/hash.h"

#include "../external/nlohmann_json/json.hpp"

//Influence rules. The game itself has no options, the only setting is the technical move limit.
struct Rules {
  //Maximum number of player turns (one turn = attack phase + reinforce phase).
  //When reached the game is a draw. 0 = unlimited.
  int maxMoves;

  Rules();
  explicit Rules(int maxMoves);
  ~Rules();

  bool operator==(const Rules& other) const;
  bool operator!=(const Rules& other) const;

  static Rules getTrompTaylorish();
  static Rules getDefault();

  static Rules parseRules(const std::string& str);
  static bool tryParseRules(const std::string& str, Rules& buf);

  static Rules updateRules(const std::string& key, const std::string& value, Rules priorRules);

  friend std::ostream& operator<<(std::ostream& out, const Rules& rules);
  std::string toString() const;
  std::string toStringMaybeNice() const;
  std::string toJsonString() const;
  nlohmann::json toJson() const;

  static const Hash128 ZOBRIST_MAXMOVES_HASH_BASE;
};

#endif  // GAME_RULES_H_
