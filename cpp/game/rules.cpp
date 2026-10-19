#include "../game/rules.h"

#include "../external/nlohmann_json/json.hpp"

#include <sstream>

using namespace std;
using json = nlohmann::json;

Rules::Rules() {
  maxMoves = 0;
  forbidPassUnlessNoOtherMove = false;
}

Rules::Rules(int mm) : maxMoves(mm), forbidPassUnlessNoOtherMove(false) {}

Rules::~Rules() {}

bool Rules::operator==(const Rules& other) const {
  return maxMoves == other.maxMoves && forbidPassUnlessNoOtherMove == other.forbidPassUnlessNoOtherMove;
}

bool Rules::operator!=(const Rules& other) const {
  return !(*this == other);
}

Rules Rules::getTrompTaylorish() {
  return Rules();
}

Rules Rules::getDefault() {
  return Rules();
}

ostream& operator<<(ostream& out, const Rules& rules) {
  out << "maxmoves" << rules.maxMoves;
  if(rules.forbidPassUnlessNoOtherMove)
    out << "forbidpass";
  return out;
}

string Rules::toString() const {
  ostringstream out;
  out << (*this);
  return out.str();
}

json Rules::toJson() const {
  json ret;
  ret["maxmoves"] = maxMoves;
  ret["forbidPassUnlessNoOtherMove"] = forbidPassUnlessNoOtherMove;
  return ret;
}

string Rules::toJsonString() const {
  return toJson().dump();
}

Rules Rules::updateRules(const string& k, const string& v, Rules oldRules) {
  Rules rules = oldRules;
  string key = Global::toLower(Global::trim(k));
  string value = Global::trim(Global::toUpper(v));
  if(key == "maxmoves") {
    rules.maxMoves = Global::stringToInt(value);
    if(rules.maxMoves < 0)
      throw IOError("maxmoves must be >= 0");
  } else if(key == "forbidpassunlessnoothermove" || key == "forbidpass") {
    if(value == "1" || value == "0")
      rules.forbidPassUnlessNoOtherMove = value == "1";
    else if(!Global::tryStringToBool(value, rules.forbidPassUnlessNoOtherMove))
      throw IOError("Could not parse " + key + " value: " + v);
  } else
    throw IOError("Unknown rules option: " + key);
  return rules;
}

static Rules parseRulesHelper(const string& sOrig) {
  Rules rules;
  string lowercased = Global::trim(Global::toLower(sOrig));
  if(lowercased == "influence" || lowercased == "default" || lowercased == "chinese") {
  } else if(sOrig.length() > 0 && sOrig[0] == '{') {
    rules = Rules::getDefault();
    try {
      json input = json::parse(sOrig);
      for(json::iterator iter = input.begin(); iter != input.end(); ++iter) {
        string key = iter.key();
        string v = iter.value().is_string() ? iter.value().get<string>() : to_string(iter.value());
        rules = Rules::updateRules(key, v, rules);
      }
    } catch(nlohmann::detail::exception&) {
      throw IOError("Could not parse rules: " + sOrig);
    }
  } else if(Global::isPrefix(lowercased, "maxmoves")) {
    //Short form as written by toString, e.g. "maxmoves180" or "maxmoves180forbidpass"
    string rest = lowercased.substr(8);
    if(Global::isSuffix(rest, "forbidpass")) {
      rules.forbidPassUnlessNoOtherMove = true;
      rest = rest.substr(0, rest.size() - 10);
    }
    rules.maxMoves = Global::stringToInt(rest);
  } else {
    throw IOError("Could not parse rules: " + sOrig);
  }
  return rules;
}

Rules Rules::parseRules(const string& sOrig) {
  return parseRulesHelper(sOrig);
}

bool Rules::tryParseRules(const string& sOrig, Rules& buf) {
  Rules rules;
  try {
    rules = parseRulesHelper(sOrig);
  } catch(const StringError&) {
    return false;
  }
  buf = rules;
  return true;
}

string Rules::toStringMaybeNice() const {
  if(*this == Rules::getDefault())
    return "influence";
  return toString();
}

const Hash128 Rules::ZOBRIST_MAXMOVES_HASH_BASE = Hash128(0x8aba00580c378fe8ULL, 0x7f6c1210e74fb440ULL);
const Hash128 Rules::ZOBRIST_FORBID_PASS_HASH = Hash128(0x7c3e9a51d02b84f6ULL, 0x1f86c4e3a95d2b07ULL);
