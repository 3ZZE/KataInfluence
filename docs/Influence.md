# Influence fork: implementation notes

Rules: see [RULES.md](../RULES.md).

## Board

- Hex cell `(row, col)` is `Loc` `(x = col, y = row)` in the usual KataGo padded array.
  Neighbours are `(x±1, y)`, `(x, y±1)`, `(x+1, y+1)`, `(x−1, y−1)`, i.e. Loc offsets
  `±1`, `±(x_size+1)`, `±(x_size+2)` (`Board::adj_offsets[0..5]`).
- A map is an `x_size × y_size` box. Hexes outside the map are `C_WALL` inside the box.
- Per cell: `colors[]` (owner or wall), `strength[]`, `cap[]` (8 small, 12 big).
  Big cells are supported by the game logic, NN inputs and text format, but selfplay only generates small cells for now.
- Turn state lives in `Board`: `nextPla`, `stage`, `selectedSrc`, `reinforceLeft`, `turnNumber`.
  `pos_hash` covers cells, walls, sizes, turn state and the turn number
  (the turn number keeps graph search acyclic, because positions can repeat across turns).

## Moves

Every action is a single `Loc`. A player's turn is a variable-length run of moves by the same player:

| Stage | Legal moves | Effect |
|---|---|---|
| `ATTACK_SELECT` | own cell with strength ≥ 2 and a non-own neighbour | select source |
| | `pass` | end attack phase, points = area |
| `ATTACK_TARGET` | non-own neighbour of the selected source | resolve attack |
| `REINFORCE` | own cell below its cap | +1 strength, one point spent |

Forced reinforcement (points ≥ total room, or only one cell with room) is applied automatically.
Leftover points are dropped, and the turn ends.
The player to move is always `board.nextPla`. Search, selfplay, GTP, SGF and the training writer never assume players alternate.

## Symmetries

Only 4 symmetries keep the hex adjacency and a rectangular box:
- C++ `SymmetryHelpers`: `NUM_SYMMETRIES = 4`. Bit 0 is rotate 180, bit 1 is transpose.
- Python training: `symm ∈ {0, 2, 4, 6}`.

`runinfluencetests` checks that every symmetry commutes with play and that NN tensor symmetries match board symmetries.

## Neural net

Model version 103 / inputs version 102: 32 binary spatial planes, 64 globals.
The layout is documented above `NNInputs::fillRowV102` in `cpp/neuralnet/nninputs.cpp`.
The policy is the usual board plane plus pass, where pass means "end attack phase". The value head is W/L/draw.

## Text board format

Rows are separated by `/` (or newlines), cells by spaces:
- `#` wall
- `.` empty
- `+5` White 5, `-7` Black 7
- a trailing `*` marks a big cell

An optional last row `state <W|B> <stage> <src|null> <pointsLeft> <turn>` sets the turn state.
SGFs store the initial position in a root `IB[...]` property in this format.

## Commands

- `katago runinfluencetests`: rules, hashing, symmetry and NN-input tests.
- GTP:
  - `genturn <color>`: plays the whole turn and returns the move list.
  - `set_board <rows>`: sets any position.
  - `maxmoves N`: move limit in player turns.
  - `genmove` and `play` work per move.
- Analysis engine: `initialBoard` (string in the text format above) replaces `initialStones`.

## Selfplay settings (scripts/selfplay.cfg)

- `maxTurns`: draw limit in player turns.
- `randomStartProb`, `randomWallsProb`, `randomWallsMaxDensity`: starting position variety.
- `reinforceCheapSearchProb`: cheap-search probability during the reinforce stage
  (one move per reinforcement point, so these moves dominate game length).
- `chosenMoveTemperatureHalflife` counts player turns. `policyInitAvgMoveNum` and `maxMovesPerGame` count moves.
