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

Model version 104 / inputs version 103 is the default; 103 / 102 still loads. Both use 32 binary spatial planes and 64 globals.
The layout is documented above `NNInputs::fillRowV102` in `cpp/neuralnet/nninputs.cpp`.
The policy is the usual board plane plus pass, where pass means "end attack phase". The value head is W/L/draw.

V103 adds these globals to V102 (slots 0–13 are unchanged):

| Slots | Meaning |
|---|---|
| 14–29 | Reinforce stage only: one-hot bucket of the points left (exact for 1–8, then ≤10, ≤12, ≤14, ≤16, ≤24, ≤32, ≤64, 65+) |
| 30 | Value ratio: own value / opponent value in the attack stages, (own value + points left) / opponent value in the reinforce stage (the value at the end of the turn). Opponent value is floored at 1, the ratio is capped at 32 |
| 31 | Attack stages only: a non-pass move is legal |
| 32–47 | Rule flags, filled by `NNInputs::fillRuleFeatures`. 32 = `forbidPassUnlessNoOtherMove`, the rest are reserved |
| 48–63 | 0 (63 must stay 0: the python scorebelief head reads the last global) |

To add a rule input, add the field to `Rules` (hash, parse, string, json), then set the next free slot in `fillRuleFeatures`.
Nets already trained on v104 keep working, because that slot was always 0 for them.
`forbidPassUnlessNoOtherMoveProb` in the selfplay config mixes both settings across games.

Outputs:
- Moves left: the scoremean head predicts the remaining player turns (`movesLeft`, ×20). The search's moves-left utility uses it.
- Micro-moves left (v104): score-value channel 2, formerly lead, predicts the remaining moves (decisions, not counting forced reinforcement), ×800.
  - Shown as `microMovesLeft` in `kata-raw-nn` and as `rawMicroMovesLeft` in the analysis JSON.
  - Training target C21, weight C29. The loss is the turns loss in units of 40 moves, logged as `mmloss`.

Training data: C47 of `globalTargetsNC` holds the inputs version of the row (0 in older data, meaning 102).
`python/data_processing_pytorch.py` converts rows on load: V102 rows are rebuilt exactly for v104 models, and V103 rows lose their extra globals for v103 models.
So old and new data can be mixed in one shuffle. `python/migrate_v103_to_v104.py` switches a v103 checkpoint (the weights stay the same).

### Transformer trunks

Ported from upstream KataGo (tf2/tf3 nets). Configs are `b<blocks>c<channels>h<heads>[kv<kvheads>]nbt[3]tflrs` in `python/modelconfigs.py`, usually with `-fson-silu-rsnh`. Each trunk block is a nested bottleneck block:
1. a 1x1 conv to `mid_num_channels`;
2. 2 (`nbt`) or 3 (`nbt3`) pre-norm transformer blocks, each RMSNorm → attention → RMSNorm → SwiGLU FFN. Attention uses learnable 2D RoPE with x = column and y = row, optional grouped-query heads, and ignores off-board keys;
3. a 1x1 conv back.

`-rsnh` puts a spatial RMSNorm at the trunk tip; its trunk norm kind is stored in the v104 trunk header.

Backends:
- Eigen: CPU reference.
- OpenCL: unpadded NCHW. Projections are 1x1 convolutions; the attention kernel is compiled per head dims.
- CUDA: forced to NHWC, following upstream's `cudaandrocmbackend.inc`; its kernels are copied unchanged into `cudahelpers.cu` and `cudaflashmma.cuh`.
  - Q/K/V projections, and the FFN linear1 and gate, each run as one strided-batched cuBLAS GEMM, unless grouped-query heads make the shapes differ.
  - RoPE rotates Q and K in one launch.
  - fp16 attention uses upstream's tensor-core flash kernel (sm_75+, head dims 32 and 64).
  - Otherwise a scalar online-softmax kernel runs, with head dims 16, 32 and 64.
- TensorRT: refuses transformer models.

Masking: walls inside the box are off-board (spatial plane 0 is 0), so every backend applies the board mask even when the box fills the NN buffer exactly.
Upstream CUDA and TensorRT skip the mask in that case (`requireExactNNLen`, used by selfplay when all games have one size), which is only valid without walls.

`katago dumpnnparity` writes NN inputs and outputs on random positions, evaluated in batches.
`-exact-nn-len` uses only full-size boards with `requireExactNNLen`, the selfplay path; test both modes.
`python/test_nn_parity.py` compares them with the python checkpoint, or two backends with each other. Use it after any backend change.

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
