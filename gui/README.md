# Influence GUI

Small GUI for Influence (see `../RULES.md`), written in Rust with macroquad.
Both sides can be played by hand, by the engine, or analysed by it.

```
cargo run --release -- [position.txt] [--engine "<engine command line>"]... [--replay <game.sgf|games.sgfs> [--game <n>]]
```

The engine is this repo's KataGo fork in GTP mode, for example:

```
cargo run --release -- --engine "../cpp/build_opencl/katago gtp -model ../models/run1/b10c128-s1766528-d8186957.bin.gz -config gtp.cfg"
```

`--engine` takes the whole engine command line as one quoted argument. Give it several times to load several
engines (E1, E2, ... in the order given), e.g. two models to play against each other:

```
cargo run --release -- --engine "../cpp/build_opencl/katago gtp -model ../models/run1/A.bin.gz -config gtp.cfg" \
                       --engine "../cpp/build_opencl/katago gtp -model ../models/run1/B.bin.gz -config gtp.cfg"
```

Each command line is split into the program and its
arguments at whitespace; '...' and "..." quote and `\` escapes, as in a shell, but nothing is expanded
(no `~`, `$VAR` or globs) and no shell is involved.

`gtp.cfg` is a copy of `../cpp/configs/gtp_example.cfg` with resignation disabled; it logs to `gtp_logs/`.
A single command can also be given in `INFLUENCE_ENGINE`. Engine stderr (and any
non-GTP stdout, such as a usage message) is echoed with an `[engine]` / `[engine stdout]` prefix. Winrates assume the default `reportAnalysisWinratesAs`
(side to move). The engine's board size limit is `COMPILE_MAX_BOARD_LEN` (11), larger boards are rejected.

Engine panel:
- `Blue: Human/Engine`, `Red: Human/Engine` choose who plays each side. With several engines the buttons
  cycle Human -> E1 -> E2 -> ..., so each side can be played by a different engine; the panel lists each
  engine's status and model name. The engine plays its turn one
  micro-move at a time with `genmove` (search strength from the config's `maxVisits`). Undo steps back
  over engine moves to your last move. If the engine resigns, its side switches back to Human.
  The engine that is not on move stays idle, and a stopped engine can be restarted from the panel.
- `Analysis` (A) runs `kata-analyze` (with several engines, `by En` picks which engine analyses) on the current position (also in edit mode) until it changes.
  Candidate cells get a ring (cyan for the best move) and their winrate for the side to move;
  the best move's (or the hovered candidate's) principal variation for the current turn is drawn as
  numbered attack arrows and `+n` reinforcement marks. The bar shows Blue's winrate.

Positions use the engine's text format (`Board::parseBoard`, GTP `set_board`): rows separated
by newlines or `/`, cells `#` wall, `.` empty, `+5` Blue (White in GTP/SGF), `-7` Red (Black), trailing `*` for a big
cell, and an optional final `state <W|B> <stage> <src> <points> <turn>` row.
Ctrl+C copies the current position in that format, Ctrl+V pastes one.

Play mode: click an own cell to select an attack source, click a neighbour to attack,
Space ends the attack phase, then click own cells to spend reinforcement points
(Shift+click spends as many as fit). Forced reinforcement is applied automatically, as in the engine.
Z / Y (or Left / Right) undo and redo, Shift+Z / Y or PageUp / PageDown step a whole turn, Home / End jump to
the start / end of the history; the keys repeat while held. A white dot marks the cell the last step changed.

Random rules (`Rules: standard / random` button): a variant only the GUI knows. An attack on an enemy cell
whose strength is at most 1 away from the attacker's is decided by chance: the attacker captures it with
75% (attacker 1 stronger), 50% (equal) or 25% (attacker 1 weaker). Either way the source drops to 1 and the
target ends at strength 1 for whoever won it, so an owned cell never becomes empty again. All other attacks
and the rest of the rules are unchanged. They apply to every attack played in the GUI, including the
engines'. The engines still search with the standard rules; after an engine attack the GUI resolves it
with the random rules and sends the engine the actual position (`set_board`) before its next move. Replays and the
analysis PV arrows use the standard rules.

Edit mode (E): LMB/RMB change a cell's value by +1/-1, MMB clears it, Shift+LMB toggles a wall,
arrows resize the board, T swaps the side to move, C clears the board.

Loading a game (`--replay`): the main line of an SGF game written by the engine is replayed with the GUI's
rules code from its `IB[]` start position and put into the undo history, as if the moves had been played by
hand; the GUI starts at the final position. Selfplay `.sgfs` files hold one game per line, `--game` picks one
(0-based, default 0). A box over the board shows the game, the history index (`move i / n`) and the last and
next move with their SGF comments (a move's comment holds the training targets for the position it was
played from). Everything else works as usual: undo to any position, analyse it, or play on from there.
