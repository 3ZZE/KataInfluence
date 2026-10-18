# Net strength rating

`rate.py` keeps a persistent pool of match results between nets in `models/run1/`,
fits Bayesian Elo (Bradley-Terry) ratings and picks the next match between nets of
similar strength. Ideas are borrowed from katago-server (`bayesian_elo.py`,
`rating_network_pairer.py`).

Run it with the existing venv:

```sh
PY=/home/anton/board_games/KataDot/scripts/.venv/bin/python
$PY scripts/rating/rate.py run --minutes 60        # play rating batches for an hour
$PY scripts/rating/rate.py run --batches 10 --every 3 --min-samples 5000000
$PY scripts/rating/rate.py run --dry-run --batches 5   # only show pairings
$PY scripts/rating/rate.py status [--pairs]        # Elo table (+ per-pair, per-colour results)
$PY scripts/rating/rate.py plot [--out elo.png]    # Elo vs training samples, with error bars
$PY scripts/rating/rate.py import DIR...           # add old `katago match` SGF dirs
```

Use `--state FILE` with any command to choose the state file. The default is
`scripts/rating/rating_state.json`.

## Architectures

Several architectures can share one pool. A net's architecture is its name before
`-s<samples>-d<rows>`, e.g. `b10c128` or `b10c256nbt`. Sample counts restart from 0
for each architecture, so nets are ordered by (architecture, samples), with the
primary architecture (the one with the most nets; ties go to the alphabetically
first) first. Short names are `s49733k` for the primary architecture and
`b10c256nbt-s15000k` for the others. `--nets` and `import` accept the full name,
the short name and the `ARCH-sNNNk` form for any architecture.

## State

There is one JSON file. Writes are atomic (temp file, fsync, rename), and a lock
file stops two `run`/`import` processes from writing it at once. It holds:

- `nets`: name, training samples and path of each net it has seen.
- `pools`: one pool per match setting. The default pool name is `v<visits>`, plus
  `-t<maxTurns>` when maxTurns is not 180. Each pool stores its `settings`, its
  `anchor` net, the `pairs` and a `batches` log. Each pair is stored from the
  point of view of the lexicographically first net, `a`, as `a_black` and
  `a_white` lists of `[wins, draws, losses]`.
  Results from different visits or maxTurns are never mixed. `--visits N` selects
  or creates a pool, and asking for an existing pool name with different settings
  is an error.
- `imported_files`: hashes of imported SGF files, so importing the same files
  twice adds nothing.

Each batch's config, log and SGFs are kept in `<state>_work/<pool>/<time>_A_vs_B/`.
For the default state that is `scripts/rating/rating_state_work/`.

## Rating

- Draws count as half a win.
- Prior: `--prior-draws` (default 2) virtual draws between each pair of
  consecutive nets of the same architecture, ordered by training samples. A new
  net starts at its predecessor's rating. Different architectures are linked
  only by real games and by a tiny all-pairs prior (0.01 virtual draws in total
  per net), which keeps the fit finite: the first net of a new architecture
  starts near the pool's average rating before it has games, and a one-sided
  first result (say 0-32) gives a low but finite rating with a large error. Two draws are roughly a Gaussian
  prior with an s.d. of about 245 Elo on each step. katago-server uses 4, but our
  steps between nets are bigger.
- The MAP is found by a full Newton solve with the anchor fixed at 0. The anchor
  is the oldest net in the pool when the first results were recorded; `--anchor
  NAME` overrides it.
- `±err` is 1 s.d. relative to the anchor, from the inverse Hessian. It grows
  along the chain and includes the anchor's own uncertainty.
- `local` is 1/sqrt(the net's own Fisher information), counting its own games
  and the prior. It measures how well the net is pinned down against its
  neighbours, and pairing uses it.
- The fit includes every net that has games, plus the nets selected for `run`.

## Pairing (`run`)

Each batch is one `katago match` between two nets, with `--games` games (default
32, colours balanced) at 16 visits:

1. A selected net with no games plays its neighbour in training order within its
   architecture, preferring a neighbour that already has games. Newest nets go
   first. If no neighbour of its architecture has games (the first net of a new
   architecture), it plays the newest primary-architecture net instead.
2. Otherwise a reference net is drawn from the top 5 of one of three lists, each
   list chosen with equal chance: highest `local` uncertainty, fewest games, or
   highest Elo + local error (the frontier).
3. The opponent comes from the nets within `--window` Elo (default 200) of the
   reference net's rating, after that rating is jittered by its uncertainty. The
   choice is weighted by the outcome variance p(1-p), with a mild penalty for
   pairs that have already played a lot. If fewer than 2 nets are in the window,
   the opponent is one of the 2 nearest nets below or the 2 nearest above.

The models directory is re-scanned before every batch, so new nets join
automatically. You can limit which nets play with `--min-samples`,
`--max-samples` (applied to each architecture's own sample count), `--every K`
(every K-th net counting back from the newest of each architecture, so the newest
of each is always in) or `--nets s43089k,s44785k`. Pairing ignores architecture
after step 1, so for a focused comparison list the nets, e.g. `--nets
b10c256nbt-s15000k,s49733k,s48036k`.

`plot` draws one series per architecture, with x = that architecture's training
samples.

`--batches N` and `--minutes M` both stop the run. `--minutes` only stops new
batches from starting; a batch already running finishes. With neither, `run`
plays one batch.

Ctrl-C or SIGTERM sends SIGINT to the katago PID this script started (and only
that PID). katago drops unfinished games, the finished ones are recorded, and the
script exits. A second Ctrl-C kills that katago outright.

## Import

`import DIR...` reads SGFs written by `katago match`. Bot names must be the full
net name (`b10c128-s44784512-d34590128`) or a short form (`s44785k`,
`b10c256nbt-s15000k`). A file is
skipped when:

- a bot name is unknown, such as the visit-ladder runs named `v32`/`v64`;
- the same net plays both sides;
- the `RU[maxmovesN]` value does not match the pool;
- a move comment shows more visits (`v=`) than the pool's maxVisits.

The visit check catches files played at more visits, but not files played at
fewer, so pass the right `--visits`.
