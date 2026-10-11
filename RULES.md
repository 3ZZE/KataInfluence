# Influence (deterministic): rules

Influence is a two-player game of territory control on a hexagonal board.
Players take turns. On each turn a player attacks neighbouring cells, then
reinforces the cells they own. Nothing in the game is random.

---

## 1. Board

### 1.1 Cells and walls

The board is a **finite, arbitrary subset of the infinite hexagonal grid**.

- Every hex in the subset is a **cell**. Cells are where the game is played.
- Every hex outside the subset is a **wall**. A wall cannot be owned, attacked
  or reinforced, and it has no value. Walls can lie outside the playing area
  or inside it as holes and obstacles.

### 1.1.1 Cell sizes

Every cell is either **small** or **big**. The size of a cell is fixed by the
map and never changes during the game. The size sets the cell's **cap**, the
largest strength the cell can hold:

| Size  | Cap  |
|-------|------|
| small | `8`  |
| big   | `12` |

Unless the map marks a cell as big, it is small.


### 1.2 Coordinates and adjacency

Cells are addressed by integer pairs `(row, col)`. Two cells are **adjacent**
if both are on the board (neither is a wall) and their difference
`(Δrow, Δcol)` is one of these six vectors:

```
(0, +1)   (0, -1)
(+1, 0)   (-1, 0)
(+1, +1)  (-1, -1)
```

Each cell has at most 6 neighbours. Cells on the board edge and cells next to
walls have fewer.

When drawn, the cell `(row, col)` is centred at
`x = col − row/2`, `y = row · √3/2`, which gives a grid of pointy-top hexes.

### 1.3 Default board

If no custom map is given, the board is the full `N × N` rhombus
`0 ≤ row, col < N` with no interior walls and no big cells (every cell is
small). The implementation uses `N = 11`.

---

## 2. Cell state

Each cell holds a signed integer `v`:

| Value   | Meaning                                                      |
|---------|--------------------------------------------------------------|
| `v > 0` | owned by **White** (drawn blue), with strength `v`           |
| `v < 0` | owned by **Black** (drawn red), with strength `abs(v)`       |
| `v = 0` | **empty** (neutral, drawn grey)                              |

Below, *strength* means `abs(v)`, and *own* or *enemy* is from the point of
view of the player whose turn it is.

- **Area** of a player = the number of cells they own.
- **Value** of a player = the total strength of the cells they own.

### 2.1 Strength cap

A cell's strength never exceeds its cap (1.1.1): **8** for a small cell,
**12** for a big cell.

- Reinforcement cannot raise a cell's strength above its cap.
- Combat never increases total strength (see 4.2). If a captured cell would
  get a strength above its own cap, it gets its cap and the excess stays in
  the attacking cell. This can only happen when a big cell captures a small
  one.
- The map cannot give a cell a starting strength above its cap.


## 3. Setup

A map gives the set of cells, which of them are big, and an initial value for
each one. All cells not listed are empty. A starting value's strength must not
exceed the cap of its cell.

**Default setup** (`N = 11`, offset `d = 1`):

| Cell                     | Value                     |
|--------------------------|---------------------------|
| `(N−1−d, d)` = `(9, 1)`  | `+5` (White, strength 5)  |
| `(d, N−1−d)` = `(1, 9)`  | `−7` (Black, strength 7)  |

White moves first. Black's higher starting strength makes up for this.

On a custom board with walls, the map must put every starting cell on a
non-wall hex.

---

## 4. Turn structure

Play goes around this cycle:

```
White Attack → White Reinforce → Black Attack → Black Reinforce → White Attack → …
```

A player's turn is an Attack phase followed by a Reinforce phase.

### 4.1 Attack phase

During the Attack phase the active player may make **any number of attacks,
including none**, in any order. The player decides when the phase ends.

An **attack** goes from a source cell `S` to a target cell `T`. It is legal
when all of these hold:

1. `S` is owned by the active player.
2. `S` has strength **at least 2**.
3. `T` is adjacent to `S`. In particular, `T` is not a wall.
4. `T` is **not** owned by the active player, so it is either empty or an
   enemy cell.

The source can be any own cell that meets these conditions. It does not have
to be connected to earlier attacks this turn.

### 4.2 Resolving an attack

Let `s` be the strength of `S`.

- **Target is empty:** the attacker captures `T`, which gets strength `s − 1`
  (always ≥ 1).
- **Target is an enemy cell with strength `e`:**
  - `s > e`: the attacker captures `T`, which gets strength `s − e`.
  - `s = e`: `T` becomes **empty** (value 0).
  - `s < e`: the defender keeps `T`, which drops to strength `e − s`.

In every case the source cell is then left at **strength 1** and still
belongs to the attacker.

**Overflow.** A captured `T` never gets more than its own cap (2.1). If the
strength computed above is larger than the cap of `T`, then `T` gets its
cap and the excess stays in `S`, which is left at `1 + excess` instead of 1.
For example, a big cell at 12 that captures an empty small cell leaves `T`
at 8 and `S` at `1 + (11 − 8) = 4`. Overflow only happens when a big cell
captures a small one. `S` is a big cell, so it can hold the excess, and if
it is left at 2 or more it can attack again in the same phase.

Consequences:

- A cell at strength 1 cannot attack again until it is reinforced in a later
  turn. (Only overflow can leave a source above 1.)
- A cell captured this phase with strength ≥ 2 can attack straight away, so
  attacks can chain deep into territory.
- Taking an empty cell costs 1 strength. Fighting an enemy cell has no extra
  cost beyond the defender's strength.
- Attacks never increase strength.

### 4.3 Reinforce phase

When the Attack phase ends, the active player receives **reinforcement
points** equal to their **area** at that moment, which is the number of cells
they own after all their attacks.

The player then spends the points one at a time. Each point adds +1 strength
to one of their own cells whose strength is below its cap (8 for small, 12
for big). The same cell can get several points, up to its cap.

- All points must be spent before the phase can end.
- If every own cell is already at its cap while points remain, the leftover
  points are lost and the phase ends.
- Reinforcement can only go to cells the player owns. Empty cells and walls
  cannot be reinforced.

After the Reinforce phase, play passes to the other player's Attack phase.

---

## 5. End of the game

- A player who owns **no cells** has lost, and the other player wins. Such a
  player also gets 0 reinforcement points, so they cannot recover.
- On a move limit there is a tie (technical rule). 


---

## 6. Summary of the map extensions

Compared with the base game on a full rhombus of small cells, maps can change
only these points.

Walls:

1. The board is any finite set of hexes. Hexes outside it are walls.
2. Adjacency (1.2) applies only between two non-wall cells, so walls block
   attacks.
3. Walls have no owner and no value, and they cannot be attacked or
   reinforced. They do not count toward area or value.
4. Starting positions must be on non-wall cells.

Big cells:

5. The map may mark cells as big. A big cell has cap 12 instead of 8.
6. The cap limits reinforcement (4.3) and the strength a captured cell
   receives (4.2). When a big cell captures a small one, the excess over the
   small cell's cap stays in the big cell (overflow, 4.2). Starting values
   must respect the cap (3).

All other rules (Sections 2–5) stay the same.
