//! Influence rules (see RULES.md), mirroring cpp/game/board.cpp.
//! Coordinates follow the engine: x is the hex column, y is the hex row.

pub const SMALL_CAP: i32 = 8;
pub const BIG_CAP: i32 = 12;
pub const MAX_LEN: usize = 40;
pub const DEFAULT_LEN: usize = 11;

/// Hex neighbour offsets as (dx, dy).
pub const ADJ: [(i32, i32); 6] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Player {
    White,
    Black,
}

impl Player {
    pub fn opp(self) -> Player {
        match self {
            Player::White => Player::Black,
            Player::Black => Player::White,
        }
    }
    /// Sign of the cell values owned by this player.
    pub fn sign(self) -> i32 {
        match self {
            Player::White => 1,
            Player::Black => -1,
        }
    }
    /// Display name, after the colour the side is drawn in (GTP and SGF still say W / B).
    pub fn name(self) -> &'static str {
        match self {
            Player::White => "Blue",
            Player::Black => "Red",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    AttackSelect,
    AttackTarget(usize, usize),
    Reinforce,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Cell {
    pub wall: bool,
    pub big: bool,
    /// > 0 White, < 0 Black, 0 empty.
    pub v: i32,
}

impl Cell {
    pub fn cap(&self) -> i32 {
        if self.wall {
            0
        } else if self.big {
            BIG_CAP
        } else {
            SMALL_CAP
        }
    }
    pub fn owner(&self) -> Option<Player> {
        if self.wall || self.v == 0 {
            None
        } else if self.v > 0 {
            Some(Player::White)
        } else {
            Some(Player::Black)
        }
    }
    pub fn strength(&self) -> i32 {
        self.v.abs()
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Board {
    pub w: usize,
    pub h: usize,
    cells: Vec<Cell>,
    pub next: Player,
    pub stage: Stage,
    pub reinforce_left: i32,
    /// Number of finished player turns.
    pub turn: i32,
}

/// Predicted result of an attack: new target value and new source value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttackResult {
    pub target_v: i32,
    pub source_v: i32,
}

impl Board {
    pub fn new(w: usize, h: usize) -> Board {
        Board {
            w,
            h,
            cells: vec![Cell::default(); w * h],
            next: Player::White,
            stage: Stage::AttackSelect,
            reinforce_left: 0,
            turn: 0,
        }
    }

    /// RULES.md 3, same as Board::getDefaultStart in the engine.
    pub fn default_start(w: usize, h: usize) -> Board {
        let mut b = Board::new(w, h);
        let d = if w >= 4 && h >= 4 { 1 } else { 0 };
        let (wx, wy) = (d.min(w - 1), (h - 1).saturating_sub(d));
        let (bx, by) = ((w - 1).saturating_sub(d), d.min(h - 1));
        if (wx, wy) != (bx, by) {
            b.cell_mut(wx, wy).v = 5;
            b.cell_mut(bx, by).v = -7;
        }
        b
    }

    pub fn in_box(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h
    }

    pub fn cell(&self, x: usize, y: usize) -> &Cell {
        &self.cells[y * self.w + x]
    }

    pub fn cell_mut(&mut self, x: usize, y: usize) -> &mut Cell {
        &mut self.cells[y * self.w + x]
    }

    pub fn neighbours(&self, x: usize, y: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        ADJ.iter().filter_map(move |&(dx, dy)| {
            let (nx, ny) = (x as i32 + dx, y as i32 + dy);
            if self.in_box(nx, ny) && !self.cell(nx as usize, ny as usize).wall {
                Some((nx as usize, ny as usize))
            } else {
                None
            }
        })
    }

    pub fn is_adjacent(&self, a: (usize, usize), b: (usize, usize)) -> bool {
        self.neighbours(a.0, a.1).any(|n| n == b)
    }

    pub fn area(&self, p: Player) -> i32 {
        self.cells.iter().filter(|c| c.owner() == Some(p)).count() as i32
    }

    pub fn value(&self, p: Player) -> i32 {
        self.cells.iter().filter(|c| c.owner() == Some(p)).map(|c| c.strength()).sum()
    }

    /// Some(winner) if the game is over, Some(None) for a draw (nobody owns anything).
    pub fn winner(&self) -> Option<Option<Player>> {
        match (self.area(Player::White), self.area(Player::Black)) {
            (0, 0) => Some(None),
            (0, _) => Some(Some(Player::Black)),
            (_, 0) => Some(Some(Player::White)),
            _ => None,
        }
    }

    pub fn can_attack_from(&self, x: usize, y: usize) -> bool {
        let c = self.cell(x, y);
        c.owner() == Some(self.next)
            && c.strength() >= 2
            && self.neighbours(x, y).any(|(nx, ny)| self.cell(nx, ny).owner() != Some(self.next))
    }

    pub fn can_attack(&self, src: (usize, usize), dst: (usize, usize)) -> bool {
        self.can_attack_from(src.0, src.1)
            && self.is_adjacent(src, dst)
            && self.cell(dst.0, dst.1).owner() != Some(self.next)
    }

    pub fn can_reinforce(&self, x: usize, y: usize) -> bool {
        let c = self.cell(x, y);
        self.stage == Stage::Reinforce && c.owner() == Some(self.next) && c.strength() < c.cap()
    }

    /// RULES.md 4.2, including overflow.
    pub fn attack_result(&self, src: (usize, usize), dst: (usize, usize)) -> AttackResult {
        let sign = self.next.sign();
        let s = self.cell(src.0, src.1).strength();
        let t = *self.cell(dst.0, dst.1);
        let e = t.strength();
        let mut overflow = 0;
        let target_v = if t.v == 0 || s > e {
            let captured = if t.v == 0 { s - 1 } else { s - e };
            overflow = (captured - t.cap()).max(0);
            sign * (captured - overflow)
        } else if s == e {
            0
        } else {
            t.v.signum() * (e - s)
        };
        AttackResult { target_v, source_v: sign * (1 + overflow) }
    }

    // Moves ---------------------------------------------------------------

    pub fn select(&mut self, x: usize, y: usize) -> bool {
        if self.stage != Stage::AttackSelect || !self.can_attack_from(x, y) {
            return false;
        }
        self.stage = Stage::AttackTarget(x, y);
        true
    }

    /// Sandbox convenience, not an engine move.
    pub fn cancel_select(&mut self) {
        if let Stage::AttackTarget(..) = self.stage {
            self.stage = Stage::AttackSelect;
        }
    }

    pub fn attack(&mut self, x: usize, y: usize) -> bool {
        let Stage::AttackTarget(sx, sy) = self.stage else { return false };
        if !self.can_attack((sx, sy), (x, y)) {
            return false;
        }
        let r = self.attack_result((sx, sy), (x, y));
        self.cell_mut(x, y).v = r.target_v;
        self.cell_mut(sx, sy).v = r.source_v;
        self.stage = Stage::AttackSelect;
        true
    }

    /// PASS: ends the attack phase.
    pub fn end_attack(&mut self) -> bool {
        if self.stage == Stage::Reinforce {
            return false;
        }
        self.stage = Stage::Reinforce;
        self.reinforce_left = self.area(self.next);
        self.maybe_auto_reinforce();
        true
    }

    /// Adds up to `n` points to one cell.
    pub fn reinforce(&mut self, x: usize, y: usize, n: i32) -> bool {
        if !self.can_reinforce(x, y) {
            return false;
        }
        let c = *self.cell(x, y);
        let add = n.min(self.reinforce_left).min(c.cap() - c.strength()).max(1);
        self.cell_mut(x, y).v += self.next.sign() * add;
        self.reinforce_left -= add;
        self.maybe_auto_reinforce();
        true
    }

    /// Same as the engine: if spending the remaining points involves no choice, spend them and end the turn.
    fn maybe_auto_reinforce(&mut self) {
        let p = self.next;
        let room: Vec<usize> = (0..self.cells.len())
            .filter(|&i| self.cells[i].owner() == Some(p) && self.cells[i].strength() < self.cells[i].cap())
            .collect();
        let capacity: i32 = room.iter().map(|&i| self.cells[i].cap() - self.cells[i].strength()).sum();
        if self.reinforce_left > 0 && self.reinforce_left < capacity && room.len() > 1 {
            return;
        }
        for i in room {
            let c = &mut self.cells[i];
            let add = self.reinforce_left.min(c.cap() - c.strength());
            c.v += p.sign() * add;
            self.reinforce_left -= add;
        }
        self.end_turn();
    }

    fn end_turn(&mut self) {
        self.next = self.next.opp();
        self.stage = Stage::AttackSelect;
        self.reinforce_left = 0;
        self.turn += 1;
    }

    // Editing -------------------------------------------------------------

    /// Sets the side to move and resets the turn to the start of its attack phase.
    pub fn set_next(&mut self, p: Player) {
        self.next = p;
        self.stage = Stage::AttackSelect;
        self.reinforce_left = 0;
    }

    pub fn set_value(&mut self, x: usize, y: usize, v: i32) {
        let c = self.cell_mut(x, y);
        if !c.wall {
            c.v = v.clamp(-c.cap(), c.cap());
        }
        self.fix_stage();
    }

    pub fn toggle_wall(&mut self, x: usize, y: usize) {
        let c = self.cell_mut(x, y);
        *c = Cell { wall: !c.wall, big: false, v: 0 };
        self.fix_stage();
    }

    pub fn toggle_big(&mut self, x: usize, y: usize) {
        let c = self.cell_mut(x, y);
        if !c.wall {
            c.big = !c.big;
            c.v = c.v.clamp(-c.cap(), c.cap());
        }
        self.fix_stage();
    }

    /// Resizes the box, keeping cells that still fit.
    pub fn resize(&mut self, w: usize, h: usize) {
        let mut b = Board::new(w, h);
        for y in 0..h.min(self.h) {
            for x in 0..w.min(self.w) {
                *b.cell_mut(x, y) = *self.cell(x, y);
            }
        }
        b.next = self.next;
        b.turn = self.turn;
        *self = b;
    }

    /// After an edit, drop a turn state that no longer makes sense.
    fn fix_stage(&mut self) {
        match self.stage {
            Stage::AttackTarget(x, y) if !self.can_attack_from(x, y) => self.stage = Stage::AttackSelect,
            Stage::Reinforce => {
                let room = self.cells.iter().any(|c| c.owner() == Some(self.next) && c.strength() < c.cap());
                if !room {
                    self.end_turn();
                }
            }
            _ => {}
        }
    }

    // Text format (Board::toStringSimple / parseBoard, GTP set_board) -----

    pub fn to_text(&self, delim: char) -> String {
        let mut s = String::new();
        for y in 0..self.h {
            let row: Vec<String> = (0..self.w)
                .map(|x| {
                    let c = self.cell(x, y);
                    if c.wall {
                        return "#".to_string();
                    }
                    let mut t = match c.v {
                        0 => ".".to_string(),
                        v if v > 0 => format!("+{v}"),
                        v => format!("{v}"),
                    };
                    if c.big {
                        t.push('*');
                    }
                    t
                })
                .collect();
            s += &row.join(" ");
            s.push(delim);
        }
        let (stage, src) = match self.stage {
            Stage::AttackSelect => (0, "null".to_string()),
            Stage::AttackTarget(x, y) => (1, format!("({x},{y})")),
            Stage::Reinforce => (2, "null".to_string()),
        };
        let pla = if self.next == Player::White { "W" } else { "B" };
        s += &format!("state {pla} {stage} {src} {} {}", self.reinforce_left, self.turn);
        s.push(delim);
        s
    }

    pub fn parse(text: &str) -> Result<Board, String> {
        let mut rows: Vec<Vec<&str>> = Vec::new();
        let mut state: Option<Vec<&str>> = None;
        for line in text.split(['\n', '/', '|']) {
            let toks: Vec<&str> = line.split_whitespace().collect();
            if toks.is_empty() {
                continue;
            }
            if toks[0] == "state" {
                state = Some(toks);
            } else {
                rows.push(toks);
            }
        }
        let h = rows.len();
        let w = rows.first().map_or(0, |r| r.len());
        if w < 2 || h < 2 || w > MAX_LEN || h > MAX_LEN {
            return Err(format!("bad board size {w}x{h}"));
        }
        let mut b = Board::new(w, h);
        for (y, row) in rows.iter().enumerate() {
            if row.len() != w {
                return Err(format!("row {y} has {} cells, expected {w}", row.len()));
            }
            for (x, &tok) in row.iter().enumerate() {
                if tok == "#" {
                    b.cell_mut(x, y).wall = true;
                    continue;
                }
                let (num, big) = match tok.strip_suffix('*') {
                    Some(t) => (t, true),
                    None => (tok, false),
                };
                let v: i32 = match num {
                    "." | "" => 0,
                    _ => num.parse().map_err(|_| format!("bad cell token '{tok}'"))?,
                };
                let c = b.cell_mut(x, y);
                c.big = big;
                if v.abs() > c.cap() {
                    return Err(format!("cell value above cap: '{tok}'"));
                }
                c.v = v;
            }
        }
        if let Some(st) = state {
            if st.len() != 6 {
                return Err("bad state line".into());
            }
            b.next = match st[1].to_ascii_lowercase().as_str() {
                "w" | "white" => Player::White,
                "b" | "black" => Player::Black,
                _ => return Err(format!("bad player '{}'", st[1])),
            };
            let num = |s: &str| s.parse::<i32>().map_err(|_| format!("bad number '{s}'"));
            let stage = num(st[2])?;
            b.reinforce_left = num(st[4])?;
            b.turn = num(st[5])?;
            b.stage = match stage {
                0 => Stage::AttackSelect,
                1 => {
                    let xy = st[3].trim_matches(|c| c == '(' || c == ')');
                    let (x, y) = xy.split_once(',').ok_or("bad selected source")?;
                    let (x, y) = (num(x)? as usize, num(y)? as usize);
                    if x >= w || y >= h || !b.can_attack_from(x, y) {
                        return Err("selected source cannot attack".into());
                    }
                    Stage::AttackTarget(x, y)
                }
                2 => Stage::Reinforce,
                _ => return Err(format!("bad stage {stage}")),
            };
            if (stage == 2) != (b.reinforce_left > 0) || b.turn < 0 {
                return Err("inconsistent state line".into());
            }
        }
        Ok(b)
    }
}

/// An engine micro-move (GTP vertex or pass).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Move {
    Pass,
    Cell(usize, usize),
}

const GTP_LETTERS: &[u8] = b"ABCDEFGHJKLMNOPQRSTUVWXYZ";

impl Move {
    /// Parses a GTP vertex as written by the engine (Location::toString): column letter, row counted from the bottom.
    pub fn parse(s: &str, w: usize, h: usize) -> Option<Move> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("pass") {
            return Some(Move::Pass);
        }
        if let Some(xy) = s.strip_prefix('(').and_then(|t| t.strip_suffix(')')) {
            let (x, y) = xy.split_once(',')?;
            let (x, y) = (x.trim().parse().ok()?, y.trim().parse().ok()?);
            return (x < w && y < h).then_some(Move::Cell(x, y));
        }
        let letter = s.bytes().next()?.to_ascii_uppercase();
        let x = GTP_LETTERS.iter().position(|&c| c == letter)?;
        let row: usize = s[1..].parse().ok()?;
        if x >= w || row == 0 || row > h {
            return None;
        }
        Some(Move::Cell(x, h - row))
    }

    pub fn to_gtp(self, h: usize) -> String {
        match self {
            Move::Pass => "pass".into(),
            Move::Cell(x, y) => format!("{}{}", GTP_LETTERS[x] as char, h - y),
        }
    }
}

impl Board {
    /// Plays one engine micro-move for the current stage. Returns false if illegal.
    pub fn play(&mut self, mv: Move) -> bool {
        match (self.stage, mv) {
            (Stage::AttackSelect, Move::Pass) => self.end_attack(),
            (Stage::AttackSelect, Move::Cell(x, y)) => self.select(x, y),
            (Stage::AttackTarget(..), Move::Cell(x, y)) => self.attack(x, y),
            (Stage::Reinforce, Move::Cell(x, y)) => self.reinforce(x, y, 1),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_start() {
        let b = Board::default_start(11, 11);
        assert_eq!(b.cell(1, 9).v, 5);
        assert_eq!(b.cell(9, 1).v, -7);
    }

    #[test]
    fn attack_outcomes() {
        let mut b = Board::new(4, 4);
        b.set_value(0, 0, 5);
        b.set_value(1, 0, -3);
        b.set_value(0, 1, -5);
        b.set_value(1, 1, -7);
        assert_eq!(b.attack_result((0, 0), (1, 0)), AttackResult { target_v: 2, source_v: 1 });
        assert_eq!(b.attack_result((0, 0), (0, 1)), AttackResult { target_v: 0, source_v: 1 });
        assert_eq!(b.attack_result((0, 0), (1, 1)), AttackResult { target_v: -2, source_v: 1 });
        // (1,0) and (0,1) are not adjacent on this hex grid
        assert!(!b.is_adjacent((1, 0), (0, 1)));
        assert!(b.is_adjacent((0, 0), (1, 1)));
    }

    #[test]
    fn overflow() {
        let mut b = Board::new(3, 3);
        b.toggle_big(0, 0);
        b.set_value(0, 0, 12);
        b.set_value(2, 2, -1);
        assert!(b.select(0, 0));
        assert!(b.attack(1, 0));
        assert_eq!(b.cell(1, 0).v, 8);
        assert_eq!(b.cell(0, 0).v, 4);
        // the source can attack again
        assert!(b.select(0, 0));
    }

    #[test]
    fn turn_cycle_and_auto_reinforce() {
        let mut b = Board::default_start(11, 11);
        assert!(b.select(1, 9));
        assert!(b.attack(2, 9));
        assert_eq!(b.cell(2, 9).v, 4);
        assert_eq!(b.cell(1, 9).v, 1);
        assert!(b.end_attack());
        assert_eq!(b.reinforce_left, 2);
        assert!(b.reinforce(1, 9, 1));
        // one point left, forced choice between two cells is not forced
        assert_eq!(b.stage, Stage::Reinforce);
        assert!(b.reinforce(2, 9, 1));
        assert_eq!(b.next, Player::Black);
        assert_eq!(b.turn, 1);
        // Black has one cell: its reinforcement is forced
        assert!(b.end_attack());
        assert_eq!(b.cell(9, 1).v, -8);
        assert_eq!(b.next, Player::White);
    }

    #[test]
    fn engine_moves() {
        let mut b = Board::default_start(11, 11);
        assert_eq!(Move::parse("B2", 11, 11), Some(Move::Cell(1, 9)));
        assert_eq!(Move::Cell(1, 9).to_gtp(11), "B2");
        assert_eq!(Move::parse("J10", 11, 11), Some(Move::Cell(8, 1)));
        assert_eq!(Move::parse("pass", 11, 11), Some(Move::Pass));
        assert_eq!(Move::parse("L12", 11, 11), None);
        for m in ["B2", "C2", "pass", "B2"] {
            assert!(b.play(Move::parse(m, 11, 11).unwrap()), "{m}");
        }
        assert_eq!(b.reinforce_left, 1);
    }

    #[test]
    fn text_roundtrip() {
        let mut b = Board::default_start(5, 4);
        b.toggle_wall(2, 2);
        b.toggle_big(3, 0);
        b.set_value(3, 0, -11);
        b.select(1, 2);
        let t = b.to_text('/');
        assert_eq!(Board::parse(&t).unwrap(), b);
        let t = b.to_text('\n');
        assert_eq!(Board::parse(&t).unwrap(), b);
    }
}
