//! GUI for Influence (RULES.md): play by hand or against the engine, analyse, load SGF games.

mod engine;
mod game;
mod replay;

use engine::{Candidate, EngineCtl, Want};
use game::{Board, Move, Player, Stage, DEFAULT_LEN, MAX_LEN};
use replay::Replay;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use macroquad::prelude::*;

const PANEL_W: f32 = 290.0;
/// Shift+RMB toggles big cells in edit mode. Off while big cells are not used in selfplay.
const EDIT_BIG_CELLS: bool = false;
const SQRT3: f32 = 1.732_050_8;

const BG: Color = Color::new(0.10, 0.11, 0.13, 1.0);
const PANEL_BG: Color = Color::new(0.14, 0.15, 0.18, 1.0);
const TEXT: Color = Color::new(0.90, 0.91, 0.93, 1.0);
const DIM: Color = Color::new(0.58, 0.60, 0.65, 1.0);
const EMPTY: Color = Color::new(0.36, 0.37, 0.40, 1.0);
const WALL: Color = Color::new(0.17, 0.17, 0.19, 1.0);
const WHITE_PLA: Color = Color::new(0.24, 0.47, 0.90, 1.0);
const BLACK_PLA: Color = Color::new(0.88, 0.27, 0.25, 1.0);
const HILITE: Color = Color::new(1.0, 0.85, 0.25, 1.0);
const BUTTON: Color = Color::new(0.22, 0.24, 0.28, 1.0);
const BUTTON_HOVER: Color = Color::new(0.30, 0.32, 0.38, 1.0);
const BEST: Color = Color::new(0.30, 0.95, 0.85, 1.0);

fn pla_color(p: Player) -> Color {
    match p {
        Player::White => WHITE_PLA,
        Player::Black => BLACK_PLA,
    }
}

/// Lighter for weak cells, full colour at cap.
fn cell_color(b: &Board, x: usize, y: usize) -> Color {
    let c = b.cell(x, y);
    match c.owner() {
        None => EMPTY,
        Some(p) => {
            let t = 0.45 + 0.55 * c.strength() as f32 / c.cap() as f32;
            let base = pla_color(p);
            Color::new(EMPTY.r + (base.r - EMPTY.r) * t, EMPTY.g + (base.g - EMPTY.g) * t, EMPTY.b + (base.b - EMPTY.b) * t, 1.0)
        }
    }
}

/// Maps hex cells to the screen: cell (x=col, y=row) is centred at (col - row/2, row*sqrt3/2) in units
/// of the neighbour distance (RULES.md 1.2).
struct Layout {
    scale: f32,
    ox: f32,
    oy: f32,
}

impl Layout {
    fn new(b: &Board) -> Layout {
        let (w, h) = (b.w as f32, b.h as f32);
        // unit extents of the cell centres, plus one hex of margin
        let min_x = -(h - 1.0) / 2.0;
        let max_x = w - 1.0;
        let span_x = max_x - min_x + 1.0;
        let span_y = (h - 1.0) * SQRT3 / 2.0 + 2.0 / SQRT3;
        let avail_w = screen_width() - PANEL_W - 30.0;
        let avail_h = screen_height() - 30.0;
        let scale = (avail_w / span_x).min(avail_h / span_y);
        let ox = 15.0 + (avail_w - span_x * scale) / 2.0 + (0.5 - min_x) * scale;
        let oy = 15.0 + (avail_h - span_y * scale) / 2.0 + scale / SQRT3;
        Layout { scale, ox, oy }
    }

    fn center(&self, x: usize, y: usize) -> Vec2 {
        vec2(self.ox + (x as f32 - y as f32 / 2.0) * self.scale, self.oy + y as f32 * SQRT3 / 2.0 * self.scale)
    }

    /// Circumradius of a hex.
    fn radius(&self) -> f32 {
        self.scale / SQRT3
    }

    fn pick(&self, b: &Board, p: Vec2) -> Option<(usize, usize)> {
        let mut best = None;
        let mut best_d = self.scale * 0.5;
        for y in 0..b.h {
            for x in 0..b.w {
                let d = self.center(x, y).distance(p);
                if d < best_d {
                    best_d = d;
                    best = Some((x, y));
                }
            }
        }
        best
    }
}

struct App {
    board: Board,
    undo: Vec<Board>,
    redo: Vec<Board>,
    edit: bool,
    message: String,
    engines: Vec<EngineSlot>,
    /// Engine (index into `engines`) playing each side: [White, Black]; None is a human.
    players: [Option<usize>; 2],
    analyze: bool,
    /// Engine that runs the analysis.
    analysis_engine: usize,
    /// Random rules variant (`Board::attack_random`) for all attacks played here, by hand or by an engine.
    random_rules: bool,
    /// Move and SGF comment of a loaded game (--replay), by the position text after the move.
    notes: HashMap<String, String>,
    /// Game index, result and players of the loaded game.
    game_info: String,
    /// Held navigation key and when it repeats next.
    repeat: Option<(KeyCode, f64)>,
}

impl App {
    fn new() -> App {
        App {
            board: Board::default_start(DEFAULT_LEN, DEFAULT_LEN),
            undo: Vec::new(),
            redo: Vec::new(),
            edit: false,
            message: String::new(),
            engines: Vec::new(),
            players: [None, None],
            analyze: false,
            analysis_engine: 0,
            random_rules: false,
            notes: HashMap::new(),
            game_info: String::new(),
            repeat: None,
        }
    }

    fn add_engine(&mut self, cmd: String) {
        let name = engine_name(&cmd);
        self.engines.push(EngineSlot { cmd, name, ctl: None });
        self.start_engine(self.engines.len() - 1);
    }

    fn start_engine(&mut self, i: usize) {
        let slot = &mut self.engines[i];
        slot.ctl = None;
        match EngineCtl::new(&slot.cmd) {
            Ok(e) => slot.ctl = Some(e),
            Err(e) => self.message = e,
        }
    }

    fn engine_alive(&self, i: usize) -> bool {
        self.engines.get(i).and_then(|s| s.ctl.as_ref()).is_some_and(|e| e.alive())
    }

    fn engine_side(&self, p: Player) -> bool {
        self.players[p as usize].is_some_and(|i| self.engine_alive(i))
    }

    /// Short label of engine `i` for buttons: "Engine" if there is only one, else "E1", "E2", ...
    fn engine_label(&self, i: usize) -> String {
        if self.engines.len() == 1 { "Engine".to_string() } else { format!("E{}", i + 1) }
    }

    /// Is the side to move played by the engine right now (blocks human input)?
    fn engine_to_move(&self) -> bool {
        !self.edit && !self.game_over() && self.engine_side(self.board.next)
    }

    /// Ticks every engine: the one playing the side to move generates a move, the analysis engine
    /// analyses when no engine is to move, the others stay idle.
    fn engine_tick(&mut self) {
        let mover = if self.engine_to_move() { self.players[self.board.next as usize] } else { None };
        let analyzer = (mover.is_none() && self.analyze && !self.game_over()).then_some(self.analysis_engine);
        let board = self.board.clone();
        for i in 0..self.engines.len() {
            let want = if mover == Some(i) {
                Want::Genmove
            } else if analyzer == Some(i) {
                Want::Analyze
            } else {
                Want::Idle
            };
            let Some(ctl) = self.engines[i].ctl.as_mut() else { continue };
            if let Some((pos, vertex)) = ctl.tick(&board, want) {
                // the board may change, the other engines catch up on the next tick
                self.apply_engine_move(i, pos, vertex);
                return;
            }
        }
    }

    fn apply_engine_move(&mut self, i: usize, pos: Board, vertex: String) {
        let label = self.engine_label(i);
        if vertex.eq_ignore_ascii_case("resign") {
            self.message = format!("{label} ({}) resigns", pos.next.name());
            self.players[pos.next as usize] = None;
            return;
        }
        let mut after = pos.clone();
        let legal = Move::parse(&vertex, pos.w, pos.h).is_some_and(|m| after.play(m));
        if !legal {
            self.message = format!("{label} played illegal move '{vertex}'");
            self.players = [None, None];
            return;
        }
        // The engine has played the move under the standard rules. If the random rules give another
        // result, the engine is told the real position with set_board before it is asked anything.
        if let Some(ctl) = self.engines[i].ctl.as_mut() {
            ctl.mark_synced(&after);
        }
        // ignore the move if the board changed while the engine was thinking
        if pos == self.board {
            if let (Stage::AttackTarget(..), Some(Move::Cell(x, y))) = (pos.stage, Move::parse(&vertex, pos.w, pos.h)) {
                self.attack(x, y);
            } else {
                self.change(|b| *b = after);
            }
        }
    }

    /// Attacks (x, y) from the selected source, under the random rules if they are on.
    fn attack(&mut self, x: usize, y: usize) {
        let Stage::AttackTarget(sx, sy) = self.board.stage else { return };
        let chance = self.board.capture_chance((sx, sy), (x, y)).filter(|_| self.random_rules);
        let Some(chance) = chance else {
            self.change(|b| {
                b.attack(x, y);
            });
            return;
        };
        let roll = macroquad::rand::gen_range(0.0, 1.0);
        let pla = self.board.next;
        self.change(|b| {
            b.attack_random(x, y, roll);
        });
        let won = self.board.cell(x, y).owner() == Some(pla);
        self.message = format!(
            "{} attack {}: {:.0}% chance, {}",
            pla.name(),
            Move::Cell(x, y).to_gtp(self.board.h),
            chance * 100.0,
            if won { "captured" } else { "failed" }
        );
    }

    /// Current analysis, if it belongs to the board on screen.
    fn analysis(&self) -> Option<&[Candidate]> {
        if !self.analyze || self.engine_to_move() {
            return None;
        }
        let (b, c) = self.engines.get(self.analysis_engine)?.ctl.as_ref()?.analysis.as_ref()?;
        (*b == self.board && !c.is_empty()).then_some(c.as_slice())
    }

    /// Applies `f` to a copy of the board and records it in the history if anything changed.
    fn change(&mut self, f: impl FnOnce(&mut Board)) {
        let mut b = self.board.clone();
        f(&mut b);
        if b != self.board {
            self.undo.push(std::mem::replace(&mut self.board, b));
            self.redo.clear();
        }
    }

    /// Steps back over engine moves too, so the engine does not immediately replay them.
    fn undo(&mut self) {
        while let Some(b) = self.undo.pop() {
            self.redo.push(std::mem::replace(&mut self.board, b));
            if !self.engine_to_move() {
                break;
            }
        }
    }

    fn redo(&mut self) {
        if let Some(b) = self.redo.pop() {
            self.undo.push(std::mem::replace(&mut self.board, b));
        }
    }

    /// Back to the start of the current turn, or of the previous one if already there.
    /// Skips turns the engine would replay.
    fn undo_turn(&mut self) {
        while let Some(b) = self.undo.pop() {
            self.redo.push(std::mem::replace(&mut self.board, b));
            let at_turn_start = self.undo.last().is_none_or(|prev| prev.turn != self.board.turn);
            if at_turn_start && !self.engine_to_move() {
                break;
            }
        }
    }

    /// Forward to the start of the next turn (or as far as the redo history goes).
    fn redo_turn(&mut self) {
        let turn = self.board.turn;
        while self.board.turn == turn {
            let Some(b) = self.redo.pop() else { break };
            self.undo.push(std::mem::replace(&mut self.board, b));
        }
    }

    /// True when `k` is pressed, and repeatedly while it is held.
    fn key_repeat(&mut self, k: KeyCode) -> bool {
        let now = get_time();
        if is_key_pressed(k) {
            self.repeat = Some((k, now + 0.4));
            return true;
        }
        match self.repeat {
            Some((rk, at)) if rk == k && is_key_down(k) && now >= at => {
                self.repeat = Some((k, now + 0.03));
                true
            }
            _ => false,
        }
    }

    fn undo_all(&mut self) {
        while let Some(b) = self.undo.pop() {
            self.redo.push(std::mem::replace(&mut self.board, b));
        }
    }

    fn redo_all(&mut self) {
        while let Some(b) = self.redo.pop() {
            self.undo.push(std::mem::replace(&mut self.board, b));
        }
    }

    /// Puts a game's main line into the history as if its moves had been played here, showing the final position.
    fn load_game(&mut self, r: Replay) {
        self.notes = (0..r.len()).map(|i| (r.positions[i + 1].to_text('/'), format!("{}  {}", r.describe(i), r.moves[i].2))).collect();
        let players = if r.white == r.black { r.white.clone() } else { format!("Blue {}  Red {}", r.white, r.black) };
        self.game_info = format!("game {}  result {}  {players}", r.game, r.result);
        let mut positions = r.positions;
        self.board = positions.pop().unwrap();
        self.undo = positions;
        self.redo.clear();
    }

    fn game_over(&self) -> bool {
        self.board.winner().is_some()
    }

    fn copy(&mut self) {
        let text = self.board.to_text('/');
        miniquad::window::clipboard_set(&text);
        println!("{text}");
        self.message = "Copied position (GTP set_board format)".into();
    }

    fn paste(&mut self) {
        let Some(text) = miniquad::window::clipboard_get() else {
            self.message = "Clipboard is empty".into();
            return;
        };
        match Board::parse(&text) {
            Ok(b) => {
                self.change(|cur| *cur = b);
                self.message = "Pasted position".into();
            }
            Err(e) => self.message = format!("Paste failed: {e}"),
        }
    }

    fn resize(&mut self, dw: i32, dh: i32) {
        let w = (self.board.w as i32 + dw).clamp(2, MAX_LEN as i32) as usize;
        let h = (self.board.h as i32 + dh).clamp(2, MAX_LEN as i32) as usize;
        self.change(|b| b.resize(w, h));
    }

    fn handle_board_click(&mut self, cell: (usize, usize)) {
        let (x, y) = cell;
        let shift = is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift);
        let left = is_mouse_button_pressed(MouseButton::Left);
        let right = is_mouse_button_pressed(MouseButton::Right);
        let middle = is_mouse_button_pressed(MouseButton::Middle);

        if self.edit {
            let v = self.board.cell(x, y).v;
            if shift && left {
                self.change(|b| b.toggle_wall(x, y));
            } else if shift && right {
                if EDIT_BIG_CELLS {
                    self.change(|b| b.toggle_big(x, y));
                }
            } else if left {
                self.change(|b| b.set_value(x, y, v + 1));
            } else if right {
                self.change(|b| b.set_value(x, y, v - 1));
            } else if middle {
                self.change(|b| b.set_value(x, y, 0));
            }
            return;
        }

        if self.game_over() || self.engine_to_move() {
            return;
        }
        if right {
            self.change(|b| b.cancel_select());
            return;
        }
        if !left {
            return;
        }
        match self.board.stage {
            Stage::AttackSelect => self.change(|b| {
                b.select(x, y);
            }),
            Stage::AttackTarget(sx, sy) => {
                if (sx, sy) == (x, y) {
                    self.change(|b| b.cancel_select());
                } else if self.board.can_attack((sx, sy), (x, y)) {
                    // chain: select the captured cell if it can attack, else keep the source (overflow)
                    self.attack(x, y);
                    // one history step per click: the selection goes with the attack
                    let mut b = self.board.clone();
                    if !b.select(x, y) {
                        b.select(sx, sy);
                    }
                    if b != self.board {
                        self.board = b;
                    }
                } else {
                    // clicking another own source switches the selection
                    self.change(|b| {
                        b.cancel_select();
                        b.select(x, y);
                    });
                }
            }
            Stage::Reinforce => {
                let n = if shift { i32::MAX } else { 1 };
                self.change(|b| {
                    b.reinforce(x, y, n);
                });
            }
        }
    }

    fn handle_keys(&mut self) {
        let ctrl = is_key_down(KeyCode::LeftControl) || is_key_down(KeyCode::RightControl);
        let shift = is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift);
        // history keys repeat while held; Left / Right / Home / End only outside edit mode
        let (z, y) = (self.key_repeat(KeyCode::Z), self.key_repeat(KeyCode::Y));
        let (left, right) = (!self.edit && self.key_repeat(KeyCode::Left), !self.edit && self.key_repeat(KeyCode::Right));
        if (shift && z) || self.key_repeat(KeyCode::PageUp) {
            self.undo_turn();
        } else if z || left || is_key_pressed(KeyCode::Backspace) {
            self.undo();
        }
        if (shift && y) || self.key_repeat(KeyCode::PageDown) {
            self.redo_turn();
        } else if y || right {
            self.redo();
        }
        if !self.edit && is_key_pressed(KeyCode::Home) {
            self.undo_all();
        }
        if !self.edit && is_key_pressed(KeyCode::End) {
            self.redo_all();
        }
        if ctrl && is_key_pressed(KeyCode::C) {
            self.copy();
        }
        if ctrl && is_key_pressed(KeyCode::V) {
            self.paste();
        }
        if is_key_pressed(KeyCode::E) {
            self.edit = !self.edit;
            self.change(|b| b.cancel_select());
        }
        if is_key_pressed(KeyCode::A) {
            self.analyze = !self.analyze;
        }
        if is_key_pressed(KeyCode::N) {
            let (w, h) = (self.board.w, self.board.h);
            self.change(|b| *b = Board::default_start(w, h));
        }
        if self.edit {
            if is_key_pressed(KeyCode::Right) {
                self.resize(1, 0);
            }
            if is_key_pressed(KeyCode::Left) {
                self.resize(-1, 0);
            }
            if is_key_pressed(KeyCode::Down) {
                self.resize(0, 1);
            }
            if is_key_pressed(KeyCode::Up) {
                self.resize(0, -1);
            }
            if is_key_pressed(KeyCode::T) {
                self.change(|b| b.set_next(b.next.opp()));
            }
            if !ctrl && is_key_pressed(KeyCode::C) {
                let (w, h) = (self.board.w, self.board.h);
                self.change(|b| *b = Board::new(w, h));
            }
        } else if !self.game_over() && !self.engine_to_move() {
            if is_key_pressed(KeyCode::Space) || is_key_pressed(KeyCode::Enter) {
                self.change(|b| {
                    b.end_attack();
                });
            }
            if is_key_pressed(KeyCode::Escape) {
                self.change(|b| b.cancel_select());
            }
        }
    }

    fn draw_board(&self, layout: &Layout, hover: Option<(usize, usize)>) {
        let b = &self.board;
        let r = layout.radius();
        let font = (layout.scale * 0.5).max(10.0) as u16;
        let selected = match b.stage {
            Stage::AttackTarget(x, y) => Some((x, y)),
            _ => None,
        };
        let playing = !self.edit && !self.game_over() && !self.engine_to_move();
        let cand_cells: Vec<(usize, usize)> = shown_candidates(self.analysis().unwrap_or(&[]))
            .filter_map(|c| match c.mv {
                Move::Cell(x, y) => Some((x, y)),
                Move::Pass => None,
            })
            .collect();

        for y in 0..b.h {
            for x in 0..b.w {
                let c = b.cell(x, y);
                let p = layout.center(x, y);
                if c.wall {
                    draw_poly(p.x, p.y, 6, r * 0.96, 30.0, WALL);
                    continue;
                }
                draw_poly(p.x, p.y, 6, r * 0.94, 30.0, cell_color(b, x, y));
                if c.big {
                    draw_poly_lines(p.x, p.y, 6, r * 0.80, 30.0, (r * 0.07).max(1.5), Color::new(1.0, 1.0, 1.0, 0.55));
                }

                // hints for what can be clicked
                if playing {
                    let hint = match b.stage {
                        Stage::AttackSelect => b.can_attack_from(x, y),
                        Stage::AttackTarget(sx, sy) => b.can_attack((sx, sy), (x, y)),
                        Stage::Reinforce => b.can_reinforce(x, y),
                    };
                    if hint {
                        draw_poly_lines(p.x, p.y, 6, r * 0.90, 30.0, (r * 0.06).max(1.0), Color::new(1.0, 1.0, 1.0, 0.35));
                    }
                }

                if c.v != 0 {
                    let s = c.strength().to_string();
                    let shifted = cand_cells.contains(&(x, y));
                    let f = if shifted { font * 4 / 5 } else { font };
                    let dims = measure_text(&s, None, f, 1.0);
                    let cy = if shifted { p.y - layout.scale * 0.13 } else { p.y };
                    draw_text(&s, p.x - dims.width / 2.0, cy + dims.offset_y / 2.0, f as f32, WHITE);
                }
            }
        }

        if let Some((x, y)) = selected {
            let p = layout.center(x, y);
            draw_poly_lines(p.x, p.y, 6, r * 0.95, 30.0, (r * 0.12).max(2.0), HILITE);
        }
        if let Some((x, y)) = self.last_move_cell() {
            let p = layout.center(x, y);
            draw_circle(p.x, p.y + r * 0.62, (r * 0.1).max(2.0), WHITE);
        }
        if let Some(cands) = self.analysis() {
            self.draw_analysis(layout, cands, hover);
        }
        if let Some((x, y)) = hover {
            if !b.cell(x, y).wall || self.edit {
                let p = layout.center(x, y);
                draw_poly_lines(p.x, p.y, 6, r * 0.97, 30.0, (r * 0.05).max(1.0), Color::new(1.0, 1.0, 1.0, 0.8));
            }
        }
    }

    /// The cell the last history step changed (besides an attack source), if it changed just one.
    fn last_move_cell(&self) -> Option<(usize, usize)> {
        let (prev, b) = (self.undo.last()?, &self.board);
        if self.edit || (prev.w, prev.h) != (b.w, b.h) {
            return None;
        }
        let src = match prev.stage {
            Stage::AttackTarget(x, y) => Some((x, y)),
            _ => None,
        };
        let mut changed = (0..b.h)
            .flat_map(|y| (0..b.w).map(move |x| (x, y)))
            .filter(|&(x, y)| prev.cell(x, y) != b.cell(x, y) && src != Some((x, y)));
        let first = changed.next();
        if changed.next().is_some() {
            return None;
        }
        first
    }

    /// Move number and SGF notes of a loaded game, top left over the board area.
    fn draw_notes(&self) {
        if self.notes.is_empty() {
            return;
        }
        let mut lines = vec![self.game_info.clone(), format!("move {} / {}", self.undo.len(), self.undo.len() + self.redo.len())];
        if let Some(s) = self.notes.get(&self.board.to_text('/')) {
            lines.push(format!("last  {s}"));
        }
        if let Some(s) = self.redo.last().and_then(|b| self.notes.get(&b.to_text('/'))) {
            lines.push(format!("next  {s}"));
        }
        let w = lines.iter().map(|l| measure_text(l, None, 16, 1.0).width).fold(0.0, f32::max);
        draw_rectangle(8.0, 8.0, w + 16.0, lines.len() as f32 * 18.0 + 10.0, Color::new(0.0, 0.0, 0.0, 0.55));
        for (i, l) in lines.iter().enumerate() {
            draw_text(l, 16.0, 26.0 + i as f32 * 18.0, 16.0, TEXT);
        }
    }

    fn draw_analysis(&self, layout: &Layout, cands: &[Candidate], hover: Option<(usize, usize)>) {
        let b = &self.board;
        let r = layout.radius();
        let total: u64 = cands.iter().map(|c| c.visits).sum::<u64>().max(1);
        let font = (layout.scale * 0.3).max(9.0) as u16;
        for c in shown_candidates(cands) {
            let Move::Cell(x, y) = c.mv else { continue };
            let p = layout.center(x, y);
            let frac = c.visits as f32 / total as f32;
            let (color, thick) = if c.order == 0 { (BEST, 0.13) } else { (Color::new(1.0, 1.0, 1.0, 0.3 + 0.7 * frac.sqrt()), 0.07) };
            draw_poly_lines(p.x, p.y, 6, r * 0.86, 30.0, (r * thick).max(1.5), color);
            let s = format!("{:.0}", c.winrate * 100.0);
            let dims = measure_text(&s, None, font, 1.0);
            let ty = if b.cell(x, y).v != 0 { p.y + layout.scale * 0.25 } else { p.y };
            draw_text(&s, p.x - dims.width / 2.0, ty + dims.offset_y / 2.0, font as f32, color);
        }

        // principal variation of the hovered candidate, else of the best move, until the turn ends
        let shown = hover.and_then(|(x, y)| cands.iter().find(|c| c.mv == Move::Cell(x, y))).unwrap_or(&cands[0]);
        let (arrows, plus) = pv_marks(b, &shown.pv);
        let pc = pla_color(b.next);
        let arrow_color = Color::new(1.0 - (1.0 - pc.r) * 0.4, 1.0 - (1.0 - pc.g) * 0.4, 1.0 - (1.0 - pc.b) * 0.4, 0.95);
        for (i, &(src, dst)) in arrows.iter().enumerate() {
            let (a, z) = (layout.center(src.0, src.1), layout.center(dst.0, dst.1));
            let dir = (z - a).normalize();
            let (a, z) = (a + dir * r * 0.35, z - dir * r * 0.35);
            let th = (r * 0.1).max(2.0);
            draw_line(a.x, a.y, z.x, z.y, th, arrow_color);
            let side = vec2(-dir.y, dir.x) * r * 0.22;
            draw_triangle(z + dir * r * 0.12, z - dir * r * 0.2 + side, z - dir * r * 0.2 - side, arrow_color);
            let m = (a + z) / 2.0;
            let s = (i + 1).to_string();
            let dims = measure_text(&s, None, font, 1.0);
            draw_circle(m.x, m.y, dims.width.max(dims.height) * 0.75, Color::new(0.0, 0.0, 0.0, 0.7));
            draw_text(&s, m.x - dims.width / 2.0, m.y + dims.offset_y / 2.0, font as f32, arrow_color);
        }
        for (&(x, y), &n) in &plus {
            let p = layout.center(x, y);
            let s = format!("+{n}");
            let dims = measure_text(&s, None, font, 1.0);
            draw_text(&s, p.x - dims.width / 2.0, p.y - layout.scale * 0.32 + dims.offset_y / 2.0, font as f32, HILITE);
        }
    }

    fn draw_engine_panel(&mut self, x: f32, y: &mut f32, hover: Option<(usize, usize)>) {
        let line = |s: &str, y: &mut f32, size: f32, color: Color| {
            draw_text(s, x, *y, size, color);
            *y += size * 1.15;
        };
        let bw = (PANEL_W - 32.0 - 8.0) / 2.0;
        let x2 = x + bw + 8.0;
        *y += 8.0;
        if self.engines.is_empty() {
            line("Engine: none (--engine \"<cmd>\")", y, 18.0, TEXT);
            *y += 4.0;
            return;
        }
        for (i, slot) in self.engines.iter().enumerate() {
            let state = match &slot.ctl {
                None => "not running".to_string(),
                Some(e) if !e.alive() => "exited".to_string(),
                Some(e) if !e.ready => format!("starting... {}", e.last_stderr()),
                Some(e) if e.thinking() => "thinking".to_string(),
                Some(e) if e.analyzing() => "analyzing".to_string(),
                Some(_) => "ready".to_string(),
            };
            let status = format!("{}: {state}", self.engine_label(i));
            line(&status.chars().take(36).collect::<String>(), y, 18.0, TEXT);
            if self.engines.len() > 1 {
                line(&slot.name.chars().take(38).collect::<String>(), y, 15.0, DIM);
            }
            if let Some(err) = slot.ctl.as_ref().and_then(|e| e.error.clone()) {
                line(&err.chars().take(36).collect::<String>(), y, 16.0, BLACK_PLA);
            }
        }
        *y += 4.0;
        let dead: Vec<usize> = (0..self.engines.len()).filter(|&i| !self.engine_alive(i)).collect();
        if !dead.is_empty() {
            let label = if self.engines.len() == 1 { "Restart engine".to_string() } else { "Restart stopped engines".to_string() };
            if self.button(&label, x, *y, PANEL_W - 32.0, true) {
                for i in dead {
                    self.start_engine(i);
                }
            }
            *y += 34.0;
        }
        // each side cycles Human -> engine 1 -> engine 2 -> ... -> Human
        for (i, p) in [Player::White, Player::Black].into_iter().enumerate() {
            let who = self.players[i].map_or("Human".to_string(), |e| self.engine_label(e));
            if self.button(&format!("{}: {who}", p.name()), if i == 0 { x } else { x2 }, *y, bw, true) {
                self.players[i] = match self.players[i] {
                    None => Some(0),
                    Some(e) if e + 1 < self.engines.len() => Some(e + 1),
                    Some(_) => None,
                };
            }
        }
        *y += 34.0;
        let analysis_label = if self.analyze { "Analysis: on [A]" } else { "Analysis: off [A]" };
        if self.engines.len() == 1 {
            if self.button(analysis_label, x, *y, PANEL_W - 32.0, true) {
                self.analyze = !self.analyze;
            }
        } else {
            if self.button(analysis_label, x, *y, bw, true) {
                self.analyze = !self.analyze;
            }
            if self.button(&format!("by {}", self.engine_label(self.analysis_engine)), x2, *y, bw, true) {
                self.analysis_engine = (self.analysis_engine + 1) % self.engines.len();
            }
        }
        *y += 34.0;

        let Some(cands) = self.analysis() else { return };
        let best = &cands[0];
        let visits: u64 = cands.iter().map(|c| c.visits).sum();
        let white_wr = if self.board.next == Player::White { best.winrate } else { 1.0 - best.winrate };
        // winrate bar, Blue (White) on the left
        let w = PANEL_W - 32.0;
        draw_rectangle(x, *y, w * white_wr as f32, 14.0, WHITE_PLA);
        draw_rectangle(x + w * white_wr as f32, *y, w * (1.0 - white_wr as f32), 14.0, BLACK_PLA);
        *y += 32.0;
        line(&format!("Blue {:.1}%   {} visits", white_wr * 100.0, visits), y, 18.0, TEXT);
        let h = self.board.h;
        let describe = |c: &Candidate| {
            format!("{:<5} {:5.1}% {:>6}v  prior {:.0}%", c.mv.to_gtp(h), c.winrate * 100.0, c.visits, c.prior * 100.0)
        };
        line(&format!("best  {}", describe(best)), y, 16.0, BEST);
        if let Some(c) = hover.and_then(|(hx, hy)| cands.iter().find(|c| c.mv == Move::Cell(hx, hy))) {
            line(&format!("hover {}", describe(c)), y, 16.0, TEXT);
        }
        if let Some(c) = cands.iter().find(|c| c.mv == Move::Pass) {
            line(&format!("end attack {:.1}% {}v", c.winrate * 100.0, c.visits), y, 16.0, DIM);
        }
        let pv: Vec<String> = best.pv.iter().take(10).map(|m| m.to_gtp(h)).collect();
        line(&format!("pv {}", pv.join(" ")), y, 16.0, DIM);
    }

    /// Returns true if clicked.
    fn button(&self, label: &str, x: f32, y: f32, w: f32, enabled: bool) -> bool {
        let h = 28.0;
        let m = mouse_position();
        let inside = m.0 >= x && m.0 <= x + w && m.1 >= y && m.1 <= y + h;
        draw_rectangle(x, y, w, h, if inside && enabled { BUTTON_HOVER } else { BUTTON });
        let dims = measure_text(label, None, 18, 1.0);
        draw_text(label, x + (w - dims.width) / 2.0, y + h / 2.0 + dims.offset_y / 2.0, 18.0, if enabled { TEXT } else { DIM });
        enabled && inside && is_mouse_button_pressed(MouseButton::Left)
    }

    fn draw_panel(&mut self, hover: Option<(usize, usize)>) {
        let x0 = screen_width() - PANEL_W;
        draw_rectangle(x0, 0.0, PANEL_W, screen_height(), PANEL_BG);
        let x = x0 + 16.0;
        let mut y = 34.0;
        let line = |s: &str, y: &mut f32, size: f32, color: Color| {
            draw_text(s, x, *y, size, color);
            *y += size * 1.15;
        };

        let b = self.board.clone();
        line(if self.edit { "Influence - EDIT" } else { "Influence" }, &mut y, 26.0, TEXT);
        y += 6.0;

        match b.winner() {
            Some(Some(p)) => line(&format!("{} wins", p.name()), &mut y, 24.0, pla_color(p)),
            Some(None) => line("No cells owned", &mut y, 24.0, DIM),
            None => {
                line(&format!("{} to move", b.next.name()), &mut y, 24.0, pla_color(b.next));
                let stage = match b.stage {
                    Stage::AttackSelect => "Attack: choose source".to_string(),
                    Stage::AttackTarget(..) => "Attack: choose target".to_string(),
                    Stage::Reinforce => format!("Reinforce: {} point(s) left", b.reinforce_left),
                };
                line(&stage, &mut y, 20.0, TEXT);
            }
        }
        line(&format!("Turn {}   board {}x{}", b.turn, b.w, b.h), &mut y, 18.0, DIM);
        y += 8.0;
        for p in [Player::White, Player::Black] {
            line(&format!("{:<6} area {:>3}   value {:>4}", p.name(), b.area(p), b.value(p)), &mut y, 20.0, pla_color(p));
        }
        y += 6.0;

        // attack preview / hovered cell info
        let info = hover.map(|(hx, hy)| {
            let c = b.cell(hx, hy);
            if let Stage::AttackTarget(sx, sy) = b.stage {
                if !self.edit && b.can_attack((sx, sy), (hx, hy)) {
                    if let Some(chance) = b.capture_chance((sx, sy), (hx, hy)).filter(|_| self.random_rules) {
                        return format!("attack: {:.0}% capture, target -> 1, source -> 1", chance * 100.0);
                    }
                    let r = b.attack_result((sx, sy), (hx, hy));
                    return format!("attack: target {:+} -> {:+}, source -> {}", c.v, r.target_v, r.source_v.abs());
                }
            }
            if c.wall {
                format!("({hx},{hy}) wall")
            } else {
                format!("({hx},{hy}) {} {:+} / cap {}", if c.big { "big" } else { "small" }, c.v, c.cap())
            }
        });
        line(info.as_deref().unwrap_or(""), &mut y, 17.0, DIM);
        y += 8.0;

        // buttons
        let bw = (PANEL_W - 32.0 - 8.0) / 2.0;
        let x2 = x + bw + 8.0;
        let playing = !self.edit && !self.game_over() && !self.engine_to_move();
        if self.button("End attack [Space]", x, y, PANEL_W - 32.0, playing && b.stage != Stage::Reinforce) {
            self.change(|b| {
                b.end_attack();
            });
        }
        y += 34.0;
        let qw = (PANEL_W - 32.0 - 3.0 * 6.0) / 4.0;
        let (can_undo, can_redo) = (!self.undo.is_empty(), !self.redo.is_empty());
        if self.button("<<", x, y, qw, can_undo) {
            self.undo_turn();
        }
        if self.button("<", x + qw + 6.0, y, qw, can_undo) {
            self.undo();
        }
        if self.button(">", x + 2.0 * (qw + 6.0), y, qw, can_redo) {
            self.redo();
        }
        if self.button(">>", x + 3.0 * (qw + 6.0), y, qw, can_redo) {
            self.redo_turn();
        }
        y += 34.0;
        if self.button(if self.edit { "Play [E]" } else { "Edit [E]" }, x, y, bw, true) {
            self.edit = !self.edit;
            self.change(|b| b.cancel_select());
        }
        if self.button("New game [N]", x2, y, bw, true) {
            let (w, h) = (b.w, b.h);
            self.change(|b| *b = Board::default_start(w, h));
        }
        y += 34.0;
        if self.button("Copy [^C]", x, y, bw, true) {
            self.copy();
        }
        if self.button("Paste [^V]", x2, y, bw, true) {
            self.paste();
        }
        y += 34.0;
        let rules = if self.random_rules { "Rules: random" } else { "Rules: standard" };
        if self.button(rules, x, y, PANEL_W - 32.0, true) {
            self.random_rules = !self.random_rules;
        }
        y += 34.0;
        if self.edit {
            if self.button("Swap side to move [T]", x, y, PANEL_W - 32.0, true) {
                self.change(|b| b.set_next(b.next.opp()));
            }
            y += 34.0;
            if self.button("W-", x, y, bw / 2.0 - 4.0, b.w > 2) {
                self.resize(-1, 0);
            }
            if self.button("W+", x + bw / 2.0 + 4.0, y, bw / 2.0 - 4.0, b.w < MAX_LEN) {
                self.resize(1, 0);
            }
            if self.button("H-", x2, y, bw / 2.0 - 4.0, b.h > 2) {
                self.resize(0, -1);
            }
            if self.button("H+", x2 + bw / 2.0 + 4.0, y, bw / 2.0 - 4.0, b.h < MAX_LEN) {
                self.resize(0, 1);
            }
            y += 34.0;
        }
        self.draw_engine_panel(x, &mut y, hover);
        y += 14.0;

        let help: &[&str] = if self.edit {
            &[
                "LMB / RMB   value +1 / -1",
                "            (+ White, - Black)",
                "MMB         clear cell",
                "Shift+LMB   toggle wall",
                "Shift+RMB   toggle big cell",
                "Arrows      resize board",
                "C           clear board",
            ]
        } else {
            &[
                "LMB own cell    select source",
                "LMB neighbour   attack",
                "RMB / Esc       cancel selection",
                "LMB own cell    reinforce +1",
                "Shift+LMB       reinforce max",
                "Z/Y, Left/Right undo / redo",
                "Shift+Z/Y, PgUp/PgDn  by turn",
                "Home / End      history start/end",
                "A               toggle analysis",
            ]
        };
        for h in help.iter().filter(|h| EDIT_BIG_CELLS || !h.contains("big cell")) {
            line(h, &mut y, 17.0, DIM);
        }

        if !self.message.is_empty() {
            draw_text(&self.message, x, screen_height() - 16.0, 17.0, DIM);
        }
    }
}

struct EngineSlot {
    cmd: String,
    /// Model file name (or program name) for the panel.
    name: String,
    ctl: Option<EngineCtl>,
}

/// The `-model` file name of an engine command line without its extension, else the program name.
fn engine_name(cmd: &str) -> String {
    let words = engine::split_command(cmd).unwrap_or_default();
    let path = words.iter().position(|w| w == "-model").and_then(|i| words.get(i + 1)).or(words.first());
    let Some(path) = path else { return String::new() };
    let file = path.rsplit('/').next().unwrap_or(path);
    let file = file.strip_suffix(".gz").unwrap_or(file);
    file.strip_suffix(".bin").or_else(|| file.strip_suffix(".txt")).unwrap_or(file).to_string()
}

/// Candidates worth drawing on the board.
fn shown_candidates(cands: &[Candidate]) -> impl Iterator<Item = &Candidate> {
    let total: u64 = cands.iter().map(|c| c.visits).sum::<u64>().max(1);
    cands.iter().filter(move |c| c.order == 0 || (c.order < 10 && c.visits * 50 >= total))
}

/// Attacks (source, target) and reinforcement counts along a PV, for the side to move's current turn.
#[allow(clippy::type_complexity)]
fn pv_marks(board: &Board, pv: &[Move]) -> (Vec<((usize, usize), (usize, usize))>, HashMap<(usize, usize), i32>) {
    let mut b = board.clone();
    let pla = b.next;
    let mut arrows = Vec::new();
    let mut plus = HashMap::new();
    for &mv in pv {
        if b.next != pla || b.winner().is_some() {
            break;
        }
        match (b.stage, mv) {
            (Stage::AttackTarget(sx, sy), Move::Cell(x, y)) => arrows.push(((sx, sy), (x, y))),
            (Stage::Reinforce, Move::Cell(x, y)) => *plus.entry((x, y)).or_insert(0) += 1,
            _ => {}
        }
        if !b.play(mv) {
            break;
        }
    }
    (arrows, plus)
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Influence".to_owned(),
        window_width: 1200,
        window_height: 800,
        high_dpi: true,
        ..Default::default()
    }
}

#[derive(Debug, Default, PartialEq)]
struct Args {
    engine_cmds: Vec<String>,
    position: Option<String>,
    replay: Option<String>,
    game: usize,
}

const USAGE: &str = "usage: influence-gui [position.txt] [--engine \"<engine command line>\"]... [--replay <file.sgf|file.sgfs>] [--game <n>]";

/// Parses the command line (without the program name), see `USAGE`. `--engine` takes the whole engine
/// command line as one (quoted) argument and can be given several times to load several engines.
fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut args = args.into_iter();
    let mut out = Args::default();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--engine" => {
                let cmd = args.next().ok_or("--engine needs the engine command line as one quoted argument")?;
                if !cmd.trim().is_empty() {
                    out.engine_cmds.push(cmd);
                }
            }
            "--replay" => out.replay = Some(args.next().ok_or("--replay needs a file")?),
            "--game" => out.game = args.next().and_then(|n| n.parse().ok()).ok_or("--game needs a number (0-based game index)")?,
            _ if a.starts_with('-') => {
                return Err(format!("unknown option '{a}' (quote the whole engine command: --engine \"katago gtp ...\")"));
            }
            _ if out.position.is_some() => return Err(format!("unexpected argument '{a}'")),
            _ => out.position = Some(a),
        }
    }
    Ok(out)
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut app = App::new();
    let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
    macroquad::rand::srand(seed);
    let args = parse_args(std::env::args().skip(1)).unwrap_or_else(|e| {
        eprintln!("{e}\n{USAGE}");
        std::process::exit(2);
    });
    let (path, replay_path, game) = (args.position, args.replay, args.game);
    let mut cmds = args.engine_cmds;
    if cmds.is_empty() {
        cmds.extend(std::env::var("INFLUENCE_ENGINE").ok().filter(|s| !s.trim().is_empty()));
    }
    for cmd in cmds {
        app.add_engine(cmd);
    }
    if let Some(path) = path {
        match std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|t| Board::parse(&t)) {
            Ok(b) => app.board = b,
            Err(e) => app.message = format!("Could not load {path}: {e}"),
        }
    }
    if let Some(path) = replay_path {
        match Replay::load(&path, game) {
            Ok(r) => app.load_game(r),
            Err(e) => {
                eprintln!("Could not replay {path} game {game}: {e}");
                app.message = format!("Could not replay {path}: {e}");
            }
        }
    }

    // The engine is driven from its own thread: the render loop stops while the window is hidden
    // (next_frame blocks until the compositor wants a frame), engine games and analysis should not.
    let app = Arc::new(Mutex::new(app));
    let weak = Arc::downgrade(&app);
    std::thread::spawn(move || {
        while let Some(app) = weak.upgrade() {
            app.lock().unwrap().engine_tick();
            drop(app);
            std::thread::sleep(Duration::from_millis(5));
        }
    });

    loop {
        // not held across next_frame, which may block for as long as the window is hidden
        let mut app = app.lock().unwrap();
        clear_background(BG);
        let layout = Layout::new(&app.board);
        let m = mouse_position();
        let hover = if m.0 < screen_width() - PANEL_W { layout.pick(&app.board, vec2(m.0, m.1)) } else { None };

        app.handle_keys();
        if let Some(cell) = hover {
            app.handle_board_click(cell);
        }
        // the board may have been resized
        let layout = Layout::new(&app.board);
        let hover = hover.filter(|&(x, y)| x < app.board.w && y < app.board.h);
        app.draw_board(&layout, hover);
        app.draw_notes();
        app.draw_panel(hover);
        drop(app);

        next_frame().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_line() {
        let args = |argv: &[&str]| parse_args(argv.iter().map(|s| s.to_string()));
        let engine = "../cpp/build_opencl/katago gtp -model m.bin.gz -config gtp.cfg";
        let a = args(&["--engine", engine, "--replay", "g.sgfs", "--game", "2"]).unwrap();
        assert_eq!(a, Args { engine_cmds: vec![engine.into()], position: None, replay: Some("g.sgfs".into()), game: 2 });
        // one engine per --engine
        let a = args(&["--engine", engine, "--engine", "katago gtp -model other.bin.gz"]).unwrap();
        assert_eq!(a.engine_cmds.len(), 2);
        assert_eq!(engine_name(engine), "m");
        assert_eq!(engine_name("katago gtp -model ../models/b10c128-s1.bin.gz"), "b10c128-s1");
        assert_eq!(engine_name("../katago gtp"), "katago");
        // the position file can come anywhere
        let a = args(&["--engine", engine, "pos.txt", "--game", "1"]).unwrap();
        assert_eq!((a.position.as_deref(), a.game), (Some("pos.txt"), 1));
        // unquoted engine command
        let e = args(&["--engine", "../katago", "gtp", "-model", "m.bin.gz"]).unwrap_err();
        assert!(e.contains("'-model'"), "{e}");
        assert!(args(&["--game", "x"]).is_err());
        assert!(args(&["--engine"]).is_err());
        assert_eq!(args(&[]).unwrap(), Args::default());
    }

    #[test]
    fn loaded_game_is_history() {
        let sgf = "(;SZ[4:3]PB[p]PW[p]RE[W+]IB[+5 . . ./. . . ./. . . -1/state W 0 null 0 0/]\
                   ;W[aa];W[ba]C[x];W[];W[aa];W[ba];B[]C[last])";
        let mut app = App::new();
        app.load_game(Replay::from_text(sgf, 0).unwrap());
        assert_eq!((app.undo.len(), app.redo.len(), app.board.turn), (6, 0, 2));
        assert_eq!(app.notes.get(&app.board.to_text('/')).map(|s| s.as_str()), Some("Red end attack  last"));
        app.undo_turn();
        assert_eq!((app.undo.len(), app.board.turn), (5, 1));
        app.undo();
        // the last step reinforced (0,0)
        assert_eq!(app.last_move_cell(), Some((0, 0)));
        app.undo_all();
        assert_eq!((app.undo.len(), app.redo.len()), (0, 6));
        app.redo();
        app.redo();
        // the attack changed the source and the target, the dot marks the target
        assert_eq!(app.last_move_cell(), Some((1, 0)));
        assert_eq!(app.notes.get(&app.board.to_text('/')).map(|s| s.as_str()), Some("Blue attack B3  x"));
        app.redo_all();
        assert_eq!(app.undo.len(), 6);
    }
}
