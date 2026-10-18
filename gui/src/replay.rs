//! Replays the main line of an SGF game as written by the engine (cpp/dataio/sgf.cpp, WriteSgf::writeSgf):
//! the root holds SZ, PB/PW, RE, PL and the initial position IB[<set_board text>], every following node
//! one micro-move W[xy] / B[xy] (x column, y row from the top, letters a..) or W[] / B[] for a pass.
//! Selfplay `.sgfs` files hold one game per line. The GUI puts the replayed positions into its undo history.

use crate::game::{Board, Move, Player, Stage};

/// One SGF node: (property, values) in file order.
type Node = Vec<(String, Vec<String>)>;

const SGF_LETTERS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

pub struct Replay {
    pub game: usize,
    pub white: String,
    pub black: String,
    pub result: String,
    /// Main line micro-moves with the comment of their node.
    pub moves: Vec<(Player, Move, String)>,
    /// positions[i] is the board after the first i moves.
    pub positions: Vec<Board>,
}

impl Replay {
    pub fn load(path: &str, game: usize) -> Result<Replay, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        Replay::from_text(&text, game)
    }

    pub fn from_text(text: &str, game: usize) -> Result<Replay, String> {
        let starts = game_starts(text);
        let start = *starts.get(game).ok_or(format!("game {game} not found, the file has {} game(s)", starts.len()))?;
        let nodes = parse_main_line(&text[start..])?;
        let root = nodes.first().ok_or("empty game")?;
        let text_prop = |key: &str| prop(root, key).unwrap_or("").to_string();

        let (w, h) = match prop(root, "SZ") {
            None => (crate::game::DEFAULT_LEN, crate::game::DEFAULT_LEN),
            Some(sz) => {
                let num = |s: &str| s.trim().parse::<usize>().map_err(|_| format!("bad SZ[{sz}]"));
                match sz.split_once(':') {
                    Some((a, b)) => (num(a)?, num(b)?),
                    None => (num(sz)?, num(sz)?),
                }
            }
        };
        let mut board = match prop(root, "IB") {
            Some(ib) => Board::parse(ib).map_err(|e| format!("bad IB: {e}"))?,
            None => Board::default_start(w, h),
        };
        if (board.w, board.h) != (w, h) {
            return Err(format!("IB is {}x{} but SZ is {w}x{h}", board.w, board.h));
        }
        // the IB state row sets the side to move; PL only matters without it
        if !prop(root, "IB").is_some_and(|ib| ib.contains("state")) {
            match prop(root, "PL") {
                Some("W") => board.set_next(Player::White),
                Some("B") => board.set_next(Player::Black),
                _ => {}
            }
        }

        let mut moves = Vec::new();
        let mut positions = vec![board.clone()];
        for node in &nodes[1..] {
            let (pla, v) = match (prop(node, "W"), prop(node, "B")) {
                (Some(v), _) => (Player::White, v),
                (None, Some(v)) => (Player::Black, v),
                _ => continue,
            };
            let n = moves.len() + 1;
            let mv = sgf_move(v, w, h).ok_or(format!("move {n}: bad coordinate [{v}]"))?;
            if pla != board.next {
                return Err(format!("move {n}: played by {} but {} is to move", pla.name(), board.next.name()));
            }
            if !board.play(mv) {
                return Err(format!("move {n}: {} {} is illegal", pla.name(), mv.to_gtp(h)));
            }
            moves.push((pla, mv, prop(node, "C").unwrap_or("").to_string()));
            positions.push(board.clone());
        }
        Ok(Replay {
            game,
            white: text_prop("PW"),
            black: text_prop("PB"),
            result: text_prop("RE"),
            moves,
            positions,
        })
    }

    pub fn len(&self) -> usize {
        self.moves.len()
    }

    /// "W attack C1" for move `i` (0-based), describing it by the stage it was played in.
    pub fn describe(&self, i: usize) -> String {
        let (pla, mv, _) = self.moves[i];
        let b = &self.positions[i];
        let what = match (b.stage, mv) {
            (Stage::AttackSelect, Move::Pass) => return format!("{} end attack", pla.name()),
            (Stage::AttackSelect, _) => "select",
            (Stage::AttackTarget(..), _) => "attack",
            (Stage::Reinforce, _) => "+1",
        };
        format!("{} {what} {}", pla.name(), mv.to_gtp(b.h))
    }
}

fn prop<'a>(node: &'a Node, key: &str) -> Option<&'a str> {
    node.iter().find(|(k, _)| k == key).and_then(|(_, v)| v.first()).map(|s| s.as_str())
}

/// SGF coordinate: column letter, row letter (row a at the top); empty is a pass.
fn sgf_move(v: &str, w: usize, h: usize) -> Option<Move> {
    if v.is_empty() {
        return Some(Move::Pass);
    }
    let b = v.as_bytes();
    if b.len() != 2 {
        return None;
    }
    let x = SGF_LETTERS.iter().position(|&c| c == b[0])?;
    let y = SGF_LETTERS.iter().position(|&c| c == b[1])?;
    (x < w && y < h).then_some(Move::Cell(x, y))
}

/// Byte offsets of the top-level games, i.e. of every '(' at depth 0.
fn game_starts(text: &str) -> Vec<usize> {
    let mut starts = Vec::new();
    let (mut depth, mut in_value, mut escaped) = (0, false, false);
    for (i, c) in text.char_indices() {
        if in_value {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == ']' {
                in_value = false;
            }
            continue;
        }
        match c {
            '[' => in_value = true,
            '(' => {
                if depth == 0 {
                    starts.push(i);
                }
                depth += 1;
            }
            ')' => depth = (depth - 1).max(0),
            _ => {}
        }
    }
    starts
}

/// Nodes of the main line of the game starting at `text`: follows the first variation at each branch
/// and stops at the first ')', ignoring later variations.
fn parse_main_line(text: &str) -> Result<Vec<Node>, String> {
    let mut nodes: Vec<Node> = Vec::new();
    let mut key = String::new();
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        match c {
            '(' => {}
            ')' => break,
            ';' => nodes.push(Vec::new()),
            '[' => {
                let mut v = String::new();
                loop {
                    match chars.next() {
                        None => return Err("unterminated property value".into()),
                        Some(']') => break,
                        Some('\\') => match chars.next() {
                            Some('\n') | None => {}
                            Some(e) => v.push(e),
                        },
                        Some(ch) => v.push(ch),
                    }
                }
                let node = nodes.last_mut().ok_or("property outside a node")?;
                if key.is_empty() {
                    // another value of the previous property
                    if let Some(last) = node.last_mut() {
                        last.1.push(v);
                    }
                } else {
                    node.push((std::mem::take(&mut key), vec![v]));
                }
            }
            c if c.is_ascii_uppercase() => key.push(c),
            _ => {}
        }
    }
    Ok(nodes)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GAME: &str = "(;FF[4]GM[1]SZ[4:3]PB[b]PW[w]RE[W+]PL[W]IB[+5 . . ./. . . ./. . . -1/state W 0 null 0 0/]C[root]\
        ;W[aa];W[ba]C[c\\]1];W[];W[aa];W[ba];B[]C[0.10 0.90 0.00 v=20 weight=1.00 result=W+])\n";

    #[test]
    fn parse_and_replay() {
        // second game in the file, with variations that must be ignored
        let varied = GAME.replacen(";B[]", "(;B[]", 1).replacen("W+])", "W+])(;W[aa];W[ca]))", 1);
        let text = format!("(;SZ[3])\n{varied}");
        let r = Replay::from_text(&text, 1).unwrap();
        assert_eq!((r.white.as_str(), r.black.as_str(), r.result.as_str()), ("w", "b", "W+"));
        assert_eq!(r.len(), 6);
        assert_eq!(r.moves[1], (Player::White, Move::Cell(1, 0), "c]1".to_string()));
        assert_eq!(r.describe(0), "Blue select A3");
        assert_eq!(r.describe(1), "Blue attack B3");
        assert_eq!(r.describe(2), "Blue end attack");
        assert_eq!(r.describe(3), "Blue +1 A3");
        // Black's single cell gets its point automatically
        let end = &r.positions[6];
        assert_eq!((end.cell(0, 0).v, end.cell(1, 0).v, end.cell(3, 2).v), (2, 5, -2));
        assert_eq!((end.next, end.turn), (Player::White, 2));
        assert!(Replay::from_text(&text, 2).is_err());
        // not adjacent
        assert!(Replay::from_text(&GAME.replacen("W[ba]", "W[ca]", 1), 0).is_err());
        // wrong side
        assert!(Replay::from_text(&GAME.replacen("W[aa]", "B[aa]", 1), 0).is_err());
    }

    /// Final positions of real selfplay games, checked against the engine's showboard after replaying
    /// the same moves through GTP (`set_board` + `play`). Skipped if the files are not there.
    #[test]
    fn selfplay_draws() {
        let Ok(text) = std::fs::read_to_string("../draws/draws_recent.sgfs") else { return };
        // (White area, value, Black area, value) at the end of each game, all ending at turn 180
        let expected = [(2, 9, 119, 952), (1, 2, 120, 960), (120, 822, 1, 5), (108, 864, 9, 72), (4, 27, 116, 864), (93, 744, 14, 112), (21, 149, 99, 663)];
        for (g, &e) in expected.iter().enumerate() {
            let r = Replay::from_text(&text, g).unwrap();
            let b = &r.positions[r.len()];
            assert_eq!((b.area(Player::White), b.value(Player::White), b.area(Player::Black), b.value(Player::Black)), e, "game {g}");
            assert_eq!((b.turn, b.next, b.stage), (180, Player::White, Stage::AttackSelect), "game {g}");
        }
    }
}
