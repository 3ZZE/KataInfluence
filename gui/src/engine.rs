//! KataGo (Influence fork) GTP engine running as a child process.

use crate::game::{Board, Move, Player};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};

pub enum Line {
    Stdout(String),
    Exited,
}

pub struct Engine {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<Line>,
    next_id: u32,
    /// Last line the engine wrote to stderr (startup progress, errors).
    pub last_stderr: Arc<Mutex<String>>,
    pub alive: bool,
}

/// A GTP response: id, success, text after the id.
pub struct Response {
    pub id: u32,
    pub ok: bool,
    pub text: String,
}

pub enum Event {
    Response(Response),
    /// One `kata-analyze` report (the candidates of a whole line).
    Analysis(String),
    Exited,
}

impl Engine {
    /// `cmd` is the whole command line, split by `split_command`.
    pub fn spawn(cmd: &str) -> Result<Engine, String> {
        let words = split_command(cmd)?;
        let (prog, args) = words.split_first().ok_or("empty engine command")?;
        let mut child = Command::new(prog)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("could not start engine '{prog}': {e}"))?;
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();

        let (tx, rx) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if tx.send(Line::Stdout(line)).is_err() {
                    return;
                }
            }
            let _ = tx.send(Line::Exited);
        });
        let last_stderr = Arc::new(Mutex::new(String::new()));
        let last = last_stderr.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines() {
                let Ok(line) = line else { break };
                eprintln!("[engine] {line}");
                if !line.trim().is_empty() {
                    *last.lock().unwrap() = line;
                }
            }
        });
        Ok(Engine { child, stdin, rx, next_id: 1, last_stderr, alive: true })
    }

    /// Sends a command, returns its GTP id.
    pub fn send(&mut self, cmd: &str) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        if writeln!(self.stdin, "{id} {cmd}").and_then(|_| self.stdin.flush()).is_err() {
            self.alive = false;
        }
        id
    }

    /// Non-blocking: everything the engine has written since the last call.
    pub fn poll(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        while let Ok(line) = self.rx.try_recv() {
            let line = match line {
                Line::Stdout(l) => l,
                Line::Exited => {
                    self.alive = false;
                    events.push(Event::Exited);
                    continue;
                }
            };
            let line = line.trim();
            if line.starts_with("info ") {
                events.push(Event::Analysis(line.to_string()));
            } else if let Some(rest) = line.strip_prefix('=').or_else(|| line.strip_prefix('?')) {
                let ok = line.starts_with('=');
                let digits = rest.chars().take_while(|c| c.is_ascii_digit()).count();
                if let Ok(id) = rest[..digits].parse() {
                    events.push(Event::Response(Response { id, ok, text: rest[digits..].trim().to_string() }));
                }
            } else if !line.is_empty() {
                // not GTP, e.g. a usage message when the command line is wrong
                eprintln!("[engine stdout] {line}");
                *self.last_stderr.lock().unwrap() = line.to_string();
            }
        }
        events
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.child.kill();
    }
}

/// Splits a command line into words like a shell, without any expansion: whitespace separates words,
/// '...' and "..." quote, and a backslash escapes the next character outside single quotes.
pub fn split_command(cmd: &str) -> Result<Vec<String>, String> {
    let mut words = Vec::new();
    let mut cur = String::new();
    let mut in_word = false;
    let mut quote: Option<char> = None;
    let mut chars = cmd.chars();
    while let Some(c) = chars.next() {
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (Some('"') | None, '\\') => {
                cur.extend(chars.next());
                in_word = true;
            }
            (Some(_), c) => cur.push(c),
            (None, '\'' | '"') => {
                quote = Some(c);
                in_word = true;
            }
            (None, c) if c.is_whitespace() => {
                if in_word {
                    words.push(std::mem::take(&mut cur));
                    in_word = false;
                }
            }
            (None, c) => {
                cur.push(c);
                in_word = true;
            }
        }
    }
    if quote.is_some() {
        return Err(format!("unterminated quote in engine command: {cmd}"));
    }
    if in_word {
        words.push(cur);
    }
    Ok(words)
}

/// One candidate move of a `kata-analyze` report. Winrate is for the side to move.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub mv: Move,
    pub visits: u64,
    pub winrate: f64,
    pub prior: f64,
    pub order: u32,
    pub pv: Vec<Move>,
}

/// Parses `info move .. visits .. winrate .. pv .. info move ..`.
pub fn parse_analysis(line: &str, w: usize, h: usize) -> Vec<Candidate> {
    let mut out = Vec::new();
    for chunk in line.split("info ").filter(|c| !c.trim().is_empty()) {
        let toks: Vec<&str> = chunk.split_whitespace().collect();
        let mut c = Candidate { mv: Move::Pass, visits: 0, winrate: 0.5, prior: 0.0, order: 0, pv: Vec::new() };
        let mut has_move = false;
        let mut i = 0;
        while i + 1 < toks.len() {
            let (k, v) = (toks[i], toks[i + 1]);
            match k {
                "move" => match Move::parse(v, w, h) {
                    Some(m) => {
                        c.mv = m;
                        has_move = true;
                    }
                    None => break,
                },
                "visits" => c.visits = v.parse().unwrap_or(0),
                "winrate" => c.winrate = v.parse().unwrap_or(0.5),
                "prior" => c.prior = v.parse().unwrap_or(0.0),
                "order" => c.order = v.parse().unwrap_or(0),
                "pv" => {
                    let mut j = i + 1;
                    while j < toks.len() {
                        match Move::parse(toks[j], w, h) {
                            Some(m) => c.pv.push(m),
                            None => break,
                        }
                        j += 1;
                    }
                    i = j;
                    continue;
                }
                _ => {}
            }
            i += 2;
        }
        if has_move {
            out.push(c);
        }
    }
    out.sort_by_key(|c| c.order);
    out
}


#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Want {
    Idle,
    Analyze,
    Genmove,
}

/// Keeps the engine in sync with the GUI board and runs analysis / move generation.
/// The engine is told the position with `set_board` whenever it may differ from the GUI.
pub struct EngineCtl {
    engine: Engine,
    ready_id: u32,
    pub ready: bool,
    /// Position text the engine is believed to be at.
    synced: Option<String>,
    /// Pending set_board commands.
    sync_ids: Vec<(u32, String)>,
    /// Position the engine rejected; nothing is asked about it.
    bad_pos: Option<String>,
    analyze: Option<(u32, Board)>,
    analyze_live: bool,
    /// Latest analysis and the position it belongs to.
    pub analysis: Option<(Board, Vec<Candidate>)>,
    genmove: Option<(u32, Board)>,
    pub error: Option<String>,
}

impl EngineCtl {
    pub fn new(cmd: &str) -> Result<EngineCtl, String> {
        let mut engine = Engine::spawn(cmd)?;
        let ready_id = engine.send("name");
        Ok(EngineCtl {
            engine,
            ready_id,
            ready: false,
            synced: None,
            sync_ids: Vec::new(),
            bad_pos: None,
            analyze: None,
            analyze_live: false,
            analysis: None,
            genmove: None,
            error: None,
        })
    }

    pub fn alive(&self) -> bool {
        self.engine.alive
    }

    pub fn thinking(&self) -> bool {
        self.genmove.is_some()
    }

    pub fn analyzing(&self) -> bool {
        self.analyze.is_some()
    }

    pub fn last_stderr(&self) -> String {
        self.engine.last_stderr.lock().unwrap().clone()
    }

    /// The engine has played a move itself (genmove), so it is now at `board`.
    pub fn mark_synced(&mut self, board: &Board) {
        self.synced = Some(board.to_text('/'));
    }

    fn sync(&mut self, board: &Board) {
        let text = board.to_text('/');
        if self.synced.as_deref() != Some(text.as_str()) {
            let id = self.engine.send(&format!("set_board {text}"));
            self.sync_ids.push((id, text.clone()));
            self.synced = Some(text);
        }
    }

    /// Call regularly (every few ms). Returns a generated move as (position it was generated for, GTP vertex).
    pub fn tick(&mut self, board: &Board, want: Want) -> Option<(Board, String)> {
        let mut result = None;
        for ev in self.engine.poll() {
            match ev {
                Event::Response(r) => {
                    if r.id == self.ready_id {
                        self.ready = true;
                    } else if let Some(i) = self.sync_ids.iter().position(|(id, _)| *id == r.id) {
                        let (_, text) = self.sync_ids.remove(i);
                        if !r.ok {
                            self.error = Some(format!("set_board: {}", r.text));
                            self.bad_pos = Some(text);
                            self.synced = None;
                        }
                    } else if self.genmove.as_ref().is_some_and(|(id, _)| *id == r.id) {
                        let (_, b) = self.genmove.take().unwrap();
                        if !r.ok {
                            self.error = Some(format!("genmove: {}", r.text));
                            self.synced = None;
                        } else if self.bad_pos.as_deref() != Some(b.to_text('/').as_str()) {
                            result = Some((b, r.text));
                        }
                    } else if self.analyze.as_ref().is_some_and(|(id, _)| *id == r.id) {
                        if r.ok {
                            self.analyze_live = true;
                        } else {
                            self.error = Some(format!("kata-analyze: {}", r.text));
                        }
                    }
                }
                Event::Analysis(line) => {
                    if let (true, Some((_, b))) = (self.analyze_live, &self.analyze) {
                        if self.bad_pos.as_deref() != Some(b.to_text('/').as_str()) {
                            let cands = parse_analysis(&line, b.w, b.h);
                            self.analysis = Some((b.clone(), cands));
                        }
                    }
                }
                Event::Exited => {
                    self.error = Some(format!("engine exited: {}", self.last_stderr()));
                }
            }
        }
        // the caller applies a generated move first, the next command is sent on the next tick
        if !self.engine.alive || self.genmove.is_some() || result.is_some() {
            return result;
        }

        let bad = self.bad_pos.as_deref() == Some(board.to_text('/').as_str());
        match want {
            Want::Genmove if !bad => {
                // any command stops a running analysis
                self.analyze = None;
                self.analyze_live = false;
                self.sync(board);
                let id = self.engine.send(&format!("genmove {}", pla_str(board.next)));
                self.genmove = Some((id, board.clone()));
            }
            Want::Analyze if !bad => {
                if self.analyze.as_ref().map(|(_, b)| b) != Some(board) {
                    self.sync(board);
                    let id = self.engine.send(&format!("kata-analyze {} 20", pla_str(board.next)));
                    self.analyze = Some((id, board.clone()));
                    self.analyze_live = false;
                }
            }
            _ => {
                if self.analyze.is_some() {
                    self.engine.send("protocol_version");
                    self.analyze = None;
                    self.analyze_live = false;
                }
            }
        }
        result
    }
}

fn pla_str(p: Player) -> &'static str {
    match p {
        Player::White => "W",
        Player::Black => "B",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split() {
        let w = split_command("  ../katago gtp -config 'my gtp.cfg' -model \"a b\\\"c\" x\\ y '' ").unwrap();
        assert_eq!(w, ["../katago", "gtp", "-config", "my gtp.cfg", "-model", "a b\"c", "x y", ""]);
        assert!(split_command("katago 'gtp").is_err());
        assert!(split_command("   ").unwrap().is_empty());
    }

    #[test]
    fn parse_info() {
        let line = "info move B2 visits 10 utility 0.1 winrate 0.6 scoreMean 0 scoreStdev 0 scoreLead 0 prior 0.3 lcb 0.5 \
                    utilityLcb 0 weight 10 order 0 pv B2 C2 pass info move pass visits 3 utility 0 winrate 0.4 prior 0.1 order 1 pv pass B2";
        let c = parse_analysis(line, 11, 11);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].mv, Move::Cell(1, 9));
        assert_eq!(c[0].visits, 10);
        assert_eq!(c[0].pv, vec![Move::Cell(1, 9), Move::Cell(2, 9), Move::Pass]);
        assert_eq!(c[1].mv, Move::Pass);
        assert!((c[1].winrate - 0.4).abs() < 1e-9);
    }
}
