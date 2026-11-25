// Ported from engine/server/src/parse/inkling_parser.cpp (grammar, `StreamingParserEngine`, `ParserEngine::on_terminal`
// and argument streaming) @ 6100ab8, itself a port of vLLM's vllm/parser/inkling.py (`inkling_config`) and
// vllm/parser/engine/{parser_engine,streaming_parser_engine}.py (Apache-2.0, Copyright contributors to the vLLM
// project).
//! The Inkling grammar as a state machine over lexer and scanner output, producing events.

use std::sync::Arc;

use crate::lexer::{LexToken, Lexer, TERMINALS, Term};
use crate::scanner::{Item, Scanner};
use crate::tables::ParserTables;

/// Grammar states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum State {
    Content,
    Reasoning,
    MessageHeader,
    ToolArgs,
}

/// What a transition emits (the data comes from the terminal and the machine).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Emit {
    Text,
    ReasoningStart,
    ReasoningEnd,
    ToolCallStart,
    ToolCallEnd,
}

/// Parser events.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Event {
    Text(String),
    ReasoningStart,
    ReasoningChunk(String),
    ReasoningEnd,
    ToolCallStart(usize),
    ArgChunk(usize, String),
    ToolCallEnd(usize),
}

struct Transition {
    next: State,
    emit: &'static [Emit],
}

const fn to(next: State, emit: &'static [Emit]) -> Option<Transition> {
    Some(Transition { next, emit })
}

fn transition(state: State, term: Term) -> Option<Transition> {
    use Emit::{ReasoningEnd, ReasoningStart, Text, ToolCallEnd, ToolCallStart};
    use State::{Content, MessageHeader, Reasoning, ToolArgs};
    match (state, term) {
        (Content, Term::MsgModel) => to(MessageHeader, &[]),
        (Content, Term::TextStart | Term::ToolText | Term::ToolError) => to(Content, &[ReasoningEnd]),
        (Content, Term::ThinkStart) => to(Reasoning, &[ReasoningStart]),
        (Content, Term::ToolStart) => to(ToolArgs, &[ReasoningEnd, ToolCallStart]),
        (Content, Term::ThinkEnd | Term::EndSampling) => to(Content, &[]),
        (MessageHeader, Term::MsgModel) => to(MessageHeader, &[]),
        (MessageHeader, Term::TextStart | Term::ToolText | Term::ToolError) => to(Content, &[ReasoningEnd]),
        (MessageHeader, Term::ThinkStart) => to(Reasoning, &[ReasoningStart]),
        (MessageHeader, Term::ToolStart) => to(ToolArgs, &[ReasoningEnd, ToolCallStart]),
        (MessageHeader, Term::ThinkEnd | Term::EndSampling) => to(Content, &[Text]),
        (Reasoning, Term::ThinkStart) => to(Reasoning, &[]),
        (Reasoning, Term::ToolStart) => to(ToolArgs, &[ReasoningEnd, ToolCallStart]),
        (Reasoning, Term::ThinkEnd) => to(Content, &[]),
        (Reasoning, Term::EndSampling) => to(Content, &[ReasoningEnd]),
        (ToolArgs, Term::ThinkEnd | Term::EndSampling) => to(Content, &[ToolCallEnd]),
        _ => None,
    }
}

fn is_tool_terminal(term: Term) -> bool {
    matches!(term, Term::ToolStart | Term::ThinkEnd | Term::EndSampling)
}

fn is_tool_exit_terminal(term: Term) -> bool {
    matches!(term, Term::ThinkEnd | Term::EndSampling)
}

/// Lexer, scanner and grammar state of one parser engine.
#[derive(Debug)]
pub(crate) struct Machine {
    has_drops: bool,
    /// Tool terminals are plain text (the reasoning engine) instead of opening and closing calls.
    skip_tool_parsing: bool,
    lexer: Lexer,
    scanner: Scanner,
    state: State,
    /// Index of the tool call being read; `None` before the first.
    tool_index: Option<usize>,
    ever_had_ids: bool,
    header: String,
    in_skipped_tool_span: bool,
    args_buffer: String,
    /// Bytes of `args_buffer` that may be emitted (a closing brace of the outermost object is held back).
    args_safe_end: usize,
    args_depth: u32,
    args_in_string: bool,
    args_escape: bool,
}

impl Machine {
    pub fn new(tables: Arc<ParserTables>, skip_tool_parsing: bool) -> Self {
        let literals = TERMINALS
            .iter()
            .map(|(lit, term)| ((*lit).to_owned(), *term))
            .chain(tables.drop_literals.iter().map(|d| (d.clone(), Term::Drop)))
            .collect();
        let mut machine = Self {
            has_drops: !tables.drop_ids.is_empty(),
            skip_tool_parsing,
            lexer: Lexer::new(literals),
            scanner: Scanner::new(tables),
            state: State::MessageHeader,
            tool_index: None,
            ever_had_ids: false,
            header: String::new(),
            in_skipped_tool_span: false,
            args_buffer: String::new(),
            args_safe_end: 0,
            args_depth: 0,
            args_in_string: false,
            args_escape: false,
        };
        machine.reset(State::MessageHeader);
        machine
    }

    /// Clears all stream state and starts in `initial`.
    pub fn reset(&mut self, initial: State) {
        self.state = initial;
        self.tool_index = None;
        self.ever_had_ids = false;
        self.scanner.reset();
        self.lexer.clear();
        self.header.clear();
        self.in_skipped_tool_span = false;
        self.args_buffer.clear();
        self.args_safe_end = 0;
        self.args_depth = 0;
        self.args_in_string = false;
        self.args_escape = false;
    }

    /// Feeds one step: its text and the token ids that produced it.
    pub fn feed(&mut self, text: &str, ids: &[u32]) -> Vec<Event> {
        if !ids.is_empty() {
            self.ever_had_ids = true;
        }
        if !text.is_empty() && self.lexer.is_idle() && !self.scanner.has_deferred() && !text.contains('<') {
            let has_special = ids.iter().any(|id| self.scanner.is_drop(*id));
            if !has_special {
                return self.emit_for_state(text);
            }
        }
        let items = self.scanner.scan(text, ids);
        if items.len() == 1 && !items[0].terminal {
            let mut lex = self.lexer.feed(&items[0].text);
            if lex.len() == 1 && lex[0].term == Term::Content {
                return self.emit_for_state(&lex.remove(0).value);
            }
            return self.process_lex(lex);
        }
        self.process_items(items)
    }

    /// End of stream: flushes everything held back and closes the open construct.
    pub fn finish(&mut self) -> Vec<Event> {
        let pending = self.scanner.flush_pending();
        let mut events = self.process_items(pending);
        let flushed = self.lexer.flush();
        events.extend(self.process_lex(flushed));
        if !self.args_buffer.is_empty()
            && let Some(idx) = self.tool_index
        {
            events.push(Event::ArgChunk(idx, std::mem::take(&mut self.args_buffer)));
            self.args_safe_end = 0;
        }
        match self.state {
            State::ToolArgs => {
                if let Some(idx) = self.tool_index {
                    events.push(Event::ToolCallEnd(idx));
                }
                self.state = State::Content;
            }
            State::Reasoning => {
                events.push(Event::ReasoningEnd);
                self.state = State::Content;
            }
            State::MessageHeader => {
                if !self.header.is_empty() {
                    events.push(Event::Text(std::mem::take(&mut self.header)));
                }
                self.state = State::Content;
            }
            State::Content => {}
        }
        events
    }

    fn process_items(&mut self, items: Vec<Item>) -> Vec<Event> {
        let mut events = Vec::new();
        for item in items {
            if item.terminal {
                let flushed = self.lexer.flush();
                events.extend(self.process_lex(flushed));
                events.extend(self.on_terminal(Term::Drop, &item.text));
            } else if item.text.is_empty() && item.token_count > 0 {
                events.extend(self.emit_for_state(""));
            } else {
                let lexed = self.lexer.feed(&item.text);
                events.extend(self.process_lex(lexed));
            }
        }
        events
    }

    fn process_lex(&mut self, tokens: Vec<LexToken>) -> Vec<Event> {
        let mut events = Vec::new();
        let strict = self.ever_had_ids && self.has_drops;
        for tok in tokens {
            if tok.term == Term::Content || (strict && tok.term == Term::Drop) {
                if !tok.value.is_empty() {
                    events.extend(self.emit_for_state(&tok.value));
                }
            } else {
                events.extend(self.on_terminal(tok.term, &tok.value));
            }
        }
        events
    }

    fn on_terminal(&mut self, term: Term, value: &str) -> Vec<Event> {
        let Some(tr) = transition(self.state, term) else {
            if self.has_drops && term == Term::Drop {
                return Vec::new();
            }
            if self.skip_tool_parsing && is_tool_exit_terminal(term) {
                self.in_skipped_tool_span = false;
            }
            return self.emit_for_state(value);
        };
        if self.skip_tool_parsing && is_tool_terminal(term) {
            let is_opener = tr.next == State::ToolArgs;
            let is_exit = is_tool_exit_terminal(term);
            let plain_closer = is_exit && !is_opener && !self.in_skipped_tool_span;
            if !plain_closer {
                if is_opener {
                    self.in_skipped_tool_span = true;
                } else if is_exit {
                    self.in_skipped_tool_span = false;
                }
                let leaving_header = self.state == State::MessageHeader;
                if leaving_header {
                    self.header.clear();
                }
                let text = || Event::Text(value.to_owned());
                if tr.emit.contains(&Emit::ReasoningEnd) {
                    self.state = State::Content;
                    return vec![Event::ReasoningEnd, text()];
                }
                if leaving_header {
                    self.state = State::Content;
                    return vec![text()];
                }
                return match self.state {
                    State::Content => vec![text()],
                    State::Reasoning => vec![Event::ReasoningChunk(value.to_owned())],
                    State::ToolArgs => self
                        .tool_index
                        .map_or_else(Vec::new, |i| vec![Event::ArgChunk(i, value.to_owned())]),
                    State::MessageHeader => Vec::new(),
                };
            }
        }
        self.apply_transition(&tr)
    }

    fn emit_for_state(&mut self, text: &str) -> Vec<Event> {
        match self.state {
            State::MessageHeader => {
                self.header.push_str(text);
                Vec::new()
            }
            State::ToolArgs => self.feed_args_text(text),
            State::Content => vec![Event::Text(text.to_owned())],
            State::Reasoning => vec![Event::ReasoningChunk(text.to_owned())],
        }
    }

    fn apply_transition(&mut self, tr: &Transition) -> Vec<Event> {
        let mut events = Vec::new();
        let prev = self.state;
        if prev == State::ToolArgs
            && tr.next != State::ToolArgs
            && !self.args_buffer.is_empty()
            && let Some(idx) = self.tool_index
        {
            events.push(Event::ArgChunk(idx, std::mem::take(&mut self.args_buffer)));
        }
        let header = if prev == State::MessageHeader {
            std::mem::take(&mut self.header)
        } else {
            String::new()
        };
        self.state = tr.next;
        for emit in tr.emit {
            match emit {
                Emit::ToolCallStart => {
                    let idx = self.tool_index.map_or(0, |i| i + 1);
                    self.tool_index = Some(idx);
                    events.push(Event::ToolCallStart(idx));
                }
                Emit::ToolCallEnd => {
                    if let Some(idx) = self.tool_index {
                        events.push(Event::ToolCallEnd(idx));
                    }
                }
                Emit::Text => {
                    // Only `MessageHeader` transitions emit text: the header text collected so far.
                    if !header.is_empty() {
                        events.push(Event::Text(header.clone()));
                    }
                }
                Emit::ReasoningStart => events.push(Event::ReasoningStart),
                Emit::ReasoningEnd => events.push(Event::ReasoningEnd),
            }
        }
        if self.state == State::ToolArgs {
            self.args_depth = 0;
            self.args_in_string = false;
            self.args_escape = false;
            self.args_safe_end = 0;
        }
        events
    }

    fn feed_args_text(&mut self, text: &str) -> Vec<Event> {
        text.chars().flat_map(|c| self.feed_args_char(c)).collect()
    }

    /// Argument text is released as soon as it cannot be the closing brace of the outermost object.
    fn feed_args_char(&mut self, ch: char) -> Option<Event> {
        self.args_buffer.push(ch);
        if self.args_escape {
            self.args_escape = false;
        } else if self.args_in_string {
            match ch {
                '\\' => self.args_escape = true,
                '"' => self.args_in_string = false,
                _ => {}
            }
        } else {
            match ch {
                '"' => self.args_in_string = true,
                '{' | '[' => self.args_depth += 1,
                '}' | ']' => {
                    self.args_depth = self.args_depth.saturating_sub(1);
                    if self.args_depth == 0 {
                        return None;
                    }
                }
                _ => {}
            }
        }
        self.args_safe_end = self.args_buffer.len();
        self.flush_safe_args()
    }

    fn flush_safe_args(&mut self) -> Option<Event> {
        if self.args_safe_end == 0 {
            return None;
        }
        let idx = self.tool_index?;
        let tail = self.args_buffer.split_off(self.args_safe_end);
        let out = std::mem::replace(&mut self.args_buffer, tail);
        self.args_safe_end = 0;
        Some(Event::ArgChunk(idx, out))
    }
}
