// Ported from engine/server/src/parse/inkling_parser.cpp (`ParserEngine`: events to deltas, tool slots, extraction)
// @ 6100ab8, itself a port of vLLM's vllm/parser/engine/parser_engine.py and vllm/parser/inkling.py (Apache-2.0,
// Copyright contributors to the vLLM project).
//! Turns parser events into streaming deltas and into finished tool calls.

use std::collections::BTreeSet;
use std::sync::Arc;

use ink_text::pyfmt::py_strip;
use serde_json::Value;

use crate::args::{arg_converter, safe_arg_prefix, search_name};
use crate::coerce::{fix_arg_types, py_rstrip, streamable_string_keys, tool_properties};
use crate::delta::{CallKind, DeltaMessage, FunctionCall, FunctionDelta, TextField, ToolCallDelta};
use crate::machine::Event;

/// Makes the id of a tool call.
pub type IdFactory = Arc<dyn Fn() -> String + Send + Sync>;

#[derive(Debug, Default)]
struct ToolSlot {
    id: String,
    name: String,
    args: String,
    name_sent: bool,
    string_keys: Option<BTreeSet<String>>,
    streamed_json: String,
    json_name: Option<String>,
}

/// Result of [`Assembler::extract_tool_calls_from_content`].
#[derive(Debug, Default)]
pub(crate) struct Extracted {
    pub tools_called: bool,
    pub tool_calls: Vec<FunctionCall>,
    pub content: Option<String>,
}

pub(crate) struct Assembler {
    tools: Vec<Value>,
    /// `tool_choice` is `none` while tools were sent: calls are parsed but never reported.
    suppress: bool,
    id_factory: IdFactory,
    slots: Vec<ToolSlot>,
    deferred_content: String,
    deferred_reasoning: String,
    content_has_nonws: bool,
    /// A `ReasoningEnd` event was seen.
    pub reasoning_ended: bool,
}

impl std::fmt::Debug for Assembler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Assembler")
            .field("slots", &self.slots)
            .field("suppress", &self.suppress)
            .finish_non_exhaustive()
    }
}

impl Assembler {
    pub fn new(tools: Vec<Value>, suppress: bool, id_factory: IdFactory) -> Self {
        Self {
            tools,
            suppress,
            id_factory,
            slots: Vec::new(),
            deferred_content: String::new(),
            deferred_reasoning: String::new(),
            content_has_nonws: false,
            reasoning_ended: false,
        }
    }

    pub fn reset(&mut self) {
        self.reasoning_ended = false;
        self.slots.clear();
        self.deferred_content.clear();
        self.deferred_reasoning.clear();
        self.content_has_nonws = false;
    }

    /// Whether content held back behind tool events is waiting.
    pub fn has_deferred_content(&self) -> bool {
        !self.deferred_content.is_empty()
    }

    /// `ParserEngine._events_to_delta`.
    pub fn events_to_delta(&mut self, events: &[Event], finished: bool) -> Option<DeltaMessage> {
        if events.is_empty() && self.deferred_content.is_empty() {
            return None;
        }
        let mut deltas: Vec<ToolCallDelta> = Vec::new();
        let mut content = String::new();
        let mut reasoning = String::new();
        let carried = self.deferred_content.clone();
        let mut seen_tool = false;
        let suppress = self.suppress;
        for event in events {
            match event {
                Event::Text(v) => {
                    if seen_tool {
                        self.deferred_content.push_str(v);
                    } else {
                        content.push_str(v);
                    }
                }
                Event::ReasoningChunk(v) => reasoning.push_str(v),
                Event::ReasoningEnd => self.reasoning_ended = true,
                Event::ReasoningStart => {}
                Event::ToolCallStart(idx) if !suppress => {
                    seen_tool = true;
                    self.ensure_slot(*idx);
                }
                Event::ArgChunk(idx, v) if !suppress => {
                    seen_tool = true;
                    self.handle_arg_chunk(*idx, v, &mut deltas);
                }
                Event::ToolCallEnd(idx) if !suppress => {
                    seen_tool = true;
                    self.handle_tool_end(*idx, &mut deltas);
                }
                Event::ToolCallStart(_) | Event::ArgChunk(..) | Event::ToolCallEnd(_) => {}
            }
        }
        if deltas.len() > 1 {
            deltas = coalesce(deltas);
        }
        if !self.deferred_content.is_empty() && (!seen_tool || deltas.is_empty()) {
            let now = self.deferred_content[carried.len()..].to_owned();
            content = format!("{carried}{content}{now}");
            self.deferred_content.clear();
        }
        if !self.content_has_nonws && !content.is_empty() {
            if !py_strip(&content).is_empty() {
                self.content_has_nonws = true;
            } else if !self.slots.is_empty() {
                content.clear();
            } else if !finished {
                self.deferred_content = std::mem::take(&mut content);
            }
        }
        if content.is_empty() && deltas.is_empty() && reasoning.is_empty() {
            return None;
        }
        let mut delta = DeltaMessage::default();
        if !content.is_empty() {
            delta.content = TextField::Text(content);
        }
        if !reasoning.is_empty() {
            delta.reasoning = TextField::Text(reasoning);
        }
        delta.tool_calls = deltas;
        Some(delta)
    }

    /// `InklingParser`'s trailing-white-space hold-back: reasoning white space is released only when more reasoning
    /// follows.
    pub fn strip_trailing_reasoning(&mut self, delta: Option<DeltaMessage>) -> Option<DeltaMessage> {
        match delta {
            Some(mut d) if d.reasoning.text().is_some() => {
                let combined = format!("{}{}", self.deferred_reasoning, d.reasoning.text().unwrap_or_default());
                let trimmed_len = py_rstrip(&combined).len();
                self.deferred_reasoning = combined[trimmed_len..].to_owned();
                let trimmed = &combined[..trimmed_len];
                d.reasoning.set((!trimmed.is_empty()).then(|| trimmed.to_owned()));
                (d.reasoning.text().is_some() || d.content.text().is_some() || !d.tool_calls.is_empty()).then_some(d)
            }
            other => {
                if self.reasoning_ended {
                    self.deferred_reasoning.clear();
                }
                other
            }
        }
    }

    /// Non-streaming reasoning adapter: `(reasoning, content)` of a whole output's events.
    pub fn extract_reasoning(&mut self, events: &[Event]) -> (Option<String>, Option<String>) {
        let (mut reasoning, mut content) = (String::new(), String::new());
        for event in events {
            match event {
                Event::ReasoningChunk(v) => reasoning.push_str(v),
                Event::Text(v) => content.push_str(v),
                Event::ReasoningEnd => self.reasoning_ended = true,
                _ => {}
            }
        }
        let reasoning = py_rstrip(&reasoning);
        (
            (!reasoning.is_empty()).then(|| reasoning.to_owned()),
            (!content.is_empty()).then_some(content),
        )
    }

    /// Non-streaming tool adapter: the finished calls and the content of a whole output's events.
    pub fn extract_tool_calls(&mut self, events: &[Event]) -> Extracted {
        let delta = self.events_to_delta(events, true);
        let mut info = self.build_extracted_result();
        let mut parsed = delta.and_then(|d| d.content.text().map(str::to_owned));
        parsed = match parsed {
            Some(c) if !c.is_empty() => strip_content_ws(c, info.tools_called),
            _ => None,
        };
        // `_single_pass_parse`: content that was deferred behind tool events comes last.
        if !self.deferred_content.is_empty() {
            let trailing = std::mem::take(&mut self.deferred_content);
            parsed = strip_content_ws(parsed.unwrap_or_default() + &trailing, info.tools_called);
            info.content.clone_from(&parsed);
        }
        if info.content.is_none() {
            info.content = parsed;
        }
        info
    }

    fn ensure_slot(&mut self, idx: usize) {
        if self.slots.len() <= idx {
            self.slots.resize_with(idx + 1, ToolSlot::default);
        }
    }

    fn ensure_tool_id(&mut self, idx: usize) {
        if self.slots[idx].id.is_empty() {
            self.slots[idx].id = (self.id_factory)();
        }
    }

    /// The tool name found in the raw call text so far.
    fn try_extract_name(&mut self, idx: usize) -> Option<String> {
        let slot = &mut self.slots[idx];
        if slot.json_name.is_none() {
            slot.json_name = search_name(&slot.args).map(str::to_owned);
        }
        slot.json_name.clone().filter(|n| !n.is_empty())
    }

    fn emit_name_delta(&mut self, idx: usize, deltas: &mut Vec<ToolCallDelta>, name: Option<String>) {
        let Some(name) = name.filter(|n| !n.is_empty()) else {
            return;
        };
        self.announce(idx, &name);
        deltas.push(ToolCallDelta {
            id: Some(self.slots[idx].id.clone()),
            kind: Some(CallKind::Function),
            index: idx,
            function: Some(FunctionDelta {
                name: Some(name),
                arguments: None,
            }),
        });
    }

    /// Marks the call's name as sent and prepares its argument streaming.
    fn announce(&mut self, idx: usize, name: &str) {
        self.slots[idx].string_keys = streamable_string_keys(tool_properties(&self.tools, name));
        let slot = &mut self.slots[idx];
        slot.name = name.to_owned();
        slot.name_sent = true;
        self.ensure_tool_id(idx);
    }

    fn handle_arg_chunk(&mut self, idx: usize, value: &str, deltas: &mut Vec<ToolCallDelta>) {
        self.ensure_slot(idx);
        self.slots[idx].args.push_str(value);
        if value.is_empty() {
            return;
        }
        if self.slots[idx].name_sent {
            if let Some(args) = self.compute_arg_delta(idx) {
                deltas.push(argument_delta(idx, args));
            }
        } else {
            let name = self.try_extract_name(idx);
            self.emit_name_delta(idx, deltas, name);
        }
    }

    fn handle_tool_end(&mut self, idx: usize, deltas: &mut Vec<ToolCallDelta>) {
        if idx >= self.slots.len() {
            return;
        }
        let mut remaining = self.flush_arg_converter(idx);
        if !self.slots[idx].name_sent {
            let name = match self.slots[idx].name.as_str() {
                "" => self.try_extract_name(idx).unwrap_or_default(),
                known => known.to_owned(),
            };
            if !name.is_empty() {
                self.announce(idx, &name);
                deltas.push(ToolCallDelta {
                    id: Some(self.slots[idx].id.clone()),
                    kind: Some(CallKind::Function),
                    index: idx,
                    function: Some(FunctionDelta {
                        name: Some(name),
                        arguments: Some(remaining.take().unwrap_or_default()),
                    }),
                });
            }
        }
        if let Some(rest) = remaining.filter(|r| !r.is_empty())
            && self.slots[idx].name_sent
        {
            deltas.push(argument_delta(idx, rest));
        }
    }

    /// New argument text that is safe to stream: what `args` now holds beyond what was already sent.
    fn compute_arg_delta(&mut self, idx: usize) -> Option<String> {
        let slot = &self.slots[idx];
        let current = arg_converter(&slot.args, true).ok().filter(|c| !c.is_empty())?;
        let current = if slot.name.is_empty() {
            current.into()
        } else {
            fix_arg_types(current, &slot.name, &self.tools)
        };
        let prev = &slot.streamed_json;
        let safe = safe_arg_prefix(&current, slot.string_keys.as_ref());
        if safe.is_empty() || (!prev.is_empty() && !safe.starts_with(prev.as_str())) {
            return None;
        }
        let diff = safe[prev.len()..].to_owned();
        let safe = safe.to_owned();
        self.slots[idx].streamed_json = safe;
        (!diff.is_empty()).then_some(diff)
    }

    /// At the end of a call: the arguments not yet streamed.
    fn flush_arg_converter(&mut self, idx: usize) -> Option<String> {
        let slot = &self.slots[idx];
        let fin = arg_converter(&slot.args, false).ok()?;
        let fin = if fin.is_empty() {
            fin.into()
        } else {
            fix_arg_types(fin, &slot.name, &self.tools)
        };
        let prev = &slot.streamed_json;
        if fin.len() <= prev.len() || (!prev.is_empty() && !fin.starts_with(prev.as_str())) {
            return None;
        }
        let diff = fin[prev.len()..].to_owned();
        let fin = fin.into_owned();
        self.slots[idx].streamed_json = fin;
        Some(diff)
    }

    /// `InklingParser._extract_name_and_args`: a call whose name is not in the stream: `{"name": ..., "args": ...}`.
    fn extract_name_and_args(raw: &str) -> (String, String) {
        let raw = py_strip(raw);
        let Some(Value::Object(mut parsed)) = serde_json::from_str(raw).ok() else {
            return (String::new(), raw.to_owned());
        };
        let name = parsed
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        for key in ["args", "arguments", "parameters"] {
            if let Some(v) = parsed.get(key) {
                let args = match v {
                    Value::String(s) => s.clone(),
                    other => dumps_spaced(other),
                };
                return (name, args);
            }
        }
        parsed.shift_remove("name");
        (name, dumps_spaced(&Value::Object(parsed)))
    }

    fn build_extracted_result(&mut self) -> Extracted {
        let mut info = Extracted::default();
        for idx in 0..self.slots.len() {
            let slot = &self.slots[idx];
            if slot.name.is_empty() && slot.args.is_empty() {
                continue;
            }
            let mut name = py_strip(&slot.name).to_owned();
            let raw = slot.args.as_str();
            let raw_nonblank = !py_strip(raw).is_empty();
            let mut args_json = if name.is_empty() && raw_nonblank {
                let (n, a) = Self::extract_name_and_args(raw);
                name = n;
                a
            } else if raw_nonblank {
                match arg_converter(raw, false) {
                    Ok(a) => a.to_owned(),
                    Err(_) => Self::extract_name_and_args(raw).1,
                }
            } else {
                "{}".to_owned()
            };
            if !name.is_empty() {
                self.ensure_tool_id(idx);
                args_json = fix_arg_types(&args_json, &name, &self.tools).into_owned();
                info.tool_calls.push(FunctionCall {
                    id: self.slots[idx].id.clone(),
                    name,
                    arguments: args_json,
                });
            }
        }
        info.tools_called = !info.tool_calls.is_empty();
        info
    }
}

fn argument_delta(idx: usize, arguments: String) -> ToolCallDelta {
    ToolCallDelta {
        id: None,
        kind: None,
        index: idx,
        function: Some(FunctionDelta {
            name: None,
            arguments: Some(arguments),
        }),
    }
}

fn dumps_spaced(value: &Value) -> String {
    use ink_text::pyfmt::{NonAscii, Separators, dumps};
    dumps(value, NonAscii::Keep, Separators::Spaced)
}

/// Merges fragments of the same call into one, in order of first appearance.
fn coalesce(deltas: Vec<ToolCallDelta>) -> Vec<ToolCallDelta> {
    let mut merged: Vec<ToolCallDelta> = Vec::new();
    for tc in deltas {
        let Some(existing) = merged.iter_mut().find(|m| m.index == tc.index) else {
            merged.push(tc);
            continue;
        };
        if existing.id.is_none() {
            existing.id = tc.id;
        }
        if existing.kind.is_none() {
            existing.kind = tc.kind;
        }
        match (&mut existing.function, tc.function) {
            (None, f) => existing.function = f,
            (Some(ef), Some(f)) => {
                if ef.name.is_none() {
                    ef.name = f.name;
                }
                if let Some(a) = f.arguments {
                    ef.arguments.get_or_insert_with(String::new).push_str(&a);
                }
            }
            (Some(_), None) => {}
        }
    }
    merged
}

/// `_strip_content_ws`: content that is only white space before tool calls is dropped; empty content is `None`.
pub(crate) fn strip_content_ws(mut content: String, tools_called: bool) -> Option<String> {
    if tools_called && py_strip(&content).is_empty() {
        content.clear();
    }
    (!content.is_empty()).then_some(content)
}
