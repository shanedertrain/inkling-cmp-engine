// Ported from engine/server/src/parse/inkling_parser.cpp (`ParserEngine` entry points) @ 6100ab8 (Apache-2.0,
// Copyright contributors to the vLLM project).
//! One parser engine: a grammar [`Machine`] feeding an [`Assembler`]. The parser builds two, one for reasoning and one
//! for tool calls.

use std::sync::Arc;

use serde_json::Value;

use crate::assemble::{Assembler, Extracted, IdFactory};
use crate::delta::DeltaMessage;
use crate::machine::{Machine, State};
use crate::tables::ParserTables;

#[derive(Debug)]
pub(crate) struct Engine {
    tables: Arc<ParserTables>,
    machine: Machine,
    assembler: Assembler,
    streaming_initialized: bool,
}

impl Engine {
    pub fn new(
        tables: Arc<ParserTables>,
        skip_tool_parsing: bool,
        tools: Vec<Value>,
        suppress: bool,
        id_factory: IdFactory,
    ) -> Self {
        Self {
            machine: Machine::new(Arc::clone(&tables), skip_tool_parsing),
            assembler: Assembler::new(tools, suppress, id_factory),
            tables,
            streaming_initialized: false,
        }
    }

    pub fn reasoning_ended(&self) -> bool {
        self.assembler.reasoning_ended
    }

    fn reset(&mut self, initial: State) {
        self.machine.reset(initial);
        self.assembler.reset();
    }

    pub fn initialize_streaming(&mut self, initial: State) {
        if !self.streaming_initialized {
            self.streaming_initialized = true;
            self.reset(initial);
        }
    }

    /// `InklingParser.adjust_initial_state_from_prompt`: starts where the prompt left the model (inside a thinking,
    /// text or model message), found from the last grammar token of the prompt.
    pub fn adjust_initial_state_from_prompt(&mut self, prompt: &[u32]) {
        let t = &self.tables;
        for &id in prompt.iter().rev() {
            let initial = if Some(id) == t.thinking {
                State::Reasoning
            } else if Some(id) == t.text {
                State::Content
            } else if Some(id) == t.model {
                State::MessageHeader
            } else if t.special_ids.contains(&id) {
                return;
            } else {
                continue;
            };
            self.machine.reset(initial);
            self.streaming_initialized = true;
            return;
        }
    }

    pub fn stream_feed(&mut self, text: &str, ids: &[u32]) -> Option<DeltaMessage> {
        let events = self.machine.feed(text, ids);
        let delta = self.assembler.events_to_delta(&events, false);
        self.assembler.strip_trailing_reasoning(delta)
    }

    pub fn finish_streaming(&mut self) -> Option<DeltaMessage> {
        let events = self.machine.finish();
        if events.is_empty() && !self.assembler.has_deferred_content() {
            return None;
        }
        self.assembler.events_to_delta(&events, true)
    }

    fn whole_output_events(&mut self, text: &str, initial: State) -> Vec<crate::machine::Event> {
        self.reset(initial);
        let mut events = self.machine.feed(text, &[]);
        events.extend(self.machine.finish());
        events
    }

    /// Non-streaming reasoning adapter.
    pub fn extract_reasoning(&mut self, output: &str) -> (Option<String>, Option<String>) {
        let events = self.whole_output_events(output, State::MessageHeader);
        self.assembler.extract_reasoning(&events)
    }

    /// Non-streaming tool adapter, over the content the reasoning adapter left.
    pub fn extract_tool_calls_from_content(&mut self, content: &str) -> Extracted {
        let events = self.whole_output_events(content, State::Content);
        self.assembler.extract_tool_calls(&events)
    }
}
