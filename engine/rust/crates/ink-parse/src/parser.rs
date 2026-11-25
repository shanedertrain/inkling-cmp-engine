// Ported from engine/server/src/parse/inkling_parser.cpp (`InklingParser`) @ 6100ab8, itself a port of vLLM's
// vllm/parser/inkling.py (`InklingParser`) and vllm/parser/abstract_parser.py (`DelegatingParser.parse`,
// `parse_delta`) (Apache-2.0, Copyright contributors to the vLLM project).
//! The Inkling parser: reasoning engine and tool engine, streaming and non-streaming.

use std::hash::{BuildHasher, Hasher, RandomState};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ink_text::pyfmt::py_strip;

use crate::assemble::IdFactory;
use crate::delta::{DeltaMessage, Parsed, TextField};
use crate::engine::Engine;
use crate::machine::State;
use crate::request::{ParseRequest, ToolChoice};
use crate::tables::ParserTables;

/// `chatcmpl-tool-` and 16 random hex digits, as config B names tool calls.
fn random_tool_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
    format!("chatcmpl-tool-{:016x}", hasher.finish())
}

/// Parses one choice's output into reasoning, content and tool calls (`DelegatingParser` of the Inkling reasoning
/// and tool adapters). One instance per request choice; streaming and non-streaming use separate instances.
#[derive(Debug)]
pub struct InklingParser {
    tables: Arc<ParserTables>,
    request: ParseRequest,
    reasoning: Engine,
    tool: Engine,
    reasoning_ended: bool,
    tool_call_text_started: bool,
    prompt_reasoning_checked: bool,
}

impl InklingParser {
    /// A parser for one request. `id_factory` names tool calls; `None` makes `chatcmpl-tool-<16 hex>` ids.
    #[must_use]
    pub fn new(tables: Arc<ParserTables>, request: ParseRequest, id_factory: Option<IdFactory>) -> Self {
        let id_factory = id_factory.unwrap_or_else(|| Arc::new(random_tool_id));
        let suppress = request.tool_choice == ToolChoice::None && !request.tools.is_empty();
        Self {
            reasoning: Engine::new(Arc::clone(&tables), true, Vec::new(), false, Arc::clone(&id_factory)),
            tool: Engine::new(Arc::clone(&tables), false, request.tools.clone(), suppress, id_factory),
            tables,
            request,
            reasoning_ended: false,
            tool_call_text_started: false,
            prompt_reasoning_checked: false,
        }
    }

    /// `InklingParser.count_reasoning_tokens`: the output ids that lie inside reasoning.
    #[must_use]
    pub fn count_reasoning_tokens(&self, ids: &[u32]) -> usize {
        self.tables.count_reasoning_tokens(ids)
    }

    /// `InklingParser.is_reasoning_end`: whether the ids leave the model outside reasoning.
    #[must_use]
    pub fn is_reasoning_end(&self, ids: &[u32]) -> bool {
        self.tables.is_reasoning_end(ids)
    }

    /// Streaming: `Parser.parse_delta` for one model step. `delta_text` and `delta_ids` are the step's new text and
    /// token ids; `prompt_ids` is read on the first call only (the serving layer passes it every time); `finished`
    /// marks the last step.
    pub fn parse_delta(
        &mut self,
        delta_text: &str,
        delta_ids: &[u32],
        prompt_ids: &[u32],
        finished: bool,
    ) -> Option<DeltaMessage> {
        if !self.prompt_reasoning_checked {
            self.prompt_reasoning_checked = true;
            if self.tables.is_reasoning_end(prompt_ids) {
                self.reasoning_ended = true;
            } else {
                self.reasoning.adjust_initial_state_from_prompt(prompt_ids);
            }
        }
        let mut tool_text = delta_text.to_owned();
        let mut dm: Option<DeltaMessage> = None;
        if !self.reasoning_ended {
            self.reasoning.initialize_streaming(State::MessageHeader);
            dm = self.reasoning.stream_feed(delta_text, delta_ids);
            if self.reasoning.reasoning_ended() {
                self.reasoning_ended = true;
                // The text that follows the end of reasoning in this step is the tool engine's first input.
                let flush = self.reasoning.finish_streaming();
                let content = |d: &Option<DeltaMessage>| d.as_ref().and_then(|d| d.content.text().map(str::to_owned));
                tool_text = content(&dm).unwrap_or_default() + &content(&flush).unwrap_or_default();
            }
        }
        if self.reasoning_ended {
            if self.tool_call_text_started {
                delta_text.clone_into(&mut tool_text);
            }
            self.tool_call_text_started = true;
            let r_batch = dm.as_ref().and_then(|d| d.reasoning.text().map(str::to_owned));
            self.tool.initialize_streaming(State::Content);
            dm = self.tool.stream_feed(&tool_text, delta_ids);
            if self.request.tool_choice == ToolChoice::None
                && let Some(d) = dm.as_mut()
            {
                d.tool_calls.clear();
            }
            if let Some(batch) = r_batch {
                match dm.as_mut() {
                    None => {
                        dm = Some(DeltaMessage {
                            reasoning: TextField::Text(batch),
                            ..DeltaMessage::default()
                        })
                    }
                    Some(d) if !d.reasoning.has_text() => d.reasoning.set(Some(batch)),
                    Some(_) => {}
                }
            }
        }
        if finished {
            dm = self.finalize(dm);
        }
        if !self.request.include_reasoning
            && let Some(d) = dm.as_mut()
        {
            d.reasoning.set(None);
            if !d.content.has_text() && d.tool_calls.is_empty() {
                dm = None;
            }
        }
        dm
    }

    /// `finalize_generation`: the last tool call's `arguments` are set to `""` when unset (vLLM's
    /// `_append_unstreamed_tool_args`), then both engines flush what they held back into the delta.
    fn finalize(&mut self, mut dm: Option<DeltaMessage>) -> Option<DeltaMessage> {
        if let Some(function) = dm
            .as_mut()
            .and_then(|d| d.tool_calls.last_mut())
            .and_then(|tc| tc.function.as_mut())
        {
            function.arguments.get_or_insert_with(String::new);
        }
        let mut flushes = Vec::new();
        if !self.reasoning_ended {
            flushes.push(self.reasoning.finish_streaming());
        }
        flushes.push(self.tool.finish_streaming());
        for flush in flushes.into_iter().flatten() {
            let Some(d) = dm.as_mut() else {
                dm = Some(flush);
                continue;
            };
            if let Some(text) = flush.content.text().filter(|t| !t.is_empty()) {
                d.content
                    .set(Some(format!("{}{}", d.content.text().unwrap_or_default(), text)));
            }
            if let Some(text) = flush.reasoning.text().filter(|t| !t.is_empty()) {
                d.reasoning
                    .set(Some(format!("{}{}", d.reasoning.text().unwrap_or_default(), text)));
            }
            d.tool_calls.extend(flush.tool_calls);
        }
        dm
    }

    /// Non-streaming: `DelegatingParser.parse` of the whole output text.
    pub fn parse(&mut self, model_output: &str) -> Parsed {
        let (reasoning, content) = self.reasoning.extract_reasoning(model_output);
        let info = self
            .tool
            .extract_tool_calls_from_content(content.as_deref().unwrap_or_default());
        let mut out = Parsed {
            reasoning,
            ..Parsed::default()
        };
        if self.request.tool_choice == ToolChoice::None {
            out.tool_calls = Some(Vec::new());
            out.content = info.content;
        } else if info.tools_called {
            out.tool_calls = Some(info.tool_calls);
            out.content = info.content.filter(|c| c.is_empty() || !py_strip(c).is_empty());
        } else if matches!(self.request.tool_choice, ToolChoice::Required | ToolChoice::Named)
            && content.as_deref().is_none_or(|c| py_strip(c).is_empty())
        {
            out.tool_calls = Some(Vec::new());
        } else {
            out.content = info.content.filter(|c| !c.is_empty());
        }
        out
    }
}
