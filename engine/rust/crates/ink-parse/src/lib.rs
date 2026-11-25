//! Inkling reasoning and tool-call parser: config B's `--reasoning-parser inkling --tool-call-parser inkling`.
//!
//! [`InklingParser`] turns the text and token ids of a model's output into reasoning, content and tool calls, one
//! step at a time ([`InklingParser::parse_delta`]) or all at once ([`InklingParser::parse`]). Tool arguments are
//! coerced to the types of the tool's JSON schema.
//!
//! Ported from `engine/server/src/parse/inkling_parser.*` (package F3 of `docs/RUST-PLAN.md`), itself a port of vLLM's
//! Inkling parser (Apache-2.0, Copyright contributors to the vLLM project); see the crate README.
#![forbid(unsafe_code)]

mod args;
mod assemble;
mod coerce;
mod delta;
mod engine;
mod lexer;
mod machine;
mod parser;
mod request;
mod scanner;
mod tables;

pub use args::{MemberNotObject, extract_json_member_object};
pub use assemble::IdFactory;
pub use delta::{CallKind, DeltaMessage, FunctionCall, FunctionDelta, Parsed, TextField, ToolCallDelta};
pub use parser::InklingParser;
pub use request::{ParseRequest, ToolChoice};
pub use tables::{ParserTables, SpecialTokensError, special_tokens_from_dir};
