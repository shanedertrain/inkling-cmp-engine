//! Text side of the front end: tokenizer, Python-compatible formatting, chat rendering and detokenisation.
//!
//! * [`Tokenizer`] wraps the Hugging Face `tokenizers` crate (the library config B itself runs under Python).
//! * [`pyfmt`] and [`pyjson`] reproduce the Python spellings config B's renderer and parser put on the wire
//!   (`json.dumps` separators, `repr`, `float` and `int` parsing, `round`, `%g`).
//!
//! * [`ChatRenderer`] renders chat requests to prompt token ids as config B's `render_inkling2_messages` does.
//! * [`OutputState`] and [`DecodeStream`] detokenise a request's tokens as they stream and apply its stop rules.
//!
//! Ported from `engine/server/src/text/*` (packages F1 and F2 of `docs/RUST-PLAN.md`).
#![forbid(unsafe_code)]

mod detok;
pub mod pyfmt;
pub mod pyjson;
mod render;
mod tokenizer;

pub use detok::{DecodeStream, DetokError, FinishReason, OutputState, StepResult, StopParams, StopReason};
pub use render::{
    ChatRenderParams, ChatRenderer, MissingSpecialToken, RequestError, SpecialIds, effort_text, resolve_chat_params,
    resolve_effort,
};
pub use tokenizer::{AddedToken, Specials, Tokenizer, TokenizerError};
