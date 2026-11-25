// Ported from engine/server/src/parse/inkling_parser.h @ 6100ab8 (Apache-2.0, Copyright contributors to the vLLM
// project).
//! The request fields the parsers read.

use serde_json::Value;

/// The request's `tool_choice`, as far as the parsers distinguish it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ToolChoice {
    /// `"none"`: tool calls are parsed away and never reported.
    #[default]
    None,
    /// `"auto"` (or any other string): calls are reported when found.
    Auto,
    /// `"required"`.
    Required,
    /// A named function (`{"type": "function", "function": {...}}`).
    Named,
}

impl ToolChoice {
    /// Classifies a request's `tool_choice` value (after the serving layer's defaulting).
    #[must_use]
    pub fn from_value(value: &Value) -> Self {
        match value {
            Value::String(s) if s == "none" => Self::None,
            Value::String(s) if s == "required" => Self::Required,
            Value::Object(_) => Self::Named,
            _ => Self::Auto,
        }
    }
}

/// The request fields the parsers read.
#[derive(Clone, Debug, PartialEq)]
pub struct ParseRequest {
    /// `request.tools`: the tool definitions as sent (empty when the request has none).
    pub tools: Vec<Value>,
    /// `request.tool_choice` after the serving layer's defaulting.
    pub tool_choice: ToolChoice,
    /// Whether reasoning text is reported.
    pub include_reasoning: bool,
}

impl Default for ParseRequest {
    fn default() -> Self {
        Self {
            tools: Vec::new(),
            tool_choice: ToolChoice::None,
            include_reasoning: true,
        }
    }
}
