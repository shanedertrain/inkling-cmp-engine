// Ported from engine/server/src/parse/inkling_parser.{h,cpp} @ 6100ab8, itself a port of vLLM's Inkling parser
// (Apache-2.0, Copyright contributors to the vLLM project).
//! Parser output: streaming deltas and non-streaming results.

use serde_json::{Map, Value};

/// A text field of a [`DeltaMessage`] with pydantic's "set" semantics: serialised only when it was set, and a field
/// set to `None` is written as `null`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum TextField {
    /// Never assigned: omitted from the wire.
    #[default]
    Unset,
    /// Assigned `None`: written as `null`.
    Null,
    /// Assigned text.
    Text(String),
}

impl TextField {
    /// The text, if any (a field that is unset or null has none).
    #[must_use]
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s),
            Self::Unset | Self::Null => None,
        }
    }

    /// Assigns `value`; `None` marks the field as set to null.
    pub fn set(&mut self, value: Option<String>) {
        *self = value.map_or(Self::Null, Self::Text);
    }

    /// Whether the field carries non-empty text.
    #[must_use]
    pub fn has_text(&self) -> bool {
        self.text().is_some_and(|s| !s.is_empty())
    }

    fn to_value(&self) -> Option<Value> {
        match self {
            Self::Unset => None,
            Self::Null => Some(Value::Null),
            Self::Text(s) => Some(Value::String(s.clone())),
        }
    }
}

/// The only tool-call type OpenAI defines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallKind {
    /// `"function"`.
    Function,
}

/// Name and argument fragments of a streamed tool call.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FunctionDelta {
    /// The function name, sent once with the first fragment.
    pub name: Option<String>,
    /// A fragment of the JSON arguments.
    pub arguments: Option<String>,
}

/// One streamed tool-call fragment (`DeltaToolCall`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCallDelta {
    /// Call id, sent once with the first fragment.
    pub id: Option<String>,
    /// Call type, sent once with the first fragment.
    pub kind: Option<CallKind>,
    /// Position of the call in the message.
    pub index: usize,
    /// Name and argument fragments.
    pub function: Option<FunctionDelta>,
}

impl ToolCallDelta {
    fn to_value(&self) -> Value {
        let mut m = Map::new();
        if let Some(id) = &self.id {
            m.insert("id".into(), Value::String(id.clone()));
        }
        if let Some(CallKind::Function) = self.kind {
            m.insert("type".into(), Value::String("function".into()));
        }
        m.insert("index".into(), Value::from(self.index));
        if let Some(f) = &self.function {
            let mut fm = Map::new();
            if let Some(n) = &f.name {
                fm.insert("name".into(), Value::String(n.clone()));
            }
            if let Some(a) = &f.arguments {
                fm.insert("arguments".into(), Value::String(a.clone()));
            }
            m.insert("function".into(), Value::Object(fm));
        }
        Value::Object(m)
    }
}

/// One streaming step of the assistant message (`DeltaMessage`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeltaMessage {
    /// Message role (set by the API layer, never by the parser).
    pub role: Option<String>,
    /// Visible text.
    pub content: TextField,
    /// Reasoning text.
    pub reasoning: TextField,
    /// Tool-call fragments; an empty list is never serialised.
    pub tool_calls: Vec<ToolCallDelta>,
}

impl DeltaMessage {
    /// The wire object config B serialises (pydantic `exclude_unset`): keys in the order role, content, reasoning,
    /// tool_calls; unset fields and an empty `tool_calls` list are left out.
    #[must_use]
    pub fn to_value(&self) -> Value {
        let mut m = Map::new();
        if let Some(role) = &self.role {
            m.insert("role".into(), Value::String(role.clone()));
        }
        if let Some(v) = self.content.to_value() {
            m.insert("content".into(), v);
        }
        if let Some(v) = self.reasoning.to_value() {
            m.insert("reasoning".into(), v);
        }
        if !self.tool_calls.is_empty() {
            m.insert(
                "tool_calls".into(),
                Value::Array(self.tool_calls.iter().map(ToolCallDelta::to_value).collect()),
            );
        }
        Value::Object(m)
    }
}

/// A finished tool call of a non-streaming parse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionCall {
    /// Call id.
    pub id: String,
    /// Function name.
    pub name: String,
    /// JSON arguments, schema-coerced and spelled as Python's `json.dumps` when coercion changed them.
    pub arguments: String,
}

/// Result of [`InklingParser::parse`](crate::InklingParser::parse) (`DelegatingParser.parse`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Parsed {
    /// Reasoning text, trailing white space removed.
    pub reasoning: Option<String>,
    /// Visible text.
    pub content: Option<String>,
    /// Tool calls; `None` where config B returns `None` (no call was found), an empty list where it returns `[]`.
    pub tool_calls: Option<Vec<FunctionCall>>,
}
