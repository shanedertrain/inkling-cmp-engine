// Ported from engine/server/src/text/render.{h,cpp} @ 4171493, which ports vllm/renderers/inkling_encoding.py
// (render_inkling2_messages and helpers), vllm/renderers/inkling.py (_resolve_inkling2_reasoning_effort) and the
// request validators of vllm/entrypoints/openai/chat_completion/protocol.py (check_tool_usage,
// check_generation_prompt, build_chat_params). Apache-2.0, Copyright contributors to the vLLM project.
//! Inkling-2 chat rendering to token ids, identical to config B's `render_inkling2_messages`.
//!
//! [`resolve_chat_params`] turns a chat-completions request body into [`ChatRenderParams`] (and rejects what vLLM's
//! request validation rejects); [`ChatRenderer::render`] turns those into prompt token ids.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use serde_json::{Map, Value, json};

use crate::pyfmt::{Exponent, KeyOrder, NonAscii, dumps_string, format_g, py_strip, round_half_even, tml_dumps};
use crate::pyjson::{float_repr, parse_float, repr, truthy};
use crate::{Specials, Tokenizer};

const NULL: &Value = &Value::Null;

/// A request config B answers with HTTP 400. The message is what config B returns (as far as it is reproduced).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestError {
    message: String,
    param: Option<String>,
}

impl RequestError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            param: None,
        }
    }

    fn with_param(message: impl Into<String>, param: &str) -> Self {
        Self {
            message: message.into(),
            param: Some(param.to_owned()),
        }
    }

    /// The error message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The request parameter the error names (`"messages"`, `"tool_choice"`, ...), if any.
    #[must_use]
    pub fn param(&self) -> Option<&str> {
        self.param.as_deref()
    }
}

impl fmt::Display for RequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for RequestError {}

/// The tokenizer has no token with this content, so it is not an Inkling-2 tokenizer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MissingSpecialToken(pub String);

impl fmt::Display for MissingSpecialToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Inkling tokenizer is missing special token {}", self.0)
    }
}

impl Error for MissingSpecialToken {}

/// The special-token ids the renderer and the parsers use, resolved from the vocabulary and never hard-coded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpecialIds {
    /// `<|endoftext|>`
    pub endoftext: u32,
    /// `<|message_user|>`
    pub message_user: u32,
    /// `<|message_model|>`
    pub message_model: u32,
    /// `<|message_system|>`
    pub message_system: u32,
    /// `<|message_tool|>`
    pub message_tool: u32,
    /// `<|content_text|>`
    pub content_text: u32,
    /// `<|content_image|>`
    pub content_image: u32,
    /// `<|content_model_end_sampling|>`
    pub end_sampling: u32,
    /// `<|content_thinking|>`
    pub content_thinking: u32,
    /// `<|end_message|>`
    pub end_message: u32,
    /// `<|content_tool_error|>`
    pub content_tool_error: u32,
    /// `<|content_xml|>`
    pub content_xml: u32,
    /// `<|invoke_tool_json|>`
    pub invoke_tool_json: u32,
    /// `<|invoke_tool_text|>`
    pub invoke_tool_text: u32,
    /// `<|begin_of_text|>`
    pub begin_of_text: u32,
}

impl SpecialIds {
    /// Looks every token up in `tok`.
    ///
    /// # Errors
    /// A token is missing.
    pub fn from_tokenizer(tok: &Tokenizer) -> Result<Self, MissingSpecialToken> {
        let need = |s: &str| tok.token_id(s).ok_or_else(|| MissingSpecialToken(s.to_owned()));
        Ok(Self {
            endoftext: need("<|endoftext|>")?,
            message_user: need("<|message_user|>")?,
            message_model: need("<|message_model|>")?,
            message_system: need("<|message_system|>")?,
            message_tool: need("<|message_tool|>")?,
            content_text: need("<|content_text|>")?,
            content_image: need("<|content_image|>")?,
            end_sampling: need("<|content_model_end_sampling|>")?,
            content_thinking: need("<|content_thinking|>")?,
            end_message: need("<|end_message|>")?,
            content_tool_error: need("<|content_tool_error|>")?,
            content_xml: need("<|content_xml|>")?,
            invoke_tool_json: need("<|content_invoke_tool_json|>")?,
            invoke_tool_text: need("<|content_invoke_tool_text|>")?,
            begin_of_text: need("<|begin_of_text|>")?,
        })
    }
}

/// What `render_inkling2_messages` receives, resolved from the request as config B does.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatRenderParams {
    /// The request's messages (`reasoning_content` already folded into `reasoning`).
    pub messages: Vec<Value>,
    /// The request's tools in `model_dump` shape; `None` when absent or empty.
    pub tools: Option<Value>,
    /// `"auto"`, `"none"`, `"required"` or `{"type":"function","function":{"name":..}}`; `None` without tools.
    pub tool_choice: Option<Value>,
    /// `parallel_tool_calls` (default true).
    pub parallel_tool_calls: bool,
    /// Resolved reasoning effort in `[0.0, 0.99]` (default 0.9).
    pub reasoning_effort: f64,
    /// `continue_final_message`.
    pub continue_final_message: bool,
}

impl Default for ChatRenderParams {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            tools: None,
            tool_choice: None,
            parallel_tool_calls: true,
            reasoning_effort: 0.9,
            continue_final_message: false,
        }
    }
}

const NAMED_EFFORTS: [(&str, f64); 7] = [
    ("none", 0.0),
    ("minimal", 0.1),
    ("low", 0.2),
    ("medium", 0.7),
    ("high", 0.9),
    ("xhigh", 0.99),
    ("max", 0.99),
];

fn named_effort(name: &str) -> Option<f64> {
    NAMED_EFFORTS.iter().find(|(n, _)| *n == name).map(|(_, e)| *e)
}

fn type_error(message: &str) -> RequestError {
    RequestError::new(message)
}

fn get<'v>(obj: &'v Value, key: &str) -> Option<&'v Value> {
    obj.as_object()?.get(key)
}

fn get_or_null<'v>(obj: &'v Value, key: &str) -> &'v Value {
    get(obj, key).unwrap_or(NULL)
}

/// Python truthiness of a possibly missing value.
fn is_truthy(v: Option<&Value>) -> bool {
    v.is_some_and(truthy)
}

fn str_of(v: &Value) -> Option<&str> {
    v.as_str()
}

/// `_resolve_inkling2_reasoning_effort`: a name, a number, a numeric string or null (0.9).
///
/// # Errors
/// The value is none of those.
pub fn resolve_effort(value: &Value) -> Result<f64, RequestError> {
    let effort = match value {
        Value::Null => Some(0.9),
        Value::String(s) => named_effort(s).or_else(|| parse_float(s)),
        Value::Number(n) => n.as_f64(),
        _ => None,
    };
    effort.ok_or_else(|| RequestError::new(format!("invalid Inkling reasoning_effort: {}", repr(value))))
}

/// `_inkling2_effort_text`: `Thinking effort level: <effort>` with the effort rounded to two decimals.
///
/// # Errors
/// `effort` is outside `[0.0, 0.99]` or not a number.
pub fn effort_text(effort: f64) -> Result<String, RequestError> {
    if !(0.0..=0.99).contains(&effort) {
        return Err(RequestError::new(format!(
            "Inkling reasoning_effort must be in [0.0, 0.99], got {}",
            float_repr(effort)
        )));
    }
    let milli = round_half_even(effort * 1000.0);
    let rounded = round_half_even((milli / 1000.0) * 100.0) / 100.0;
    Ok(format!("Thinking effort level: {}", format_g(rounded)))
}

/// Request body (chat completions) to render parameters.
///
/// # Errors
/// What vLLM's request validation rejects with HTTP 400: empty `messages`, non-text parts in system and tool
/// messages, `continue_final_message` with `add_generation_prompt`, an inconsistent `tool_choice`, an unknown
/// `reasoning_effort`.
pub fn resolve_chat_params(body: &Value) -> Result<ChatRenderParams, RequestError> {
    let messages = match get(body, "messages").and_then(Value::as_array) {
        Some(m) if !m.is_empty() => m,
        _ => {
            return Err(RequestError::with_param(
                "messages must be a non-empty list.",
                "messages",
            ));
        }
    };
    // pydantic validation of the message params: tool and system message content parts must be text parts
    for m in messages {
        let role = str_of(get_or_null(m, "role"));
        let Some(parts) = get_or_null(m, "content").as_array() else {
            continue;
        };
        if !matches!(role, Some("tool" | "system")) {
            continue;
        }
        for part in parts.iter().filter(|p| p.is_object()) {
            let t = get(part, "type");
            if t.and_then(str_of) != Some("text") {
                return Err(RequestError::new(format!(
                    "1 validation error for ValidatorIterator\n0.type\n  Input should be 'text' [type=literal_error, input_value={}]",
                    t.map_or_else(|| "None".to_owned(), repr)
                )));
            }
        }
    }
    let messages = messages.iter().map(fold_reasoning_content).collect();

    let continue_final = is_truthy(get(body, "continue_final_message"));
    let add_generation_prompt = get(body, "add_generation_prompt").is_none_or(truthy);
    if continue_final && add_generation_prompt {
        return Err(RequestError::new(
            "Cannot set both `continue_final_message` and `add_generation_prompt` to True.",
        ));
    }

    let tools = get(body, "tools").filter(|t| !t.is_null() && t.as_array().is_none_or(|a| !a.is_empty()));
    let mut tool_choice = get(body, "tool_choice").cloned().unwrap_or(Value::Null);
    if tools.is_some() {
        if tool_choice.is_null() {
            tool_choice = Value::from("auto");
        }
    } else if tool_choice.is_null() || tool_choice.as_str() == Some("auto") {
        tool_choice = Value::from("none");
    }
    check_tool_choice(&tool_choice, tools)?;

    let parallel_tool_calls = get(body, "parallel_tool_calls").and_then(Value::as_bool) != Some(false);

    // chat_template_kwargs merged with the request-level overrides; overrides that are null or "auto" are dropped
    let from_kwargs = get(body, "chat_template_kwargs").map(|k| get_or_null(k, "reasoning_effort"));
    let mut effort_value = from_kwargs.unwrap_or(NULL);
    if let Some(re) = get(body, "reasoning_effort").filter(|r| !r.is_null() && r.as_str() != Some("auto")) {
        if !re.as_str().is_some_and(|s| named_effort(s).is_some()) {
            return Err(RequestError::with_param(
                "Input should be 'none', 'minimal', 'low', 'medium', 'high', 'xhigh' or 'max'",
                "reasoning_effort",
            ));
        }
        effort_value = re;
    }
    Ok(ChatRenderParams {
        messages,
        tools: tools.cloned(),
        tool_choice: tools.map(|_| tool_choice),
        parallel_tool_calls,
        reasoning_effort: resolve_effort(effort_value)?,
        continue_final_message: continue_final,
    })
}

/// `_materialize_tool_calls_before`: `reasoning_content` becomes `reasoning` when `reasoning` is null or absent.
fn fold_reasoning_content(message: &Value) -> Value {
    let mut m = message.clone();
    if let Some(map) = m.as_object_mut()
        && let Some(rc) = map.shift_remove("reasoning_content")
        && !rc.is_null()
        && map.get("reasoning").is_none_or(Value::is_null)
    {
        map.insert("reasoning".to_owned(), rc);
    }
    m
}

/// `check_tool_usage`: a `tool_choice` other than none needs tools, and a named one must name a given tool.
fn check_tool_choice(tool_choice: &Value, tools: Option<&Value>) -> Result<(), RequestError> {
    if tool_choice.is_null() || tool_choice.as_str() == Some("none") {
        return Ok(());
    }
    let Some(tools) = tools else {
        return Err(RequestError::with_param(
            "When using `tool_choice`, `tools` must be set.",
            "tool_choice",
        ));
    };
    if matches!(tool_choice.as_str(), Some("required" | "auto")) {
        return Ok(());
    }
    if !tool_choice.is_object() {
        let shown = tool_choice.as_str().map_or_else(|| repr(tool_choice), str::to_owned);
        return Err(RequestError::with_param(
            format!(
                "Invalid value for `tool_choice`: {shown}! Only named tools, \"none\", \"auto\" or \"required\" are supported."
            ),
            "tool_choice",
        ));
    }
    const USAGE: &str = "Correct usage: `{\"type\": \"function\", \"function\": {\"name\": \"my_function\"}}`";
    let function = get(tool_choice, "function").filter(|f| f.is_object());
    let Some(function) = function else {
        let shown = get(tool_choice, "function").map_or_else(|| "None".to_owned(), repr);
        return Err(RequestError::with_param(
            format!("Invalid value for `function`: `{shown}` in `tool_choice`! {USAGE}"),
            "tool_choice.function",
        ));
    };
    let Some(name) = get(function, "name") else {
        return Err(RequestError::with_param(
            format!("Expected field `name` in `function` in `tool_choice`! {USAGE}"),
            "tool_choice.function.name",
        ));
    };
    let Some(name_str) = name.as_str().filter(|n| !n.is_empty()) else {
        let shown = name.as_str().map_or_else(|| repr(name), str::to_owned);
        return Err(RequestError::with_param(
            format!("Invalid `name` in `function`: `{shown}` in `tool_choice`! {USAGE}"),
            "tool_choice.function.name",
        ));
    };
    let found = tools.as_array().is_some_and(|ts| {
        ts.iter()
            .any(|t| get(t, "function").and_then(|f| get(f, "name")) == Some(name))
    });
    if found {
        Ok(())
    } else {
        Err(RequestError::with_param(
            format!("The tool `{name_str}` specified in `tool_choice` does not match any of the specified `tools`"),
            "tool_choice",
        ))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    User,
    Assistant,
    System,
    Developer,
    Tool,
}

impl Role {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "user" => Self::User,
            "assistant" => Self::Assistant,
            "system" => Self::System,
            "developer" => Self::Developer,
            "tool" => Self::Tool,
            _ => return None,
        })
    }

    fn is_system_like(self) -> bool {
        matches!(self, Self::System | Self::Developer)
    }
}

/// The content kinds a rendered message can carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Text,
    Thinking,
    ToolError,
    Image,
    Xml,
    InvokeToolJson,
}

struct Block {
    kind: Kind,
    text: String,
    author: Option<String>,
}

impl Block {
    fn new(kind: Kind, text: String) -> Self {
        Self {
            kind,
            text,
            author: None,
        }
    }
}

/// `_iter_inkling2_parts`
fn iter_parts(content: Option<&Value>) -> Result<Vec<(Kind, String)>, RequestError> {
    let Some(content) = content.filter(|c| !c.is_null()) else {
        return Ok(Vec::new());
    };
    if let Some(s) = content.as_str() {
        return Ok(vec![(Kind::Text, s.to_owned())]);
    }
    let parts = content
        .as_array()
        .ok_or_else(|| type_error("message content must be a string or a sequence of parts"))?;
    let mut out = Vec::with_capacity(parts.len());
    for part in parts {
        if let Some(s) = part.as_str() {
            out.push((Kind::Text, s.to_owned()));
            continue;
        }
        if !part.is_object() {
            return Err(type_error("content part must be mapping"));
        }
        let text_or_empty = |key: &str| -> Result<String, RequestError> {
            match get(part, key).filter(|v| truthy(v)) {
                None => Ok(String::new()),
                Some(Value::String(s)) => Ok(s.clone()),
                Some(_) => Err(type_error("can only concatenate str")),
            }
        };
        let kind = get(part, "type").filter(|t| !t.is_null());
        match kind.map(|t| t.as_str().unwrap_or("?")) {
            None | Some("text" | "input_text") => out.push((Kind::Text, text_or_empty("text")?)),
            Some("thinking") => out.push((Kind::Thinking, text_or_empty("thinking")?)),
            Some("tool_error") => out.push((Kind::ToolError, text_or_empty("text")?)),
            Some("image" | "input_image" | "image_url") => out.push((Kind::Image, String::new())),
            Some(_) => {
                return Err(RequestError::new(format!(
                    "unsupported Inkling2 content part type: {}",
                    repr(kind.unwrap_or(NULL))
                )));
            }
        }
    }
    Ok(out)
}

fn assistant_reasoning(m: &Value) -> &Value {
    match get(m, "reasoning").filter(|r| !r.is_null()) {
        Some(r) => r,
        None => get_or_null(m, "reasoning_content"),
    }
}

/// `str(value)` for an id or name that is normally a string.
fn id_text(v: &Value) -> String {
    v.as_str().map_or_else(|| repr(v), str::to_owned)
}

/// `_tool_call_name_and_args`
fn tool_call_name_and_args(tc: &Value) -> Result<(String, Map<String, Value>), RequestError> {
    if !tc.is_object() {
        return Err(type_error("expected mapping"));
    }
    let empty = Value::Object(Map::new());
    let function = get(tc, "function").unwrap_or(&empty);
    if !function.is_object() {
        return Err(type_error("expected mapping"));
    }
    let name = get(function, "name")
        .and_then(Value::as_str)
        .ok_or_else(|| type_error("tool call function name must be a string"))?;
    let args = match get(function, "arguments").filter(|a| truthy(a)) {
        None => Value::Object(Map::new()),
        Some(Value::String(s)) if py_strip(s).is_empty() => Value::Object(Map::new()),
        Some(Value::String(s)) => serde_json::from_str(s).map_err(|e| RequestError::new(e.to_string()))?,
        Some(raw) => raw.clone(),
    };
    match args {
        Value::Object(map) => Ok((name.to_owned(), map)),
        _ => Err(type_error("tool call function arguments must decode to an object")),
    }
}

#[derive(Debug, PartialEq, Eq)]
enum ToolChoice {
    Auto,
    None,
    Required,
    Named(String),
    /// Any other string; no instruction is added for it.
    Other,
}

/// `_inkling2_tool_choice`
fn inkling2_tool_choice(tc: Option<&Value>) -> Result<ToolChoice, RequestError> {
    let Some(tc) = tc.filter(|t| !t.is_null()) else {
        return Ok(ToolChoice::Auto);
    };
    if let Some(s) = tc.as_str() {
        return Ok(match s {
            "auto" => ToolChoice::Auto,
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            _ => ToolChoice::Other,
        });
    }
    if !tc.is_object() {
        return Err(type_error("expected mapping"));
    }
    if get(tc, "type").and_then(Value::as_str) != Some("function") {
        return Err(RequestError::new(format!("unsupported tool_choice: {}", repr(tc))));
    }
    let holder = get(tc, "function").filter(|f| truthy(f)).unwrap_or(tc);
    let name = get(holder, "name")
        .and_then(Value::as_str)
        .ok_or_else(|| RequestError::new("tool_choice function name must be a string"))?;
    Ok(ToolChoice::Named(name.to_owned()))
}

fn tool_choice_instruction(choice: &ToolChoice, parallel: bool) -> Option<String> {
    match choice {
        ToolChoice::Named(name) => {
            let once = if parallel { "" } else { " exactly once" };
            Some(format!(
                "Call the function named {}{once} in your next response.",
                dumps_string(name, NonAscii::Escape)
            ))
        }
        ToolChoice::Required => {
            let q = if parallel { "at least one" } else { "exactly one" };
            Some(format!("Call {q} of the provided functions in your next response."))
        }
        ToolChoice::Auto if !parallel => {
            Some("If you call a function, call at most one function in your next response.".to_owned())
        }
        _ => None,
    }
}

/// `_with_appended_text`
fn with_appended_text(message: &Value, text: &str) -> Result<Value, RequestError> {
    let mut m = message.clone();
    let content = get(message, "content");
    let new_content = match content {
        Some(Value::String(s)) if !s.is_empty() => Value::from(format!("{s}\n\n{text}")),
        c if !is_truthy(c) => Value::from(text),
        Some(Value::Array(parts)) => {
            let mut parts = parts.clone();
            parts.push(json!({"type": "text", "text": text}));
            Value::Array(parts)
        }
        _ => return Err(type_error("message content must be a string or a sequence of parts")),
    };
    if let Some(map) = m.as_object_mut() {
        map.insert("content".to_owned(), new_content);
    }
    Ok(m)
}

/// `_collect_tools`: request tools, then the tools of developer messages.
fn collect_tools(messages: &[Value], tools: Option<&Value>) -> Result<Vec<Value>, RequestError> {
    let mut all: Vec<Value> = tools.and_then(Value::as_array).cloned().unwrap_or_default();
    for m in messages {
        if get(m, "role").and_then(Value::as_str) == Some("developer")
            && let Some(mt) = get(m, "tools").filter(|t| truthy(t))
        {
            let list = mt
                .as_array()
                .ok_or_else(|| type_error("developer tools must be a list"))?;
            all.extend(list.iter().cloned());
        }
    }
    Ok(all)
}

/// The `tool_declare` message text: the tools as sorted-key JSON.
fn tool_declare_json(tools: &[Value]) -> Result<String, RequestError> {
    let mut specs = Vec::with_capacity(tools.len());
    for tool in tools {
        if !tool.is_object() {
            return Err(type_error("expected mapping"));
        }
        let function: Map<String, Value> = match get(tool, "function") {
            None => Map::new(),
            Some(Value::Object(f)) => f
                .iter()
                .filter(|(_, v)| !v.is_null())
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            Some(_) => return Err(type_error("expected mapping")),
        };
        let parameters = function.get("parameters").cloned().unwrap_or_else(|| json!({}));
        if !parameters.is_object() {
            return Err(type_error("tool parameters must be a JSON object"));
        }
        let name = function.get("name").ok_or_else(|| RequestError::new("'name'"))?;
        let kind = get(tool, "type")
            .filter(|t| truthy(t))
            .cloned()
            .unwrap_or_else(|| Value::from("function"));
        specs.push(json!({
            "description": function.get("description").cloned().unwrap_or_else(|| Value::from("")),
            "name": name,
            "parameters": parameters,
            "type": kind,
        }));
    }
    Ok(tml_dumps(&Value::Array(specs), Exponent::Signed, KeyOrder::Sorted))
}

/// `_inkling2_blocks`: the content blocks of one message; assistant tool calls are registered in `id_to_name`.
fn inkling2_blocks(
    message: &Value,
    role: Role,
    id_to_name: &mut HashMap<String, String>,
) -> Result<Vec<Block>, RequestError> {
    let content = get(message, "content");
    if role != Role::Assistant {
        return Ok(iter_parts(content)?
            .into_iter()
            .map(|(k, t)| Block::new(k, t))
            .collect());
    }
    let reasoning = assistant_reasoning(message);
    if !reasoning.is_null() && !reasoning.is_string() {
        return Err(type_error("assistant reasoning_content must be a string"));
    }
    let tool_calls = get(message, "tool_calls").filter(|t| truthy(t));
    let has_reasoning = truthy(reasoning);
    let content = content.filter(|c| !(c.as_str() == Some("") && (tool_calls.is_some() || has_reasoning)));
    let mut blocks = Vec::new();
    if has_reasoning && let Some(r) = reasoning.as_str() {
        blocks.push(Block::new(Kind::Thinking, r.to_owned()));
    }
    blocks.extend(iter_parts(content)?.into_iter().map(|(k, t)| Block::new(k, t)));
    let Some(tool_calls) = tool_calls else {
        return Ok(blocks);
    };
    for tc in tool_calls
        .as_array()
        .ok_or_else(|| type_error("tool_calls must be a list"))?
    {
        let (name, args) = tool_call_name_and_args(tc)?;
        if let Some(id) = get(tc, "id").filter(|i| truthy(i)) {
            id_to_name.insert(id_text(id), name.clone());
        }
        // _tml_json_loads(json.dumps(args)), top-level keys sorted, nested order kept
        let mut entries: Vec<(String, Value)> = args.into_iter().collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        let sorted: Map<String, Value> = entries.into_iter().collect();
        let args_json = tml_dumps(&Value::Object(sorted), Exponent::Bare, KeyOrder::Insertion);
        let text = format!(
            "{{\"name\":{},\"args\":{args_json}}}",
            dumps_string(&name, NonAscii::Keep)
        );
        blocks.push(Block {
            kind: Kind::InvokeToolJson,
            text,
            author: Some(name),
        });
    }
    Ok(blocks)
}

/// The text of a final assistant message that the model continues.
fn continuation_prefix(m: &Value) -> Option<String> {
    if get(m, "role").and_then(Value::as_str) != Some("assistant")
        || is_truthy(get(m, "tool_calls"))
        || truthy(assistant_reasoning(m))
    {
        return None;
    }
    get(m, "content").and_then(Value::as_str).map(str::to_owned)
}

/// Renders chat requests to prompt token ids with one tokenizer.
#[derive(Debug)]
pub struct ChatRenderer<'t> {
    tok: &'t Tokenizer,
    ids: SpecialIds,
}

impl<'t> ChatRenderer<'t> {
    /// A renderer over `tok`.
    ///
    /// # Errors
    /// `tok` lacks one of the Inkling-2 special tokens.
    pub fn new(tok: &'t Tokenizer) -> Result<Self, MissingSpecialToken> {
        Ok(Self {
            tok,
            ids: SpecialIds::from_tokenizer(tok)?,
        })
    }

    /// The special-token ids in use.
    #[must_use]
    pub fn ids(&self) -> &SpecialIds {
        &self.ids
    }

    fn encode_literal(&self, out: &mut Vec<u32>, text: &str) -> Result<(), RequestError> {
        let ids = self
            .tok
            .encode(text, Specials::Literal)
            .map_err(|e| RequestError::new(e.to_string()))?;
        out.extend(ids);
        Ok(())
    }

    fn append_message(
        &self,
        out: &mut Vec<u32>,
        role: Role,
        kind: Kind,
        text: &str,
        author: Option<&str>,
    ) -> Result<(), RequestError> {
        out.push(match role {
            Role::User => self.ids.message_user,
            Role::Assistant => self.ids.message_model,
            Role::System | Role::Developer => self.ids.message_system,
            Role::Tool => self.ids.message_tool,
        });
        if let Some(author) = author.filter(|a| !a.is_empty()) {
            self.encode_literal(out, author)?;
        }
        let kind_id = match kind {
            Kind::Image => {
                out.push(self.ids.content_image);
                out.push(self.ids.end_message);
                return Ok(());
            }
            Kind::Text => self.ids.content_text,
            Kind::Thinking => self.ids.content_thinking,
            Kind::ToolError => self.ids.content_tool_error,
            Kind::Xml => self.ids.content_xml,
            Kind::InvokeToolJson => self.ids.invoke_tool_json,
        };
        out.push(kind_id);
        self.encode_literal(out, text)?;
        out.push(self.ids.end_message);
        Ok(())
    }

    /// Prompt token ids of `p`, as config B renders them.
    ///
    /// # Errors
    /// The messages, tools or tool choice are ones `render_inkling2_messages` raises on (HTTP 400).
    pub fn render(&self, p: &ChatRenderParams) -> Result<Vec<u32>, RequestError> {
        let mut messages = p.messages.clone();
        let assistant_prefix = if p.continue_final_message {
            let prefix = messages.last().and_then(continuation_prefix);
            if prefix.is_some() {
                messages.pop();
            }
            prefix
        } else {
            None
        };
        let mut ids = Vec::new();
        let choice = inkling2_tool_choice(p.tool_choice.as_ref())?;
        let request_tools = if choice == ToolChoice::None {
            None
        } else {
            p.tools.as_ref()
        };
        let mut declared = collect_tools(&messages, request_tools)?;
        if let ToolChoice::Named(wanted) = &choice {
            declared
                .retain(|t| get(t, "function").and_then(|f| get(f, "name")).and_then(Value::as_str) == Some(wanted));
        }
        if !declared.is_empty() {
            let text = tool_declare_json(&declared)?;
            self.append_message(&mut ids, Role::System, Kind::Xml, &text, Some("tool_declare"))?;
        }
        let mut effort = Some(effort_text(p.reasoning_effort)?);
        if !declared.is_empty()
            && let Some(instruction) = tool_choice_instruction(&choice, p.parallel_tool_calls)
        {
            let first_role = messages.first().and_then(|m| get(m, "role")).and_then(Value::as_str);
            if first_role.and_then(Role::parse).is_some_and(Role::is_system_like) {
                messages[0] = with_appended_text(&messages[0], &instruction)?;
            } else if let Some(e) = effort.as_mut() {
                e.push_str("\n\n");
                e.push_str(&instruction);
            }
        }
        let mut id_to_name = HashMap::new();
        for m in &messages {
            let role_value = get(m, "role");
            let Some(role) = role_value.and_then(Value::as_str).and_then(Role::parse) else {
                return Err(RequestError::new(format!(
                    "unsupported Inkling message role {}; expected one of ['assistant', 'developer', 'system', 'tool', 'user']",
                    role_value.map_or_else(|| "None".to_owned(), repr)
                )));
            };
            if !role.is_system_like()
                && let Some(text) = effort.take()
            {
                self.append_message(&mut ids, Role::System, Kind::Text, &text, None)?;
            }
            let author = if role == Role::Tool {
                tool_author(m, &id_to_name)
            } else {
                None
            };
            let blocks = inkling2_blocks(m, role, &mut id_to_name)?;
            for b in &blocks {
                let a = b.author.as_deref().or(author.as_deref());
                self.append_message(&mut ids, role, b.kind, &b.text, a)?;
            }
            if role == Role::Assistant && !blocks.is_empty() {
                ids.push(self.ids.end_sampling);
            }
        }
        if let Some(text) = effort {
            self.append_message(&mut ids, Role::System, Kind::Text, &text, None)?;
        }
        if let Some(prefix) = assistant_prefix {
            ids.push(self.ids.message_model);
            ids.push(self.ids.content_text);
            self.encode_literal(&mut ids, &prefix)?;
        }
        Ok(ids)
    }
}

/// The author of a tool message: its `name`, else the function name its `tool_call_id` was registered under.
fn tool_author(m: &Value, id_to_name: &HashMap<String, String>) -> Option<String> {
    if let Some(name) = get(m, "name").filter(|n| truthy(n)) {
        return Some(id_text(name));
    }
    let id = get(m, "tool_call_id").filter(|i| truthy(i))?;
    id_to_name.get(&id_text(id)).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effort_text_rounds_half_even() {
        // Port of test_text.cpp `text_effort_and_floats` (effort part).
        assert_eq!(effort_text(0.9).unwrap(), "Thinking effort level: 0.9");
        assert_eq!(effort_text(0.0).unwrap(), "Thinking effort level: 0");
        assert_eq!(effort_text(0.99).unwrap(), "Thinking effort level: 0.99");
        assert_eq!(effort_text(0.125).unwrap(), "Thinking effort level: 0.12");
        assert!(effort_text(1.5).is_err());
        assert!(effort_text(f64::NAN).is_err());
    }

    #[test]
    fn resolve_effort_accepts_names_numbers_and_strings() {
        assert_eq!(resolve_effort(&json!("low")).unwrap(), 0.2);
        assert_eq!(resolve_effort(&json!("0.35")).unwrap(), 0.35);
        assert_eq!(resolve_effort(&Value::Null).unwrap(), 0.9);
        assert_eq!(resolve_effort(&json!(1)).unwrap(), 1.0);
        assert!(resolve_effort(&json!("huge")).is_err());
        assert!(resolve_effort(&json!(true)).is_err());
    }

    #[test]
    fn tool_choice_instruction_escapes_non_ascii_names() {
        let named = ToolChoice::Named("é\u{7f}".to_owned());
        assert_eq!(
            tool_choice_instruction(&named, false).unwrap(),
            r#"Call the function named "\u00e9\u007f" exactly once in your next response."#
        );
        assert_eq!(tool_choice_instruction(&ToolChoice::None, true), None);
    }

    #[test]
    fn request_validation_rejects_like_vllm() {
        // Port of test_text.cpp `text_render_errors`, the part that needs no tokenizer.
        let rejects = |body: Value| resolve_chat_params(&body).is_err();
        assert!(rejects(json!({"messages": []})));
        assert!(rejects(
            json!({"messages": [{"role": "user", "content": "x"}], "tool_choice": "required"})
        ));
        assert!(rejects(
            json!({"messages": [{"role": "user", "content": "x"}], "continue_final_message": true})
        ));
        assert!(rejects(
            json!({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "huge"})
        ));
        assert!(rejects(json!({
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {"name": "a"}}],
            "tool_choice": {"type": "function", "function": {"name": "b"}}
        })));
        let ok = resolve_chat_params(&json!({"messages": [{"role": "user", "content": "x"}]})).unwrap();
        assert_eq!(ok.tool_choice, None);
        assert!((ok.reasoning_effort - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn reasoning_content_is_folded_into_reasoning() {
        let p = resolve_chat_params(&json!({"messages": [
            {"role": "assistant", "content": "a", "reasoning_content": "r"},
            {"role": "assistant", "content": "b", "reasoning": "keep", "reasoning_content": "drop"}
        ]}))
        .unwrap();
        assert_eq!(
            p.messages[0],
            json!({"role": "assistant", "content": "a", "reasoning": "r"})
        );
        assert_eq!(
            p.messages[1],
            json!({"role": "assistant", "content": "b", "reasoning": "keep"})
        );
    }
}
