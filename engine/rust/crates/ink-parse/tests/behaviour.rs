//! Behaviour of the Inkling parser on a small byte-level tokenizer (no model files needed). The first five tests are
//! the ports of `engine/server/tests/test_parser.cpp`; the rest pin behaviours the C++ port found against config B.

mod common;

use std::collections::BTreeMap;

use common::{encode, fixed_ids, mini_tables, steps_of};
use ink_parse::{DeltaMessage, InklingParser, ParseRequest, ParserTables, TextField, ToolChoice};
use ink_text::Tokenizer;
use serde_json::{Value, json};

struct Fixture {
    tok: std::sync::Arc<Tokenizer>,
    tables: std::sync::Arc<ParserTables>,
}

impl Fixture {
    fn new() -> Self {
        let (tok, tables) = mini_tables();
        Self { tok, tables }
    }

    fn id(&self, special: &str) -> u32 {
        self.tok.token_id(special).expect("special token")
    }

    fn text(&self, s: &str) -> Vec<u32> {
        encode(&self.tok, s)
    }

    fn ids(&self, parts: &[Part<'_>]) -> Vec<u32> {
        parts
            .iter()
            .flat_map(|p| match p {
                Part::Tok(s) => vec![self.id(s)],
                Part::Text(s) => self.text(s),
            })
            .collect()
    }

    fn eos(&self) -> u32 {
        self.id("<|eos|>")
    }

    fn prompt(&self) -> Vec<u32> {
        self.ids(&[
            Part::Tok("<|message_user|>"),
            Part::Tok("<|content_text|>"),
            Part::Text("hi"),
            Part::Tok("<|end_message|>"),
        ])
    }

    fn parser(&self, request: &ParseRequest) -> InklingParser {
        InklingParser::new(self.tables.clone(), request.clone(), Some(fixed_ids()))
    }

    /// Server-shaped streaming: groups of `chunk` ids, then one finishing step.
    fn stream_deltas(&self, ids: &[u32], chunk: usize, request: &ParseRequest) -> Vec<Option<DeltaMessage>> {
        let groups: Vec<Vec<u32>> = ids.chunks(chunk).map(<[u32]>::to_vec).collect();
        let prompt = self.prompt();
        let mut parser = self.parser(request);
        steps_of(&self.tok, self.eos(), 100_000, &groups)
            .iter()
            .map(|s| parser.parse_delta(&s.text, &s.ids, &prompt, s.finished))
            .collect()
    }

    fn stream(&self, ids: &[u32], chunk: usize, request: &ParseRequest) -> Collected {
        let mut c = Collected::default();
        for d in self.stream_deltas(ids, chunk, request).into_iter().flatten() {
            c.reasoning.push_str(d.reasoning.text().unwrap_or_default());
            c.content.push_str(d.content.text().unwrap_or_default());
            for tc in &d.tool_calls {
                if let Some(f) = &tc.function {
                    c.names
                        .entry(tc.index)
                        .or_default()
                        .push_str(f.name.as_deref().unwrap_or_default());
                    c.args
                        .entry(tc.index)
                        .or_default()
                        .push_str(f.arguments.as_deref().unwrap_or_default());
                }
            }
        }
        c
    }

    fn nonstream(&self, ids: &[u32], request: &ParseRequest) -> Collected {
        let text: String = steps_of(&self.tok, self.eos(), 100_000, &[ids.to_vec()])
            .iter()
            .map(|s| s.text.clone())
            .collect();
        let parsed = self.parser(request).parse(&text);
        let mut c = Collected {
            reasoning: parsed.reasoning.unwrap_or_default(),
            content: parsed.content.unwrap_or_default(),
            ..Collected::default()
        };
        for (i, tc) in parsed.tool_calls.unwrap_or_default().into_iter().enumerate() {
            c.names.insert(i, tc.name);
            c.args.insert(i, tc.arguments);
        }
        c
    }
}

enum Part<'a> {
    Tok(&'a str),
    Text(&'a str),
}
use Part::{Text, Tok};

#[derive(Debug, Default, PartialEq, Eq)]
struct Collected {
    reasoning: String,
    content: String,
    names: BTreeMap<usize, String>,
    args: BTreeMap<usize, String>,
}

fn request_with_tools() -> ParseRequest {
    ParseRequest {
        tools: vec![
            json!({"type": "function", "function": {"name": "calc", "parameters": {"type": "object",
            "properties": {"expr": {"type": "string"}, "n": {"type": "integer"}}}}}),
        ],
        tool_choice: ToolChoice::Auto,
        include_reasoning: true,
    }
}

#[test]
fn parser_reasoning_then_text() {
    let f = Fixture::new();
    let ids = f.ids(&[
        Tok("<|message_model|>"),
        Tok("<|content_thinking|>"),
        Text("Let me think about a < b.  "),
        Tok("<|end_message|>"),
        Tok("<|message_model|>"),
        Tok("<|content_text|>"),
        Text("Answer: x<y and <b>."),
        Tok("<|end_message|>"),
        Tok("<|content_model_end_sampling|>"),
    ]);
    let reference = f.nonstream(&ids, &ParseRequest::default());
    assert_eq!(reference.reasoning, "Let me think about a < b.");
    assert_eq!(reference.content, "Answer: x<y and <b>.");
    for chunk in [1, 2, 3, 5, 8, 100] {
        assert_eq!(
            f.stream(&ids, chunk, &ParseRequest::default()),
            reference,
            "chunk {chunk}"
        );
    }
}

fn two_call_ids(f: &Fixture) -> Vec<u32> {
    f.ids(&[
        Tok("<|message_model|>"),
        Tok("<|content_thinking|>"),
        Text("Need calc."),
        Tok("<|end_message|>"),
        Tok("<|message_model|>"),
        Text("calc"),
        Tok("<|content_invoke_tool_json|>"),
        Text(r#"{"name":"calc","args":{"expr":"1+1","n":"3"}}"#),
        Tok("<|end_message|>"),
        Tok("<|message_model|>"),
        Text("calc"),
        Tok("<|content_invoke_tool_json|>"),
        Text(r#"{"name":"calc","args":{"expr":"2*{3}"}}"#),
        Tok("<|end_message|>"),
        Tok("<|content_model_end_sampling|>"),
    ])
}

#[test]
fn parser_tool_call_streaming_equals_nonstreaming() {
    let f = Fixture::new();
    let ids = two_call_ids(&f);
    let request = request_with_tools();
    let reference = f.nonstream(&ids, &request);
    assert_eq!(reference.names.len(), 2);
    assert_eq!(reference.names[&0], "calc");
    // "n" is an integer in the tool schema: coerced and re-serialised with Python's default separators
    assert_eq!(reference.args[&0], r#"{"expr": "1+1", "n": 3}"#);
    assert_eq!(reference.args[&1], r#"{"expr":"2*{3}"}"#);
    assert_eq!(reference.reasoning, "Need calc.");
    for chunk in [1, 2, 4, 7, 100] {
        let s = f.stream(&ids, chunk, &request);
        assert_eq!(s.names, reference.names, "chunk {chunk}");
        assert_eq!(s.reasoning, reference.reasoning, "chunk {chunk}");
        assert_eq!(s.args[&1], reference.args[&1], "chunk {chunk}");
    }
}

#[test]
fn parser_truncated_tool_call() {
    let f = Fixture::new();
    let ids = f.ids(&[
        Tok("<|message_model|>"),
        Text("calc"),
        Tok("<|content_invoke_tool_json|>"),
        Text(r#"{"name":"calc","args":{"expr":"1+"#),
    ]);
    let request = request_with_tools();
    assert_eq!(f.stream(&ids, 3, &request).names[&0], "calc");
    let n = f.nonstream(&ids, &request);
    assert_eq!(n.names[&0], "calc");
    assert_eq!(n.args[&0], r#"{"expr":"1+"#);
}

#[test]
fn parser_tool_choice_none_suppresses_calls() {
    let f = Fixture::new();
    let ids = f.ids(&[
        Tok("<|message_model|>"),
        Text("calc"),
        Tok("<|content_invoke_tool_json|>"),
        Text(r#"{"name":"calc","args":{}}"#),
        Tok("<|end_message|>"),
        Tok("<|message_model|>"),
        Tok("<|content_text|>"),
        Text("done"),
        Tok("<|end_message|>"),
        Tok("<|content_model_end_sampling|>"),
    ]);
    let request = ParseRequest {
        tool_choice: ToolChoice::None,
        ..request_with_tools()
    };
    let s = f.stream(&ids, 2, &request);
    assert!(s.names.is_empty());
    assert_eq!(s.content, "done");
    let n = f.nonstream(&ids, &request);
    assert!(n.names.is_empty());
    assert_eq!(n.content, "done");
}

#[test]
fn parser_helpers() {
    let f = Fixture::new();
    let parser = f.parser(&ParseRequest::default());
    let thinking = f.ids(&[
        Tok("<|message_model|>"),
        Tok("<|content_thinking|>"),
        Text("a b c"),
        Tok("<|end_message|>"),
    ]);
    assert_eq!(parser.count_reasoning_tokens(&thinking), f.text("a b c").len());
    assert!(!parser.is_reasoning_end(&f.prompt()));
    assert!(parser.is_reasoning_end(&f.ids(&[Tok("<|message_model|>"), Tok("<|content_text|>"), Text("partial")])));
}

#[test]
fn last_tool_call_arguments_become_empty_at_stream_end() {
    // Config B's `_append_unstreamed_tool_args` sets `arguments: ""` on the last call of the final delta when it has
    // none; the engine's own flush (the call is closed with `{}`) follows as a second fragment.
    let f = Fixture::new();
    let mut ids = f.ids(&[
        Tok("<|message_model|>"),
        Text("calc"),
        Tok("<|content_invoke_tool_json|>"),
        Text(r#"{"name":"calc""#),
    ]);
    ids.push(f.eos());
    let deltas = f.stream_deltas(&ids, 1000, &request_with_tools());
    let last = deltas.last().expect("one step").as_ref().expect("a delta");
    assert_eq!(
        last.to_value(),
        json!({"tool_calls": [
            {"id": "chatcmpl-tool-0000000000000000", "type": "function", "index": 0,
             "function": {"name": "calc", "arguments": ""}},
            {"index": 0, "function": {"arguments": "{}"}}]})
    );
}

#[test]
fn empty_tool_call_lists_are_never_serialised() {
    let mut d = DeltaMessage::default();
    d.content.set(Some("x".into()));
    assert_eq!(d.to_value(), json!({"content": "x"}));
    assert_eq!(DeltaMessage::default().to_value(), json!({}));
}

#[test]
fn unset_and_null_fields_serialise_differently_and_keep_order() {
    let mut d = DeltaMessage {
        role: Some("assistant".into()),
        ..DeltaMessage::default()
    };
    d.reasoning.set(None);
    d.content.set(Some("c".into()));
    let v = d.to_value();
    assert_eq!(v, json!({"role": "assistant", "content": "c", "reasoning": null}));
    assert_eq!(
        v.as_object().expect("object").keys().collect::<Vec<_>>(),
        ["role", "content", "reasoning"]
    );
    assert_eq!(DeltaMessage::default().content, TextField::Unset);
}

#[test]
fn include_reasoning_false_writes_null_reasoning_and_drops_reasoning_only_steps() {
    let f = Fixture::new();
    let ids = f.ids(&[
        Tok("<|message_model|>"),
        Tok("<|content_thinking|>"),
        Text("hidden"),
        Tok("<|end_message|>"),
        Tok("<|message_model|>"),
        Tok("<|content_text|>"),
        Text("shown"),
        Tok("<|end_message|>"),
        Tok("<|content_model_end_sampling|>"),
    ]);
    let request = ParseRequest {
        include_reasoning: false,
        ..ParseRequest::default()
    };
    let values: Vec<Value> = f
        .stream_deltas(&ids, 1, &request)
        .into_iter()
        .flatten()
        .map(|d| d.to_value())
        .collect();
    assert!(
        values.iter().all(|v| v.get("reasoning") == Some(&Value::Null)),
        "{values:?}"
    );
    let content: String = values.iter().filter_map(|v| v["content"].as_str()).collect();
    assert_eq!(content, "shown");
}

#[test]
fn whitespace_only_content_before_tool_calls_is_dropped() {
    let f = Fixture::new();
    let ids = f.ids(&[
        Tok("<|message_model|>"),
        Tok("<|content_text|>"),
        Text("  \n"),
        Tok("<|end_message|>"),
        Tok("<|message_model|>"),
        Text("calc"),
        Tok("<|content_invoke_tool_json|>"),
        Text(r#"{"name":"calc","args":{"expr":"x"}}"#),
        Tok("<|end_message|>"),
        Tok("<|content_model_end_sampling|>"),
    ]);
    let request = request_with_tools();
    for chunk in [1, 4, 100] {
        let s = f.stream(&ids, chunk, &request);
        assert_eq!(s.content, "", "chunk {chunk}");
        assert_eq!(s.names[&0], "calc", "chunk {chunk}");
    }
    let n = f.nonstream(&ids, &request);
    assert_eq!(n.content, "");
    assert_eq!(n.args[&0], r#"{"expr":"x"}"#);
}

#[test]
fn required_tool_choice_without_a_call_reports_an_empty_call_list() {
    let f = Fixture::new();
    let ids = f.ids(&[
        Tok("<|message_model|>"),
        Tok("<|content_thinking|>"),
        Text("thinking only"),
        Tok("<|content_model_end_sampling|>"),
    ]);
    let mut request = request_with_tools();
    request.tool_choice = ToolChoice::Required;
    let text: String = steps_of(&f.tok, f.eos(), 1000, std::slice::from_ref(&ids))
        .iter()
        .map(|s| s.text.clone())
        .collect();
    let parsed = f.parser(&request).parse(&text);
    assert_eq!(parsed.tool_calls, Some(vec![]));
    assert_eq!(parsed.content, None);
    assert_eq!(parsed.reasoning.as_deref(), Some("thinking only"));
    request.tool_choice = ToolChoice::Auto;
    assert_eq!(f.parser(&request).parse(&text).tool_calls, None);
}

#[test]
fn special_tokens_outside_the_grammar_are_dropped_by_id() {
    let f = Fixture::new();
    let ids = f.ids(&[
        Tok("<|message_model|>"),
        Tok("<|content_text|>"),
        Text("a"),
        Tok("<|content_image|>"),
        Text("b"),
        Tok("<|end_message|>"),
        Tok("<|content_model_end_sampling|>"),
    ]);
    for chunk in [1, 3, 100] {
        assert_eq!(
            f.stream(&ids, chunk, &ParseRequest::default()).content,
            "ab",
            "chunk {chunk}"
        );
    }
}

#[test]
fn non_ascii_arguments_stream_and_keep_their_characters() {
    let f = Fixture::new();
    let ids = f.ids(&[
        Tok("<|message_model|>"),
        Text("calc"),
        Tok("<|content_invoke_tool_json|>"),
        Text(r#"{"name":"calc","args":{"expr":"héllo ✓","n":"7"}}"#),
        Tok("<|end_message|>"),
        Tok("<|content_model_end_sampling|>"),
    ]);
    let request = request_with_tools();
    assert_eq!(f.nonstream(&ids, &request).args[&0], r#"{"expr": "héllo ✓", "n": 7}"#);
    assert_eq!(
        f.stream(&ids, 1, &request).args[&0],
        f.stream(&ids, 100, &request).args[&0]
    );
}

#[test]
fn integers_beyond_64_bits_become_floating_point_when_coercion_rewrites_arguments() {
    // Known limit, shared with the C++ port: serde_json has no `arbitrary_precision` here, Python keeps the digits.
    let f = Fixture::new();
    let ids = f.ids(&[
        Tok("<|message_model|>"),
        Text("calc"),
        Tok("<|content_invoke_tool_json|>"),
        Text(r#"{"name":"calc","args":{"n":"3","big":123456789012345678901234567890}}"#),
        Tok("<|end_message|>"),
        Tok("<|content_model_end_sampling|>"),
    ]);
    let args = &f.nonstream(&ids, &request_with_tools()).args[&0];
    assert_eq!(args, r#"{"n": 3, "big": 1.2345678901234568e+29}"#);
}
