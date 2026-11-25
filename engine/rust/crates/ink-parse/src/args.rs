// Ported from engine/server/src/parse/inkling_parser.cpp (`safe_arg_prefix`, `scan_json_object_end`,
// `extract_json_member_object`, `search_name`) @ 6100ab8, itself a port of vLLM's vllm/parser/utils.py
// (`extract_json_member_object`) and vllm/parser/engine/parser_engine.py (Apache-2.0, Copyright contributors to the
// vLLM project).
//! Scanning of the generated tool-call JSON (`{"name": ..., "args": {...}}`) that may still be incomplete.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

/// A member that must hold a JSON object holds something else (Python's `ValueError`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemberNotObject {
    key: String,
}

impl fmt::Display for MemberNotObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "member '{}' must be a JSON object", self.key)
    }
}

impl Error for MemberNotObject {}

const fn is_json_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

/// End (exclusive) of the object that starts at `start`, or `None` while it is still open.
fn object_end(raw: &[u8], start: usize) -> Option<usize> {
    let (mut depth, mut in_string, mut escape) = (0i32, false, false);
    for (i, &b) in raw.iter().enumerate().skip(start) {
        if escape {
            escape = false;
        } else if in_string {
            match b {
                b'\\' => escape = true,
                b'"' => in_string = false,
                _ => {}
            }
        } else {
            match b {
                b'"' => in_string = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i + 1);
                    }
                }
                _ => {}
            }
        }
    }
    None
}

/// The text of the top-level member `key` of the (possibly truncated) JSON object `raw`, which must itself be an
/// object: the whole value when it is closed, the rest of `raw` when it is still open. `None` when the member is
/// absent or has no value yet.
///
/// # Errors
/// The member's value starts with something other than `{`.
pub fn extract_json_member_object<'a>(raw: &'a str, key: &str) -> Result<Option<&'a str>, MemberNotObject> {
    let bytes = raw.as_bytes();
    let (mut depth, mut in_string, mut escape) = (0i32, false, false);
    let mut string_start = 0;
    let mut last_string: Option<&str> = None;
    for (i, &b) in bytes.iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        if in_string {
            match b {
                b'\\' => escape = true,
                b'"' => {
                    in_string = false;
                    if depth == 1 {
                        last_string = Some(&raw[string_start + 1..i]);
                    }
                }
                _ => {}
            }
            continue;
        }
        match b {
            b'"' => {
                in_string = true;
                string_start = i;
            }
            b':' if depth == 1 && last_string == Some(key) => {
                let value_start = bytes[i + 1..]
                    .iter()
                    .position(|&c| !is_json_space(c))
                    .map(|p| i + 1 + p);
                let Some(vs) = value_start else { return Ok(None) };
                if bytes[vs] != b'{' {
                    return Err(MemberNotObject { key: key.to_owned() });
                }
                return Ok(Some(match object_end(bytes, vs) {
                    Some(end) => &raw[vs..end],
                    None => &raw[vs..],
                }));
            }
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth -= 1,
            _ => {}
        }
    }
    Ok(None)
}

/// `_inkling_arg_converter`: the `args` object of a tool call's raw JSON; `""` (partial) or `"{}"` when absent.
pub(crate) fn arg_converter(raw: &str, partial: bool) -> Result<&str, MemberNotObject> {
    Ok(extract_json_member_object(raw, "args")?.unwrap_or(if partial { "" } else { "{}" }))
}

/// The longest prefix of `js` (an incomplete JSON object) that can be streamed without ever having to be taken back:
/// up to the last top-level colon, extended into a string value when its key is one of `string_keys` (all keys when
/// `None`).
pub(crate) fn safe_arg_prefix<'a>(js: &'a str, string_keys: Option<&BTreeSet<String>>) -> &'a str {
    let bytes = js.as_bytes();
    let mut last_colon: Option<usize> = None;
    let (mut last_key, mut pending_key): (Option<&str>, Option<&str>) = (None, None);
    let (mut in_string, mut escape) = (false, false);
    let mut string_start = 0;
    let mut depth = 0i32;
    for (i, &c) in bytes.iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        if in_string {
            match c {
                b'\\' => escape = true,
                b'"' => {
                    in_string = false;
                    if depth == 1 {
                        pending_key = Some(&js[string_start + 1..i]);
                    }
                }
                _ => {}
            }
            continue;
        }
        match c {
            b'"' => {
                in_string = true;
                string_start = i;
            }
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth -= 1,
            b':' if depth == 1 => {
                last_colon = Some(i);
                last_key = pending_key.take();
            }
            _ => {}
        }
    }
    let Some(colon) = last_colon else { return "" };
    let mut end = colon + 1;
    while end < bytes.len() && is_json_space(bytes[end]) {
        end += 1;
    }
    if end >= bytes.len() || bytes[end] != b'"' {
        return &js[..end];
    }
    if string_keys.is_some_and(|keys| !last_key.is_some_and(|k| keys.contains(k))) {
        return &js[..end];
    }
    let mut escape = false;
    for (i, &c) in bytes.iter().enumerate().skip(end + 1) {
        if escape {
            escape = false;
        } else if c == b'\\' {
            escape = true;
        } else if c == b'"' {
            return &js[..i];
        }
    }
    js
}

/// The tool name in a raw call: the first match of the regular expression `"name"\s*:\s*"([^"]*)"`, `\s` being
/// Python's Unicode white space.
pub(crate) fn search_name(text: &str) -> Option<&str> {
    let mut from = 0;
    while let Some(found) = text[from..].find("\"name\"") {
        let p = from + found;
        let after = skip_py_space(text, p + 6);
        if text[after..].starts_with(':') {
            let value = skip_py_space(text, after + 1);
            if text[value..].starts_with('"')
                && let Some(len) = text[value + 1..].find('"')
            {
                return Some(&text[value + 1..value + 1 + len]);
            }
        }
        from = p + 1;
    }
    None
}

fn skip_py_space(text: &str, from: usize) -> usize {
    let rest = &text[from..];
    from + (rest.len() - rest.trim_start_matches(is_py_space).len())
}

/// Python's `str.isspace` for one character: White_Space plus U+001C..U+001F.
pub(crate) fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn member_object_closed_open_and_absent() {
        assert_eq!(
            extract_json_member_object(r#"{"name":"a","args":{"x":"}"}}"#, "args"),
            Ok(Some(r#"{"x":"}"}"#))
        );
        assert_eq!(
            extract_json_member_object(r#"{"name":"a","args":{"x":"#, "args"),
            Ok(Some(r#"{"x":"#))
        );
        assert_eq!(extract_json_member_object(r#"{"name":"a""#, "args"), Ok(None));
        assert_eq!(extract_json_member_object(r#"{"args":   "#, "args"), Ok(None));
    }

    #[test]
    fn member_object_rejects_non_object_value() {
        let err = extract_json_member_object(r#"{"args": 5}"#, "args").unwrap_err();
        assert_eq!(err.to_string(), "member 'args' must be a JSON object");
    }

    #[test]
    fn member_object_ignores_nested_keys() {
        assert_eq!(
            extract_json_member_object(r#"{"x":{"args":{"a":1}},"args":{"b":2}}"#, "args"),
            Ok(Some(r#"{"b":2}"#))
        );
    }

    #[test]
    fn arg_converter_defaults() {
        assert_eq!(arg_converter(r#"{"name":"a"}"#, true), Ok(""));
        assert_eq!(arg_converter(r#"{"name":"a"}"#, false), Ok("{}"));
    }

    #[test]
    fn safe_prefix_streams_string_values_only_for_string_keys() {
        let keys: BTreeSet<String> = ["s".to_owned()].into();
        assert_eq!(safe_arg_prefix(r#"{"s":"ab"#, Some(&keys)), r#"{"s":"ab"#);
        assert_eq!(safe_arg_prefix(r#"{"n":"1"#, Some(&keys)), r#"{"n":"#);
        assert_eq!(safe_arg_prefix(r#"{"n":"1"#, None), r#"{"n":"1"#);
        assert_eq!(safe_arg_prefix(r#"{"s""#, None), "");
        assert_eq!(safe_arg_prefix(r#"{"s": "a\"b"#, None), r#"{"s": "a\"b"#);
    }

    #[test]
    fn name_search_skips_unquoted_and_non_matching_candidates() {
        assert_eq!(search_name(r#"{"name" : "calc","args":{}}"#), Some("calc"));
        assert_eq!(search_name(r#"{"x":"name","name":"b"}"#), Some("b"));
        assert_eq!(search_name(r#"{"name":"unterminated"#), None);
        assert_eq!(search_name(r#"{"name":5}"#), None);
        assert_eq!(search_name("{\"name\"\u{3000}:\u{2003}\"u\"}"), Some("u"));
    }
}
