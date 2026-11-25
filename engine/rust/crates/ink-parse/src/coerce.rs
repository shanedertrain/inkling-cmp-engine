// Ported from engine/server/src/parse/inkling_parser.cpp (schema coercion) @ 6100ab8, itself a port of vLLM's
// vllm/tool_parsers/utils.py (type coercion) and vllm/parser/engine/parser_engine.py (`_coerce_value`) (Apache-2.0,
// Copyright contributors to the vLLM project).
//! Coercion of generated tool arguments to the types of the tool's JSON schema.
//!
//! Limits shared with the C++ port: integers beyond `[-2^63, 2^64)` are read as floating point by `serde_json`
//! (`arbitrary_precision` is not enabled in this tree), where Python keeps them exact; `NaN` and `Infinity` literals
//! are not JSON to `serde_json`, where Python reads them.

use std::borrow::Cow;
use std::collections::BTreeSet;

use ink_text::pyfmt::{self, NonAscii, Separators};
use ink_text::pyjson;
use serde_json::{Map, Number, Value};

use crate::args::is_py_space;

fn aliased(name: &str) -> &str {
    match name {
        "str" | "text" | "varchar" | "char" | "enum" => "string",
        "int" | "int32" | "int64" | "uint" | "uint32" | "uint64" | "long" | "short" | "unsigned" => "integer",
        "float" | "float32" | "float64" | "double" => "number",
        "bool" => "boolean",
        "dict" => "object",
        "arr" | "list" | "sequence" => "array",
        other => other,
    }
}

/// The JSON-schema types a property accepts: `type`, the types of `enum` values, and those of `anyOf`/`oneOf`/`allOf`
/// members; `{"string"}` when none is declared.
pub(crate) fn extract_types(schema: &Value) -> BTreeSet<String> {
    let Value::Object(schema) = schema else {
        return BTreeSet::from(["string".to_owned()]);
    };
    let mut types = BTreeSet::new();
    match schema.get("type") {
        Some(Value::String(t)) => {
            types.insert(t.clone());
        }
        Some(Value::Array(ts)) => types.extend(ts.iter().filter_map(Value::as_str).map(str::to_owned)),
        _ => {}
    }
    if let Some(Value::Array(values)) = schema.get("enum") {
        for v in values {
            let t = match v {
                Value::Null => "null",
                Value::Bool(_) => "boolean",
                Value::Number(n) if n.is_f64() => "number",
                Value::Number(_) => "integer",
                Value::String(_) => "string",
                Value::Array(_) => "array",
                Value::Object(_) => "object",
            };
            types.insert(t.to_owned());
        }
    }
    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(Value::Array(members)) = schema.get(key) {
            for m in members {
                types.extend(extract_types(m));
            }
        }
    }
    if types.is_empty() {
        types.insert("string".to_owned());
    }
    types
}

fn parse_json(text: &str) -> Option<Value> {
    serde_json::from_str(text).ok()
}

fn is_finite(value: &Value) -> bool {
    match value {
        Value::Number(n) => n.as_f64().is_none_or(f64::is_finite),
        Value::Array(a) => a.iter().all(is_finite),
        Value::Object(o) => o.values().all(is_finite),
        _ => true,
    }
}

/// `coerce_to_schema_type`: the value as the first matching schema type reads it; `None` keeps the string.
fn coerce_to_schema_type(value: &str, schema_types: &BTreeSet<String>) -> Option<Value> {
    let normalised: BTreeSet<String> = schema_types
        .iter()
        .map(|t| aliased(&pyfmt::py_strip(t).to_lowercase()).to_owned())
        .collect();
    for candidate in ["null", "integer", "number", "boolean", "object", "array", "string"] {
        if !normalised.contains(candidate) {
            continue;
        }
        match candidate {
            "null" => {
                if value.to_lowercase() == "null" {
                    return Some(Value::Null);
                }
            }
            "string" => return None,
            "integer" => {
                if let Some(n) = pyfmt::parse_int(value) {
                    return Some(Value::Number(n));
                }
            }
            "number" => {
                if let Some(n) = pyfmt::parse_int(value) {
                    return Some(Value::Number(n));
                }
                let Some(f) = pyjson::parse_float(value).filter(|f| f.is_finite()) else {
                    continue;
                };
                let t = f.trunc();
                if f != t || t.abs() >= 9.2e18 {
                    return Number::from_f64(f).map(Value::Number);
                }
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "t is integral and below 9.2e18 in magnitude"
                )]
                return Some(Value::from(t as i64));
            }
            "boolean" => match pyfmt::py_strip(value).to_lowercase().as_str() {
                "true" | "1" => return Some(Value::Bool(true)),
                "false" | "0" => return Some(Value::Bool(false)),
                _ => {}
            },
            _ => {
                if let Some(p) = parse_json(value).filter(is_finite) {
                    return Some(p);
                }
            }
        }
    }
    parse_json(value).filter(is_finite)
}

/// Python `==` on JSON values, where numbers (and booleans) compare by value.
fn py_equal(a: &Value, b: &Value) -> bool {
    let num = |v: &Value| match v {
        Value::Bool(x) => Some(f64::from(u8::from(*x))),
        Value::Number(n) => n.as_f64(),
        _ => None,
    };
    match (a, b) {
        (Value::Number(x), Value::Number(y)) if !x.is_f64() && !y.is_f64() => x == y,
        _ => match (num(a), num(b)) {
            (Some(x), Some(y)) => x == y,
            _ => a == b,
        },
    }
}

/// `type(x)` identity for the coercion check.
fn type_tag(v: &Value) -> u8 {
    match v {
        Value::Null => 0,
        Value::Bool(_) => 1,
        Value::Number(n) if !n.is_f64() => 2,
        Value::Number(_) => 3,
        Value::String(_) => 4,
        Value::Array(_) => 5,
        Value::Object(_) => 6,
    }
}

/// `_coerce_value`: coerces `value` in place to `schema`; whether it changed.
fn coerce_value(value: &mut Value, schema: &Value) -> bool {
    match value {
        Value::String(s) => match coerce_to_schema_type(s, &extract_types(schema)) {
            Some(c) => {
                *value = c;
                true
            }
            None => false,
        },
        Value::Object(map) => match schema.get("properties") {
            Some(Value::Object(props)) => coerce_dict(map, props),
            _ => false,
        },
        Value::Array(items) => match schema.get("items") {
            Some(item_schema @ Value::Object(_)) => {
                let mut changed = false;
                for item in items {
                    changed |= coerce_value(item, item_schema);
                }
                changed
            }
            _ => false,
        },
        _ => {
            let as_str = pyfmt::dumps(value, NonAscii::Keep, Separators::Spaced);
            let coerced = coerce_to_schema_type(&as_str, &extract_types(schema)).unwrap_or(Value::String(as_str));
            if type_tag(&coerced) != type_tag(value) || !py_equal(&coerced, value) {
                *value = coerced;
                true
            } else {
                false
            }
        }
    }
}

fn coerce_dict(args: &mut Map<String, Value>, properties: &Map<String, Value>) -> bool {
    let mut changed = false;
    for (key, value) in args.iter_mut() {
        if let Some(prop @ Value::Object(_)) = properties.get(key) {
            changed |= coerce_value(value, prop);
        }
    }
    changed
}

/// `find_tool_properties`: the `properties` of the first tool called `name` (empty when it declares none).
pub(crate) fn tool_properties<'a>(tools: &'a [Value], name: &str) -> Option<&'a Map<String, Value>> {
    for tool in tools {
        let Some(function) = tool.get("function").filter(|f| f.is_object()) else {
            continue;
        };
        if function.get("name").and_then(Value::as_str) == Some(name) {
            return function
                .get("parameters")
                .and_then(|p| p.get("properties"))
                .and_then(Value::as_object);
        }
    }
    None
}

/// Keys of `properties` whose schema is a plain string: their values stream character by character. `None` when the
/// tool declares no properties (every string value streams).
pub(crate) fn streamable_string_keys(properties: Option<&Map<String, Value>>) -> Option<BTreeSet<String>> {
    let props = properties.filter(|p| !p.is_empty())?;
    let only_string = BTreeSet::from(["string".to_owned()]);
    Some(
        props
            .iter()
            .filter(|(_, schema)| extract_types(schema) == only_string)
            .map(|(k, _)| k.clone())
            .collect(),
    )
}

/// `fix_arg_types`: the arguments JSON with its values coerced to the tool's schema, re-serialised as Python's
/// `json.dumps` writes it when anything changed, else untouched.
pub(crate) fn fix_arg_types<'a>(args_json: &'a str, name: &str, tools: &[Value]) -> Cow<'a, str> {
    if tools.is_empty() || name.is_empty() {
        return Cow::Borrowed(args_json);
    }
    let Some(Value::Object(mut args)) = parse_json(args_json) else {
        return Cow::Borrowed(args_json);
    };
    let Some(props) = tool_properties(tools, name).filter(|p| !p.is_empty()) else {
        return Cow::Borrowed(args_json);
    };
    if coerce_dict(&mut args, props) {
        Cow::Owned(pyfmt::dumps(&Value::Object(args), NonAscii::Keep, Separators::Spaced))
    } else {
        Cow::Borrowed(args_json)
    }
}

/// Python `str.rstrip()`.
pub(crate) fn py_rstrip(text: &str) -> &str {
    text.trim_end_matches(is_py_space)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn tools() -> Vec<Value> {
        vec![
            json!({"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
            "i": {"type": "integer"}, "n": {"type": "number"}, "b": {"type": "boolean"}, "s": {"type": "string"},
            "o": {"type": "object", "properties": {"k": {"type": "integer"}}},
            "a": {"type": "array", "items": {"type": "integer"}}, "u": {"type": ["null", "integer"]}}}}}),
        ]
    }

    fn fixed(args: &str) -> String {
        fix_arg_types(args, "f", &tools()).into_owned()
    }

    #[test]
    fn strings_become_schema_types_and_respell_as_python() {
        assert_eq!(
            fixed(r#"{"i":"3","n":"2.5","b":"True","s":"x"}"#),
            r#"{"i": 3, "n": 2.5, "b": true, "s": "x"}"#
        );
    }

    #[test]
    fn untouched_arguments_keep_their_spelling() {
        let args = r#"{"s":"a","i":4}"#;
        assert!(matches!(fix_arg_types(args, "f", &tools()), Cow::Borrowed(a) if a == args));
    }

    #[test]
    fn nested_values_follow_properties_and_items() {
        assert_eq!(
            fixed(r#"{"o":{"k":"7"},"a":["1","2"]}"#),
            r#"{"o": {"k": 7}, "a": [1, 2]}"#
        );
    }

    #[test]
    fn integral_floats_and_big_values() {
        assert_eq!(fixed(r#"{"n":"4.0"}"#), r#"{"n": 4}"#);
        assert_eq!(fixed(r#"{"n":"1e300"}"#), r#"{"n": 1e+300}"#);
        assert_eq!(fixed(r#"{"u":"null"}"#), r#"{"u": null}"#);
    }

    #[test]
    fn native_numbers_are_coerced_through_their_text() {
        assert_eq!(fixed(r#"{"s":5}"#), r#"{"s": "5"}"#);
        assert_eq!(fixed(r#"{"i":5}"#), r#"{"i":5}"#);
    }

    #[test]
    fn unknown_tool_or_bad_json_is_untouched() {
        assert_eq!(fix_arg_types(r#"{"i":"3"}"#, "g", &tools()), r#"{"i":"3"}"#);
        assert_eq!(fix_arg_types(r#"{"i":"3""#, "f", &tools()), r#"{"i":"3""#);
    }

    #[test]
    fn streamable_keys_are_the_plain_string_properties() {
        let tools = tools();
        let keys = streamable_string_keys(tool_properties(&tools, "f")).unwrap();
        assert_eq!(keys.into_iter().collect::<Vec<_>>(), ["s"]);
        assert!(streamable_string_keys(tool_properties(&tools, "missing")).is_none());
    }

    #[test]
    fn rstrip_removes_python_white_space() {
        assert_eq!(py_rstrip("a b \u{3000}\n\u{1f}"), "a b");
    }
}
