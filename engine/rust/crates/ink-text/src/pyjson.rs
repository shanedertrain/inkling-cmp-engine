// Ported from engine/server/src/text/pyjson.{h,cpp} @ 4171493 (adapted there from ingine @ 306eaa8, Apache-2.0,
// Copyright 2026 satellitedown). `json.loads` is not ported: callers use `serde_json`.
//! Python `repr`, `float()` and truthiness of JSON values.

use serde_json::{Number, Value};

/// `repr(float)`: shortest round-trip digits, positional for exponents -4..16, else `1.5e+16`.
#[must_use]
pub fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value.is_infinite() {
        return if value < 0.0 { "-inf" } else { "inf" }.into();
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    let sci = shortest_digits(value);
    let (mantissa, exp) = sci.split_once('e').expect("`{:e}` of a finite float has an exponent");
    let exp10: i32 = exp.parse().expect("`{:e}` exponent is an integer");
    let (sign, mantissa) = mantissa.strip_prefix('-').map_or(("", mantissa), |m| ("-", m));
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if (-4..16).contains(&exp10) {
        return match usize::try_from(exp10) {
            Ok(exp) if digits.len() <= exp + 1 => format!("{sign}{digits}{}.0", "0".repeat(exp + 1 - digits.len())),
            Ok(exp) => format!("{sign}{}.{}", &digits[..=exp], &digits[exp + 1..]),
            Err(_) => format!(
                "{sign}0.{}{digits}",
                "0".repeat(usize::try_from(-exp10 - 1).unwrap_or(0))
            ),
        };
    }
    let (head, tail) = digits.split_at(1);
    let point = if tail.is_empty() { "" } else { "." };
    let exp_sign = if exp10 < 0 { '-' } else { '+' };
    format!("{sign}{head}{point}{tail}e{exp_sign}{:02}", exp10.abs())
}

/// The shortest digits that read back as `value`, as `{:e}` spells them, with Python's choice among the candidates:
/// the one closest to the exact value, and half-even when two are equally close. Rust's shortest-digits routine
/// rounds such a tie up (an `f32` widened to `f64` often ends in an exact `5` one digit past the 17th).
fn shortest_digits(value: f64) -> String {
    let shortest = format!("{value:e}");
    let digits = shortest
        .split_once('e')
        .map_or(0, |(mantissa, _)| mantissa.chars().filter(char::is_ascii_digit).count());
    // exact rounding to the same number of digits; it is the closest candidate when it reads back as `value`
    let exact = format!("{value:.*e}", digits.saturating_sub(1));
    if exact != shortest && exact.parse::<f64>() == Ok(value) {
        exact
    } else {
        shortest
    }
}

/// `repr(str)`. Printable means "not a control character and not white space other than a plain space";
/// Python's `isprintable` also rejects format (Cf), unassigned and private-use characters, which std cannot tell apart.
#[must_use]
pub fn str_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for c in text.chars() {
        match c {
            _ if c == quote || c == '\\' => {
                out.push('\\');
                out.push(c);
            }
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            ' '..='~' => out.push(c),
            _ if c.is_ascii() || c.is_control() || c.is_whitespace() => escape_hex(&mut out, c),
            _ => out.push(c),
        }
    }
    out.push(quote);
    out
}

fn escape_hex(out: &mut String, c: char) {
    use std::fmt::Write as _;
    let v = u32::from(c);
    let _ = match v {
        0..=0xFF => write!(out, "\\x{v:02x}"),
        0x100..=0xFFFF => write!(out, "\\u{v:04x}"),
        _ => write!(out, "\\U{v:08x}"),
    };
}

/// `repr()` of the Python object `json.loads` would produce from `value`.
#[must_use]
pub fn repr(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) => number_repr(n),
        Value::String(s) => str_repr(s),
        Value::Array(items) => {
            let items: Vec<String> = items.iter().map(repr).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Object(map) => {
            let items: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", str_repr(k), repr(v)))
                .collect();
            format!("{{{}}}", items.join(", "))
        }
    }
}

/// `repr` of a JSON number: integers as written, floats through [`float_repr`].
#[must_use]
pub fn number_repr(n: &Number) -> String {
    if n.is_f64() {
        float_repr(n.as_f64().unwrap_or(f64::NAN))
    } else {
        n.to_string()
    }
}

/// Python `float(text)` for a `str`; `None` where Python raises `ValueError`.
///
/// Leading and trailing white space is stripped, single underscores between digits are dropped, `inf`/`nan` spellings
/// are accepted. Non-ASCII decimal digits, which Python accepts, are rejected.
#[must_use]
pub fn parse_float(text: &str) -> Option<f64> {
    let stripped = crate::pyfmt::py_strip(text);
    let cleaned = drop_digit_underscores(stripped)?;
    cleaned.parse::<f64>().ok()
}

/// Removes `_` that sit between two ASCII digits; `None` for any other underscore (PEP 515).
pub(crate) fn drop_digit_underscores(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    for (i, c) in s.char_indices() {
        if c == '_' {
            let between = i > 0 && bytes[i - 1].is_ascii_digit() && bytes.get(i + 1).is_some_and(u8::is_ascii_digit);
            if !between {
                return None;
            }
        } else {
            out.push(c);
        }
    }
    Some(out)
}

/// Python truthiness of a JSON value.
#[must_use]
pub fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_none_or(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn float_repr_matches_python() {
        let cases: &[(f64, &str)] = &[
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (2.5, "2.5"),
            (0.1, "0.1"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (123_456_789_012_345_678.0, "1.2345678901234568e+17"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (-12.375, "-12.375"),
            (1e100, "1e+100"),
            (f64::MAX, "1.7976931348623157e+308"),
            (5e-324, "5e-324"),
            // f32 values widened to f64: the 17th digit is a tie, which Python rounds to even
            (-22.065_628_051_757_813, "-22.065628051757812"),
            (-22.210_586_547_851_563, "-22.210586547851562"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
        ];
        for (v, want) in cases {
            assert_eq!(float_repr(*v), *want, "value {v:e}");
        }
        assert_eq!(float_repr(f64::NAN), "nan");
    }

    #[test]
    fn repr_of_values() {
        let v = json!({"a": [1, 2.0, "it's", null, true], "b\"": {"c": "x\ny\u{7f}é\u{1f600}\u{a0}"}});
        assert_eq!(
            repr(&v),
            "{'a': [1, 2.0, \"it's\", None, True], 'b\"': {'c': 'x\\ny\\x7fé\u{1f600}\\xa0'}}"
        );
        assert_eq!(str_repr("a'b\"c"), "'a\\'b\"c'");
        assert_eq!(str_repr("\u{2028}\u{0}"), "'\\u2028\\x00'");
        assert_eq!(repr(&json!(18_446_744_073_709_551_615_u64)), "18446744073709551615");
    }

    #[test]
    fn parse_float_matches_python() {
        let ok: &[(&str, f64)] = &[
            ("0.35", 0.35),
            (" 1_000.5 ", 1000.5),
            ("1e3", 1000.0),
            (".5", 0.5),
            ("5.", 5.0),
            ("-INF", f64::NEG_INFINITY),
            ("\u{1c}7\u{3000}", 7.0),
        ];
        for (s, want) in ok {
            assert_eq!(parse_float(s), Some(*want), "input {s:?}");
        }
        assert!(parse_float("nan").is_some_and(f64::is_nan));
        for s in [
            "", " ", "1__0", "_1", "1_", "1_.5", "1e", "e5", ".", "0x10", "1,5", "12abc", "٣",
        ] {
            assert_eq!(parse_float(s), None, "input {s:?}");
        }
    }

    #[test]
    fn truthiness() {
        assert!(!truthy(&json!(null)) && !truthy(&json!(0)) && !truthy(&json!(0.0)) && !truthy(&json!("")));
        assert!(!truthy(&json!([])) && !truthy(&json!({})) && !truthy(&json!(false)));
        assert!(truthy(&json!(-1)) && truthy(&json!(" ")) && truthy(&json!([0])) && truthy(&json!({"a": 0})));
    }
}
