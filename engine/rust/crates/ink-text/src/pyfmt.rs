// Ported from engine/server/src/text/pyfmt.{h,cpp} @ 4171493. The Inkling2 JSON spelling is a port of
// vllm/renderers/inkling_encoding.py (_tml_json_dumps), Apache-2.0, Copyright contributors to the vLLM project.
//! Python value formatting the renderer and parser put on the wire: `json.dumps`, the Inkling2 `_tml_json_dumps`,
//! `int()`, `round()` and `%g`.

use std::fmt::Write as _;

use serde_json::{Number, Value};

use crate::pyjson::{drop_digit_underscores, float_repr};

/// Whether `json.dumps` escapes non-ASCII characters (`ensure_ascii`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NonAscii {
    /// `\uXXXX` escapes, surrogate pairs above the BMP.
    Escape,
    /// Characters are written as they are.
    Keep,
}

/// Item and key separators of `json.dumps`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Separators {
    /// `(",", ":")`.
    Compact,
    /// The default `(", ", ": ")`.
    Spaced,
}

/// Exponent spelling of [`tml_float`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exponent {
    /// `1e16`, `1.5e-7` (`legacy_floats=True`).
    Bare,
    /// `1e+16`, as Python's `repr`.
    Signed,
}

/// Key order of [`tml_dumps`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyOrder {
    /// Keys sorted by code point.
    Sorted,
    /// Insertion order.
    Insertion,
}

/// `json.dumps(value, ensure_ascii=..., separators=...)`. NaN and infinities cannot occur in a [`Value`].
///
/// With [`NonAscii::Escape`] everything outside `' '..='~'` is escaped, U+007F included, as Python's
/// `ESCAPE_ASCII` does; [`NonAscii::Keep`] writes U+007F raw, as `ESCAPE` does.
#[must_use]
pub fn dumps(value: &Value, non_ascii: NonAscii, separators: Separators) -> String {
    let mut out = String::new();
    dump_any(&mut out, value, non_ascii, separators);
    out
}

/// `json.dumps(s, ensure_ascii=...)` of one string.
#[must_use]
pub fn dumps_string(s: &str, non_ascii: NonAscii) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    dump_str(&mut out, s, non_ascii);
    out
}

fn dump_str(out: &mut String, s: &str, non_ascii: NonAscii) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            _ if c < ' ' => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            _ if non_ascii == NonAscii::Keep || c <= '~' => out.push(c),
            _ => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
}

fn dump_any(out: &mut String, v: &Value, non_ascii: NonAscii, sep: Separators) {
    let (item_sep, key_sep) = match sep {
        Separators::Compact => (",", ":"),
        Separators::Spaced => (", ", ": "),
    };
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&number_json(n)),
        Value::String(s) => dump_str(out, s, non_ascii),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                dump_any(out, item, non_ascii, sep);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (key, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                dump_str(out, key, non_ascii);
                out.push_str(key_sep);
                dump_any(out, item, non_ascii, sep);
            }
            out.push('}');
        }
    }
}

fn number_json(n: &Number) -> String {
    crate::pyjson::number_repr(n)
}

/// `_tml_json_dumps(value, legacy_floats=..., sort_keys=...)`: compact separators, no ASCII escaping, floats spelled
/// by [`tml_float`].
#[must_use]
pub fn tml_dumps(value: &Value, exponent: Exponent, order: KeyOrder) -> String {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            if order == KeyOrder::Sorted {
                entries.sort_by(|a, b| a.0.cmp(b.0));
            }
            let items: Vec<String> = entries
                .into_iter()
                .map(|(k, v)| format!("{}:{}", dumps_string(k, NonAscii::Keep), tml_dumps(v, exponent, order)))
                .collect();
            format!("{{{}}}", items.join(","))
        }
        Value::Array(items) => {
            let items: Vec<String> = items.iter().map(|v| tml_dumps(v, exponent, order)).collect();
            format!("[{}]", items.join(","))
        }
        Value::Number(n) if n.is_f64() => tml_float(n.as_f64().unwrap_or(f64::NAN), exponent),
        _ => dumps(value, NonAscii::Keep, Separators::Compact),
    }
}

/// `_tml_float`: Python's `repr` with exponents -5..15 written positionally and the exponent sign per `exponent`.
/// `value` must be finite.
#[must_use]
pub fn tml_float(value: f64, exponent: Exponent) -> String {
    let repr = float_repr(value);
    let Some((mantissa, exp)) = repr.split_once('e') else {
        return repr;
    };
    let exp10: i32 = exp.parse().expect("float_repr writes an integer exponent");
    if !(-5..16).contains(&exp10) {
        let sign = if exp10 >= 0 && exponent == Exponent::Signed {
            "+"
        } else {
            ""
        };
        return format!("{mantissa}e{sign}{exp10}");
    }
    let (sign, mantissa) = mantissa.strip_prefix('-').map_or(("", mantissa), |m| ("-", m));
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let decimal = exp10 + 1;
    let body = match usize::try_from(decimal) {
        Err(_) | Ok(0) => format!("0.{}{digits}", "0".repeat(usize::try_from(-decimal).unwrap_or(0))),
        Ok(n) if n >= digits.len() => format!("{digits}{}.0", "0".repeat(n - digits.len())),
        Ok(n) => format!("{}.{}", &digits[..n], &digits[n..]),
    };
    format!("{sign}{body}")
}

/// Python `str.strip()`: removes characters Python's `isspace` accepts (std white space plus U+001C..U+001F).
#[must_use]
pub fn py_strip(text: &str) -> &str {
    text.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

/// Python `int(text)` for a `str`; `None` where Python raises `ValueError` or the value is outside
/// `[-2^63, 2^64)` (documented limit). Non-ASCII decimal digits, which Python accepts, are rejected.
#[must_use]
pub fn parse_int(text: &str) -> Option<Number> {
    let s = py_strip(text);
    let (negative, digits) = match s.as_bytes().first()? {
        b'-' => (true, &s[1..]),
        b'+' => (false, &s[1..]),
        _ => (false, s),
    };
    let digits = drop_digit_underscores(digits)?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if negative {
        format!("-{digits}").parse::<i64>().ok().map(Number::from)
    } else {
        digits.parse::<u64>().ok().map(Number::from)
    }
}

/// Python `round(x)` to an integer: halves go to the even neighbour.
#[must_use]
pub fn round_half_even(x: f64) -> f64 {
    x.round_ties_even()
}

/// `f"{x:g}"`: six significant digits, trailing zeros removed, exponent form below 1e-4 and from 1e6.
#[must_use]
pub fn format_g(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf" } else { "inf" }.into();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0" } else { "0" }.into();
    }
    let sci = format!("{x:.5e}");
    let (mantissa, exp) = sci.split_once('e').expect("`{:e}` has an exponent");
    let exp10: i32 = exp.parse().expect("`{:e}` exponent is an integer");
    if (-4..6).contains(&exp10) {
        let decimals = usize::try_from(5 - exp10).unwrap_or(0);
        return trim_zeros(&format!("{x:.decimals$}")).to_string();
    }
    let sign = if exp10 < 0 { '-' } else { '+' };
    format!("{}e{sign}{:02}", trim_zeros(mantissa), exp10.abs())
}

fn trim_zeros(s: &str) -> &str {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.')
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tml_float_spellings() {
        // Port of the tml_float checks of test_text.cpp `text_effort_and_floats`.
        let cases: &[(f64, Exponent, &str)] = &[
            (1e16, Exponent::Bare, "1e16"),
            (1e16, Exponent::Signed, "1e+16"),
            (1e-05, Exponent::Bare, "0.00001"),
            (1.5e-07, Exponent::Bare, "1.5e-7"),
            (123_456_789_012_345_678.0, Exponent::Bare, "1.2345678901234568e17"),
            (2.5, Exponent::Bare, "2.5"),
            (-1.5e-05, Exponent::Signed, "-0.000015"),
            (1e-6, Exponent::Signed, "1e-6"),
        ];
        for (v, e, want) in cases {
            assert_eq!(tml_float(*v, *e), *want, "value {v:e}");
        }
    }

    #[test]
    fn dumps_spellings() {
        let v: Value = serde_json::from_str(r#"{"a":[1,2.0,"é"],"b":null}"#).expect("valid json");
        assert_eq!(
            dumps(&v, NonAscii::Keep, Separators::Spaced),
            r#"{"a": [1, 2.0, "é"], "b": null}"#
        );
        assert_eq!(
            dumps(&v, NonAscii::Keep, Separators::Compact),
            r#"{"a":[1,2.0,"é"],"b":null}"#
        );
        assert_eq!(dumps_string("é😀\n", NonAscii::Escape), r#""\u00e9\ud83d\ude00\n""#);
        assert_eq!(dumps_string("é😀\n", NonAscii::Keep), r#""é😀\n""#);
        assert_eq!(
            dumps_string("\u{1}\u{8}\u{c}\"\\", NonAscii::Keep),
            r#""\u0001\b\f\"\\""#
        );
        // Python: json.dumps("\x7f") == '"\\u007f"', json.dumps("\x7f", ensure_ascii=False) == '"\x7f"'
        assert_eq!(dumps_string("\u{7f}", NonAscii::Escape), r#""\u007f""#);
        assert_eq!(dumps_string("\u{7f}", NonAscii::Keep), "\"\u{7f}\"");
    }

    #[test]
    fn tml_dumps_sorts_and_spells() {
        let v = json!({"b": [1.0, 1e16], "a": {"z": "é", "y": 0.5}});
        assert_eq!(
            tml_dumps(&v, Exponent::Signed, KeyOrder::Sorted),
            r#"{"a":{"y":0.5,"z":"é"},"b":[1.0,1e+16]}"#
        );
        assert_eq!(
            tml_dumps(&v, Exponent::Bare, KeyOrder::Insertion),
            r#"{"b":[1.0,1e16],"a":{"z":"é","y":0.5}}"#
        );
    }

    #[test]
    fn parse_int_matches_python() {
        // Port of the parse_int checks of test_text.cpp.
        assert_eq!(parse_int(" 1_000 "), Some(Number::from(1000)));
        assert_eq!(parse_int("1__0"), None);
        assert_eq!(parse_int("1.5"), None);
        assert_eq!(parse_int("-0"), Some(Number::from(0)));
        assert_eq!(parse_int("+007"), Some(Number::from(7)));
        assert_eq!(parse_int("18446744073709551615"), Some(Number::from(u64::MAX)));
        assert_eq!(parse_int("-9223372036854775808"), Some(Number::from(i64::MIN)));
        for s in [
            "",
            "-",
            "+",
            "_1",
            "1_",
            "18446744073709551616",
            "-9223372036854775809",
            "٣",
            "0x1",
        ] {
            assert_eq!(parse_int(s), None, "input {s:?}");
        }
    }

    #[test]
    fn round_and_g() {
        assert_eq!(round_half_even(0.5), 0.0);
        assert_eq!(round_half_even(1.5), 2.0);
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(-2.5), -2.0);
        let g: &[(f64, &str)] = &[
            (0.9, "0.9"),
            (0.0, "0"),
            (0.99, "0.99"),
            (0.12, "0.12"),
            (100.0, "100"),
            (1e-5, "1e-05"),
            (123_456.7, "123457"),
            (1_234_567.0, "1.23457e+06"),
            (0.0001, "0.0001"),
            (-2.5, "-2.5"),
        ];
        for (v, want) in g {
            assert_eq!(format_g(*v), *want, "value {v:e}");
        }
    }

    #[test]
    fn effort_text_arithmetic() {
        // Port of the effort_text checks: round(round(e*1000)/1000*100)/100 formatted with %g.
        let effort = |e: f64| format_g(round_half_even((round_half_even(e * 1000.0) / 1000.0) * 100.0) / 100.0);
        assert_eq!(effort(0.9), "0.9");
        assert_eq!(effort(0.0), "0");
        assert_eq!(effort(0.99), "0.99");
        assert_eq!(effort(0.125), "0.12");
    }
}
