//! One JSON object per line in the format of the C++ benchmark tools (`p6::Line`, `common.hpp`): keys in call order,
//! doubles as `%.7g` (non-finite as `null`), strings escaped as the C++ writer does.

use std::fmt::Write as _;
use std::io::Write as _;

use anyhow::Result;
use ink_record::Source;
use serde_json::Value;

use crate::fmt::g;

/// A line under construction.
#[derive(Debug, Clone)]
pub struct Line {
    s: String,
}

impl Line {
    /// Starts `{"probe":<probe>,"test":<test>`.
    #[must_use]
    pub fn new(probe: &str, test: &str) -> Self {
        let mut line = Self { s: "{".to_owned() };
        line.key("probe");
        line.string(probe);
        line.key("test");
        line.string(test);
        line
    }

    fn key(&mut self, k: &str) {
        if self.s.len() > 1 {
            self.s.push(',');
        }
        self.string(k);
        self.s.push(':');
    }

    fn string(&mut self, v: &str) {
        self.s.push('"');
        for c in v.chars() {
            match c {
                '"' | '\\' => {
                    self.s.push('\\');
                    self.s.push(c);
                }
                c if u32::from(c) < 0x20 => {
                    let _ = write!(self.s, "\\u{:04x}", u32::from(c));
                }
                c => self.s.push(c),
            }
        }
        self.s.push('"');
    }

    /// `"k":"v"`.
    #[must_use]
    pub fn text(mut self, k: &str, v: &str) -> Self {
        self.key(k);
        self.string(v);
        self
    }

    /// `"k":true|false`.
    #[must_use]
    pub fn flag(mut self, k: &str, v: bool) -> Self {
        self.key(k);
        self.s.push_str(if v { "true" } else { "false" });
        self
    }

    /// `"k":<integer>`.
    #[must_use]
    pub fn int(mut self, k: &str, v: impl Into<i64>) -> Self {
        self.key(k);
        let _ = write!(self.s, "{}", v.into());
        self
    }

    /// `"k":<unsigned integer>`.
    #[must_use]
    pub fn uint(mut self, k: &str, v: impl Into<u64>) -> Self {
        self.key(k);
        let _ = write!(self.s, "{}", v.into());
        self
    }

    /// `"k":<%.7g>`, `null` when not finite.
    #[must_use]
    pub fn num(mut self, k: &str, v: f64) -> Self {
        self.key(k);
        if v.is_finite() {
            self.s.push_str(&g(v, 7));
        } else {
            self.s.push_str("null");
        }
        self
    }

    /// `"k":<json>`, the text as given.
    #[must_use]
    pub fn raw(mut self, k: &str, json: &str) -> Self {
        self.key(k);
        self.s.push_str(json);
        self
    }

    /// The runtime library's identity fields as the C++ tools print them: `engine`, `engine_version`, `commit`,
    /// `dirty`.
    #[must_use]
    pub fn identity(mut self, source: &Source) -> Self {
        let fields = crate::engine::identity_fields(source);
        self.s.push(',');
        self.s.push_str(&fields);
        self
    }

    /// The finished line without a newline.
    #[must_use]
    pub fn finish(mut self) -> String {
        self.s.push('}');
        self.s
    }

    /// Prints the line on stdout (flushed) and returns it as a JSON value for the record.
    ///
    /// # Errors
    ///
    /// Fails when stdout cannot be written or the line is not valid JSON.
    pub fn emit(self) -> Result<Value> {
        let text = self.finish();
        let mut out = std::io::stdout().lock();
        writeln!(out, "{text}")?;
        out.flush()?;
        Ok(serde_json::from_str(&text)?)
    }
}

#[cfg(test)]
mod tests {
    use super::Line;

    #[test]
    fn keys_in_call_order_with_c_formats() {
        let s = Line::new("native_rt", "decode_step")
            .text("a", "q\"\\\n")
            .flag("b", true)
            .int("c", -3_i32)
            .uint("d", 7_u64)
            .num("e", 1_234.567_891_2)
            .num("f", f64::NAN)
            .num("g", 1e-5)
            .raw("h", "{\"x\":1}")
            .finish();
        assert_eq!(
            s,
            r#"{"probe":"native_rt","test":"decode_step","a":"q\"\\\u000a","b":true,"c":-3,"d":7,"e":1234.568,"f":null,"g":1e-05,"h":{"x":1}}"#
        );
    }
}
