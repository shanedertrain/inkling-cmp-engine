// Ported from engine/server/src/parse/inkling_parser.cpp (`Lexer`) @ 6100ab8, itself a port of vLLM's
// vllm/parser/engine/incremental_lexer.py (Apache-2.0, Copyright contributors to the vLLM project).
//! Incremental lexer: splits streamed text into grammar terminals and content, holding back a possible terminal prefix.

/// A lexer token kind: the grammar's terminals, special tokens outside the grammar, and plain content.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Term {
    MsgModel,
    TextStart,
    ThinkStart,
    ThinkEnd,
    EndSampling,
    ToolStart,
    ToolText,
    ToolError,
    Drop,
    Content,
}

/// Terminal literals of `inkling_config`.
pub(crate) const TERMINALS: [(&str, Term); 8] = [
    ("<|message_model|>", Term::MsgModel),
    ("<|content_text|>", Term::TextStart),
    ("<|content_thinking|>", Term::ThinkStart),
    ("<|end_message|>", Term::ThinkEnd),
    ("<|content_model_end_sampling|>", Term::EndSampling),
    ("<|content_invoke_tool_json|>", Term::ToolStart),
    ("<|content_invoke_tool_text|>", Term::ToolText),
    ("<|content_tool_error|>", Term::ToolError),
];

#[derive(Debug)]
pub(crate) struct LexToken {
    pub term: Term,
    pub value: String,
}

impl LexToken {
    fn content(value: String) -> Self {
        Self {
            term: Term::Content,
            value,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Lexer {
    literals: Vec<(String, Term)>,
    buffer: String,
}

impl Lexer {
    pub fn new(literals: Vec<(String, Term)>) -> Self {
        Self {
            literals,
            buffer: String::new(),
        }
    }

    pub fn is_idle(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    pub fn feed(&mut self, text: &str) -> Vec<LexToken> {
        if self.buffer.is_empty() && !text.contains('<') {
            return vec![LexToken::content(text.to_owned())];
        }
        self.buffer.push_str(text);
        self.drain(false)
    }

    pub fn flush(&mut self) -> Vec<LexToken> {
        let mut tokens = if self.buffer.is_empty() {
            Vec::new()
        } else {
            self.drain(true)
        };
        if !self.buffer.is_empty() {
            tokens.push(LexToken::content(std::mem::take(&mut self.buffer)));
        }
        tokens
    }

    /// Byte offset of the first `<` after the start of the buffer that begins (or may still grow into) a literal.
    fn content_boundary(&self) -> usize {
        let buf = self.buffer.as_bytes();
        for i in 1..buf.len() {
            if buf[i] != b'<' {
                continue;
            }
            let rest = &buf[i..];
            if self.literals.iter().any(|(lit, _)| {
                let len = rest.len().min(lit.len());
                rest[..len] == lit.as_bytes()[..len]
            }) {
                return i;
            }
        }
        buf.len()
    }

    /// The longest literal the buffer starts with.
    fn longest_match(&self) -> Option<(String, Term)> {
        let mut best: Option<&(String, Term)> = None;
        for lt in &self.literals {
            if self.buffer.starts_with(lt.0.as_str()) && best.is_none_or(|b| lt.0.len() > b.0.len()) {
                best = Some(lt);
            }
        }
        best.cloned()
    }

    /// Whether the buffer is a proper prefix of some literal.
    fn is_literal_prefix(&self) -> bool {
        let buf = self.buffer.as_bytes();
        self.literals
            .iter()
            .any(|(lit, _)| lit.len() > buf.len() && lit.as_bytes().starts_with(buf))
    }

    fn drain(&mut self, last: bool) -> Vec<LexToken> {
        let mut tokens = Vec::new();
        while !self.buffer.is_empty() {
            if !self.buffer.contains('<') {
                tokens.push(LexToken::content(std::mem::take(&mut self.buffer)));
                break;
            }
            let best = self.longest_match();
            if self.is_literal_prefix() && !last {
                let Some((lit, term)) = best else { break };
                let buf = self.buffer.as_bytes();
                let longer = self
                    .literals
                    .iter()
                    .any(|(l, _)| l.len() > lit.len() && l.len() >= buf.len() && l.as_bytes().starts_with(buf));
                if longer {
                    break;
                }
                self.buffer.drain(..lit.len());
                tokens.push(LexToken { term, value: lit });
                continue;
            }
            if let Some((lit, term)) = best {
                self.buffer.drain(..lit.len());
                tokens.push(LexToken { term, value: lit });
            } else {
                let end = self.content_boundary().max(1);
                let rest = self.buffer.split_off(end);
                let content = std::mem::replace(&mut self.buffer, rest);
                tokens.push(LexToken::content(content));
            }
        }
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lexer() -> Lexer {
        Lexer::new(TERMINALS.iter().map(|(l, t)| ((*l).to_owned(), *t)).collect())
    }

    fn kinds(tokens: &[LexToken]) -> Vec<(Term, &str)> {
        tokens.iter().map(|t| (t.term, t.value.as_str())).collect()
    }

    #[test]
    fn plain_text_passes_through_without_buffering() {
        let mut lx = lexer();
        let toks = lx.feed("hello");
        assert_eq!(kinds(&toks), [(Term::Content, "hello")]);
        assert!(lx.is_idle());
    }

    #[test]
    fn terminal_split_across_feeds_is_held_then_emitted() {
        let mut lx = lexer();
        assert!(lx.feed("a<|content_").iter().all(|t| t.term == Term::Content));
        let toks = lx.feed("text|>b");
        assert_eq!(
            kinds(&toks),
            [(Term::TextStart, "<|content_text|>"), (Term::Content, "b")]
        );
    }

    #[test]
    fn lone_angle_bracket_is_content_at_flush() {
        let mut lx = lexer();
        assert!(lx.feed("x <").iter().all(|t| t.value != "<"));
        let toks = lx.flush();
        assert_eq!(kinds(&toks), [(Term::Content, "<")]);
    }

    #[test]
    fn non_terminal_angle_bracket_is_content() {
        let mut lx = lexer();
        let toks = lx.feed("a < b <b>c");
        let text: String = toks.iter().map(|t| t.value.as_str()).collect();
        assert!(toks.iter().all(|t| t.term == Term::Content));
        assert_eq!(
            text + &lx.flush().iter().map(|t| t.value.as_str()).collect::<String>(),
            "a < b <b>c"
        );
    }
}
