// Ported from engine/server/src/text/tokenizer.{h,cpp} @ 4171493 (adapted there from ingine @ 306eaa8, Apache-2.0,
// Copyright 2026 satellitedown). The BPE, the split regex and the added-token matcher of the C++ are replaced by the
// `tokenizers` crate.

use std::error::Error;
use std::fmt;
use std::path::Path;

use tokenizers::Tokenizer as HfTokenizer;

type BoxError = Box<dyn Error + Send + Sync>;

/// How added tokens in the text are treated when encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Specials {
    /// Added tokens such as `<|end_message|>` become their single id (vLLM's `encode(text, add_special_tokens=False)`).
    Recognize,
    /// The whole text is ordinary text (`encode_special_tokens=True`).
    Literal,
}

/// An added (special) token of the tokenizer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AddedToken {
    /// Token id.
    pub id: u32,
    /// Literal text of the token.
    pub content: String,
    /// Whether the token is special (skipped when decoding with `skip_special`).
    pub special: bool,
}

/// Failure to load, encode or decode.
#[derive(Debug)]
pub struct TokenizerError {
    what: &'static str,
    source: BoxError,
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tokenizer {}: {}", self.what, self.source)
    }
}

impl Error for TokenizerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&*self.source)
    }
}

/// The Inkling-2 tokenizer (`tokenizer.json`: byte-level BPE with `ignore_merges`, o200k-style split, added tokens).
#[derive(Debug)]
pub struct Tokenizer {
    recognize: HfTokenizer,
    literal: HfTokenizer,
    added: Vec<AddedToken>,
}

impl Tokenizer {
    /// Reads `<model_dir>/tokenizer.json`.
    ///
    /// # Errors
    /// The file is missing or is not a valid `tokenizers` definition.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let bytes = std::fs::read(model_dir.as_ref().join("tokenizer.json")).map_err(|e| TokenizerError {
            what: "load",
            source: e.into(),
        })?;
        Self::from_json(&bytes)
    }

    /// Builds the tokenizer from the text of a `tokenizer.json`.
    ///
    /// # Errors
    /// The text is not a valid `tokenizers` definition.
    pub fn from_json(json: &[u8]) -> Result<Self, TokenizerError> {
        let recognize = HfTokenizer::from_bytes(json).map_err(|source| TokenizerError { what: "load", source })?;
        let mut literal = recognize.clone();
        literal.set_encode_special_tokens(true);
        let mut added: Vec<AddedToken> = recognize
            .get_added_tokens_decoder()
            .into_iter()
            .map(|(id, t)| AddedToken {
                id,
                content: t.content,
                special: t.special,
            })
            .collect();
        added.sort_by_key(|t| t.id);
        Ok(Self {
            recognize,
            literal,
            added,
        })
    }

    /// Token ids of `text`, without any special tokens added around it.
    ///
    /// # Errors
    /// The underlying tokenizer fails (not observed on valid UTF-8).
    pub fn encode(&self, text: &str, specials: Specials) -> Result<Vec<u32>, TokenizerError> {
        let tok = match specials {
            Specials::Recognize => &self.recognize,
            Specials::Literal => &self.literal,
        };
        let encoding = tok
            .encode_fast(text, false)
            .map_err(|source| TokenizerError { what: "encode", source })?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Text of `ids` as Python's `tokenizer.decode` returns it: invalid UTF-8 becomes U+FFFD, unknown ids are dropped.
    /// With `skip_special`, added special tokens contribute nothing.
    ///
    /// # Errors
    /// The underlying decoder fails (not observed).
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String, TokenizerError> {
        self.recognize
            .decode(ids, skip_special)
            .map_err(|source| TokenizerError { what: "decode", source })
    }

    /// Id of an added token or vocabulary entry, by its literal content.
    #[must_use]
    pub fn token_id(&self, content: &str) -> Option<u32> {
        self.recognize.token_to_id(content)
    }

    /// Whether `id` is an added special token.
    #[must_use]
    pub fn is_special(&self, id: u32) -> bool {
        self.added
            .binary_search_by_key(&id, |t| t.id)
            .is_ok_and(|i| self.added[i].special)
    }

    /// The added tokens, ordered by id.
    #[must_use]
    pub fn added_tokens(&self) -> &[AddedToken] {
        &self.added
    }

    /// Number of ids: vocabulary plus added tokens.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.recognize.get_vocab_size(true)
    }

    /// The underlying `tokenizers` object, for `decode_stream` (package F2).
    #[must_use]
    pub fn inner(&self) -> &HfTokenizer {
        &self.recognize
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Tokenizer>();
};
