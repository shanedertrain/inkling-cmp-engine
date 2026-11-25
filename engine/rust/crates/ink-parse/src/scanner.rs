// Ported from engine/server/src/parse/inkling_parser.cpp (`Scanner`) @ 6100ab8, itself a port of vLLM's
// vllm/parser/engine/token_id_scanner.py (Apache-2.0, Copyright contributors to the vLLM project).
//! Token-id scanner: finds special tokens by id (the detokeniser may have dropped or moved their text) and aligns
//! them with the step's text.

use std::sync::Arc;

use crate::tables::ParserTables;

/// A piece of a step: text with the number of tokens that produced it, or a special token seen by id.
#[derive(Clone, Debug)]
pub(crate) struct Item {
    pub terminal: bool,
    pub text: String,
    pub token_count: usize,
    pub token_texts: Vec<String>,
}

impl Item {
    fn text(text: String, token_count: usize, token_texts: Vec<String>) -> Self {
        Self {
            terminal: false,
            text,
            token_count,
            token_texts,
        }
    }

    fn bare(text: &str, token_count: usize) -> Self {
        Self::text(text.to_owned(), token_count, Vec::new())
    }
}

#[derive(Debug)]
pub(crate) struct Scanner {
    tables: Arc<ParserTables>,
    /// Terminals whose text has not appeared in the text stream yet (`PreLexedTerminals`).
    deferred: Vec<Item>,
    deferred_prefix_counts: Vec<usize>,
    deferred_trailing: usize,
    deferred_post_text: String,
}

impl Scanner {
    pub fn new(tables: Arc<ParserTables>) -> Self {
        Self {
            tables,
            deferred: Vec::new(),
            deferred_prefix_counts: Vec::new(),
            deferred_trailing: 0,
            deferred_post_text: String::new(),
        }
    }

    pub fn reset(&mut self) {
        self.deferred.clear();
        self.deferred_prefix_counts.clear();
        self.deferred_trailing = 0;
        self.deferred_post_text.clear();
    }

    pub fn has_deferred(&self) -> bool {
        !self.deferred.is_empty()
    }

    pub fn is_drop(&self, id: u32) -> bool {
        self.tables.drop_ids.contains(&id)
    }

    pub fn scan(&mut self, delta_text: &str, ids: &[u32]) -> Vec<Item> {
        let mut prefix_items = Vec::new();
        let mut effective = delta_text.to_owned();
        let mut trailing_count = 0;
        if self.has_deferred() {
            (effective, trailing_count) = self.resolve_deferred(delta_text, ids.len(), &mut prefix_items);
        }
        if self.tables.drop_ids.is_empty() {
            if !effective.is_empty() {
                prefix_items.push(Item::bare(&effective, trailing_count + ids.len()));
            }
            return prefix_items;
        }
        if !ids.iter().any(|&id| self.is_drop(id)) {
            if !effective.is_empty() {
                let token_texts: Vec<String> = ids.iter().map(|&id| self.tables.decode_token(id)).collect();
                let count = token_texts.len();
                if prefix_items.is_empty() {
                    return vec![Item::text(effective, count, token_texts)];
                }
                prefix_items.push(Item::text(effective, trailing_count + count, token_texts));
            }
            return prefix_items;
        }
        let mut results: Vec<Item> = Vec::new();
        let mut acc = String::new();
        let mut acc_texts: Vec<String> = Vec::new();
        for &id in ids {
            let decoded = self.tables.decode_token(id);
            if self.is_drop(id) {
                if !acc_texts.is_empty() {
                    let (text, texts) = (std::mem::take(&mut acc), std::mem::take(&mut acc_texts));
                    if !text.is_empty() {
                        results.push(Item::text(text, texts.len(), texts));
                    }
                }
                results.push(Item {
                    terminal: true,
                    text: decoded,
                    token_count: 0,
                    token_texts: Vec::new(),
                });
            } else {
                acc.push_str(&decoded);
                acc_texts.push(decoded);
            }
        }
        if !acc_texts.is_empty() && !acc.is_empty() {
            results.push(Item::text(acc, acc_texts.len(), acc_texts));
        }
        if trailing_count > 0 {
            results.insert(0, Item::bare("", trailing_count));
        }
        if effective.is_empty() {
            let mut prefix_count = 0;
            for item in results {
                if item.terminal {
                    self.deferred.push(item);
                    self.deferred_prefix_counts.push(prefix_count);
                    prefix_count = 0;
                } else {
                    prefix_count += item.token_count;
                }
            }
            self.deferred_trailing += prefix_count;
        } else {
            prefix_items.extend(self.recover_holdback(&effective, results));
        }
        prefix_items
    }

    pub fn flush_pending(&mut self) -> Vec<Item> {
        if self.deferred.is_empty() && self.deferred_post_text.is_empty() {
            return Vec::new();
        }
        let mut res = Vec::new();
        if !self.deferred_post_text.is_empty() {
            let count = self.deferred_prefix_counts.first().copied().unwrap_or(0);
            res.push(Item::text(
                std::mem::take(&mut self.deferred_post_text),
                count,
                Vec::new(),
            ));
        }
        res.append(&mut self.deferred);
        if self.deferred_trailing > 0 {
            res.push(Item::bare("", self.deferred_trailing));
        }
        self.reset();
        res
    }

    /// Matches terminals deferred by earlier steps against this step's text. Returns the remaining text and the
    /// trailing token count, and appends what resolved to `results`.
    fn resolve_deferred(&mut self, delta_text: &str, current_count: usize, results: &mut Vec<Item>) -> (String, usize) {
        let deferred = std::mem::take(&mut self.deferred);
        let counts = std::mem::take(&mut self.deferred_prefix_counts);
        let mut trailing = self.deferred_trailing;
        self.deferred_trailing = 0;
        let mut remaining = delta_text.to_owned();
        if !self.deferred_post_text.is_empty() {
            remaining.insert_str(0, &std::mem::take(&mut self.deferred_post_text));
        }
        for (item, mut prefix_count) in deferred.into_iter().zip(counts) {
            match remaining.find(&item.text) {
                Some(pos) => {
                    if pos > 0 {
                        results.push(Item::bare(&remaining[..pos], prefix_count));
                    }
                    let rest = remaining.split_off(pos + item.text.len());
                    results.push(item);
                    remaining = rest;
                }
                None => {
                    if !remaining.is_empty() {
                        self.deferred_post_text.push_str(&remaining);
                        prefix_count += current_count;
                        remaining.clear();
                    }
                    self.deferred.push(item);
                    self.deferred_prefix_counts.push(prefix_count);
                }
            }
        }
        if !self.deferred.is_empty() {
            self.deferred_trailing = trailing;
            trailing = 0;
        }
        (remaining, trailing)
    }

    fn recover_holdback(&mut self, delta_text: &str, results: Vec<Item>) -> Vec<Item> {
        if results.is_empty() {
            return vec![Item::bare(delta_text, 0)];
        }
        let recon: String = results.iter().map(|r| r.text.as_str()).collect();
        if recon.is_empty() {
            let mut out = vec![Item::bare(delta_text, 0)];
            out.extend(results);
            return out;
        }
        match delta_text.find(&recon) {
            Some(0) => results,
            Some(pos) => {
                let mut out = vec![Item::bare(&delta_text[..pos], 0)];
                out.extend(results);
                out
            }
            None => self.rebuild_from_anchors(delta_text, &results),
        }
    }

    /// The step's text no longer equals the concatenated token texts (the detokeniser held bytes back): place each
    /// terminal at its last occurrence in the text and give the pieces between them their token groups.
    fn rebuild_from_anchors(&mut self, delta_text: &str, results: &[Item]) -> Vec<Item> {
        let anchors: Vec<&Item> = results.iter().filter(|r| r.terminal).collect();
        if anchors.is_empty() {
            return vec![Item::bare(delta_text, 0)];
        }
        let mut groups: Vec<Vec<String>> = vec![Vec::new(); anchors.len() + 1];
        let mut count_groups = vec![0usize; anchors.len() + 1];
        let mut g = 0;
        for r in results {
            if r.terminal {
                g += 1;
            } else {
                groups[g].extend(r.token_texts.iter().cloned());
                count_groups[g] += r.token_count;
            }
        }
        let mut positions: Vec<Option<usize>> = vec![None; anchors.len()];
        let mut search_end = delta_text.len();
        for i in (0..anchors.len()).rev() {
            if let Some(pos) = delta_text[..search_end].rfind(&anchors[i].text) {
                positions[i] = Some(pos);
                search_end = pos;
            }
        }
        let mut out = Vec::new();
        let mut consumed = 0;
        for (i, anchor) in anchors.iter().enumerate() {
            match positions[i] {
                Some(pos) if pos >= consumed => {
                    if pos > consumed {
                        out.push(Item::text(
                            delta_text[consumed..pos].to_owned(),
                            count_groups[i],
                            std::mem::take(&mut groups[i]),
                        ));
                    }
                    out.push((*anchor).clone());
                    consumed = pos + anchor.text.len();
                }
                _ => {
                    self.deferred.push((*anchor).clone());
                    self.deferred_prefix_counts.push(count_groups[i]);
                }
            }
        }
        if consumed < delta_text.len() {
            let last = anchors.len();
            out.push(Item::text(
                delta_text[consumed..].to_owned(),
                count_groups[last],
                std::mem::take(&mut groups[last]),
            ));
        }
        out
    }
}
