// Ported from engine/native/src/engine.cu @ 1c7ac46 (Engine::forward's graph cache 907-961, apply_ablation,
// set_ablation, graph_region_counts) and engine/native/capi/ink_rt_capi.cpp (ink_rt_set_ablation's refusal). Own code
// of this project.
//! The CUDA-graph cache of decode calls (package R6 of docs/RUST-PLAN.md): one captured graph per key, the region of
//! every node (from the capture marks, `ink_cuda::Graph`), the timing ablation over those regions, and a fixed
//! device-memory reserve. Pattern: vLLM's CUDA-graph dispatch (a small fixed set of shapes, each captured once and
//! replayed with the call's inputs copied into the fixed device buffers first).
//!
//! The key holds every input that changes a captured launch's kernels or blocks: the glue version (the C++ stage-3
//! bug was a key without it, `bench/raw/2026-11-23-native-rt/STATUS.md`), the segment count `S`, the call's rows `T`
//! (one row group: plan classes, GEMM M, the projection path, the gather and head) and the decode-attention kernel
//! variant of its widest segment (`meta::cache_key`). Neither the context length nor the per-segment widths are in
//! it: decode attention's launch depends on the call's key chunks and row bound `R` (grid x = workers per session
//! and head, combine row blocks, `Params::max_chunks`, `Params::R`), and before a replay whose values differ from the
//! graph's the engine sets those nodes to the call's (`Graph::set_kernel_param`, the pattern of in-place graph
//! updates), so a replay launches exactly what the eager call would. The server's keys are one-row decode at 1..8
//! segments and verify calls at `S` segments of `T = S m` rows, `m` from a small set (`spec.rs`). Everything else a
//! captured launch reads is device data written before the replay (`Engine::upload`).
//!
//! Memory: each graph is charged the device memory its capture took (free memory before the capture minus after its
//! first replay; 8 MiB per forward graph measured, `bench/raw/2026-11-23-graph-cache/`). Graphs are captured at first
//! use while the largest charge seen still fits the reserve; a key beyond it runs eagerly (bits do not depend on
//! whether a call runs from a graph) and is counted. Graphs live as long as the engine.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use ink_cuda::{Graph, NodeState};
use ink_rt_api::RuntimeError;

use super::{Res, Rt};

/// One held graph and the key chunks its decode-attention nodes are set to.
#[derive(Debug)]
pub(super) struct Entry {
    /// The captured graph.
    pub graph: Graph,
    /// Key chunks (global, sliding window) of the decode-attention nodes now.
    pub chunks: (usize, usize),
    /// Row bound `R` of the decode-attention nodes now.
    pub r: usize,
    /// Indices of the decode-attention and combine kernel nodes.
    pub da_nodes: Vec<usize>,
}

/// Captured decode graphs by key within a fixed byte reserve, and the regions disabled in all of them.
#[derive(Debug)]
pub(super) struct GraphCache {
    graphs: HashMap<String, Entry>,
    reserve: usize,
    held: usize,
    /// Largest charge seen: a capture starts only while this still fits.
    largest: usize,
    ablate: BTreeSet<String>,
    captures: u64,
    /// Keys that run eagerly because the reserve was full or capture failed.
    refused: BTreeSet<String>,
}

impl GraphCache {
    /// An empty cache whose graphs may hold `reserve` bytes.
    pub fn new(reserve: usize) -> GraphCache {
        GraphCache {
            graphs: HashMap::new(),
            reserve,
            held: 0,
            largest: 0,
            ablate: BTreeSet::new(),
            captures: 0,
            refused: BTreeSet::new(),
        }
    }

    /// The graph of `key`, when held.
    pub fn get(&self, key: &str) -> Option<&Graph> {
        self.graphs.get(key).map(|e| &e.graph)
    }

    /// The entry of `key`, when held.
    pub fn entry(&self, key: &str) -> Option<&Entry> {
        self.graphs.get(key)
    }

    /// The entry of `key`, when held.
    pub fn entry_mut(&mut self, key: &str) -> Option<&mut Entry> {
        self.graphs.get_mut(key)
    }

    /// Sets the bytes the held graphs may take (the engine's reserve less what other graphs of the context hold).
    pub fn set_reserve(&mut self, reserve: usize) {
        self.reserve = reserve;
    }

    /// Whether a new graph may be captured for `key`: the key has not been refused and the largest graph seen so far
    /// still fits the reserve. A key refused once stays refused (it runs eagerly; counted in [`GraphCache::refused`]).
    pub fn admits(&mut self, key: &str) -> bool {
        if self.refused.contains(key) {
            return false;
        }
        if self.held + self.largest <= self.reserve {
            return true;
        }
        if self.refused.insert(key.to_owned()) {
            tracing::debug!(
                target: "ink_step",
                phase = "graph",
                event = "refuse",
                key = %key,
                held = self.held,
                reserve = self.reserve,
                refused = self.refused.len(),
            );
        }
        false
    }

    /// Marks `key` as refused so later calls with the same key run eagerly instead of retrying capture.
    pub fn refuse(&mut self, key: &str) {
        self.refused.insert(key.to_owned());
    }

    /// Applies the current ablation to a newly captured graph (the C++ `apply_ablation` after `emplace`).
    pub fn ablate(&self, graph: &mut Graph) -> Res<()> {
        apply(graph, &self.ablate)
    }

    /// Adds a newly captured graph (ablation applied, replayed once) charged `bytes`, captured at key chunks
    /// `chunks`; its decode-attention nodes are those whose kernel is in namespace `da`.
    pub fn insert(&mut self, key: String, graph: Graph, bytes: usize, chunks: (usize, usize), r: usize) {
        self.captures += 1;
        self.largest = self.largest.max(bytes);
        self.held += bytes;
        let da_nodes = (0..graph.nodes().len())
            .filter(|&i| {
                graph.nodes()[i]
                    .kernel
                    .as_deref()
                    .is_some_and(|k| k.starts_with("_ZN2da"))
            })
            .collect();
        self.graphs.insert(
            key,
            Entry {
                graph,
                chunks,
                r,
                da_nodes,
            },
        );
    }

    /// Graphs captured so far.
    pub fn captures(&self) -> u64 {
        self.captures
    }

    /// Graphs held now.
    pub fn live(&self) -> usize {
        self.graphs.len()
    }

    /// Bytes charged to the held graphs.
    pub fn held(&self) -> usize {
        self.held
    }

    /// Keys that run eagerly because the reserve was full or capture failed.
    pub fn refused(&self) -> usize {
        self.refused.len()
    }

    /// Nodes per region over every held graph (the C++ `graph_region_counts`).
    pub fn region_counts(&self) -> BTreeMap<String, usize> {
        let mut out = BTreeMap::new();
        for e in self.graphs.values() {
            for (region, n) in e.graph.region_counts() {
                *out.entry(region).or_insert(0) += n;
            }
        }
        out
    }

    /// Disables the nodes of `regions` in every graph, now and in graphs captured later; enables every other node.
    /// Refuses a region that no held graph has, as `ink_rt_set_ablation` does; empty names are skipped (the C ABI
    /// splits a comma list).
    pub fn set_ablation(&mut self, regions: &[&str]) -> Res<()> {
        if let Some(bad) = regions.iter().find(|r| r.contains(',') || r.contains('\0')) {
            return Err(RuntimeError::request(format!(
                "region name contains a comma or NUL byte: {bad:?}"
            )));
        }
        let want: BTreeSet<String> = regions
            .iter()
            .filter(|r| !r.is_empty())
            .map(|r| (*r).to_owned())
            .collect();
        let known = self.region_counts();
        if let Some(r) = want.iter().find(|r| !known.contains_key(*r)) {
            let hint = if known.is_empty() {
                " (no graph captured yet: run a decode call first)"
            } else {
                ""
            };
            return Err(RuntimeError::request(format!(
                "region \"{r}\" is in no captured graph{hint}"
            )));
        }
        self.ablate = want;
        for e in self.graphs.values_mut() {
            apply(&mut e.graph, &self.ablate)?;
        }
        Ok(())
    }
}

/// Every node that can be disabled: off when its region is ablated, on otherwise (the C++ `apply_ablation`).
fn apply(g: &mut Graph, ablate: &BTreeSet<String>) -> Res<()> {
    for i in 0..g.nodes().len() {
        let n = &g.nodes()[i];
        if !n.kind.can_disable() {
            continue;
        }
        let state = if ablate.contains(&n.region) {
            NodeState::Disabled
        } else {
            NodeState::Enabled
        };
        g.set_node(i, state).rt()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refused_key_stays_eager_even_when_reserve_has_room() {
        let mut c = GraphCache::new(512 << 20);
        assert!(c.admits("6|8,8;"));
        c.refuse("6|8,8;");
        assert_eq!(c.refused(), 1);
        assert!(!c.admits("6|8,8;"));
        // refusing or re-checking the same key does not double-count, and other keys are still admitted
        c.refuse("6|8,8;");
        assert_eq!(c.refused(), 1);
        assert!(c.admits("6|1,1;"));
        // growing the reserve does not re-admit a refused key
        c.set_reserve(1024 << 20);
        assert!(!c.admits("6|8,8;"));
    }
}
