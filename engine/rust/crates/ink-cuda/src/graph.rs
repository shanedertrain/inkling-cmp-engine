//! CUDA graphs: stream capture (thread-local mode), instantiate, replay, node walk with region marks, and node
//! enable/disable for ablation. Same design as the C++ engine (`engine/native/src/engine.cu` @ 1c7ac46, lines
//! 281-295 `mark`, 913-977 capture, region walk, `apply_ablation` and `set_ablation`).
//!
//! Capture, instantiate and launch use cudarc's safe `CudaStream::begin_capture` / `end_capture` and `CudaGraph`
//! (which owns graph and executable). cudarc's safe layer has no node API, so the capture marks
//! (`cuStreamGetCaptureInfo`), node walk (`cuGraphGetNodes`, `cuGraphGetEdges`, `cuGraphNodeGetType`,
//! `cuGraphKernelNodeGetParams`) and `cuGraphNodeSetEnabled` go to `cudarc::driver::sys`.

use std::any::Any;
use std::collections::HashMap;
use std::hash::Hash;
use std::rc::Rc;

use cudarc::driver::sys::{CUgraphInstantiate_flags, CUgraphNodeType, CUstreamCaptureMode, CUstreamCaptureStatus};
use cudarc::driver::{CudaGraph, sys};

use crate::context::Stream;
use crate::error::{CudaError, Result, check_driver, driver};
use crate::module::{func_name, kernel_name};

/// State of an open capture, held by its [`Stream`].
#[derive(Debug, Default)]
pub(crate) struct CaptureState {
    /// `(region, last captured node at the mark)`, in mark order.
    marks: Vec<(String, Option<sys::CUgraphNode>)>,
    /// Objects the captured nodes refer to; moved into the [`Graph`].
    pub(crate) keep: Vec<Rc<dyn Any>>,
}

/// An open stream capture. Work queued on the stream until [`Capture::end`] becomes graph nodes.
///
/// Dropping it without `end` ends the capture and discards the graph, so the stream is usable again.
#[derive(Debug)]
pub struct Capture<'s> {
    stream: &'s Stream,
    open: bool,
}

impl Stream {
    /// Starts a capture in thread-local mode (`CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`, as the C++ engine).
    ///
    /// # Errors
    ///
    /// [`CudaError::Capturing`] if a capture is already open, or the driver error.
    pub fn begin_capture(&self) -> Result<Capture<'_>> {
        self.not_capturing("begin capture")?;
        self.inner
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(driver("cuStreamBeginCapture"))?;
        *self.capture.borrow_mut() = Some(CaptureState::default());
        Ok(Capture {
            stream: self,
            open: true,
        })
    }
}

impl Capture<'_> {
    /// Starts region `name`: nodes captured after this point belong to it (until the next mark).
    ///
    /// # Errors
    ///
    /// The driver error of `cuStreamGetCaptureInfo`.
    pub fn mark(&self, name: &str) -> Result<()> {
        let mut status = CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let mut deps: *const sys::CUgraphNode = std::ptr::null();
        let mut n: usize = 0;
        // SAFETY: the stream is live and capturing; all out-pointers are valid locals; the optional id, graph and
        // edge-data outputs are null (allowed). `deps` points to driver memory valid until the next stream call,
        // read right below.
        let status_code = unsafe {
            sys::cuStreamGetCaptureInfo_v3(
                self.stream.inner.cu_stream(),
                &mut status,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut deps,
                std::ptr::null_mut(),
                &mut n,
            )
        };
        check_driver("cuStreamGetCaptureInfo", status_code)?;
        if status != CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE {
            return Err(CudaError::NotCapturing("capture mark"));
        }
        // SAFETY: when n > 0, deps points to n node handles (see above); only the first is read, as the C++ does.
        let leaf = (n > 0 && !deps.is_null()).then(|| unsafe { *deps });
        if let Some(c) = self.stream.capture.borrow_mut().as_mut() {
            c.marks.push((name.to_owned(), leaf));
        }
        Ok(())
    }

    /// Ends the capture, instantiates the graph and walks its nodes.
    ///
    /// # Errors
    ///
    /// The driver error of `cuStreamEndCapture` (e.g. `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED` after an illegal call
    /// during capture), of instantiation, or of the node walk; [`CudaError::Graph`] for an empty capture.
    pub fn end(mut self) -> Result<Graph> {
        self.open = false;
        let state = self.stream.capture.borrow_mut().take().unwrap_or_default();
        let exec = self
            .stream
            .inner
            .end_capture(CUgraphInstantiate_flags(0))
            .map_err(driver("cuStreamEndCapture"))?
            .ok_or_else(|| CudaError::Graph("capture produced no graph".into()))?;
        Graph::walk(exec, state)
    }
}

impl Drop for Capture<'_> {
    fn drop(&mut self) {
        if self.open {
            self.stream.capture.borrow_mut().take();
            // best effort: end the capture so the stream leaves capture mode; the graph (if any) is dropped
            let _ = self.stream.inner.end_capture(CUgraphInstantiate_flags(0));
        }
    }
}

/// Type of a graph node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NodeKind {
    /// A kernel launch.
    Kernel,
    /// A memory copy.
    Memcpy,
    /// A memory set.
    Memset,
    /// Any other node type (raw `CUgraphNodeType` value).
    Other(u32),
}

impl NodeKind {
    fn from_raw(t: CUgraphNodeType) -> NodeKind {
        match t {
            CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL => NodeKind::Kernel,
            CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY => NodeKind::Memcpy,
            CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET => NodeKind::Memset,
            other => NodeKind::Other(other as u32),
        }
    }

    /// Whether `cuGraphNodeSetEnabled` applies (kernel, memcpy and memset nodes).
    pub fn can_disable(self) -> bool {
        matches!(self, NodeKind::Kernel | NodeKind::Memcpy | NodeKind::Memset)
    }
}

/// One node of a captured graph, in topological walk order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphNode {
    /// Node type.
    pub kind: NodeKind,
    /// Kernel symbol name for kernel nodes.
    pub kernel: Option<String>,
    /// Grid and block of kernel nodes.
    pub dims: Option<([u32; 3], [u32; 3], u32)>,
    /// Region (the last [`Capture::mark`] before the node).
    pub region: String,
    /// Whether the node runs on replay.
    pub enabled: bool,
}

/// An instantiated graph with its node map.
pub struct Graph {
    exec: CudaGraph,
    raw: Vec<sys::CUgraphNode>,
    nodes: Vec<GraphNode>,
    _keep: Vec<Rc<dyn Any>>,
}

impl std::fmt::Debug for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Graph").field("nodes", &self.nodes).finish()
    }
}

impl Graph {
    fn walk(exec: CudaGraph, state: CaptureState) -> Result<Graph> {
        let g = exec.cu_graph();
        let mut n = 0usize;
        // SAFETY: g is the live graph owned by exec; a null array asks for the count only.
        check_driver("cuGraphGetNodes", unsafe {
            sys::cuGraphGetNodes(g, std::ptr::null_mut(), &mut n)
        })?;
        let mut nodes: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); n];
        // SAFETY: nodes has room for n handles.
        check_driver("cuGraphGetNodes", unsafe {
            sys::cuGraphGetNodes(g, nodes.as_mut_ptr(), &mut n)
        })?;
        nodes.truncate(n);
        let mut ne = 0usize;
        let no_nodes: *mut sys::CUgraphNode = std::ptr::null_mut();
        let no_data: *mut sys::CUgraphEdgeData = std::ptr::null_mut();
        // SAFETY: count query (null arrays); edge data not requested.
        check_driver("cuGraphGetEdges", unsafe {
            sys::cuGraphGetEdges_v2(g, no_nodes, no_nodes, no_data, &mut ne)
        })?;
        let null_node: sys::CUgraphNode = std::ptr::null_mut();
        let (mut from, mut to) = (vec![null_node; ne], vec![null_node; ne]);
        if ne > 0 {
            // SAFETY: from and to have room for ne handles each; edge data not requested.
            let st = unsafe { sys::cuGraphGetEdges_v2(g, from.as_mut_ptr(), to.as_mut_ptr(), no_data, &mut ne) };
            check_driver("cuGraphGetEdges", st)?;
        }
        let edges: Vec<_> = from.into_iter().zip(to).take(ne).collect();
        let order = assign_regions(&nodes, &edges, &state.marks);
        let mut raw = Vec::with_capacity(order.len());
        let mut infos = Vec::with_capacity(order.len());
        for (node, region) in order {
            let mut t = CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
            // SAFETY: node is a node of g; t is a valid out-pointer.
            check_driver("cuGraphNodeGetType", unsafe { sys::cuGraphNodeGetType(node, &mut t) })?;
            let kind = NodeKind::from_raw(t);
            let (kernel, dims) = if kind == NodeKind::Kernel {
                kernel_info(node)?
            } else {
                (None, None)
            };
            raw.push(node);
            infos.push(GraphNode {
                kind,
                kernel,
                dims,
                region,
                enabled: true,
            });
        }
        Ok(Graph {
            exec,
            raw,
            nodes: infos,
            _keep: state.keep,
        })
    }

    /// Raw node handles, in [`Graph::nodes`] order (the launch trace reads kernel node parameters).
    pub(crate) fn raw_nodes(&self) -> &[sys::CUgraphNode] {
        &self.raw
    }

    /// The captured graph (owned by `exec`; the launch trace walks its nodes and edges).
    pub(crate) fn cu_graph(&self) -> sys::CUgraph {
        self.exec.cu_graph()
    }

    /// Nodes per region (the C++ `graph_region_counts` of one graph).
    pub fn region_counts(&self) -> std::collections::BTreeMap<String, usize> {
        let mut out = std::collections::BTreeMap::new();
        for n in &self.nodes {
            *out.entry(n.region.clone()).or_insert(0) += 1;
        }
        out
    }

    /// The nodes in walk order.
    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    /// Replays the graph on the stream it was captured on.
    ///
    /// # Errors
    ///
    /// The driver error of `cuGraphLaunch`.
    pub fn launch(&self) -> Result<()> {
        self.exec.launch().map_err(driver("cuGraphLaunch"))
    }

    /// Enables or disables node `index` for later replays.
    ///
    /// # Errors
    ///
    /// [`CudaError::Graph`] for an index out of range or a node type that cannot be disabled; the driver error.
    pub fn set_node(&mut self, index: usize, state: NodeState) -> Result<()> {
        let Some(info) = self.nodes.get_mut(index) else {
            return Err(CudaError::Graph(format!("node {index} of {}", self.raw.len())));
        };
        if !info.kind.can_disable() {
            return Err(CudaError::Graph(format!(
                "node {index} ({:?}) cannot be enabled or disabled",
                info.kind
            )));
        }
        let on = state == NodeState::Enabled;
        // SAFETY: raw[index] is a node of the graph exec was instantiated from (both owned by self).
        let st = unsafe { sys::cuGraphNodeSetEnabled(self.exec.cu_graph_exec(), self.raw[index], u32::from(on)) };
        check_driver("cuGraphNodeSetEnabled", st)?;
        info.enabled = on;
        Ok(())
    }

    /// The parameters of kernel node `index`, for a kernel whose one parameter is a `T` passed by value.
    fn kernel_node(&self, index: usize, size: usize) -> Result<sys::CUDA_KERNEL_NODE_PARAMS> {
        let Some(&node) = self
            .raw
            .get(index)
            .filter(|_| self.nodes[index].kind == NodeKind::Kernel)
        else {
            return Err(CudaError::Graph(format!("node {index} is not a kernel node")));
        };
        let mut p = std::mem::MaybeUninit::<sys::CUDA_KERNEL_NODE_PARAMS>::zeroed();
        // SAFETY: node is a kernel node of the graph self owns; p is a valid out-pointer.
        check_driver("cuGraphKernelNodeGetParams", unsafe {
            sys::cuGraphKernelNodeGetParams_v2(node, p.as_mut_ptr())
        })?;
        // SAFETY: the driver filled the struct (plain data; zero-initialised before).
        let p = unsafe { p.assume_init() };
        if p.func.is_null() || p.kernelParams.is_null() {
            return Err(CudaError::Graph(format!("node {index}: no function or no parameters")));
        }
        let f = p.func;
        // SAFETY: f is the node's live function handle; off and size are valid out-pointers.
        let layout = crate::module::param_layout(|i, off, sz| unsafe { sys::cuFuncGetParamInfo(f, i, off, sz) })?;
        if layout != [(0, size)] {
            return Err(CudaError::Graph(format!(
                "node {index}: parameters {layout:?}, expected one of {size} bytes"
            )));
        }
        Ok(p)
    }

    /// The one parameter of kernel node `index`, a kernel taking a single `T` by value (a launch struct).
    ///
    /// # Errors
    ///
    /// [`CudaError::Graph`] when the node is not a kernel node with exactly one parameter of `size_of::<T>()` bytes;
    /// the driver error.
    pub fn kernel_param<T: bytemuck::Pod>(&self, index: usize) -> Result<T> {
        let p = self.kernel_node(index, std::mem::size_of::<T>())?;
        // SAFETY: kernelParams[0] points to the node's copy of its one parameter, size_of::<T>() bytes (checked
        // above), valid while the graph lives; read unaligned, T is Pod.
        Ok(unsafe { std::ptr::read_unaligned((*p.kernelParams).cast::<T>()) })
    }

    /// Replaces the grid and the one parameter of kernel node `index` in the graph and in its executable
    /// (`cuGraphKernelNodeSetParams`, `cuGraphExecKernelNodeSetParams`): later replays and launch traces use them.
    /// Function, block and shared memory stay.
    ///
    /// # Errors
    ///
    /// As [`Graph::kernel_param`]; the driver error.
    pub fn set_kernel_param<T: bytemuck::Pod>(&mut self, index: usize, grid: [u32; 3], param: &T) -> Result<()> {
        let mut p = self.kernel_node(index, std::mem::size_of::<T>())?;
        let mut arg: *mut std::ffi::c_void = std::ptr::from_ref(param).cast_mut().cast();
        p.kernelParams = &mut arg;
        p.extra = std::ptr::null_mut();
        [p.gridDimX, p.gridDimY, p.gridDimZ] = grid;
        let node = self.raw[index];
        // SAFETY: node is a kernel node of the graph exec was instantiated from (both owned by self); p is the node's
        // own parameter set with a new grid and kernelParams pointing to one live T of the size the kernel takes; the
        // driver copies the parameter before returning.
        check_driver("cuGraphKernelNodeSetParams", unsafe {
            sys::cuGraphKernelNodeSetParams_v2(node, &p)
        })?;
        // SAFETY: as above, for the executable instantiated from this graph.
        check_driver("cuGraphExecKernelNodeSetParams", unsafe {
            sys::cuGraphExecKernelNodeSetParams_v2(self.exec.cu_graph_exec(), node, &p)
        })?;
        if let Some((g, _, _)) = self.nodes[index].dims.as_mut() {
            *g = grid;
        }
        Ok(())
    }

    /// Reads and caches the driver parameter template and initial value of a single-parameter kernel node `index`,
    /// so later in-place updates avoid `cuGraphKernelNodeGetParams` and `cuFuncGetParamInfo`.
    ///
    /// # Errors
    ///
    /// As [`Graph::kernel_param`].
    pub fn cache_kernel_node<T: bytemuck::Pod>(&self, index: usize) -> Result<(CachedKernelNode, T)> {
        let size = std::mem::size_of::<T>();
        let p = self.kernel_node(index, size)?;
        // SAFETY: kernelParams[0] points to the node's copy of its one parameter, `size` bytes (checked above).
        let val = unsafe { std::ptr::read_unaligned((*p.kernelParams).cast::<T>()) };
        Ok((
            CachedKernelNode {
                index,
                params: p,
                param_size: size,
            },
            val,
        ))
    }

    /// Updates a previously cached kernel node's grid and parameter value in the graph executable only
    /// (`cuGraphExecKernelNodeSetParams_v2`), with zero `GetParams` / `GetParamInfo` calls.
    ///
    /// # Errors
    ///
    /// [`CudaError::Graph`] on parameter size mismatch; the driver error.
    pub fn update_cached_kernel_node<T: bytemuck::Pod>(
        &mut self,
        cached: &mut CachedKernelNode,
        grid: [u32; 3],
        param: &T,
    ) -> Result<()> {
        if cached.param_size != std::mem::size_of::<T>() {
            return Err(CudaError::Graph(format!(
                "node {}: cached param size {}, got {}",
                cached.index,
                cached.param_size,
                std::mem::size_of::<T>()
            )));
        }
        let mut p = cached.params;
        let mut arg: *mut std::ffi::c_void = std::ptr::from_ref(param).cast_mut().cast();
        p.kernelParams = &mut arg;
        p.extra = std::ptr::null_mut();
        [p.gridDimX, p.gridDimY, p.gridDimZ] = grid;
        let node = self.raw[cached.index];
        // SAFETY: node belongs to self.exec's graph (both owned by self); p is its own validated parameter struct with
        // kernelParams pointing to one live T of checked size.
        check_driver("cuGraphExecKernelNodeSetParams", unsafe {
            sys::cuGraphExecKernelNodeSetParams_v2(self.exec.cu_graph_exec(), node, &p)
        })?;
        cached.params.gridDimX = grid[0];
        cached.params.gridDimY = grid[1];
        cached.params.gridDimZ = grid[2];
        if let Some((g, _, _)) = self.nodes[cached.index].dims.as_mut() {
            *g = grid;
        }
        Ok(())
    }

    /// Sets every node of `region` that can be disabled to `state` (the C++ `apply_ablation`); returns how many.
    ///
    /// # Errors
    ///
    /// As [`Graph::set_node`].
    pub fn set_region(&mut self, region: &str, state: NodeState) -> Result<usize> {
        let hits: Vec<usize> = (0..self.nodes.len())
            .filter(|&i| self.nodes[i].region == region && self.nodes[i].kind.can_disable())
            .collect();
        for &i in &hits {
            self.set_node(i, state)?;
        }
        Ok(hits.len())
    }
}

/// Whether a graph node runs on replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeState {
    /// The node runs.
    Enabled,
    /// The node is skipped.
    Disabled,
}

/// Cached driver parameter template for a single-parameter kernel node ([`Graph::cache_kernel_node`]),
/// used by [`Graph::update_cached_kernel_node`] to update `CUgraphExec` without `GetParams` / `GetParamInfo`.
#[derive(Debug, Clone)]
pub struct CachedKernelNode {
    index: usize,
    params: sys::CUDA_KERNEL_NODE_PARAMS,
    param_size: usize,
}

impl CachedKernelNode {
    /// Node index in [`Graph::nodes`].
    pub fn index(&self) -> usize {
        self.index
    }
}

type KernelInfo = (Option<String>, Option<([u32; 3], [u32; 3], u32)>);

fn kernel_info(node: sys::CUgraphNode) -> Result<KernelInfo> {
    let mut p = std::mem::MaybeUninit::<sys::CUDA_KERNEL_NODE_PARAMS>::zeroed();
    // SAFETY: node is a kernel node; p is a writable, zeroed parameter struct (the driver fills it).
    check_driver("cuGraphKernelNodeGetParams", unsafe {
        sys::cuGraphKernelNodeGetParams_v2(node, p.as_mut_ptr())
    })?;
    // SAFETY: the call succeeded and filled the struct; zeroed bytes are valid for every field (pointers, ints).
    let p = unsafe { p.assume_init() };
    let name = if !p.func.is_null() {
        Some(func_name(p.func)?)
    } else if !p.kern.is_null() {
        Some(kernel_name(p.kern)?)
    } else {
        None
    };
    let dims = (
        [p.gridDimX, p.gridDimY, p.gridDimZ],
        [p.blockDimX, p.blockDimY, p.blockDimZ],
        p.sharedMemBytes,
    );
    Ok((name, Some(dims)))
}

/// Topological walk of a captured graph with region labels: port of the C++ walk (`engine.cu` @ 1c7ac46, 931-955).
///
/// Roots are pushed in node order; the walk pops from the back (a stack). A node takes the current region; after a
/// node that was the capture leaf at a mark, the current region becomes that mark's. The first region is the first
/// mark's name (or `?` without marks).
pub(crate) fn assign_regions<N: Copy + Eq + Hash>(
    nodes: &[N],
    edges: &[(N, N)],
    marks: &[(String, Option<N>)],
) -> Vec<(N, String)> {
    let mut succ: HashMap<N, Vec<N>> = HashMap::new();
    let mut indeg: HashMap<N, usize> = nodes.iter().map(|&n| (n, 0)).collect();
    for &(a, b) in edges {
        succ.entry(a).or_default().push(b);
        *indeg.entry(b).or_default() += 1;
    }
    let mut leaf_region: HashMap<N, &str> = HashMap::new();
    for (name, leaf) in marks {
        if let Some(l) = leaf {
            leaf_region.insert(*l, name);
        }
    }
    let mut cur: &str = marks.first().map_or("?", |m| m.0.as_str());
    let mut stack: Vec<N> = nodes.iter().copied().filter(|n| indeg[n] == 0).collect();
    let mut out = Vec::with_capacity(nodes.len());
    while let Some(n) = stack.pop() {
        out.push((n, cur.to_owned()));
        if let Some(r) = leaf_region.get(&n) {
            cur = r;
        }
        for &m in succ.get(&n).map_or(&[][..], Vec::as_slice) {
            let d = indeg.get_mut(&m).expect("edge to a node of the graph");
            *d -= 1;
            if *d == 0 {
                stack.push(m);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regions(nodes: &[u32], edges: &[(u32, u32)], marks: &[(&str, Option<u32>)]) -> Vec<(u32, String)> {
        let marks: Vec<(String, Option<u32>)> = marks.iter().map(|(s, l)| ((*s).to_owned(), *l)).collect();
        assign_regions(nodes, edges, &marks)
    }

    #[test]
    fn linear_chain_takes_regions_from_marks() {
        // capture: mark embed (no leaf yet), k1, k2, mark attn (leaf k2), k3, mark head (leaf k3), k4
        let r = regions(
            &[1, 2, 3, 4],
            &[(1, 2), (2, 3), (3, 4)],
            &[("embed", None), ("attn", Some(2)), ("head", Some(3))],
        );
        let want: Vec<(u32, String)> = vec![
            (1, "embed".into()),
            (2, "embed".into()),
            (3, "attn".into()),
            (4, "head".into()),
        ];
        assert_eq!(r, want);
    }

    #[test]
    fn no_marks_is_question_mark() {
        let r = regions(&[7], &[], &[]);
        assert_eq!(r, vec![(7, "?".to_owned())]);
    }

    #[test]
    fn walk_visits_every_node_once_in_dependency_order() {
        // diamond 1 -> {2, 3} -> 4, plus an independent root 5
        let nodes = [1, 2, 3, 4, 5];
        let edges = [(1, 2), (1, 3), (2, 4), (3, 4)];
        let r = regions(&nodes, &edges, &[("a", None)]);
        let order: Vec<u32> = r.iter().map(|x| x.0).collect();
        assert_eq!(order.len(), 5);
        let pos = |n| order.iter().position(|&x| x == n).unwrap();
        for (a, b) in edges {
            assert!(pos(a) < pos(b), "{a} before {b} in {order:?}");
        }
        // stack semantics of the C++ walk: roots pushed [1, 5], 5 popped first
        assert_eq!(order[0], 5);
    }

    #[test]
    fn later_mark_on_the_same_leaf_wins() {
        // two marks without nodes between them share a leaf; the C++ map keeps the last
        let r = regions(&[1, 2], &[(1, 2)], &[("x", None), ("y", Some(1)), ("z", Some(1))]);
        assert_eq!(r[1], (2, "z".to_owned()));
    }

    #[test]
    fn only_kernel_copy_and_set_nodes_can_be_disabled() {
        assert!(NodeKind::Kernel.can_disable() && NodeKind::Memcpy.can_disable() && NodeKind::Memset.can_disable());
        assert!(!NodeKind::Other(3).can_disable());
        assert_eq!(
            NodeKind::from_raw(CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET),
            NodeKind::Memset
        );
    }
}
