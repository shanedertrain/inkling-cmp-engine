//! Launch trace of eager work (gate E2 of docs/RUST-PLAN.md 7.3; review R3 "What R5 must know" item 6).
//!
//! While a trace is recorded ([`Stream::start_trace`]), every kernel launch, device-to-device copy and cuBLASLt call
//! on the stream appends one line in the node format of the C++ oracle dumper (`ink-launch-trace/1`,
//! `engine/native/capi/dumps.cpp`): kernel name, grid, block, dynamic shared memory, parameter sizes, the argument
//! bytes laid out at the kernel's parameter offsets with every 8-byte word that points into the arena zeroed and
//! listed in `relocs` as `[byte offset, arena offset]`; copies as the memcpy node a captured `cudaMemcpyAsync` gives.
//! cuBLASLt launches its kernels itself, so a traced GEMM is captured into a one-call graph whose kernel nodes are
//! read exactly as the dumper reads them, and that graph is launched in place of the eager call. The lines of an
//! eager call form a chain (each depends on the previous one), as the C++ engine's captured graph does.
//!
//! [`Graph::launch_trace`](crate::Graph::launch_trace) reads a captured graph the way the C++ dumper does (package
//! R6): every node from the graph itself, so E2 compares the two engines' graphs node by node.

use std::fmt::Write as _;

use cudarc::driver::sys;

use crate::arena::Arena;
use crate::context::Stream;
use crate::error::{CudaError, Result, check_driver};
use crate::module::{LaunchCfg, func_name, kernel_name, param_layout};

/// An open trace: the arena span pointers are relocated against, and the node bodies so far.
#[derive(Debug)]
pub(crate) struct TraceState {
    base: u64,
    cap: usize,
    bodies: Vec<String>,
}

impl Stream {
    /// Starts recording a launch trace; pointers into `arena` are written as arena offsets.
    ///
    /// # Errors
    ///
    /// [`CudaError::Capturing`] during a capture, [`CudaError::Graph`] when a trace is already open.
    pub fn start_trace(&self, arena: &Arena) -> Result<()> {
        self.not_capturing("start trace")?;
        let mut t = self.trace.borrow_mut();
        if t.is_some() {
            return Err(CudaError::Graph("a launch trace is already being recorded".into()));
        }
        let (base, cap) = arena.base_and_cap();
        *t = Some(TraceState {
            base,
            cap,
            bodies: Vec::new(),
        });
        Ok(())
    }

    /// Stops recording and returns the trace's node lines (`{"i":..,"type":..,"deps":[..],...}`, one JSON object
    /// per line without a newline), or `None` when no trace was open.
    pub fn finish_trace(&self) -> Option<Vec<String>> {
        let t = self.trace.borrow_mut().take()?;
        Some(
            t.bodies
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    let deps = if i == 0 { String::new() } else { (i - 1).to_string() };
                    format!(
                        "{{\"i\":{i},{}",
                        b.replacen("\"deps\":[]", &format!("\"deps\":[{deps}]"), 1)
                    )
                })
                .collect(),
        )
    }

    /// Whether a launch trace is being recorded.
    pub fn is_tracing(&self) -> bool {
        self.trace.borrow().is_some()
    }

    /// A kernel node: `bytes` are the argument bytes laid out at the parameter offsets of `layout`.
    pub(crate) fn trace_kernel(&self, name: &str, cfg: LaunchCfg, layout: &[(usize, usize)], bytes: Vec<u8>) {
        if let Some(t) = self.trace.borrow_mut().as_mut() {
            let body = kernel_body(name, cfg, layout, bytes, t.base, t.cap);
            t.bodies.push(body);
        }
    }

    /// A device-to-device copy of `bytes` bytes, as the memcpy node of a captured `cudaMemcpyAsync`.
    pub(crate) fn trace_dtod(&self, dst: u64, src: u64, bytes: usize) {
        if let Some(t) = self.trace.borrow_mut().as_mut() {
            let p = |a: u64| ptr_json(a, t.base, t.cap);
            let body = format!(
                "\"type\":\"memcpy\",\"deps\":[],\"kind\":3,\"extent\":[{bytes},1,1],\"src_pos\":[0,0,0],\"src_pitch\":0,\
                 \"src_xy\":[0,0],\"dst_pos\":[0,0,0],\"dst_pitch\":0,\"dst_xy\":[0,0],\"src\":{},\"dst\":{}}}",
                p(src),
                p(dst)
            );
            t.bodies.push(body);
        }
    }

    /// Runs `body` (library calls that launch kernels themselves, i.e. cuBLASLt) captured into a graph, traces the
    /// graph's nodes in topological order and launches it; without an open trace, runs `body` eagerly.
    ///
    /// # Errors
    ///
    /// The error of `body`, of the capture, of reading a node, or of the launch.
    pub(crate) fn traced_library_call(&self, body: impl FnOnce() -> Result<()>) -> Result<()> {
        let Some(state) = self.trace.borrow_mut().take() else {
            return body();
        };
        let res = self.capture_and_trace(body, state.base, state.cap);
        let mut state = state;
        let out = match res {
            Ok(bodies) => {
                state.bodies.extend(bodies);
                Ok(())
            }
            Err(e) => Err(e),
        };
        *self.trace.borrow_mut() = Some(state);
        out
    }

    fn capture_and_trace(&self, body: impl FnOnce() -> Result<()>, base: u64, cap: usize) -> Result<Vec<String>> {
        let capture = self.begin_capture()?;
        body()?;
        let graph = capture.end()?;
        let mut bodies = Vec::new();
        for (node, info) in graph.raw_nodes().iter().zip(graph.nodes()) {
            match info.kind {
                crate::graph::NodeKind::Kernel => bodies.push(format!(
                    "\"type\":\"kernel\",\"deps\":[],{}}}",
                    graph_kernel_fields(*node, base, cap)?
                )),
                other => {
                    return Err(CudaError::Graph(format!(
                        "traced library call produced a {other:?} node; only kernel nodes are traced"
                    )));
                }
            }
        }
        graph.launch()?;
        // the graph is dropped on return: wait for its launch first (tracing is a measurement mode, never timed)
        self.synchronize()?;
        Ok(bodies)
    }
}

impl crate::graph::Graph {
    /// The launch trace of this captured graph, as the C++ dumper (`dumps.cpp` `dump_launch_trace` @ 1c7ac46) writes
    /// it: nodes in Kahn's order (a FIFO queue, ties in the order `cuGraphGetNodes` returns), each line
    /// `{"i":..,"type":..,"deps":[ranks of the predecessors, sorted],...}` with the kernel, memcpy or memset fields;
    /// pointers into `arena` as arena offsets. Returns the lines and the edge count.
    ///
    /// # Errors
    ///
    /// The driver error of a node or edge query; [`CudaError::Graph`] for a cycle, a memcpy node with a CUDA array or
    /// a kernel node without a function.
    pub fn launch_trace(&self, arena: &Arena) -> Result<(Vec<String>, usize)> {
        let (base, cap) = arena.base_and_cap();
        let g = self.cu_graph();
        let mut nn = 0usize;
        // SAFETY: g is the live graph owned by self; a null array asks for the count only.
        check_driver("cuGraphGetNodes", unsafe {
            sys::cuGraphGetNodes(g, std::ptr::null_mut(), &mut nn)
        })?;
        let mut nodes: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); nn];
        if nn > 0 {
            // SAFETY: nodes has room for nn handles.
            check_driver("cuGraphGetNodes", unsafe {
                sys::cuGraphGetNodes(g, nodes.as_mut_ptr(), &mut nn)
            })?;
        }
        nodes.truncate(nn);
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
        let pos: std::collections::HashMap<sys::CUgraphNode, usize> =
            nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        let index = |n: &sys::CUgraphNode| {
            pos.get(n)
                .copied()
                .ok_or_else(|| CudaError::Graph("edge to a node not in the graph".into()))
        };
        let mut succ = vec![Vec::new(); nn];
        let mut pred = vec![Vec::new(); nn];
        let mut indeg = vec![0usize; nn];
        for (a, b) in from.iter().zip(&to).take(ne) {
            let (a, b) = (index(a)?, index(b)?);
            succ[a].push(b);
            pred[b].push(a);
            indeg[b] += 1;
        }
        let order = kahn_fifo(&succ, indeg);
        if order.len() != nn {
            return Err(CudaError::Graph("launch-trace: graph has a cycle".into()));
        }
        let mut rank = vec![0usize; nn];
        for (r, &i) in order.iter().enumerate() {
            rank[i] = r;
        }
        let mut lines = Vec::with_capacity(nn);
        for (r, &i) in order.iter().enumerate() {
            let node = nodes[i];
            let mut t = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
            // SAFETY: node is a node of g; t is a valid out-pointer.
            check_driver("cuGraphNodeGetType", unsafe { sys::cuGraphNodeGetType(node, &mut t) })?;
            let mut dep: Vec<usize> = pred[i].iter().map(|&p| rank[p]).collect();
            dep.sort_unstable();
            let deps: Vec<String> = dep.iter().map(ToString::to_string).collect();
            let body = match t {
                sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL => {
                    format!(",{}", graph_kernel_fields(node, base, cap)?)
                }
                sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY => format!(",{}", memcpy_body(node, base, cap)?),
                sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET => format!(",{}", memset_body(node, base, cap)?),
                _ => String::new(),
            };
            lines.push(format!(
                "{{\"i\":{r},\"type\":\"{}\",\"deps\":[{}]{body}}}",
                type_name(t),
                deps.join(",")
            ));
        }
        Ok((lines, ne))
    }
}

/// Kahn's topological order with a FIFO queue, roots and ties in index order (the C++ dumper's `std::deque`).
fn kahn_fifo(succ: &[Vec<usize>], mut indeg: Vec<usize>) -> Vec<usize> {
    let mut q: std::collections::VecDeque<usize> = (0..succ.len()).filter(|&i| indeg[i] == 0).collect();
    let mut order = Vec::with_capacity(succ.len());
    while let Some(i) = q.pop_front() {
        order.push(i);
        for &s in &succ[i] {
            indeg[s] -= 1;
            if indeg[s] == 0 {
                q.push_back(s);
            }
        }
    }
    order
}

/// The C++ dumper's `type_name` (runtime node type names; the driver enum has the same values).
fn type_name(t: sys::CUgraphNodeType) -> &'static str {
    use sys::CUgraphNodeType as T;
    match t {
        T::CU_GRAPH_NODE_TYPE_KERNEL => "kernel",
        T::CU_GRAPH_NODE_TYPE_MEMCPY => "memcpy",
        T::CU_GRAPH_NODE_TYPE_MEMSET => "memset",
        T::CU_GRAPH_NODE_TYPE_HOST => "host",
        T::CU_GRAPH_NODE_TYPE_GRAPH => "graph",
        T::CU_GRAPH_NODE_TYPE_EMPTY => "empty",
        T::CU_GRAPH_NODE_TYPE_WAIT_EVENT => "wait_event",
        T::CU_GRAPH_NODE_TYPE_EVENT_RECORD => "event_record",
        T::CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL => "ext_sem_signal",
        T::CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT => "ext_sem_wait",
        T::CU_GRAPH_NODE_TYPE_MEM_ALLOC => "mem_alloc",
        T::CU_GRAPH_NODE_TYPE_MEM_FREE => "mem_free",
        T::CU_GRAPH_NODE_TYPE_CONDITIONAL => "conditional",
        _ => "other",
    }
}

/// `cudaMemcpyKind` of a driver copy: host / device ends (unified counts as device; the engine's copies are
/// device-to-device).
fn memcpy_kind(src: sys::CUmemorytype, dst: sys::CUmemorytype) -> u32 {
    let host = |m| m == sys::CUmemorytype::CU_MEMORYTYPE_HOST;
    match (host(src), host(dst)) {
        (true, true) => 0,
        (true, false) => 1,
        (false, true) => 2,
        (false, false) => 3,
    }
}

/// A memcpy node in the C++ dumper's `memcpy_json` fields. The runtime's `cudaMemcpy3DParms` is read there; the
/// driver's `CUDA_MEMCPY3D` here: `kind` from the memory types, `extent` = (WidthInBytes, Height, Depth), positions =
/// (XInBytes, Y, Z), `pitch` = Pitch, `xy` = (0, Height of the source / destination allocation): the driver struct
/// has no `xsize`; for the engine's 1-D copies the runtime reports 0 for both.
fn memcpy_body(node: sys::CUgraphNode, base: u64, cap: usize) -> Result<String> {
    let mut m = std::mem::MaybeUninit::<sys::CUDA_MEMCPY3D>::zeroed();
    // SAFETY: node is a memcpy node of a live graph; m is a writable, zeroed parameter struct the driver fills.
    check_driver("cuGraphMemcpyNodeGetParams", unsafe {
        sys::cuGraphMemcpyNodeGetParams(node, m.as_mut_ptr())
    })?;
    // SAFETY: the call succeeded and filled the struct; zeroed bytes are valid for every field (pointers, ints, and
    // the memory-type enum is overwritten by the driver).
    let m = unsafe { m.assume_init() };
    if !m.srcArray.is_null() || !m.dstArray.is_null() {
        return Err(CudaError::Graph("launch-trace: memcpy node with a CUDA array".into()));
    }
    let end = |t: sys::CUmemorytype, host: u64, dev: u64| {
        if t == sys::CUmemorytype::CU_MEMORYTYPE_HOST {
            host
        } else {
            dev
        }
    };
    let src = end(m.srcMemoryType, m.srcHost as u64, m.srcDevice);
    let dst = end(m.dstMemoryType, m.dstHost as u64, m.dstDevice);
    Ok(format!(
        "\"kind\":{},\"extent\":[{},{},{}],\"src_pos\":[{},{},{}],\"src_pitch\":{},\"src_xy\":[0,{}],\
         \"dst_pos\":[{},{},{}],\"dst_pitch\":{},\"dst_xy\":[0,{}],\"src\":{},\"dst\":{}",
        memcpy_kind(m.srcMemoryType, m.dstMemoryType),
        m.WidthInBytes,
        m.Height,
        m.Depth,
        m.srcXInBytes,
        m.srcY,
        m.srcZ,
        m.srcPitch,
        m.srcHeight,
        m.dstXInBytes,
        m.dstY,
        m.dstZ,
        m.dstPitch,
        m.dstHeight,
        ptr_json(src, base, cap),
        ptr_json(dst, base, cap)
    ))
}

/// A memset node in the C++ dumper's `memset_json` fields.
fn memset_body(node: sys::CUgraphNode, base: u64, cap: usize) -> Result<String> {
    let mut m = std::mem::MaybeUninit::<sys::CUDA_MEMSET_NODE_PARAMS>::zeroed();
    // SAFETY: node is a memset node of a live graph; m is a writable, zeroed parameter struct the driver fills.
    check_driver("cuGraphMemsetNodeGetParams", unsafe {
        sys::cuGraphMemsetNodeGetParams(node, m.as_mut_ptr())
    })?;
    // SAFETY: the call succeeded and filled the struct (plain integers and a device address).
    let m = unsafe { m.assume_init() };
    Ok(format!(
        "\"value\":{},\"elem\":{},\"width\":{},\"height\":{},\"pitch\":{},\"dst\":{}",
        m.value,
        m.elementSize,
        m.width,
        m.height,
        m.pitch,
        ptr_json(m.dst, base, cap)
    ))
}

/// `null`, `{"arena":<offset>}` or `{"ext":"0x.."}`, as the C++ dumper's `PtrFmt`.
fn ptr_json(p: u64, base: u64, cap: usize) -> String {
    if p == 0 {
        "null".into()
    } else if p >= base && p < base + cap as u64 {
        format!("{{\"arena\":{}}}", p - base)
    } else {
        format!("{{\"ext\":\"{p:#x}\"}}")
    }
}

/// The kernel node body of the C++ dumper's `kernel_json` (after `"i"`), with an empty `deps` placeholder.
fn kernel_body(name: &str, cfg: LaunchCfg, layout: &[(usize, usize)], bytes: Vec<u8>, base: u64, cap: usize) -> String {
    format!(
        "\"type\":\"kernel\",\"deps\":[],{}}}",
        kernel_fields(name, cfg, layout, bytes, base, cap)
    )
}

/// The fields of the C++ dumper's `kernel_json`, `"name"` to `"relocs"`.
fn kernel_fields(
    name: &str,
    cfg: LaunchCfg,
    layout: &[(usize, usize)],
    mut bytes: Vec<u8>,
    base: u64,
    cap: usize,
) -> String {
    let mut relocs = Vec::new();
    let mut o = 0;
    while o + 8 <= bytes.len() {
        let mut w = [0u8; 8];
        w.copy_from_slice(&bytes[o..o + 8]);
        let v = u64::from_le_bytes(w);
        if v >= base && v < base + cap as u64 {
            relocs.push(format!("[{o},{}]", v - base));
            bytes[o..o + 8].fill(0);
        }
        o += 8;
    }
    let mut args = String::with_capacity(bytes.len() * 2);
    for b in &bytes {
        let _ = write!(args, "{b:02x}");
    }
    let psz: Vec<String> = layout.iter().map(|&(_, s)| s.to_string()).collect();
    let [gx, gy, gz] = cfg.grid;
    let [bx, by, bz] = cfg.block;
    format!(
        "\"name\":\"{name}\",\"grid\":[{gx},{gy},{gz}],\"block\":[{bx},{by},{bz}],\
         \"smem\":{},\"psz\":[{}],\"args\":\"{args}\",\"relocs\":[{}]",
        cfg.smem,
        psz.join(","),
        relocs.join(",")
    )
}

/// A captured kernel node, read as `dumps.cpp` `kernel_json` reads it (its fields from `"name"`).
fn graph_kernel_fields(node: sys::CUgraphNode, base: u64, cap: usize) -> Result<String> {
    let mut p = std::mem::MaybeUninit::<sys::CUDA_KERNEL_NODE_PARAMS>::zeroed();
    // SAFETY: node is a kernel node of a live graph (the caller holds it); p is a valid out-pointer.
    check_driver("cuGraphKernelNodeGetParams", unsafe {
        sys::cuGraphKernelNodeGetParams_v2(node, p.as_mut_ptr())
    })?;
    // SAFETY: the driver filled the struct (plain data; zero-initialised before).
    let p = unsafe { p.assume_init() };
    let (name, layout) = if !p.func.is_null() {
        let f = p.func;
        // SAFETY: f is the node's live function handle; off and size are valid out-pointers.
        (
            func_name(f)?,
            param_layout(|i, off, size| unsafe { sys::cuFuncGetParamInfo(f, i, off, size) })?,
        )
    } else if !p.kern.is_null() {
        let k = p.kern;
        // SAFETY: k is the node's live kernel handle; off and size are valid out-pointers.
        (
            kernel_name(k)?,
            param_layout(|i, off, size| unsafe { sys::cuKernelGetParamInfo(k, i, off, size) })?,
        )
    } else {
        return Err(CudaError::Graph("kernel node without a function".into()));
    };
    let total = layout.iter().map(|&(o, s)| o + s).max().unwrap_or(0);
    let mut bytes = vec![0u8; total];
    if !p.kernelParams.is_null() {
        for (i, &(off, size)) in layout.iter().enumerate() {
            // SAFETY: kernelParams has one pointer per parameter (the driver's copy, owned by the node), each to
            // `size` bytes of the parameter's value.
            let src = unsafe { std::slice::from_raw_parts((*p.kernelParams.add(i)).cast::<u8>(), size) };
            bytes[off..off + size].copy_from_slice(src);
        }
    } else if !p.extra.is_null() {
        let (mut buf, mut len): (*const u8, usize) = (std::ptr::null(), 0);
        let mut x = p.extra;
        // CU_LAUNCH_PARAM_END = 0, CU_LAUNCH_PARAM_BUFFER_POINTER = 1, CU_LAUNCH_PARAM_BUFFER_SIZE = 2 (cuda.h)
        // SAFETY: `extra` is a key/value list terminated by CU_LAUNCH_PARAM_END, owned by the node.
        unsafe {
            while !(*x).is_null() {
                match *x as usize {
                    1 => buf = (*x.add(1)).cast::<u8>(),
                    2 => len = *(*x.add(1)).cast::<usize>(),
                    _ => {}
                }
                x = x.add(2);
            }
        }
        if buf.is_null() {
            return Err(CudaError::Graph(format!("{name}: extra without a buffer")));
        }
        // SAFETY: the buffer holds `len` bytes of argument values (CU_LAUNCH_PARAM_BUFFER_SIZE).
        bytes = unsafe { std::slice::from_raw_parts(buf, len) }.to_vec();
    } else if total > 0 {
        return Err(CudaError::Graph(format!("{name}: no argument values")));
    }
    let cfg = LaunchCfg {
        grid: [p.gridDimX, p.gridDimY, p.gridDimZ],
        block: [p.blockDimX, p.blockDimY, p.blockDimZ],
        smem: p.sharedMemBytes,
    };
    Ok(kernel_fields(&name, cfg, &layout, bytes, base, cap))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_body_relocates_arena_words_like_the_cpp_dumper() {
        // two params: a pointer into the arena (base 0x1000, cap 0x100) and an int; one pointer outside
        let base = 0x1000u64;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(base + 0x20).to_le_bytes());
        bytes.extend_from_slice(&7i32.to_le_bytes());
        bytes.extend_from_slice(&[0; 4]);
        bytes.extend_from_slice(&0x9000u64.to_le_bytes());
        let cfg = LaunchCfg {
            grid: [4, 1, 1],
            block: [256, 1, 1],
            smem: 0,
        };
        let b = kernel_body("k", cfg, &[(0, 8), (8, 4), (16, 8)], bytes, base, 0x100);
        let args = ["0000000000000000", "07000000", "00000000", "0090000000000000"].concat();
        assert_eq!(
            b,
            format!(
                "\"type\":\"kernel\",\"deps\":[],\"name\":\"k\",\"grid\":[4,1,1],\"block\":[256,1,1],\"smem\":0,\
                 \"psz\":[8,4,8],\"args\":\"{args}\",\"relocs\":[[0,32]]}}"
            )
        );
        assert_eq!(ptr_json(0, base, 0x100), "null");
        assert_eq!(ptr_json(0x1010, base, 0x100), "{\"arena\":16}");
        assert_eq!(ptr_json(0x2000, base, 0x100), "{\"ext\":\"0x2000\"}");
    }

    #[test]
    fn kahn_order_is_fifo_with_index_ties_like_the_cpp_dumper() {
        // 0 -> {1, 2} -> 3, plus a root 4: deque order 0, 4, 1, 2, 3 (the stack walk of the region map differs)
        let succ = vec![vec![1, 2], vec![3], vec![3], vec![], vec![]];
        let indeg = vec![0, 1, 1, 2, 0];
        assert_eq!(kahn_fifo(&succ, indeg), vec![0, 4, 1, 2, 3]);
        // a cycle leaves nodes out (the caller refuses it)
        assert_eq!(kahn_fifo(&[vec![1], vec![0]], vec![1, 1]), Vec::<usize>::new());
    }

    #[test]
    fn copy_kinds_follow_cuda_memcpy_kind() {
        use sys::CUmemorytype as M;
        assert_eq!(memcpy_kind(M::CU_MEMORYTYPE_DEVICE, M::CU_MEMORYTYPE_DEVICE), 3);
        assert_eq!(memcpy_kind(M::CU_MEMORYTYPE_UNIFIED, M::CU_MEMORYTYPE_DEVICE), 3);
        assert_eq!(memcpy_kind(M::CU_MEMORYTYPE_HOST, M::CU_MEMORYTYPE_DEVICE), 1);
        assert_eq!(memcpy_kind(M::CU_MEMORYTYPE_DEVICE, M::CU_MEMORYTYPE_HOST), 2);
        assert_eq!(type_name(sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET), "memset");
    }
}
