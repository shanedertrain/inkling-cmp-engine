//! The device weight set and its loader (port of `LayerW`, `Weights` in `model.hpp` and `load_weights` in
//! `weights.cpp` @ 66fe353).
//!
//! Every tensor is taken from the arena in the C++ order with the C++ name, so the weights occupy the same offsets
//! and the `weight-sums` and `arena-map` dumps compare line by line. Dense matrices stay BF16 `[out, in]`; small
//! per-channel parameters become fp32 (exact: BF16 or F32 in the checkpoint); routed experts are repacked
//! ([`crate::repack`]).

use std::path::Path;

use ink_cuda::{Arena, Bf16, DevBuf, Stream};
use safetensors::Dtype;

use crate::checkpoint::{CHECKPOINT_FILE, Checkpoint};
use crate::dims::{GATE_RANK, H, INTER, LOOP_END, LOOP_START, NEXP, NLAYERS, is_full_layer};
use crate::error::{Result, invalid};
use crate::repack::{ExpertShape, Proj, repack_mxfp8, repack_nvfp4_n1s1};

/// Storage format of a layer's routed experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertFmt {
    /// NVFP4 (layers 0-19), n1 layout.
    Nvfp4,
    /// MXFP8 (layers 20-23).
    Mxfp8,
}

/// One expert projection: repacked weights, repacked scales and the per-expert fp32 factor (NVFP4: amax /
/// denominator x 256; MXFP8: 256).
#[derive(Debug, Clone)]
pub struct ExpertW {
    /// Repacked weights.
    pub w: DevBuf<u8>,
    /// Repacked scales.
    pub s: DevBuf<u8>,
    /// Per-expert factor `[128]`.
    pub f: DevBuf<f32>,
}

/// One loop controller (order q, k, v, mlp): down bf16 `[64, 4608]`, forget and gain fp32 `[2304, 64]` and their
/// transposes `[64, 2304]`.
#[derive(Debug, Clone)]
pub struct Controller {
    /// Down projection.
    pub down: DevBuf<Bf16>,
    /// Forget gate.
    pub forget: DevBuf<f32>,
    /// Gain gate.
    pub gain: DevBuf<f32>,
    /// Forget gate, transposed.
    pub forget_t: DevBuf<f32>,
    /// Gain gate, transposed.
    pub gain_t: DevBuf<f32>,
}

/// Weights of one checkpoint layer.
#[derive(Debug, Clone)]
pub struct LayerW {
    /// Layer index.
    pub idx: usize,
    /// Full attention (layers 5, 11, 17, 23).
    pub full: bool,
    /// Inside the looped middle (6-17).
    pub looped: bool,
    /// Attention input norm (sliding layers).
    pub attn_norm: Option<DevBuf<f32>>,
    /// q / k / v input norms (full layers).
    pub q_norm: Option<DevBuf<f32>>,
    /// See `q_norm`.
    pub k_norm: Option<DevBuf<f32>>,
    /// See `q_norm`.
    pub v_norm: Option<DevBuf<f32>>,
    /// MLP input norm.
    pub mlp_norm: DevBuf<f32>,
    /// Query L2-clip scale `[24, 128]`.
    pub q_scale: DevBuf<f32>,
    /// Relative L2-clip scale `[24, 16]`.
    pub r_scale: DevBuf<f32>,
    /// Relative-logit projection bf16 `[16, 1024]`.
    pub rel_proj: DevBuf<Bf16>,
    /// Static key conv by lag `[4, 1024]` (full layers).
    pub k_static: Option<DevBuf<f32>>,
    /// Static value conv (full layers).
    pub v_static: Option<DevBuf<f32>>,
    /// Key short conv `[1024, 4]` (sliding layers).
    pub k_conv: Option<DevBuf<f32>>,
    /// Value short conv (sliding layers).
    pub v_conv: Option<DevBuf<f32>>,
    /// Attention-input short conv `[2304, 4]`.
    pub attn_conv: DevBuf<f32>,
    /// MLP-input short conv `[2304, 4]`.
    pub mlp_conv: DevBuf<f32>,
    /// Router bf16 `[128, 2304]`.
    pub router: DevBuf<Bf16>,
    /// Router bias `[128]`.
    pub router_bias: DevBuf<f32>,
    /// AttnRes queries `[1 or 3, 2304]` (layers > 0).
    pub attn_query: Option<DevBuf<f32>>,
    /// Rows of `attn_query` (0 for layer 0).
    pub n_attn_query: usize,
    /// MLP AttnRes query `[2304]`.
    pub mlp_query: DevBuf<f32>,
    /// Dense projections, bf16 `[out, in]`.
    pub wq: DevBuf<Bf16>,
    /// See `wq`.
    pub wk: DevBuf<Bf16>,
    /// See `wq`.
    pub wv: DevBuf<Bf16>,
    /// See `wq`.
    pub wr: DevBuf<Bf16>,
    /// See `wq`.
    pub wo: DevBuf<Bf16>,
    /// See `wq`.
    pub wgate: DevBuf<Bf16>,
    /// See `wq`.
    pub wsink: DevBuf<Bf16>,
    /// The input projections as one `[6552 + 2 (1024 (+ 32)), 2304]` bf16 matrix (a view over wq .. wv / wv_dyn,
    /// adjacent in the arena): the decode-row projections (D1).
    pub wqkv: DevBuf<Bf16>,
    /// Dynamic key conv weights `[32, 2304]` (full layers).
    pub wk_dyn: Option<DevBuf<Bf16>>,
    /// Dynamic value conv weights (full layers).
    pub wv_dyn: Option<DevBuf<Bf16>>,
    /// Loop controllers q, k, v, mlp (looped layers).
    pub controllers: Option<[Controller; 4]>,
    /// Expert format.
    pub fmt: ExpertFmt,
    /// Experts: w13, w2.
    pub experts: [ExpertW; 2],
}

/// The device weight set.
#[derive(Debug, Clone)]
pub struct Weights {
    /// The 24 layers.
    pub layers: Vec<LayerW>,
    /// Embedding bf16 `[201024, 2304]`.
    pub embed: DevBuf<Bf16>,
    /// Embedding norm.
    pub embed_norm: DevBuf<f32>,
    /// Final AttnRes query `[2304]`.
    pub final_query: DevBuf<f32>,
    /// Final norm.
    pub final_norm: DevBuf<f32>,
    /// Unembedding bf16 `[201024, 2304]` (padded rows).
    pub unembed: DevBuf<Bf16>,
    /// Message queries `[12, 4, 2304]`: q, k, v, mlp.
    pub msg_queries: DevBuf<f32>,
    /// Bytes uploaded (the C++ `Weights::bytes`: requested sizes, alignment gaps excluded).
    pub bytes: usize,
}

/// A weight buffer's dump name and arena range, for `weight-sums`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightRegion {
    /// Name as in the C++ dump (`layers.<l>.<field>`, `embed`, ...).
    pub name: String,
    /// Arena offset.
    pub offset: usize,
}

/// The arena estimate of the weights (`weights_bytes_estimate`).
pub const WEIGHTS_BYTES_ESTIMATE: usize = 21_000_000_000;

struct Loader<'a> {
    ar: &'a mut Arena,
    st: &'a Stream,
    ck: &'a Checkpoint,
    tot: usize,
}

impl Loader<'_> {
    /// `up_bf16`: the tensor's bytes as they are.
    fn bf16(&mut self, name: &str) -> Result<DevBuf<Bf16>> {
        let t = self.ck.tensor(name)?;
        t.expect(Dtype::BF16)?;
        let d = self.ar.take::<Bf16>(t.numel(), name)?;
        self.st.htod_bytes(&d, t.bytes)?;
        self.tot += t.bytes.len();
        Ok(d)
    }

    /// `host_f32`.
    fn host_f32(&self, name: &str) -> Result<Vec<f32>> {
        self.ck.tensor(name)?.to_f32()
    }

    /// `up_f32v`: a host vector under the arena name `what`.
    fn vec<T: bytemuck::Pod>(&mut self, v: &[T], what: &str) -> Result<DevBuf<T>> {
        let d = self.ar.take::<T>(v.len(), what)?;
        self.st.htod(&d, v)?;
        self.tot += size_of_val(v);
        Ok(d)
    }

    /// `up_f32`: the tensor as fp32 under its own name.
    fn f32(&mut self, name: &str) -> Result<DevBuf<f32>> {
        let v = self.host_f32(name)?;
        self.vec(&v, name)
    }

    /// One view over consecutive takes (the given parts, in order), refused unless each starts where the previous
    /// ends (the C++ loader's contiguity check of the input projections).
    fn contiguous(&self, parts: &[Option<&DevBuf<Bf16>>], l: usize) -> Result<DevBuf<Bf16>> {
        let mut parts = parts.iter().flatten();
        let first = parts
            .next()
            .ok_or_else(|| invalid("input projections", "no parts".to_owned()))?;
        let mut end = first.offset() + first.bytes();
        for p in parts {
            if p.offset() != end {
                return Err(invalid(
                    "input projections",
                    format!("layer {l}: not contiguous at byte {end}"),
                ));
            }
            end = p.offset() + p.bytes();
        }
        Ok(self.ar.span(first.offset()..end)?.cast::<Bf16>()?)
    }

    fn controller(&mut self, p: &str, c: &str) -> Result<Controller> {
        let base = format!("{p}controllers.{c}.");
        let down = self.bf16(&format!("{base}down_weight"))?;
        let forget = self.f32(&format!("{base}forget_weight"))?;
        let gain = self.f32(&format!("{base}gain_weight"))?;
        let forget_t = self.vec(
            &transpose_gate(&self.host_f32(&format!("{base}forget_weight"))?)?,
            "controller T",
        )?;
        let gain_t = self.vec(
            &transpose_gate(&self.host_f32(&format!("{base}gain_weight"))?)?,
            "controller T",
        )?;
        Ok(Controller {
            down,
            forget,
            gain,
            forget_t,
            gain_t,
        })
    }

    fn experts(&mut self, l: usize, e: &str, fmt: ExpertFmt, threads: usize) -> Result<[ExpertW; 2]> {
        let mut out = Vec::with_capacity(2);
        for (proj, pname) in [(Proj::W13, "w13"), (Proj::W2, "w2")] {
            let shape = match proj {
                Proj::W13 => ExpertShape {
                    experts: NEXP,
                    n: 2 * INTER,
                    k: H,
                },
                Proj::W2 => ExpertShape {
                    experts: NEXP,
                    n: H,
                    k: INTER,
                },
            };
            let ck = self.ck;
            let tw = ck.tensor(&format!("{e}{pname}_weight"))?;
            let ts = ck.tensor(&format!("{e}{pname}_weight_scale"))?;
            let mut f = vec![256.0f32; NEXP];
            let (w, s) = match fmt {
                ExpertFmt::Nvfp4 => {
                    if tw.dtype != Dtype::U8 || ts.dtype != Dtype::F8_E4M3 {
                        return Err(invalid(
                            "nvfp4 dtypes",
                            format!("layer {l}: {:?} / {:?}", tw.dtype, ts.dtype),
                        ));
                    }
                    let packed = repack_nvfp4_n1s1(tw.bytes, ts.bytes, shape, proj, threads)?;
                    let am = self.host_f32(&format!("{e}{pname}_weight_amax"))?;
                    let de = self.host_f32(&format!("{e}{pname}_weight_outer_scale_denominator"))?;
                    f = nvfp4_factors(&am, &de)?;
                    packed
                }
                ExpertFmt::Mxfp8 => {
                    if tw.dtype != Dtype::F8_E4M3 || ts.dtype != Dtype::U8 {
                        return Err(invalid(
                            "mxfp8 dtypes",
                            format!("layer {l}: {:?} / {:?}", tw.dtype, ts.dtype),
                        ));
                    }
                    repack_mxfp8(tw.bytes, ts.bytes, shape, proj, threads)?
                }
            };
            let w = self.vec(&w, "expert W")?;
            let s = self.vec(&s, "expert S")?;
            let f = self.vec(&f, "expert F")?;
            out.push(ExpertW { w, s, f });
        }
        let [w13, w2]: [ExpertW; 2] = out.try_into().map_err(|_| invalid("experts", "two projections"))?;
        Ok([w13, w2])
    }

    fn layer(&mut self, l: usize, threads: usize) -> Result<LayerW> {
        let p = format!("model.llm.layers.{l}.");
        let n = |s: &str| format!("{p}{s}");
        let full = is_full_layer(l);
        let looped = (LOOP_START..LOOP_END).contains(&l);
        let (q_norm, k_norm, v_norm, attn_norm) = if full {
            (
                Some(self.f32(&n("q_input_norm.weight"))?),
                Some(self.f32(&n("k_input_norm.weight"))?),
                Some(self.f32(&n("v_input_norm.weight"))?),
                None,
            )
        } else {
            (None, None, None, Some(self.f32(&n("attn_norm.weight"))?))
        };
        let mlp_norm = self.f32(&n("mlp_norm.weight"))?;
        // input projections back to back in the arena, one [N_total, H] matrix: q | gate | r | sink | k (| k_dyn) |
        // v (| v_dyn) (every part a multiple of 256 bytes, so the takes leave no gap; checked): decode calls project
        // a visit with one GEMM per distinct input (D1)
        let wq = self.bf16(&n("attn.wq_du.weight"))?;
        let wgate = self.bf16(&n("attn.gate_attn.weight"))?;
        let wr = self.bf16(&n("attn.wr_du.weight"))?;
        let wsink = self.bf16(&n("attn.attn_sink_proj.weight"))?;
        let wk = self.bf16(&n("attn.wk_dv.weight"))?;
        let wk_dyn = full.then(|| self.bf16(&n("attn.k_sconv.weight_proj"))).transpose()?;
        let wv = self.bf16(&n("attn.wv_dv.weight"))?;
        let wv_dyn = full.then(|| self.bf16(&n("attn.v_sconv.weight_proj"))).transpose()?;
        let wqkv = self.contiguous(
            &[
                Some(&wq),
                Some(&wgate),
                Some(&wr),
                Some(&wsink),
                Some(&wk),
                wk_dyn.as_ref(),
                Some(&wv),
                wv_dyn.as_ref(),
            ],
            l,
        )?;
        let wo = self.bf16(&n("attn.wo_ud.weight"))?;
        let q_scale = self.f32(&n("attn.q_l2_clip.scale"))?;
        let r_scale = self.f32(&n("attn.rel_l2_clip.scale"))?;
        let rel_proj = self.bf16(&n("attn.rel_logits_proj.proj"))?;
        let (k_static, v_static, k_conv, v_conv) = if full {
            (
                Some(self.f32(&n("attn.k_sconv.static_w"))?),
                Some(self.f32(&n("attn.v_sconv.static_w"))?),
                None,
                None,
            )
        } else {
            // [1024, 1, 4] -> [1024, 4]
            let k = self.f32(&n("attn.k_sconv.weight"))?;
            let v = self.f32(&n("attn.v_sconv.weight"))?;
            (None, None, Some(k), Some(v))
        };
        let attn_conv = self.f32(&n("attn_sconv.weight"))?;
        let mlp_conv = self.f32(&n("mlp_sconv.weight"))?;
        let router = self.bf16(&n("mlp.gate.weight"))?;
        let router_bias = self.f32(&n("mlp.gate.bias"))?;
        let (attn_query, n_attn_query) = if l > 0 {
            let q = self.host_f32(&format!("model.llm.attn_res.attn_queries.{}", l - 1))?;
            let rows = q.len() / H;
            if rows != if full { 3 } else { 1 } {
                return Err(invalid("attn query count", format!("layer {l}: {rows}")));
            }
            (Some(self.vec(&q, "attn_query")?), rows)
        } else {
            (None, 0)
        };
        let mlp_query = self.f32(&format!("model.llm.attn_res.mlp_queries.{l}"))?;
        let controllers = if looped {
            Some([
                self.controller(&p, "attn")?,
                self.controller(&p, "k")?,
                self.controller(&p, "v")?,
                self.controller(&p, "mlp")?,
            ])
        } else {
            None
        };
        let fmt = if l < 20 { ExpertFmt::Nvfp4 } else { ExpertFmt::Mxfp8 };
        let experts = self.experts(l, &n("mlp.experts."), fmt, threads)?;
        Ok(LayerW {
            idx: l,
            full,
            looped,
            attn_norm,
            q_norm,
            k_norm,
            v_norm,
            mlp_norm,
            q_scale,
            r_scale,
            rel_proj,
            k_static,
            v_static,
            k_conv,
            v_conv,
            attn_conv,
            mlp_conv,
            router,
            router_bias,
            attn_query,
            n_attn_query,
            mlp_query,
            wq,
            wk,
            wv,
            wr,
            wo,
            wgate,
            wsink,
            wk_dyn,
            wv_dyn,
            wqkv,
            controllers,
            fmt,
            experts,
        })
    }
}

/// `[H, 64]` -> `[64, H]` (the controllers' coalesced gate copy).
///
/// # Errors
///
/// [`crate::ModelError::Invalid`] when `v` is not `H x 64` values.
pub fn transpose_gate(v: &[f32]) -> Result<Vec<f32>> {
    if v.len() != H * GATE_RANK {
        return Err(invalid(
            "controller gate",
            format!("{} values, expected {}", v.len(), H * GATE_RANK),
        ));
    }
    let mut t = vec![0.0f32; GATE_RANK * H];
    for h in 0..H {
        for j in 0..GATE_RANK {
            t[j * H + h] = v[h * GATE_RANK + j];
        }
    }
    Ok(t)
}

/// NVFP4 per-expert factor: fp32 `amax[x] / denominator` (config B's `formats.nvfp4_factor`), then x 256.
///
/// # Errors
///
/// [`crate::ModelError::Invalid`] unless there are 128 amax values and one denominator.
pub fn nvfp4_factors(amax: &[f32], denom: &[f32]) -> Result<Vec<f32>> {
    match denom {
        [d] if amax.len() == NEXP => Ok(amax.iter().map(|a| (a / d) * 256.0).collect()),
        _ => Err(invalid(
            "amax / denominator shape",
            format!("{} / {}", amax.len(), denom.len()),
        )),
    }
}

/// `[24, H]` cross-pass queries and `[24, H]` cross-pass kv queries -> `[12, 4, H]`: per message (q[2i], kv[2i],
/// kv[2i + 1], q[2i + 1]) = (q, k, v, mlp).
///
/// # Errors
///
/// [`crate::ModelError::Invalid`] when either input is not `24 x H` values.
pub fn interleave_msg_queries(q: &[f32], kv: &[f32]) -> Result<Vec<f32>> {
    if q.len() != 24 * H || kv.len() != 24 * H {
        return Err(invalid("message query shapes", format!("{} / {}", q.len(), kv.len())));
    }
    let row = |x: &[f32], r: usize| x[r * H..(r + 1) * H].to_vec();
    Ok((0..12)
        .flat_map(|i| [row(q, 2 * i), row(kv, 2 * i), row(kv, 2 * i + 1), row(q, 2 * i + 1)])
        .flatten()
        .collect())
}

impl Weights {
    /// Maps `<model_dir>/model-00001-of-00001.safetensors` and uploads the weight set into `ar` in the C++ order,
    /// repacking experts on `threads` threads. The copies are queued on `st`; the source of each has been staged
    /// when the call returns.
    ///
    /// # Errors
    ///
    /// A missing tensor, a dtype, shape or scale the model does not accept, a full arena, or a device error.
    pub fn load(ar: &mut Arena, st: &Stream, model_dir: &Path, threads: usize) -> Result<Weights> {
        let ck = Checkpoint::open(&model_dir.join(CHECKPOINT_FILE))?;
        let mut ld = Loader {
            ar,
            st,
            ck: &ck,
            tot: 0,
        };
        let layers = (0..NLAYERS).map(|l| ld.layer(l, threads)).collect::<Result<Vec<_>>>()?;
        let embed = ld.bf16("model.llm.embed.weight")?;
        let embed_norm = ld.f32("model.llm.embed_norm.weight")?;
        let final_query = ld.f32("model.llm.attn_res.output_query")?;
        let final_norm = ld.f32("model.llm.norm.weight")?;
        let unembed = ld.bf16("model.llm.unembed.weight")?;
        let q = ld.host_f32("model.llm.middle_cycle_cross_pass_queries")?;
        let kv = ld.host_f32("model.llm.middle_cycle_cross_pass_kv_queries")?;
        let msg_queries = ld.vec(&interleave_msg_queries(&q, &kv)?, "msg_queries")?;
        Ok(Weights {
            layers,
            embed,
            embed_norm,
            final_query,
            final_norm,
            unembed,
            msg_queries,
            bytes: ld.tot,
        })
    }

    /// Every weight buffer with its `weight-sums` name, sorted by offset (the C++ `weight_regions`).
    pub fn regions(&self) -> Vec<WeightRegion> {
        let mut r = Vec::new();
        let mut add = |name: String, off: Option<usize>| {
            if let Some(offset) = off {
                r.push(WeightRegion { name, offset });
            }
        };
        fn o<T: bytemuck::Pod>(b: &DevBuf<T>) -> Option<usize> {
            Some(b.offset())
        }
        fn oo<T: bytemuck::Pod>(b: &Option<DevBuf<T>>) -> Option<usize> {
            b.as_ref().map(DevBuf::offset)
        }
        for lw in &self.layers {
            let p = format!("layers.{}.", lw.idx);
            let n = |s: &str| format!("{p}{s}");
            add(n("attn_norm"), oo(&lw.attn_norm));
            add(n("q_norm"), oo(&lw.q_norm));
            add(n("k_norm"), oo(&lw.k_norm));
            add(n("v_norm"), oo(&lw.v_norm));
            add(n("mlp_norm"), o(&lw.mlp_norm));
            add(n("q_scale"), o(&lw.q_scale));
            add(n("r_scale"), o(&lw.r_scale));
            add(n("rel_proj"), o(&lw.rel_proj));
            add(n("k_static"), oo(&lw.k_static));
            add(n("v_static"), oo(&lw.v_static));
            add(n("k_conv"), oo(&lw.k_conv));
            add(n("v_conv"), oo(&lw.v_conv));
            add(n("attn_conv"), o(&lw.attn_conv));
            add(n("mlp_conv"), o(&lw.mlp_conv));
            add(n("router"), o(&lw.router));
            add(n("router_bias"), o(&lw.router_bias));
            add(n("attn_query"), oo(&lw.attn_query));
            add(n("mlp_query"), o(&lw.mlp_query));
            add(n("wq"), o(&lw.wq));
            add(n("wk"), o(&lw.wk));
            add(n("wv"), o(&lw.wv));
            add(n("wr"), o(&lw.wr));
            add(n("wo"), o(&lw.wo));
            add(n("wgate"), o(&lw.wgate));
            add(n("wsink"), o(&lw.wsink));
            add(n("wk_dyn"), oo(&lw.wk_dyn));
            add(n("wv_dyn"), oo(&lw.wv_dyn));
            if let Some(cs) = &lw.controllers {
                for (g, c) in cs.iter().enumerate() {
                    add(n(&format!("c_down.{g}")), o(&c.down));
                    add(n(&format!("c_forget.{g}")), o(&c.forget));
                    add(n(&format!("c_gain.{g}")), o(&c.gain));
                    add(n(&format!("c_forgetT.{g}")), o(&c.forget_t));
                    add(n(&format!("c_gainT.{g}")), o(&c.gain_t));
                }
            }
            for (j, e) in lw.experts.iter().enumerate() {
                add(n(&format!("eW.{j}")), o(&e.w));
                add(n(&format!("eS.{j}")), o(&e.s));
                add(n(&format!("eF.{j}")), o(&e.f));
            }
        }
        add("embed".into(), o(&self.embed));
        add("embed_norm".into(), o(&self.embed_norm));
        add("final_query".into(), o(&self.final_query));
        add("final_norm".into(), o(&self.final_norm));
        add("unembed".into(), o(&self.unembed));
        add("msg_queries".into(), o(&self.msg_queries));
        r.sort_by_key(|x| x.offset);
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_transpose() {
        let v: Vec<f32> = (0..H * GATE_RANK).map(|i| i as f32).collect();
        let t = transpose_gate(&v).unwrap();
        // t[j * H + h] = v[h * 64 + j]
        assert_eq!(t[0], 0.0);
        assert_eq!(t[1], 64.0); // j 0, h 1
        assert_eq!(t[H], 1.0); // j 1, h 0
        assert_eq!(t[5 * H + 7], (7 * 64 + 5) as f32);
        assert!(transpose_gate(&v[1..]).is_err());
    }

    #[test]
    fn factors_are_fp32_division_then_256() {
        let mut am = vec![3.0f32; NEXP];
        am[1] = 1.0;
        let f = nvfp4_factors(&am, &[2.0]).unwrap();
        assert_eq!((f[0], f[1]), (384.0, 128.0));
        // 1/3 in fp32 then x 256 (not 256/3): 0.33333334f * 256 = 85.333336f
        am[2] = 1.0;
        let f = nvfp4_factors(&am, &[3.0]).unwrap();
        assert_eq!(f[2], (1.0f32 / 3.0f32) * 256.0);
        assert_eq!(f[2].to_bits(), 0x42aa_aaab);
        assert!(nvfp4_factors(&am, &[1.0, 2.0]).is_err());
        assert!(nvfp4_factors(&am[1..], &[1.0]).is_err());
    }

    #[test]
    fn message_queries_order() {
        let q: Vec<f32> = (0..24 * H).map(|i| (i / H) as f32).collect(); // row r of q = r
        let kv: Vec<f32> = (0..24 * H).map(|i| 100.0 + (i / H) as f32).collect(); // row r of kv = 100 + r
        let m = interleave_msg_queries(&q, &kv).unwrap();
        assert_eq!(m.len(), 12 * 4 * H);
        let at = |i: usize, s: usize| m[(i * 4 + s) * H];
        // message i: (q[2i], kv[2i], kv[2i+1], q[2i+1])
        assert_eq!([at(0, 0), at(0, 1), at(0, 2), at(0, 3)], [0.0, 100.0, 101.0, 1.0]);
        assert_eq!([at(11, 0), at(11, 1), at(11, 2), at(11, 3)], [22.0, 122.0, 123.0, 23.0]);
        assert!(interleave_msg_queries(&q[H..], &kv).is_err());
    }
}
