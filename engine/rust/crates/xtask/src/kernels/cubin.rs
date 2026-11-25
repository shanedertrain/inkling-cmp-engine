//! Kernels of an sm_80 cubin: one per `.text.<symbol>` section (the same rule as gate E0's `e0_sass.py`), with the
//! parameter list nvcc recorded in `.nv.info.<symbol>` (`EIATTR_KPARAM_INFO`: ordinal, offset, size).
//!
//! Whether a kernel is launched cooperatively is not read from here: `EIATTR_COOP_GROUP_INSTR_OFFSETS` also marks
//! plain warp-synchronous intrinsics (`__shfl_xor_sync`: 83 of 266 kernels carry it, one pair needs a cooperative
//! launch), so keys.toml lists the cooperative kernels.

use std::collections::HashMap;

use anyhow::{Context, Result, bail, ensure};
use object::{Object, ObjectSection};

/// `EIATTR_KPARAM_INFO`: one per parameter.
const EIATTR_KPARAM_INFO: u8 = 0x17;
/// `EIATTR_CBANK_PARAM_SIZE`: total parameter bytes.
const EIATTR_CBANK_PARAM_SIZE: u8 = 0x19;
/// Attribute formats: no value, byte, half-word (value in the header), sized payload.
const EIFMT_NVAL: u8 = 1;
const EIFMT_BVAL: u8 = 2;
const EIFMT_HVAL: u8 = 3;
const EIFMT_SVAL: u8 = 4;

/// One kernel parameter as nvcc laid it out in the parameter constant bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KParam {
    pub offset: usize,
    pub size: usize,
}

/// What a kernel's `.nv.info` section says about its launch.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NvInfo {
    /// Parameters in ordinal order.
    pub params: Vec<KParam>,
    /// Total parameter bytes (`EIATTR_CBANK_PARAM_SIZE`; 0 when the kernel has none).
    pub param_bytes: usize,
}

/// A kernel of the cubin.
#[derive(Debug, Clone)]
pub struct Kernel {
    /// Mangled symbol.
    pub symbol: String,
    pub info: NvInfo,
}

/// Every kernel of `cubin`, in section order.
pub fn kernels(cubin: &[u8]) -> Result<Vec<Kernel>> {
    let file = object::File::parse(cubin).context("cubin is not an ELF file")?;
    let mut infos = HashMap::new();
    let mut names = Vec::new();
    for sec in file.sections() {
        let name = sec.name().context("section name")?;
        if let Some(sym) = name.strip_prefix(".nv.info.") {
            let info = parse_nv_info(sec.data().context("nv.info data")?).with_context(|| name.to_string())?;
            infos.insert(sym.to_string(), info);
        } else if let Some(sym) = name.strip_prefix(".text.") {
            names.push(sym.to_string());
        }
    }
    names
        .into_iter()
        .map(|symbol| {
            let info = infos
                .remove(&symbol)
                .with_context(|| format!("kernel {symbol} has no .nv.info section"))?;
            Ok(Kernel { symbol, info })
        })
        .collect()
}

/// Parses one `.nv.info.<kernel>` section: 4-byte entries (format, attribute, u16 value or payload size), an
/// `EIFMT_SVAL` entry followed by its payload.
pub fn parse_nv_info(body: &[u8]) -> Result<NvInfo> {
    let mut info = NvInfo::default();
    let mut params: Vec<(u16, KParam)> = Vec::new();
    let mut i = 0;
    while i < body.len() {
        ensure!(i + 4 <= body.len(), "truncated attribute header at {i}");
        let (fmt, attr) = (body[i], body[i + 1]);
        let val = usize::from(u16::from_le_bytes([body[i + 2], body[i + 3]]));
        i += 4;
        let payload = match fmt {
            EIFMT_NVAL | EIFMT_BVAL | EIFMT_HVAL => &[][..],
            EIFMT_SVAL => {
                ensure!(
                    i + val <= body.len(),
                    "attribute {attr:#x}: payload of {val} bytes past the end"
                );
                i += val;
                &body[i - val..i]
            }
            _ => bail!("unknown attribute format {fmt:#x} at {}", i - 4),
        };
        match attr {
            EIATTR_KPARAM_INFO => {
                ensure!(payload.len() == 12, "KPARAM_INFO payload of {} bytes", payload.len());
                let ordinal = u16::from_le_bytes([payload[4], payload[5]]);
                let offset = usize::from(u16::from_le_bytes([payload[6], payload[7]]));
                let flags = u32::from_le_bytes([payload[8], payload[9], payload[10], payload[11]]);
                let size = usize::try_from(flags >> 18).context("param size")?;
                params.push((ordinal, KParam { offset, size }));
            }
            EIATTR_CBANK_PARAM_SIZE => {
                ensure!(fmt == EIFMT_HVAL, "CBANK_PARAM_SIZE format {fmt:#x}");
                info.param_bytes = val;
            }
            _ => {}
        }
    }
    params.sort_by_key(|p| p.0);
    for (i, (ordinal, _)) in params.iter().enumerate() {
        ensure!(
            usize::from(*ordinal) == i,
            "parameter ordinals are not 0..{}: {params:?}",
            params.len()
        );
    }
    info.params = params.into_iter().map(|p| p.1).collect();
    Ok(info)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kparam(ordinal: u16, offset: u16, size: u32) -> Vec<u8> {
        let mut v = vec![EIFMT_SVAL, EIATTR_KPARAM_INFO, 12, 0, 0, 0, 0, 0];
        v.extend_from_slice(&ordinal.to_le_bytes());
        v.extend_from_slice(&offset.to_le_bytes());
        v.extend_from_slice(&((size << 18) | 0x1f000).to_le_bytes());
        v
    }

    #[test]
    fn parses_params_and_size() {
        // the attribute sequence nvcc 13.3 writes for k_copy_cols(bf16 const*, int, int, int, bf16*, int), reordered
        let mut b = vec![EIFMT_SVAL, 0x37, 4, 0, 0x85, 0, 0, 0, EIFMT_NVAL, 0x35, 0, 0];
        b.extend_from_slice(&[EIFMT_HVAL, EIATTR_CBANK_PARAM_SIZE, 36, 0]);
        for (o, off, sz) in [
            (5u16, 32u16, 4u32),
            (4, 24, 8),
            (3, 16, 4),
            (2, 12, 4),
            (1, 8, 4),
            (0, 0, 8),
        ] {
            b.extend(kparam(o, off, sz));
        }
        b.extend_from_slice(&[EIFMT_BVAL, 0x4c, 1, 0]);
        let i = parse_nv_info(&b).unwrap();
        assert_eq!(i.param_bytes, 36);
        let got: Vec<_> = i.params.iter().map(|p| (p.offset, p.size)).collect();
        assert_eq!(got, [(0, 8), (8, 4), (12, 4), (16, 4), (24, 8), (32, 4)]);
    }

    #[test]
    fn rejects_damaged_sections() {
        assert!(parse_nv_info(&[EIFMT_SVAL, 0x17, 12, 0, 0]).is_err()); // payload past the end
        assert!(parse_nv_info(&[9, 0x17, 0, 0]).is_err()); // unknown format
        assert!(parse_nv_info(&[EIFMT_HVAL, 0x19]).is_err()); // truncated header
        let mut gap = kparam(0, 0, 8);
        gap.extend(kparam(2, 8, 4));
        assert!(parse_nv_info(&gap).is_err()); // ordinal 1 missing
        assert!(kernels(b"not an elf").is_err());
    }
}
