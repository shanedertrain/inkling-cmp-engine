//! The checkpoint, memory-mapped: tensor views over the file without copying (the C++ `Mapped`, `weights.cpp`).
//!
//! Pattern of candle's `MmapedSafetensors`: `memmap2` maps the file, the `safetensors` crate parses and validates the
//! header. Instead of candle's self-referential `yoke` wrapper the header is parsed once with
//! [`SafeTensors::read_metadata`], which owns its tensor table; views are cut from the map on demand.

use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::tensor::{Metadata, TensorInfo};
use safetensors::{Dtype, SafeTensors};

use crate::error::{ModelError, Result};

/// The one checkpoint file of config B's model directory.
pub const CHECKPOINT_FILE: &str = "model-00001-of-00001.safetensors";

/// A mapped safetensors file.
#[derive(Debug)]
pub struct Checkpoint {
    path: PathBuf,
    map: Mmap,
    data: usize,
    meta: Metadata,
}

/// One tensor of a [`Checkpoint`]: dtype, shape and its bytes inside the map (little-endian, any alignment).
#[derive(Debug, Clone)]
pub struct Tensor<'a> {
    /// Name in the checkpoint.
    pub name: String,
    /// Element type.
    pub dtype: Dtype,
    /// Shape.
    pub shape: &'a [usize],
    /// Raw data.
    pub bytes: &'a [u8],
}

impl Checkpoint {
    /// Maps `path` read-only and parses its header (offsets and sizes are validated against the file length).
    ///
    /// # Errors
    ///
    /// [`ModelError::Io`] when the file cannot be opened or mapped, [`ModelError::Safetensors`] for a bad header.
    pub fn open(path: &Path) -> Result<Checkpoint> {
        let io = |source| ModelError::Io {
            path: path.to_owned(),
            source,
        };
        let file = File::open(path).map_err(io)?;
        #[allow(unsafe_code)] // the one mmap (lib.rs)
        // SAFETY: the map is read-only and private to this process. Undefined behaviour would need another process to
        // truncate or rewrite the checkpoint while it is mapped; the model directory is mounted read-only in every
        // engine container (AGENTS.md hard rule 6) and nothing in the engine writes it.
        let map = unsafe { Mmap::map(&file) }.map_err(io)?;
        let (header, meta) = SafeTensors::read_metadata(&map)?;
        Ok(Checkpoint {
            path: path.to_owned(),
            map,
            data: 8 + header,
            meta,
        })
    }

    /// Path of the mapped file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The tensor `name`.
    ///
    /// # Errors
    ///
    /// [`ModelError::Missing`] when the checkpoint has no such tensor.
    pub fn tensor(&self, name: &str) -> Result<Tensor<'_>> {
        let info: &TensorInfo = self
            .meta
            .info(name)
            .ok_or_else(|| ModelError::Missing(name.to_owned()))?;
        let (a, b) = info.data_offsets;
        Ok(Tensor {
            name: name.to_owned(),
            dtype: info.dtype,
            shape: &info.shape,
            // read_metadata checked that every tensor lies inside the data section of the file
            bytes: &self.map[self.data + a..self.data + b],
        })
    }
}

impl Tensor<'_> {
    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Fails unless the dtype is `want`.
    ///
    /// # Errors
    ///
    /// [`ModelError::Dtype`].
    pub fn expect(&self, want: Dtype) -> Result<&Self> {
        if self.dtype == want {
            Ok(self)
        } else {
            Err(ModelError::Dtype {
                name: self.name.clone(),
                got: self.dtype,
                want: format!("{want:?}"),
            })
        }
    }

    /// The values as fp32: F32 copied, BF16 widened (exact), as the C++ `host_f32`.
    ///
    /// # Errors
    ///
    /// [`ModelError::Dtype`] for any other dtype.
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        match self.dtype {
            Dtype::F32 => Ok(self
                .bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()),
            Dtype::BF16 => Ok(self
                .bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::bf16::from_bits(u16::from_le_bytes(*c)).to_f32())
                .collect()),
            got => Err(ModelError::Dtype {
                name: self.name.clone(),
                got,
                want: "F32 or BF16".to_owned(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    /// A two-tensor safetensors file written by hand: header length, JSON header, data.
    fn write_file(dir: &Path) -> PathBuf {
        let header = r#"{"a":{"dtype":"BF16","shape":[3],"data_offsets":[0,6]},"b":{"dtype":"F32","shape":[1,2],"data_offsets":[6,14]},"__metadata__":{"x":"y"}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        // a: bf16 1.0, -2.0, 0.5 (0x3f80, 0xc000, 0x3f00)
        bytes.extend_from_slice(&[0x80, 0x3f, 0x00, 0xc0, 0x00, 0x3f]);
        // b: f32 3.25, -0.0, unaligned (data offset 6 of an odd-length header)
        bytes.extend_from_slice(&3.25f32.to_le_bytes());
        bytes.extend_from_slice(&(-0.0f32).to_le_bytes());
        let path = dir.join("t.safetensors");
        File::create(&path).unwrap().write_all(&bytes).unwrap();
        path
    }

    #[test]
    fn views_and_fp32() {
        let dir = std::env::temp_dir().join(format!("ink-model-ckpt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ck = Checkpoint::open(&write_file(&dir)).unwrap();
        let a = ck.tensor("a").unwrap();
        assert_eq!((a.dtype, a.shape, a.numel()), (Dtype::BF16, &[3usize][..], 3));
        assert_eq!(a.to_f32().unwrap(), [1.0, -2.0, 0.5]);
        let b = ck.tensor("b").unwrap();
        assert_eq!(b.shape, [1, 2]);
        let v = b.to_f32().unwrap();
        assert_eq!(v[0], 3.25);
        assert!(v[1] == 0.0 && v[1].is_sign_negative());
        assert!(b.expect(Dtype::F32).is_ok());
        assert!(matches!(b.expect(Dtype::BF16), Err(ModelError::Dtype { .. })));
        assert!(matches!(ck.tensor("c"), Err(ModelError::Missing(n)) if n == "c"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
