//! C++ types as the Itanium demangler and `typeid` print them (`float const*`, `float const* [4]`, `da::Params`), and
//! the Rust types the generated code mirrors them with.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};

/// A C++ type of a kernel parameter or a struct field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CType {
    /// A scalar.
    Prim(Prim),
    /// A CUDA vector type; only behind a pointer (by value its alignment exceeds the Rust array's).
    Vector(Vector),
    /// `void`; only behind a pointer.
    Void,
    /// A struct, by its qualified C++ name (`da::Params`); must be listed in `structs.def`.
    Struct(String),
    /// A pointer (const or not: the launchers do not distinguish).
    Ptr(Box<CType>),
    /// A fixed-size array (struct fields only).
    Array(Box<CType>, usize),
}

/// Scalars (alignment = size).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prim {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
    Bf16,
    F16,
}

/// CUDA vector types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vector {
    Float2,
    Int4,
    Uint4,
}

impl Prim {
    fn size(self) -> usize {
        match self {
            Prim::I8 | Prim::U8 => 1,
            Prim::I16 | Prim::U16 | Prim::Bf16 | Prim::F16 => 2,
            Prim::I32 | Prim::U32 | Prim::F32 => 4,
            Prim::I64 | Prim::U64 | Prim::F64 => 8,
        }
    }

    fn rust(self) -> &'static str {
        match self {
            Prim::I8 => "i8",
            Prim::U8 => "u8",
            Prim::I16 => "i16",
            Prim::U16 => "u16",
            Prim::I32 => "i32",
            Prim::U32 => "u32",
            Prim::I64 => "i64",
            Prim::U64 => "u64",
            Prim::F32 => "f32",
            Prim::F64 => "f64",
            Prim::Bf16 => "Bf16",
            Prim::F16 => "F16",
        }
    }
}

impl Vector {
    fn rust(self) -> &'static str {
        match self {
            Vector::Float2 => "[f32; 2]",
            Vector::Int4 => "[i32; 4]",
            Vector::Uint4 => "[u32; 4]",
        }
    }
}

/// Parses a demangled C++ type.
pub fn parse(s: &str) -> Result<CType> {
    let s = s.trim();
    if let Some(rest) = s.strip_suffix(']') {
        let open = rest
            .rfind('[')
            .with_context(|| format!("unbalanced array type {s:?}"))?;
        let n = rest[open + 1..]
            .trim()
            .parse()
            .with_context(|| format!("array length in {s:?}"))?;
        return Ok(CType::Array(Box::new(parse(&rest[..open])?), n));
    }
    if let Some(rest) = s.strip_suffix('*') {
        return Ok(CType::Ptr(Box::new(parse(rest)?)));
    }
    let base = s
        .strip_suffix(" const")
        .or_else(|| s.strip_prefix("const "))
        .unwrap_or(s)
        .trim();
    let prim = |p| Ok(CType::Prim(p));
    match base {
        "signed char" => prim(Prim::I8),
        "unsigned char" => prim(Prim::U8),
        "short" => prim(Prim::I16),
        "unsigned short" => prim(Prim::U16),
        "int" => prim(Prim::I32),
        "unsigned int" => prim(Prim::U32),
        "long" | "long long" => prim(Prim::I64),
        "unsigned long" | "unsigned long long" => prim(Prim::U64),
        "float" => prim(Prim::F32),
        "double" => prim(Prim::F64),
        "__nv_bfloat16" => prim(Prim::Bf16),
        "__half" => prim(Prim::F16),
        "float2" => Ok(CType::Vector(Vector::Float2)),
        "int4" => Ok(CType::Vector(Vector::Int4)),
        "uint4" => Ok(CType::Vector(Vector::Uint4)),
        "void" => Ok(CType::Void),
        "bool" | "char" => bail!("type {s:?} has no Pod mirror (bool) or no fixed signedness (char)"),
        _ if is_path(base) => Ok(CType::Struct(base.to_string())),
        _ => bail!("unsupported C++ type {s:?}"),
    }
}

fn is_path(s: &str) -> bool {
    !s.is_empty()
        && s.split("::").all(|p| {
            p.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                && p.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        })
}

/// Size and alignment of a struct, by C++ name (from `layout.json`).
pub type StructSizes = HashMap<String, (usize, usize)>;

impl CType {
    /// Size and alignment under the C ABI (x86-64 host and the CUDA parameter space agree for these types).
    pub fn size_align(&self, structs: &StructSizes) -> Result<(usize, usize)> {
        Ok(match self {
            CType::Prim(p) => (p.size(), p.size()),
            CType::Ptr(_) => (8, 8),
            CType::Struct(n) => *structs
                .get(n)
                .with_context(|| format!("struct {n} is not in structs.def"))?,
            CType::Array(t, n) => {
                let (s, a) = t.size_align(structs)?;
                (s * n, a)
            }
            CType::Vector(_) | CType::Void => bail!("{self:?} by value"),
        })
    }

    /// The Rust type of a struct field.
    pub fn field_rust(&self) -> Result<String> {
        Ok(match self {
            CType::Prim(p) => p.rust().to_string(),
            CType::Struct(n) => n.clone(),
            CType::Ptr(t) => format!("DevPtr<{}>", t.pointee_rust()?),
            CType::Array(t, n) => format!("[{}; {n}]", t.field_rust()?),
            CType::Vector(_) | CType::Void => bail!("{self:?} by value in a struct"),
        })
    }

    /// The Rust element type behind a pointer.
    pub fn pointee_rust(&self) -> Result<String> {
        Ok(match self {
            CType::Prim(p) => p.rust().to_string(),
            CType::Vector(v) => v.rust().to_string(),
            CType::Void => "u8".to_string(),
            CType::Struct(n) => n.clone(),
            CType::Ptr(_) | CType::Array(..) => bail!("pointer to {self:?}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_demangler_and_typeid_spellings() {
        use CType::*;
        assert_eq!(parse("int").unwrap(), Prim(super::Prim::I32));
        assert_eq!(parse("float const*").unwrap(), Ptr(Box::new(Prim(super::Prim::F32))));
        assert_eq!(parse("__nv_bfloat16*").unwrap(), Ptr(Box::new(Prim(super::Prim::Bf16))));
        assert_eq!(
            parse("unsigned long long*").unwrap(),
            Ptr(Box::new(Prim(super::Prim::U64)))
        );
        assert_eq!(
            parse("float const* [4]").unwrap(),
            Array(Box::new(Ptr(Box::new(Prim(super::Prim::F32)))), 4)
        );
        assert_eq!(
            parse("projs::Job [8]").unwrap(),
            Array(Box::new(Struct("projs::Job".into())), 8)
        );
        assert_eq!(parse("void const*").unwrap(), Ptr(Box::new(Void)));
        assert_eq!(
            parse("pfa::Seg const*").unwrap(),
            Ptr(Box::new(Struct("pfa::Seg".into())))
        );
        assert_eq!(parse("RealHead").unwrap(), Struct("RealHead".into()));
        assert!(parse("bool").is_err());
        assert!(parse("char*").is_err());
        assert!(parse("std::vector<int>").is_err());
    }

    #[test]
    fn rust_spellings_and_layout() {
        let mut s = StructSizes::new();
        s.insert("pfa::Seg".into(), (16, 4));
        let t = parse("pfa::Seg const* [3]").unwrap();
        assert_eq!(t.field_rust().unwrap(), "[DevPtr<pfa::Seg>; 3]");
        assert_eq!(t.size_align(&s).unwrap(), (24, 8));
        assert_eq!(parse("pfa::Seg [2]").unwrap().size_align(&s).unwrap(), (32, 4));
        assert_eq!(parse("float2*").unwrap().field_rust().unwrap(), "DevPtr<[f32; 2]>");
        assert_eq!(parse("void const*").unwrap().field_rust().unwrap(), "DevPtr<u8>");
        assert!(parse("int4").unwrap().size_align(&s).is_err());
        assert!(parse("da::Params").unwrap().size_align(&s).is_err());
        assert!(parse("int**").unwrap().field_rust().is_err());
    }
}
