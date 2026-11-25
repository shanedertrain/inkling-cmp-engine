//! `xtask gen-kernels PACK_DIR OUT_DIR` (package R2, docs/RUST-PLAN.md 3.2): the typed launchers of the kernel pack.
//!
//! Inputs (all in PACK_DIR, written by `engine/kernels/pack/build.sh`): `keys.toml`, `pack_{engine,drafter}.cubin`,
//! `layout.json`, `manifest.json`. Outputs (OUT_DIR, copied into `crates/ink-kernels/` and committed): `src/gen.rs`
//! (formatted with rustfmt), `kpack/layout.json`, `kpack/manifest.json`.
//!
//! Checks, each a hard error: every key resolves to exactly one kernel of its TU's cubin and every kernel has a key;
//! every kernel parameter and struct field has a supported C++ type; every struct a kernel uses is in `layout.json`;
//! each kernel's parameter offsets and sizes, computed from its C++ signature and the struct sizes, equal nvcc's
//! (`.nv.info` `KPARAM_INFO`); each struct's field offsets, computed from the field types, equal the compiler's; every
//! kernel's C++ parameter names are found in the pack's sources (`names`: one `__global__` definition per qualified
//! name and parameter count), so launchers take named argument structs (review R3 finding F13).

mod ctype;
mod cubin;
mod emit;
mod names;
mod sig;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

use ctype::{CType, StructSizes};

/// The pack's TUs: keys.toml table, cubin file.
const TUS: [(&str, &str); 2] = [("engine", "pack_engine.cubin"), ("drafter", "pack_drafter.cubin")];

#[derive(Debug, Deserialize)]
struct LayoutFile {
    format: String,
    structs: Vec<LayoutStruct>,
}

#[derive(Debug, Deserialize)]
struct LayoutStruct {
    name: String,
    size: usize,
    align: usize,
    fields: Vec<LayoutField>,
}

#[derive(Debug, Deserialize)]
struct LayoutField {
    name: String,
    #[serde(rename = "type")]
    ty: String,
    offset: usize,
    size: usize,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    format: String,
    kpack_id: String,
    cubins: Vec<ManifestCubin>,
    inputs: Vec<ManifestInput>,
}

/// A repository file the pack was compiled from (path relative to the repository root).
#[derive(Debug, Deserialize)]
struct ManifestInput {
    path: String,
}

#[derive(Debug, Deserialize)]
struct ManifestCubin {
    tu: String,
    file: String,
    bytes: usize,
    sha256: String,
}

/// A struct mirror to generate.
#[derive(Debug)]
pub struct StructDef {
    /// Qualified C++ name.
    pub cpp: String,
    pub size: usize,
    pub align: usize,
    pub fields: Vec<FieldDef>,
}

/// A field of a struct mirror.
#[derive(Debug)]
pub struct FieldDef {
    pub name: String,
    pub cpp: String,
    pub ty: CType,
    pub offset: usize,
    pub size: usize,
}

/// A cubin of the pack.
#[derive(Debug)]
pub struct CubinDef {
    pub tu: String,
    pub file: String,
    pub bytes: usize,
    pub sha256: String,
    /// FNV-1a 64 of the file: ink-kernels checks the embedded bytes against it at load (no sha256 crate there).
    pub fnv1a64: u64,
}

/// FNV-1a 64 (the hash of ink-model's weight sums; ink-kernels repeats the same six lines).
fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// One launcher to generate.
#[derive(Debug)]
pub struct LauncherDef {
    pub key: String,
    /// Index into the cubin list.
    pub tu: usize,
    pub symbol: String,
    pub demangled: String,
    pub params: Vec<CType>,
    /// The parameter types as the demangler spells them.
    pub cpp_params: Vec<String>,
    /// The C++ parameter names (from the `__global__` definition).
    pub names: Vec<String>,
    pub param_bytes: usize,
    pub coop: bool,
}

/// The generator's sources, in the order `check_fresh.py` hashes them (`engine/rust/crates/xtask/src/kernels/<file>`).
const GENERATOR_SOURCES: [&str; 6] = [
    include_str!("mod.rs"),
    include_str!("ctype.rs"),
    include_str!("cubin.rs"),
    include_str!("emit.rs"),
    include_str!("sig.rs"),
    include_str!("names.rs"),
];

/// FNV-1a 64 over the concatenated generator sources (written into gen.rs as `GENERATOR_FNV1A64`).
fn generator_fnv1a64() -> u64 {
    fnv1a64(GENERATOR_SOURCES.concat().as_bytes())
}

/// Everything `emit` writes.
#[derive(Debug)]
pub struct Pack {
    pub kpack_id: String,
    /// [`generator_fnv1a64`].
    pub generator: u64,
    pub cubins: Vec<CubinDef>,
    pub structs: Vec<StructDef>,
    pub launchers: Vec<LauncherDef>,
}

/// Entry point of `xtask gen-kernels`.
pub fn gen_kernels(pack_dir: &Path, out_dir: &Path) -> Result<()> {
    let read = |f: &str| fs::read(pack_dir.join(f)).with_context(|| format!("reading {}", pack_dir.join(f).display()));
    let keys = parse_keys(&String::from_utf8(read("keys.toml")?).context("keys.toml is not UTF-8")?)?;
    let layout_text = read("layout.json")?;
    let manifest_text = read("manifest.json")?;
    let layout: LayoutFile = serde_json::from_slice(&layout_text).context("layout.json")?;
    let manifest: Manifest = serde_json::from_slice(&manifest_text).context("manifest.json")?;
    ensure!(
        layout.format == "ink-kpack-layout/1",
        "layout.json format {}",
        layout.format
    );
    ensure!(
        manifest.format == "ink-kpack-manifest/1",
        "manifest.json format {}",
        manifest.format
    );
    let structs = build_structs(&layout)?;
    let mut cubins = Vec::new();
    let mut found = Vec::new();
    for (tu, file) in TUS {
        let m = manifest
            .cubins
            .iter()
            .find(|c| c.tu == tu)
            .with_context(|| format!("manifest has no {tu} cubin"))?;
        ensure!(m.file == file, "manifest names {} for {tu}, expected {file}", m.file);
        let data = read(file)?;
        ensure!(
            data.len() == m.bytes,
            "{file}: {} bytes, manifest says {}",
            data.len(),
            m.bytes
        );
        found.push(cubin::kernels(&data).with_context(|| file.to_string())?);
        cubins.push(CubinDef {
            tu: tu.into(),
            file: file.into(),
            bytes: m.bytes,
            sha256: m.sha256.clone(),
            fnv1a64: fnv1a64(&data),
        });
    }
    let decls = names::scan(&read_sources(&manifest)?)?;
    let launchers = build_launchers(&keys, found, &structs, &decls)?;
    let pack = Pack {
        kpack_id: manifest.kpack_id,
        generator: generator_fnv1a64(),
        cubins,
        structs,
        launchers,
    };
    let gen_rs = out_dir.join("src/gen.rs");
    fs::create_dir_all(out_dir.join("src"))?;
    fs::create_dir_all(out_dir.join("kpack"))?;
    fs::write(&gen_rs, emit::gen_rs(&pack)?)?;
    rustfmt(&gen_rs)?;
    fs::write(out_dir.join("kpack/layout.json"), &layout_text)?;
    fs::write(out_dir.join("kpack/manifest.json"), &manifest_text)?;
    println!(
        "gen-kernels: kpack {}, {} launchers ({} cooperative), {} structs -> {}",
        pack.kpack_id,
        pack.launchers.len(),
        pack.launchers.iter().filter(|l| l.coop).count(),
        pack.structs.len(),
        out_dir.display()
    );
    Ok(())
}

/// Formats the generated file with the workspace's rustfmt settings (max_width 120, edition 2024).
fn rustfmt(path: &Path) -> Result<()> {
    let st = std::process::Command::new("rustfmt")
        .args(["--edition", "2024", "--config", "max_width=120"])
        .arg(path)
        .status()
        .context("running rustfmt")?;
    ensure!(st.success(), "rustfmt failed on {}", path.display());
    Ok(())
}

/// keys.toml, parsed.
#[derive(Debug)]
struct Keys {
    /// Per TU (TUS order): (key, instance) in file order.
    tus: Vec<Vec<(String, String)>>,
    /// Keys launched with `cuLaunchCooperativeKernel`.
    cooperative: BTreeSet<String>,
}

/// keys.toml: `cooperative = [key, ...]`, then tables `[engine]` and `[drafter]`, each `key = "instance"`.
fn parse_keys(text: &str) -> Result<Keys> {
    let doc: toml_edit::DocumentMut = text.parse().context("keys.toml")?;
    let top: BTreeSet<&str> = doc.iter().map(|(k, _)| k).collect();
    ensure!(
        top == BTreeSet::from(["cooperative", "drafter", "engine"]),
        "keys.toml holds `cooperative` and the tables engine and drafter: {top:?}"
    );
    let mut all = BTreeSet::new();
    let mut tus = Vec::new();
    for (tu, _) in TUS {
        let table = doc[tu].as_table().with_context(|| format!("[{tu}] is not a table"))?;
        let mut keys = Vec::new();
        for (k, v) in table.iter() {
            let inst = v
                .as_str()
                .with_context(|| format!("{tu}.{k}: value must be a string"))?;
            ensure!(
                is_key(k),
                "{tu}.{k}: a key is a lowercase Rust identifier (a-z, 0-9, _), not a keyword"
            );
            ensure!(all.insert(k.to_string()), "key {k} appears twice");
            keys.push((k.to_string(), inst.to_string()));
        }
        tus.push(keys);
    }
    let list = doc["cooperative"]
        .as_array()
        .context("`cooperative` must be an array of keys")?;
    let mut cooperative = BTreeSet::new();
    for v in list.iter() {
        let k = v.as_str().context("`cooperative` holds strings")?;
        ensure!(all.contains(k), "cooperative key {k} is not a key");
        cooperative.insert(k.to_string());
    }
    Ok(Keys { tus, cooperative })
}

fn is_key(k: &str) -> bool {
    k.chars().next().is_some_and(|c| c.is_ascii_lowercase() || c == '_')
        && k.chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        && !emit::is_keyword(k)
}

fn align_up(x: usize, a: usize) -> usize {
    x.div_ceil(a) * a
}

/// Struct mirrors from layout.json, with the field offsets recomputed from the field types.
fn build_structs(layout: &LayoutFile) -> Result<Vec<StructDef>> {
    let mut sizes = StructSizes::new();
    let mut out = Vec::new();
    for s in &layout.structs {
        let mut fields = Vec::new();
        let mut end = 0;
        for f in &s.fields {
            let ty = ctype::parse(&f.ty).with_context(|| format!("{}.{}", s.name, f.name))?;
            // nested structs come earlier in structs.def, so their sizes are known here
            let (size, align) = ty
                .size_align(&sizes)
                .with_context(|| format!("{}.{}", s.name, f.name))?;
            ensure!(
                size == f.size,
                "{}.{}: {} is {size} bytes here, {} in C++",
                s.name,
                f.name,
                f.ty,
                f.size
            );
            ensure!(
                f.offset == align_up(end, align),
                "{}.{} at {}, the C ABI puts it at {}",
                s.name,
                f.name,
                f.offset,
                align_up(end, align)
            );
            end = f.offset + f.size;
            fields.push(FieldDef {
                name: f.name.clone(),
                cpp: f.ty.clone(),
                ty,
                offset: f.offset,
                size: f.size,
            });
        }
        ensure!(
            s.name.matches("::").count() <= 1,
            "{}: one namespace level supported",
            s.name
        );
        ensure!(
            sizes.insert(s.name.clone(), (s.size, s.align)).is_none(),
            "{} listed twice",
            s.name
        );
        out.push(StructDef {
            cpp: s.name.clone(),
            size: s.size,
            align: s.align,
            fields,
        });
    }
    Ok(out)
}

/// The C/C++ sources among the manifest's inputs (`.cu`, `.cuh`, `.h`, `.hpp`), read from the repository this
/// generator is built in (its sources are part of the same tree).
fn read_sources(manifest: &Manifest) -> Result<Vec<(String, String)>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../..");
    let mut out = Vec::new();
    for i in &manifest.inputs {
        if [".cu", ".cuh", ".h", ".hpp"].iter().any(|e| i.path.ends_with(e)) {
            let p = root.join(&i.path);
            let text = fs::read_to_string(&p).with_context(|| format!("reading kernel source {}", p.display()))?;
            out.push((i.path.clone(), text));
        }
    }
    ensure!(!out.is_empty(), "manifest.json lists no kernel sources");
    Ok(out)
}

/// Matches keys to cubin kernels and checks each kernel's parameter layout against nvcc's.
fn build_launchers(
    keys: &Keys,
    found: Vec<Vec<cubin::Kernel>>,
    structs: &[StructDef],
    decls: &[names::KernelDecl],
) -> Result<Vec<LauncherDef>> {
    let sizes: StructSizes = structs.iter().map(|s| (s.cpp.clone(), (s.size, s.align))).collect();
    let mut errors = Vec::new();
    let mut out = Vec::new();
    for (tu, (tu_keys, kernels)) in keys.tus.iter().zip(found).enumerate() {
        let tu_name = TUS[tu].0;
        let mut by_inst: HashMap<String, (cubin::Kernel, String, Vec<String>)> = HashMap::new();
        for k in kernels {
            let dem = sig::demangle(&k.symbol)?;
            let (inst, params) = sig::split(&dem)?;
            if by_inst.contains_key(&inst) {
                errors.push(format!(
                    "{tu_name}: two kernels named {inst} (overloads cannot have one key)"
                ));
            }
            by_inst.insert(inst, (k, dem, params));
        }
        for (key, inst) in tu_keys {
            let Some((k, dem, params)) = by_inst.remove(inst) else {
                errors.push(format!("{tu_name}.{key} = {inst:?}: no such kernel in {}", TUS[tu].1));
                continue;
            };
            let checked = check_params(&k, &params, &sizes)
                .and_then(|types| Ok((types, names::lookup(decls, inst, params.len())?.to_vec())));
            match checked {
                Ok((types, names)) => out.push(LauncherDef {
                    key: key.clone(),
                    tu,
                    symbol: k.symbol,
                    demangled: dem,
                    params: types,
                    cpp_params: params,
                    names,
                    param_bytes: k.info.param_bytes,
                    coop: keys.cooperative.contains(key),
                }),
                Err(e) => errors.push(format!("{tu_name}.{key} ({dem}): {e:#}")),
            }
        }
        let left: BTreeMap<_, _> = by_inst.into_iter().map(|(i, (k, ..))| (i, k.symbol)).collect();
        for (inst, sym) in left {
            errors.push(format!("{tu_name}: kernel {inst} ({sym}) has no key in keys.toml"));
        }
    }
    if !errors.is_empty() {
        bail!("{} problem(s):\n  {}", errors.len(), errors.join("\n  "));
    }
    Ok(out)
}

/// Parameter types of one kernel, with offsets and sizes equal to nvcc's.
fn check_params(k: &cubin::Kernel, params: &[String], sizes: &StructSizes) -> Result<Vec<CType>> {
    ensure!(
        params.len() == k.info.params.len(),
        "{} parameters in the signature, {} in .nv.info",
        params.len(),
        k.info.params.len()
    );
    let mut end = 0;
    let mut out = Vec::new();
    for (i, (p, kp)) in params.iter().zip(&k.info.params).enumerate() {
        let ty = ctype::parse(p)?;
        if matches!(ty, CType::Array(..)) {
            bail!("parameter {i}: array by value");
        }
        let (size, align) = ty.size_align(sizes).with_context(|| format!("parameter {i} ({p})"))?;
        let offset = align_up(end, align);
        ensure!(
            (offset, size) == (kp.offset, kp.size),
            "parameter {i} ({p}): offset {offset} size {size} here, nvcc has offset {} size {}",
            kp.offset,
            kp.size
        );
        end = offset + size;
        out.push(ty);
    }
    ensure!(
        k.info.param_bytes >= end,
        "parameter bytes {} < {end}",
        k.info.param_bytes
    );
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_need_both_tables_and_unique_identifiers() {
        let c = "cooperative = [\"k_bump\"]\n";
        let ok = format!("{c}[engine]\nda_combine_kernel = \"da::combine_kernel\"\n[drafter]\nk_bump = \"k_bump\"\n");
        let k = parse_keys(&ok).unwrap();
        assert_eq!(
            k.tus[0],
            [("da_combine_kernel".to_string(), "da::combine_kernel".to_string())]
        );
        assert_eq!(k.tus[1][0].0, "k_bump");
        assert!(k.cooperative.contains("k_bump"));
        assert!(parse_keys(&format!("{c}[engine]\na = \"x\"\n")).is_err());
        assert!(parse_keys("[engine]\na = \"x\"\n[drafter]\nk_bump = \"y\"\n").is_err()); // no cooperative list
        assert!(parse_keys("cooperative = [\"b\"]\n[engine]\na = \"x\"\n[drafter]\n").is_err()); // unknown key
        assert!(parse_keys(&format!("{c}[engine]\na = \"x\"\n[drafter]\na = \"y\"\n")).is_err());
        assert!(parse_keys(&format!("{c}[engine]\nBad = \"x\"\n[drafter]\nk_bump = \"y\"\n")).is_err());
        assert!(parse_keys(&format!("{c}[engine]\nfn = \"x\"\n[drafter]\nk_bump = \"y\"\n")).is_err());
        assert!(parse_keys(&format!("{c}[engine]\na = 1\n[drafter]\nk_bump = \"y\"\n")).is_err());
    }

    fn keys(engine: &[(&str, &str)], drafter: &[(&str, &str)]) -> Keys {
        let v = |l: &[(&str, &str)]| l.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect();
        Keys {
            tus: vec![v(engine), v(drafter)],
            cooperative: BTreeSet::from(["k_copy_cols".to_string()]),
        }
    }

    fn kernel(symbol: &str, params: &[(usize, usize)]) -> cubin::Kernel {
        let params = params
            .iter()
            .map(|&(offset, size)| cubin::KParam { offset, size })
            .collect();
        cubin::Kernel {
            symbol: symbol.into(),
            info: cubin::NvInfo {
                params,
                param_bytes: 36,
            },
        }
    }

    #[test]
    fn launchers_match_keys_and_nvcc_offsets() {
        let copy = "_Z11k_copy_colsPK13__nv_bfloat16iiiPS_i";
        let good = [(0, 8), (8, 4), (12, 4), (16, 4), (24, 8), (32, 4)];
        let keys = keys(&[], &[("k_copy_cols", "k_copy_cols")]);
        let src = "__global__ void k_copy_cols(const bf16* src, int rows, int cols, int ld, bf16* dst, int ldd) {}\n\
                   __global__ void k_bump(int* p) {}\n\
                   namespace da { __global__ void combine_kernel(Params p) {} }";
        let decls = names::scan(&[("t.cuh".into(), src.into())]).unwrap();
        let l = build_launchers(&keys, vec![vec![], vec![kernel(copy, &good)]], &[], &decls).unwrap();
        assert_eq!((l[0].tu, l[0].params.len(), l[0].coop), (1, 6, true));
        assert_eq!(l[0].names, ["src", "rows", "cols", "ld", "dst", "ldd"]);
        // the source has no definition with this many parameters
        let one = names::scan(&[("t.cuh".into(), "__global__ void k_copy_cols(int a) {}".into())]).unwrap();
        let e = build_launchers(&keys, vec![vec![], vec![kernel(copy, &good)]], &[], &one).unwrap_err();
        assert!(format!("{e:#}").contains("with 6 parameters"), "{e:#}");
        // a pointer at offset 20 instead of 24: nvcc and the signature disagree
        let bad = [(0, 8), (8, 4), (12, 4), (16, 4), (20, 8), (32, 4)];
        let e = build_launchers(&keys, vec![vec![], vec![kernel(copy, &bad)]], &[], &decls).unwrap_err();
        assert!(format!("{e:#}").contains("parameter 4"), "{e:#}");
        // a kernel without key, a key without kernel
        let bump = kernel("_Z6k_bumpPi", &[(0, 8)]);
        let e = build_launchers(&keys, vec![vec![], vec![bump]], &[], &decls).unwrap_err();
        let msg = format!("{e:#}");
        assert!(msg.contains("no such kernel") && msg.contains("has no key"), "{msg}");
        // a by-value struct not in layout.json
        let keys = super::tests::keys(&[("da_combine", "da::combine_kernel")], &[]);
        let comb = kernel("_ZN2da14combine_kernelENS_6ParamsE", &[(0, 184)]);
        assert!(build_launchers(&keys, vec![vec![comb], vec![]], &[], &decls).is_err());
    }

    #[test]
    fn struct_offsets_are_recomputed() {
        let field = |name: &str, ty: &str, offset, size| LayoutField {
            name: name.into(),
            ty: ty.into(),
            offset,
            size,
        };
        let mut l = LayoutFile {
            format: "ink-kpack-layout/1".into(),
            structs: vec![
                LayoutStruct {
                    name: "pfa::Seg".into(),
                    size: 16,
                    align: 4,
                    fields: ["q0", "len", "past", "pt_off"]
                        .iter()
                        .zip([0, 4, 8, 12])
                        .map(|(n, o)| field(n, "int", o, 4))
                        .collect(),
                },
                LayoutStruct {
                    name: "x::S".into(),
                    size: 24,
                    align: 8,
                    fields: vec![
                        field("n", "int", 0, 4),
                        field("segs", "pfa::Seg const*", 8, 8),
                        field("s", "float", 16, 4),
                    ],
                },
            ],
        };
        let s = build_structs(&l).unwrap();
        assert_eq!(s[1].fields[1].ty.field_rust().unwrap(), "DevPtr<pfa::Seg>");
        l.structs[1].fields[1].offset = 4; // the C ABI aligns a pointer to 8
        assert!(build_structs(&l).is_err());
    }
}
