//! Embeds the kernel pack when `INK_KPACK_DIR` names one (engine/kernels/pack/build.sh output), after checking that
//! it is the pack `src/gen.rs` was generated from: same kpack id (which covers every source file, keys.toml,
//! structs.def, the nvcc version and the flags) and the cubin sizes and FNV-1a 64 hashes gen.rs records (the same
//! sources rebuilt give other bytes: source mtimes in `.debug_line`). A mismatch fails the build:
//! regenerate with `xtask gen-kernels`. Without `INK_KPACK_DIR` nothing is embedded (`Kernels::load` errors).

use std::{env, fs, path::Path};

fn main() {
    println!("cargo::rustc-check-cfg=cfg(ink_kpack)");
    println!("cargo::rerun-if-env-changed=INK_KPACK_DIR");
    println!("cargo::rerun-if-changed=src/gen.rs");
    let Ok(dir) = env::var("INK_KPACK_DIR") else { return };
    let dir = Path::new(&dir);
    let manifest_path = dir.join("manifest.json");
    println!("cargo::rerun-if-changed={}", manifest_path.display());
    let manifest: serde_json::Value = serde_json::from_slice(&read(&manifest_path)).unwrap_or_else(|e| fail(&e));
    let generated = String::from_utf8(read(Path::new("src/gen.rs"))).unwrap_or_else(|e| fail(&e));
    let id = manifest["kpack_id"]
        .as_str()
        .unwrap_or_else(|| fail(&"manifest.json has no kpack_id"));
    if !generated.contains(&format!("pub const KPACK_ID: &str = \"{id}\";")) {
        fail(&format!(
            "pack {} has kpack id {id}, src/gen.rs another: run xtask gen-kernels",
            dir.display()
        ));
    }
    for c in manifest["cubins"]
        .as_array()
        .unwrap_or_else(|| fail(&"manifest.json has no cubins"))
    {
        let file = c["file"].as_str().unwrap_or_else(|| fail(&"cubin without file"));
        let path = dir.join(file);
        println!("cargo::rerun-if-changed={}", path.display());
        let image = read(&path);
        let bytes = image.len();
        // gen.rs: CubinInfo { tu: .., file: "<file>", bytes: <n>, sha256: .., fnv1a64: 0x<h> } (rustfmt may break it)
        let entry = generated
            .split_once(&format!("file: \"{file}\""))
            .map_or("", |(_, rest)| rest);
        let recorded = entry
            .split_once("bytes: ")
            .and_then(|(_, n)| n[..n.find(|c: char| !c.is_ascii_digit())?].parse::<usize>().ok());
        if recorded != Some(bytes) || c["bytes"].as_u64() != Some(bytes as u64) {
            fail(&format!(
                "{}: {bytes} bytes, not the size manifest.json and src/gen.rs record",
                path.display()
            ));
        }
        // The same sources do not give the same bytes (nvcc's -lineinfo writes each source file's mtime into
        // .debug_line): a pack rebuilt from them has gen.rs's kpack id and sizes but other bytes. Refuse it here, not
        // at Kernels::load on the GPU host.
        let recorded_fnv = entry
            .split_once("fnv1a64: 0x")
            .and_then(|(_, h)| u64::from_str_radix(&h[..h.find(|c: char| !c.is_ascii_hexdigit())?], 16).ok());
        let fnv = fnv1a64(&image);
        if recorded_fnv != Some(fnv) {
            let recorded = recorded_fnv.map_or_else(|| "none".to_owned(), |h| format!("{h:#018x}"));
            fail(&format!(
                "{}: FNV-1a {fnv:#018x}, src/gen.rs records {recorded}: another build of the same sources \
                 (cubins are not bit-reproducible); embed the pack gen.rs was generated from, or run regen.sh",
                path.display()
            ));
        }
    }
    println!("cargo::rustc-cfg=ink_kpack");
    println!("cargo::rustc-env=INK_KPACK_DIR={}", dir.display());
}

/// FNV-1a 64 (as `ink_kernels::fnv1a64` and xtask gen-kernels).
fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn read(p: &Path) -> Vec<u8> {
    fs::read(p).unwrap_or_else(|e| fail(&format!("{}: {e}", p.display())))
}

fn fail(msg: &dyn std::fmt::Display) -> ! {
    eprintln!("ink-kernels build: {msg}");
    std::process::exit(1)
}
