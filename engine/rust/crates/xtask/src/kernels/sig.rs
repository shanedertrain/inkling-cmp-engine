//! Demangled kernel signatures: `void s2f::gemm_frag<1, false, 0, 16>(s2f::GemmArgs)` -> instance
//! `s2f::gemm_frag<1, false, 0, 16>` and parameters `["s2f::GemmArgs"]`.

use anyhow::{Context, Result, bail};

/// The anonymous namespace as the demangler prints it; dropped from instance names (keys.toml leaves it out, since
/// its mangled form carries a per-TU id).
const ANON: &str = "(anonymous namespace)::";

/// Demangles a mangled kernel symbol.
pub fn demangle(symbol: &str) -> Result<String> {
    let sym = cpp_demangle::Symbol::new(symbol).with_context(|| format!("cannot demangle {symbol}"))?;
    sym.demangle(&cpp_demangle::DemangleOptions::default())
        .with_context(|| format!("cannot demangle {symbol}"))
}

/// Splits a demangled kernel signature into its instance name and its parameter types.
pub fn split(demangled: &str) -> Result<(String, Vec<String>)> {
    let s = demangled.trim();
    let Some(body) = s.strip_suffix(')') else {
        bail!("no parameter list in {s:?}")
    };
    let mut depth = 0i32;
    let mut open = None;
    for (i, c) in body.char_indices().rev() {
        match c {
            ')' | '>' | ']' => depth += 1,
            '<' | '[' => depth -= 1,
            '(' if depth == 0 => {
                open = Some(i);
                break;
            }
            '(' => depth -= 1,
            _ => {}
        }
    }
    let open = open.with_context(|| format!("unbalanced parameter list in {s:?}"))?;
    let name = body[..open].trim();
    let name = name.strip_prefix("void ").unwrap_or(name).replace(ANON, "");
    Ok((name, split_top(&body[open + 1..])))
}

/// Splits at top-level commas.
fn split_top(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let (mut depth, mut start) = (0i32, 0);
    for (i, c) in s.char_indices() {
        match c {
            '(' | '<' | '[' => depth += 1,
            ')' | '>' | ']' => depth -= 1,
            ',' if depth == 0 => {
                out.push(s[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    let last = s[start..].trim();
    if !last.is_empty() && last != "void" {
        out.push(last.to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_templates_and_anonymous_namespaces() {
        let (n, p) = split("void s2f::gemm_frag<1, false, 0, 16>(s2f::GemmArgs)").unwrap();
        assert_eq!(
            (n.as_str(), p),
            ("s2f::gemm_frag<1, false, 0, 16>", vec!["s2f::GemmArgs".to_string()])
        );
        let (n, p) = split("ink::(anonymous namespace)::verify_ids_kernel(int const*, int const*, int*, int)").unwrap();
        assert_eq!(n, "ink::verify_ids_kernel");
        assert_eq!(p, ["int const*", "int const*", "int*", "int"]);
        let (n, p) = split("k_bump(int*)").unwrap();
        assert_eq!((n.as_str(), p.len()), ("k_bump", 1));
        let (n, p) = split("void glue::rmsnorm_kernel<__nv_bfloat16>()").unwrap();
        assert_eq!((n.as_str(), p.len()), ("glue::rmsnorm_kernel<__nv_bfloat16>", 0));
        assert!(split("k_bump").is_err());
        assert!(split("k_bump(int*))").is_err());
    }

    #[test]
    fn demangles_like_the_pack_keys() {
        // symbols of the C++ engine's fatbin at ce7de6d (bench/raw/2026-11-23-rust-R2-pack/base/e0-lib.jsonl)
        let d = demangle("_ZN3ink41_GLOBAL__N__40219435_9_engine_cu_6d0b427217verify_ids_kernelEPKiS2_Pii").unwrap();
        assert_eq!(split(&d).unwrap().0, "ink::verify_ids_kernel");
        let d = demangle("_ZN4pmoe9moe_gemm2ILi2ELi2ELi3ELi8ELi0EEEvNS_6ParamsE").unwrap();
        assert_eq!(
            split(&d).unwrap(),
            (
                "pmoe::moe_gemm2<2, 2, 3, 8, 0>".to_string(),
                vec!["pmoe::Params".to_string()]
            )
        );
        let d = demangle("_ZN3s2f11markov_coopILb1EEEvNS_6MarkovE").unwrap();
        assert_eq!(split(&d).unwrap().0, "s2f::markov_coop<true>");
        let d = demangle("_Z11k_copy_colsPK13__nv_bfloat16iiiPS_i").unwrap();
        assert_eq!(
            split(&d).unwrap().1,
            ["__nv_bfloat16 const*", "int", "int", "int", "__nv_bfloat16*", "int"]
        );
    }
}
