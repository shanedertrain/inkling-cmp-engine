//! C++ parameter names of every `__global__` function in the pack's sources (review R3 finding F13: launchers with
//! positional `a0..aN` let two arguments of one type be swapped unnoticed).
//!
//! A small scanner, not a C++ parser: it removes comments, string and character literals and preprocessor lines,
//! tracks `namespace X {` nesting by braces (the anonymous namespace adds nothing to the name, as in keys.toml), and
//! reads the parameter list of every `__global__` function that has a body. A kernel is matched by its qualified name
//! (the keys.toml instance without template arguments) and its parameter count, which must select exactly one
//! definition; anything else fails the generator.

use anyhow::{Result, bail};

/// One `__global__` definition: qualified name (`glue::qkv_post_kernel`) and parameter names in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelDecl {
    pub name: String,
    pub params: Vec<String>,
    /// File it was found in (for messages).
    pub file: String,
}

/// Every `__global__` definition of `files` (path, text).
pub fn scan(files: &[(String, String)]) -> Result<Vec<KernelDecl>> {
    let mut out = Vec::new();
    for (path, text) in files {
        for (name, params) in scan_text(&clean(text))? {
            out.push(KernelDecl {
                name,
                params,
                file: path.clone(),
            });
        }
    }
    Ok(out)
}

/// The parameter names of kernel instance `instance` (`glue::rmsnorm_kernel<float>`) with `n` parameters.
pub fn lookup<'d>(decls: &'d [KernelDecl], instance: &str, n: usize) -> Result<&'d [String]> {
    let base = match instance.find('<') {
        Some(i) => &instance[..i],
        None => instance,
    };
    let hits: Vec<&KernelDecl> = decls.iter().filter(|d| d.name == base && d.params.len() == n).collect();
    match hits.as_slice() {
        [one] => Ok(&one.params),
        [] => {
            let other: Vec<String> = decls
                .iter()
                .filter(|d| d.name == base)
                .map(|d| format!("{} ({} parameters)", d.file, d.params.len()))
                .collect();
            bail!("no __global__ definition of {base} with {n} parameters in the pack's sources (found: {other:?})")
        }
        many => bail!(
            "{} __global__ definitions of {base} with {n} parameters: {:?}",
            many.len(),
            many.iter().map(|d| &d.file).collect::<Vec<_>>()
        ),
    }
}

/// Replaces comments, string / character literals and preprocessor lines (with their continuations) by spaces,
/// keeping everything else (so braces inside them do not count).
fn clean(src: &str) -> String {
    let b = src.as_bytes();
    let mut o = Vec::with_capacity(b.len());
    let mut i = 0;
    let mut line_start = true;
    while i < b.len() {
        let c = b[i];
        if line_start && c == b'#' {
            // preprocessor line, continued by a trailing backslash
            while i < b.len() {
                if b[i] == b'\n' && (i == 0 || b[i - 1] != b'\\') {
                    break;
                }
                o.push(b' ');
                i += 1;
            }
            continue;
        }
        if c == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                o.push(b' ');
                i += 1;
            }
            continue;
        }
        if c == b'/' && b.get(i + 1) == Some(&b'*') {
            o.extend_from_slice(b"  ");
            i += 2;
            while i < b.len() && !(b[i] == b'*' && b.get(i + 1) == Some(&b'/')) {
                o.push(if b[i] == b'\n' { b'\n' } else { b' ' });
                i += 1;
            }
            o.extend_from_slice(b"  ");
            i = (i + 2).min(b.len());
            continue;
        }
        if c == b'"' || c == b'\'' {
            o.push(b' ');
            i += 1;
            while i < b.len() && b[i] != c {
                if b[i] == b'\\' {
                    o.push(b' ');
                    i += 1;
                }
                o.push(b' ');
                i += 1;
            }
            o.push(b' ');
            i += 1;
            line_start = false;
            continue;
        }
        if c == b'\n' {
            line_start = true;
        } else if !c.is_ascii_whitespace() {
            line_start = false;
        }
        o.push(c);
        i += 1;
    }
    String::from_utf8_lossy(&o).into_owned()
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Punct(char),
}

fn tokens(s: &str) -> Vec<Tok> {
    let mut out = Vec::new();
    let mut it = s.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        if c.is_ascii_alphabetic() || c == '_' {
            let mut end = i + c.len_utf8();
            while let Some(&(j, d)) = it.peek() {
                if d.is_ascii_alphanumeric() || d == '_' {
                    end = j + d.len_utf8();
                    it.next();
                } else {
                    break;
                }
            }
            out.push(Tok::Ident(s[i..end].to_string()));
        } else if c.is_ascii_digit() {
            while let Some(&(_, d)) = it.peek() {
                if d.is_ascii_alphanumeric() || d == '.' || d == '_' {
                    it.next();
                } else {
                    break;
                }
            }
            out.push(Tok::Ident(String::new())); // a number: never a name
        } else if !c.is_whitespace() {
            out.push(Tok::Punct(c));
        }
    }
    out
}

/// (qualified name, parameter names) of every `__global__` function with a body in cleaned text.
fn scan_text(s: &str) -> Result<Vec<(String, Vec<String>)>> {
    let t = tokens(s);
    let mut out = Vec::new();
    // brace stack: Some(name) for a named namespace, None for the anonymous one or any other block
    let mut stack: Vec<Option<String>> = Vec::new();
    let mut pending_ns: Option<Option<String>> = None;
    let mut i = 0;
    while i < t.len() {
        match &t[i] {
            Tok::Ident(w) if w == "namespace" => {
                // `namespace a::b {`, `namespace a {`, `namespace {`; `namespace x = y;` is an alias
                let mut j = i + 1;
                let mut name = String::new();
                while let Some(tok) = t.get(j) {
                    match tok {
                        Tok::Ident(n) => name.push_str(n),
                        Tok::Punct(':') => name.push(':'),
                        _ => break,
                    }
                    j += 1;
                }
                if t.get(j) == Some(&Tok::Punct('{')) {
                    pending_ns = Some((!name.is_empty()).then_some(name));
                    i = j;
                    continue;
                }
                i = j;
            }
            Tok::Punct('{') => {
                stack.push(pending_ns.take().unwrap_or(None));
                i += 1;
            }
            Tok::Punct('}') => {
                if stack.pop().is_none() {
                    bail!("unbalanced '}}'");
                }
                i += 1;
            }
            Tok::Ident(w) if w == "__global__" => {
                let (name, params, has_body, next) = global_decl(&t, i + 1)?;
                if has_body {
                    let mut q: Vec<&str> = stack.iter().flatten().map(String::as_str).collect();
                    q.push(&name);
                    out.push((q.join("::"), params));
                }
                i = next;
            }
            _ => i += 1,
        }
    }
    if !stack.is_empty() {
        bail!("unbalanced '{{' ({} open at end of file)", stack.len());
    }
    Ok(out)
}

/// Name, parameter names, whether a body follows, index after the parameter list.
type GlobalDecl = (String, Vec<String>, bool, usize);

/// After `__global__`: skips the return type and `__launch_bounds__(...)`, reads the name and the parameter list.
/// Returns the name, the parameter names and whether a body follows, and the index after the parameter list.
fn global_decl(t: &[Tok], mut i: usize) -> Result<GlobalDecl> {
    loop {
        match t.get(i) {
            Some(Tok::Ident(w)) if w == "__launch_bounds__" => {
                i = skip_parens(t, i + 1)?;
            }
            Some(Tok::Ident(w)) if !w.is_empty() && t.get(i + 1) == Some(&Tok::Punct('(')) => {
                let name = w.clone();
                let close = skip_parens(t, i + 1)?;
                let params = params(&t[i + 2..close - 1]);
                let mut j = close;
                // trailing qualifiers before the body
                while let Some(Tok::Ident(_)) = t.get(j) {
                    j += 1;
                }
                let has_body = t.get(j) == Some(&Tok::Punct('{'));
                return Ok((name, params, has_body, close));
            }
            Some(_) => i += 1,
            None => bail!("__global__ without a function"),
        }
    }
}

/// Index after the parenthesised group that starts at `t[i] == '('`.
fn skip_parens(t: &[Tok], i: usize) -> Result<usize> {
    if t.get(i) != Some(&Tok::Punct('(')) {
        bail!("expected '('");
    }
    let mut depth = 0i32;
    for (j, tok) in t.iter().enumerate().skip(i) {
        match tok {
            Tok::Punct('(') => depth += 1,
            Tok::Punct(')') => {
                depth -= 1;
                if depth == 0 {
                    return Ok(j + 1);
                }
            }
            _ => {}
        }
    }
    bail!("unbalanced '('")
}

/// Parameter names of a parameter list's tokens: the last identifier of each top-level comma-separated part, before
/// any default value; `a<i>` for an unnamed parameter.
fn params(t: &[Tok]) -> Vec<String> {
    let mut parts: Vec<&[Tok]> = Vec::new();
    let (mut depth, mut start) = (0i32, 0);
    for (j, tok) in t.iter().enumerate() {
        match tok {
            Tok::Punct('(' | '<' | '[' | '{') => depth += 1,
            Tok::Punct(')' | '>' | ']' | '}') => depth -= 1,
            Tok::Punct(',') if depth == 0 => {
                parts.push(&t[start..j]);
                start = j + 1;
            }
            _ => {}
        }
    }
    if start < t.len() {
        parts.push(&t[start..]);
    }
    let mut names = Vec::new();
    for (k, p) in parts.iter().enumerate() {
        let p = match p.iter().position(|x| *x == Tok::Punct('=')) {
            Some(e) => &p[..e],
            None => p,
        };
        if p.len() == 1 && p[0] == Tok::Ident("void".into()) {
            continue;
        }
        let named = p.len() >= 2
            && matches!(p.last(), Some(Tok::Ident(n)) if !n.is_empty() && !QUALIFIERS.contains(&n.as_str()));
        names.push(match (named, p.last()) {
            (true, Some(Tok::Ident(n))) => n.clone(),
            _ => format!("a{k}"),
        });
    }
    names
}

/// Words that end a parameter's type without naming it.
const QUALIFIERS: &[&str] = &["const", "__restrict__", "volatile"];

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = r#"
#define CK(x) do { if (x) { abort(); } } \
  while (0)
namespace glue {
constexpr int H = 2304;   // { not a block
struct Rows { const int* pos; };
/* a { comment */
template <int Q>
__global__ void attn_res_kernel(ResSrc s, const float* queries, float* out, float eps) {
  asm("{%0}" : "=r"(x));
  if (s.nb) { out[0] = '{'; }
}
__global__ void __launch_bounds__(256) mg_in_kernel(const float* hid, int ldh, bf16* __restrict__ g, float eps,
                                                     long long hs = 0, long long gs = H) {
}
__global__ void decl_only(int a);
}  // namespace glue
namespace ink {
namespace {
__global__ void accept_kernel(const int* amax, int T) { }
}  // namespace
}  // namespace ink
__global__ void k_bump(int*) {}
"#;

    #[test]
    fn finds_kernels_with_namespaces_and_names() {
        let d = scan(&[("x.cuh".into(), SRC.into())]).unwrap();
        let got: Vec<(&str, Vec<&str>)> = d
            .iter()
            .map(|k| (k.name.as_str(), k.params.iter().map(String::as_str).collect()))
            .collect();
        assert_eq!(
            got,
            vec![
                ("glue::attn_res_kernel", vec!["s", "queries", "out", "eps"]),
                ("glue::mg_in_kernel", vec!["hid", "ldh", "g", "eps", "hs", "gs"]),
                ("ink::accept_kernel", vec!["amax", "T"]),
                ("k_bump", vec!["a0"]),
            ]
        );
        assert_eq!(lookup(&d, "glue::attn_res_kernel<3>", 4).unwrap()[1], "queries");
        assert!(lookup(&d, "glue::attn_res_kernel<3>", 3).is_err(), "count must match");
        assert!(
            lookup(&d, "glue::decl_only", 1).is_err(),
            "declarations without a body do not count"
        );
    }

    #[test]
    fn unbalanced_braces_fail() {
        assert!(scan(&[("x".into(), "namespace a { void f() {".into())]).is_err());
        assert!(scan(&[("x".into(), "}".into())]).is_err());
    }
}
