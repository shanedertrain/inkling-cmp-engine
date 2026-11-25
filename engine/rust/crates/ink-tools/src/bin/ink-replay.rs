// Ported from engine/server/tools/ink_replay.cpp @ 4171493.
//! Parity replay (gate G-FE3): config B's recorded token streams go through the front end over a scripted back end,
//! one step per recorded engine output, and the response JSON or SSE chunk sequence is compared with what config B
//! returned. The comparison ignores object key order (counted separately) and the volatile fields `id`, `created`,
//! `system_fingerprint` and tool-call ids (whose format is checked).
//!
//! ```text
//! ink-replay MODEL_DIR CAPTURE_DIR OUT.json [--only SUBSTR] [--sample N] [--http] [--dump DIR] [--lpids CAPTURE_DIR2]
//! ```
//!
//! `--http` sends every request through the real router (`ink-api::http`) over a loopback socket instead of calling
//! the front end directly (gate G-FE4: the bytes over HTTP are the replay's); `--sample N` keeps about `N` evenly
//! spaced records of each mode (non-streaming, streaming, `prod`).
//!
//! Inputs per request, written by `engine/parity/capture_srv.py`: `<id>.stream.jsonl` (SSE with per-chunk
//! `token_ids`: the exact token grouping config B streamed), `<id>.nonstream.json` and `<id>.prod.jsonl`. Logprob
//! requests need the top-k token ids, which config B returns only with `return_tokens_as_token_ids`: `--lpids` points
//! at a capture of the same requests made that way (`<id>.lpids-*`).
#![forbid(unsafe_code)]
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::{Context, Result};
use clap::Parser;
use ink_api::http::{bind_loopback, router};
use ink_api::{Frontend, Hooks, RequestMeta, Response, ServerConfig};
use ink_backend::{MockBackend, Script, TokenLogprobs};
use ink_text::Tokenizer;
use ink_text::pyjson::truthy;
use serde_json::{Map, Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Parser)]
#[command(name = "ink-replay", version = ink_record::BUILD_LINE, about = "Replay config B's recorded token streams through the front end")]
struct Cli {
    /// Directory with the model's `tokenizer.json` and `special_tokens_map.json`.
    model_dir: PathBuf,
    /// A capture directory (`<id>.stream.jsonl`, `<id>.nonstream.json`, `<id>.prod.jsonl`).
    capture_dir: PathBuf,
    /// Result file (JSON).
    out: PathBuf,
    /// Only replay files whose name contains this text.
    #[arg(long)]
    only: Option<String>,
    /// Keep about this many evenly spaced records of each mode.
    #[arg(long)]
    sample: Option<usize>,
    /// Send the requests over a loopback socket through the HTTP router.
    #[arg(long)]
    http: bool,
    /// Write our frames and bodies here.
    #[arg(long)]
    dump: Option<PathBuf>,
    /// Capture of the same requests made with `return_tokens_as_token_ids` (top-k ids for logprob requests).
    #[arg(long)]
    lpids: Option<PathBuf>,
}

fn read_json(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("{} is not JSON", path.display()))
}

/// The frames of a recorded SSE stream: `"[DONE]"` as a string, the others parsed.
fn sse_frames(record: &Value) -> Result<Vec<Value>> {
    record["sse"]
        .as_array()
        .context("record has no `sse` list")?
        .iter()
        .map(|r| {
            let data = r["data"].as_str().context("sse record without data")?;
            Ok(if data == "[DONE]" {
                Value::from("[DONE]")
            } else {
                serde_json::from_str(data).context("an SSE frame is not JSON")?
            })
        })
        .collect()
}

/// The token ids of `choice.token_ids`, when it is a list.
fn token_group(choice: &Value) -> Option<Vec<i32>> {
    Some(
        choice
            .get("token_ids")?
            .as_array()?
            .iter()
            .filter_map(|x| x.as_i64().and_then(|n| i32::try_from(n).ok()))
            .collect(),
    )
}

fn stream_groups(frames: &[Value]) -> Vec<Vec<i32>> {
    frames
        .iter()
        .filter_map(|j| token_group(j.get("choices")?.as_array()?.first()?))
        .collect()
}

/// `token_id:123` (config B's spelling with `return_tokens_as_token_ids`) to 123.
fn token_id_of(text: &str) -> Option<i32> {
    text.strip_prefix("token_id:")?.parse().ok()
}

/// One position of a chat logprobs record: the sampled token's logprob and the top candidates' ids.
fn chat_position(item: &Value) -> TokenLogprobs {
    let mut t = TokenLogprobs {
        logprob: item["logprob"].as_f64().unwrap_or(0.0) as f32,
        rank: 1,
        top_ids: Vec::new(),
        top_logprobs: Vec::new(),
    };
    for top in item["top_logprobs"].as_array().into_iter().flatten() {
        if let Some(id) = top["token"].as_str().and_then(token_id_of) {
            t.top_ids.push(id);
            t.top_logprobs.push(top["logprob"].as_f64().unwrap_or(0.0) as f32);
        }
    }
    t
}

/// One position of a completions logprobs record (the candidates come as a dict, the sampled token first).
fn completion_position(logprobs: &Value, i: usize, limit: usize) -> TokenLogprobs {
    let mut t = TokenLogprobs {
        logprob: logprobs["token_logprobs"][i].as_f64().unwrap_or(0.0) as f32,
        rank: 1,
        top_ids: Vec::new(),
        top_logprobs: Vec::new(),
    };
    for (key, value) in logprobs["top_logprobs"][i].as_object().into_iter().flatten() {
        if t.top_ids.len() < limit
            && let Some(id) = token_id_of(key)
        {
            t.top_ids.push(id);
            t.top_logprobs.push(value.as_f64().unwrap_or(0.0) as f32);
        }
    }
    t
}

/// Everything one replay needs from a record.
struct Case {
    script: Script,
    ref_frames: Vec<Value>,
    ref_body: Value,
    token_ids_from: Option<String>,
    skip: Option<&'static str>,
}

fn build_case(rec: &Value, cap: &Path, lpids_dir: Option<&Path>) -> Result<Case> {
    let rid = rec["id"].as_str().context("record id")?;
    let mode = rec["mode"].as_str().context("record mode")?;
    let endpoint = rec["endpoint"].as_str().context("record endpoint")?;
    let body = &rec["body"];
    let status = rec["status"].as_i64().context("record status")?;
    let want_lp = (endpoint == "chat" && body.get("logprobs").is_some_and(truthy))
        || (endpoint == "completions" && body.get("logprobs").is_some_and(|v| !v.is_null()));
    let mut case = Case {
        script: Script::default(),
        ref_frames: Vec::new(),
        ref_body: Value::Null,
        token_ids_from: None,
        skip: None,
    };
    if status == 200 {
        if mode == "stream" || mode == "prod" {
            // prod (no return_token_ids): the grouping is borrowed from the same request's stream capture; the
            // comparison is exact only when config B grouped the two runs' tokens identically
            if mode == "prod" {
                let stream = read_json(&cap.join(format!("{rid}.stream.jsonl")))?;
                case.script.groups.extend(stream_groups(&sse_frames(&stream)?));
                case.token_ids_from = Some(format!("{rid}.stream.jsonl"));
            }
            case.ref_frames = sse_frames(rec)?;
            case.script.groups.extend(stream_groups(&case.ref_frames));
        } else {
            case.ref_body = rec["response"].clone();
            case.script
                .groups
                .push(token_group(&case.ref_body["choices"][0]).unwrap_or_default());
        }
    }
    if status == 200 && want_lp {
        attach_logprobs(&mut case, rid, mode, endpoint, body, lpids_dir)?;
    }
    let tokens: usize = case.script.groups.iter().map(Vec::len).sum();
    if status == 200 && case.skip.is_none() && tokens == 0 {
        // include_reasoning=false hides token ids: take them from the paired include_reasoning=true run
        let name = format!("{rid}.ids-{mode}.{}", if mode == "stream" { "jsonl" } else { "json" });
        let path = cap.join(&name);
        if path.exists() {
            let rec = read_json(&path)?;
            case.script.groups = if mode == "stream" {
                stream_groups(&sse_frames(&rec)?)
            } else {
                vec![token_group(&rec["response"]["choices"][0]).unwrap_or_default()]
            };
            case.token_ids_from = Some(name);
        } else {
            case.skip = Some("no token ids in config B's response (include_reasoning=false hides them)");
        }
    }
    Ok(case)
}

/// Logprob requests: top-k ids from the `--lpids` capture, with the reference's token grouping.
fn attach_logprobs(
    case: &mut Case,
    rid: &str,
    mode: &str,
    endpoint: &str,
    body: &Value,
    lpids_dir: Option<&Path>,
) -> Result<()> {
    let path = lpids_dir.map(|d| {
        d.join(format!(
            "{rid}.lpids-{mode}.{}",
            if mode == "stream" { "jsonl" } else { "json" }
        ))
    });
    let Some(path) = path.filter(|p| p.exists()) else {
        case.skip = Some("logprobs: no top-k token-id capture");
        return Ok(());
    };
    let rec = read_json(&path)?;
    let completion_limit = usize::try_from(body["logprobs"].as_i64().unwrap_or(0) + 1).unwrap_or(1);
    let mut groups: Vec<Vec<i32>> = Vec::new();
    let mut flat: Vec<TokenLogprobs> = Vec::new();
    let mut add_choice = |c: &Value| {
        let group = token_group(c).unwrap_or_default();
        let mut l = Vec::new();
        if c["logprobs"].is_object() {
            if endpoint == "chat" {
                l.extend(
                    c["logprobs"]["content"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .map(chat_position),
                );
            } else {
                let n = c["logprobs"]["tokens"].as_array().map_or(0, Vec::len);
                l.extend((0..n).map(|i| completion_position(&c["logprobs"], i, completion_limit)));
            }
        }
        if !group.is_empty() {
            groups.push(group);
            flat.extend(l);
        }
    };
    if mode == "stream" {
        for frame in sse_frames(&rec)? {
            if let Some(c) = frame.get("choices").and_then(Value::as_array).and_then(|c| c.first()) {
                add_choice(c);
            }
        }
    } else {
        add_choice(&rec["response"]["choices"][0]);
    }
    // the token-id capture must have produced the same tokens as the reference
    let ours: Vec<i32> = groups.concat();
    let reference: Vec<i32> = case.script.groups.concat();
    if ours != reference {
        case.skip = Some("logprobs: token-id capture generated different tokens");
        return Ok(());
    }
    // keep the reference grouping, attach logprobs in order
    let mut pos = 0;
    for g in &case.script.groups {
        let end = (pos + g.len()).min(flat.len());
        case.script.logprobs.push(flat[pos.min(end)..end].to_vec());
        pos += g.len();
    }
    Ok(())
}

/// Removes what differs between runs: `id`, `created`, `system_fingerprint` and the ids of tool calls.
fn normalize(j: &mut Value) {
    let Some(obj) = j.as_object_mut() else { return };
    for key in ["id", "created", "system_fingerprint"] {
        obj.shift_remove(key);
    }
    for choice in obj
        .get_mut("choices")
        .and_then(Value::as_array_mut)
        .into_iter()
        .flatten()
    {
        for key in ["delta", "message"] {
            let calls = choice
                .get_mut(key)
                .and_then(|m| m.get_mut("tool_calls"))
                .and_then(Value::as_array_mut);
            for call in calls.into_iter().flatten() {
                if call.get("id").is_some_and(Value::is_string) {
                    call["id"] = Value::from("<ID>");
                }
            }
        }
    }
}

/// Whether every tool-call id is `chatcmpl-tool-` and 16 lower-case hex digits.
fn tool_ids_well_formed(j: &Value) -> bool {
    let ok = |id: &str| {
        id.strip_prefix("chatcmpl-tool-")
            .is_some_and(|h| h.len() == 16 && h.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
    };
    j["choices"].as_array().into_iter().flatten().all(|c| {
        ["delta", "message"].iter().all(|k| {
            c.get(k)
                .and_then(|m| m.get("tool_calls"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|t| t.get("id").and_then(Value::as_str))
                .all(ok)
        })
    })
}

/// `value` with the `logprobs` of every choice emptied, to tell a differing logprob field from a differing text.
fn without_logprobs(mut value: Value) -> Value {
    let choices: Vec<&mut Value> = match &mut value {
        Value::Array(frames) => frames.iter_mut().filter_map(|f| f.get_mut("choices")).collect(),
        other => other.get_mut("choices").into_iter().collect(),
    };
    for choices in choices {
        for choice in choices.as_array_mut().into_iter().flatten() {
            if let Some(slot) = choice.get_mut("logprobs") {
                *slot = Value::Null;
            }
        }
    }
    value
}

/// The outcome of comparing one response with config B's.
struct Compared {
    same: bool,
    /// Same, and the keys of every object also come in config B's order.
    ordered: bool,
    /// Not the same, but equal once the `logprobs` fields are left out.
    logprobs_only: bool,
    detail: String,
}

fn clip(text: &str, n: usize) -> String {
    text.chars().take(n).collect()
}

fn compare_stream(frames: &[String], reference: &[Value]) -> Compared {
    let ours: Vec<Value> = frames
        .iter()
        .map(|f| {
            let data = f
                .strip_prefix("data: ")
                .and_then(|f| f.strip_suffix("\n\n"))
                .unwrap_or(f);
            if data == "[DONE]" {
                Value::from("[DONE]")
            } else {
                serde_json::from_str(data).unwrap_or_else(|e| Value::from(format!("unparsable frame: {e}")))
            }
        })
        .collect();
    let ids_ok = ours.iter().all(tool_ids_well_formed);
    let (mut a, mut b) = (ours, reference.to_vec());
    a.iter_mut().chain(b.iter_mut()).for_each(normalize);
    let same = a == b && ids_ok;
    let ordered = same && serde_json::to_string(&a).ok() == serde_json::to_string(&b).ok();
    let detail = if same {
        format!("{} chunks", a.len())
    } else {
        let k = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
        let show = |v: &[Value]| v.get(k).map_or_else(|| "-".to_owned(), |j| clip(&j.to_string(), 400));
        format!(
            "chunks ours {} ref {}, first diff at {k}: ours {} | ref {}{}",
            a.len(),
            b.len(),
            show(&a),
            show(&b),
            if ids_ok { "" } else { " (tool-call id format)" }
        )
    };
    Compared {
        same,
        ordered,
        logprobs_only: !same && without_logprobs(Value::Array(a)) == without_logprobs(Value::Array(b)),
        detail,
    }
}

fn compare_body(body: &str, reference: &Value) -> Compared {
    let Ok(mut a) = serde_json::from_str::<Value>(body) else {
        return Compared {
            same: false,
            ordered: false,
            logprobs_only: false,
            detail: format!("ours is not JSON: {}", clip(body, 600)),
        };
    };
    let ids_ok = tool_ids_well_formed(&a);
    let mut b = reference.clone();
    normalize(&mut a);
    normalize(&mut b);
    let same = a == b && ids_ok;
    let ordered = same && serde_json::to_string(&a).ok() == serde_json::to_string(&b).ok();
    let detail = if same {
        "equal".to_owned()
    } else {
        format!("ours {} | ref {}", clip(&a.to_string(), 600), clip(&b.to_string(), 600))
    };
    Compared {
        same,
        ordered,
        logprobs_only: !same && without_logprobs(a) == without_logprobs(b),
        detail,
    }
}

/// The body of a chunked HTTP/1.1 response. Strict, so that it cannot hide a framing error: every chunk ends in
/// CRLF, sizes are bare hex (hyper writes no extensions), the last chunk is followed by the final CRLF and nothing else.
fn dechunk(mut raw: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let eol = raw.windows(2).position(|w| w == b"\r\n").context("chunk size line")?;
        let line = std::str::from_utf8(&raw[..eol])?;
        anyhow::ensure!(
            !line.is_empty() && line.bytes().all(|b| b.is_ascii_hexdigit()),
            "chunk size line {line:?}"
        );
        let size = usize::from_str_radix(line, 16)?;
        raw = &raw[eol + 2..];
        if size == 0 {
            anyhow::ensure!(raw == b"\r\n", "{} bytes after the last chunk", raw.len());
            return Ok(out);
        }
        anyhow::ensure!(raw.len() >= size + 2, "short chunk");
        anyhow::ensure!(
            &raw[size..size + 2] == b"\r\n",
            "chunk of {size} bytes not followed by CRLF"
        );
        out.extend_from_slice(&raw[..size]);
        raw = &raw[size + 2..];
    }
}

/// Posts `body` to `fe` through the router on an ephemeral loopback port and reads the answer to the end.
async fn over_http(fe: Frontend, endpoint: &str, body: &str) -> Result<(Response, Vec<String>)> {
    let listener = bind_loopback("127.0.0.1", 0).await?;
    let addr = listener.local_addr()?;
    let server = tokio::spawn(async move { axum::serve(listener, router(Arc::new(fe), String::new())).await });
    let path = if endpoint == "chat" {
        "/v1/chat/completions"
    } else {
        "/v1/completions"
    };
    let mut socket = tokio::net::TcpStream::connect(addr).await?;
    let head = format!(
        "POST {path} HTTP/1.1\r\nhost: {addr}\r\nconnection: close\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n",
        body.len()
    );
    socket.write_all(head.as_bytes()).await?;
    socket.write_all(body.as_bytes()).await?;
    let mut raw = Vec::new();
    socket.read_to_end(&mut raw).await?;
    server.abort();
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .context("no header block")?;
    let head = std::str::from_utf8(&raw[..split])?.to_ascii_lowercase();
    let status: u16 = head
        .split(' ')
        .nth(1)
        .context("status line")?
        .parse()
        .context("status code")?;
    let payload = if head.contains("transfer-encoding: chunked") {
        dechunk(&raw[split + 4..])?
    } else {
        raw[split + 4..].to_vec()
    };
    let text = String::from_utf8(payload).context("the body is UTF-8")?;
    if head.contains("content-type: text/event-stream") {
        let frames = text.split_inclusive("\n\n").map(str::to_owned).collect::<Vec<_>>();
        anyhow::ensure!(
            status == 200 && frames.concat() == text,
            "a stream is 200 and whole frames"
        );
        Ok((Response::Streamed, frames))
    } else {
        Ok((Response::Json { status, body: text }, Vec::new()))
    }
}

/// Runs one request through a fresh front end over the scripted back end, directly or over HTTP.
fn run_request(
    rt: &tokio::runtime::Runtime,
    tok: &Arc<Tokenizer>,
    model_dir: &Path,
    script: Script,
    endpoint: &str,
    body: &Value,
    http: bool,
) -> Result<(Response, Vec<String>)> {
    let backend = Arc::new(MockBackend::new());
    backend.push(script);
    let counter = Arc::new(AtomicU32::new(0));
    let hooks = Hooks {
        request_hex: Arc::new(|| "0000000000000000".to_owned()),
        tool_call_id: Arc::new(move || format!("chatcmpl-tool-{:016x}", counter.fetch_add(1, Ordering::Relaxed) + 1)),
        now: Arc::new(|| 0),
    };
    let cfg = ServerConfig {
        model_dir: model_dir.to_path_buf(),
        max_model_len: 131_072, // config B's --max-model-len
        ..ServerConfig::default()
    };
    let fe = Frontend::new(Arc::clone(tok), cfg, backend, hooks)?;
    let text = body.to_string();
    if http {
        return rt.block_on(over_http(fe, endpoint, &text));
    }
    let mut frames = Vec::new();
    let meta = RequestMeta::default();
    let write = |f: &str| {
        if !f.is_empty() {
            frames.push(f.to_owned());
        }
        true
    };
    let response = rt.block_on(async {
        if endpoint == "chat" {
            fe.chat(&text, &meta, write).await
        } else {
            fe.completions(&text, &meta, write).await
        }
    });
    Ok((response, frames))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(dump) = &cli.dump {
        fs::create_dir_all(dump)?;
    }
    let tok = Arc::new(Tokenizer::load(&cli.model_dir)?);
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    let mut files: Vec<(String, PathBuf)> = fs::read_dir(&cli.capture_dir)
        .with_context(|| format!("cannot list {}", cli.capture_dir.display()))?
        .filter_map(|e| {
            let e = e.ok()?;
            let name = e.file_name().to_string_lossy().into_owned();
            let wanted =
                name.ends_with(".stream.jsonl") || name.ends_with(".nonstream.json") || name.ends_with(".prod.jsonl");
            (wanted && cli.only.as_ref().is_none_or(|o| name.contains(o.as_str()))).then(|| (name, e.path()))
        })
        .collect();
    files.sort();
    if let Some(n) = cli.sample.filter(|&n| n > 0) {
        // per mode: the sorted list alternates `<id>.nonstream.json` / `<id>.stream.jsonl`, so one even stride over
        // all files picks a single mode (it picked only non-streaming records before)
        let mut kept = Vec::new();
        for suffix in [".nonstream.json", ".stream.jsonl", ".prod.jsonl"] {
            let group: Vec<_> = files
                .iter()
                .filter(|(name, _)| name.ends_with(suffix))
                .cloned()
                .collect();
            let stride = (group.len() / n).max(1);
            kept.extend(group.into_iter().step_by(stride).take(n));
        }
        kept.sort();
        files = kept;
    }

    let (mut n_same, mut n_total, mut n_skipped, mut order_diff) = (0usize, 0usize, 0usize, 0usize);
    let mut by_kind: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut by_endpoint: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut rows = Vec::new();
    for (_, path) in &files {
        let rec = read_json(path)?;
        let rid = rec["id"].as_str().context("record id")?;
        let mode = rec["mode"].as_str().context("record mode")?;
        let endpoint = rec["endpoint"].as_str().context("record endpoint")?;
        let status = rec["status"].as_i64().context("record status")?;
        let mut row = Map::new();
        row.insert("id".into(), rid.into());
        row.insert("mode".into(), mode.into());
        row.insert("endpoint".into(), endpoint.into());
        let case = build_case(&rec, &cli.capture_dir, cli.lpids.as_deref())?;
        if let Some(from) = &case.token_ids_from {
            row.insert("token_ids_from".into(), from.as_str().into());
        }
        if let Some(reason) = case.skip {
            row.insert("skipped".into(), reason.into());
            n_skipped += 1;
            rows.push(Value::Object(row));
            continue;
        }
        let (response, frames) = run_request(&rt, &tok, &cli.model_dir, case.script, endpoint, &rec["body"], cli.http)?;
        let (status_ours, body_ours) = match &response {
            Response::Json { status, body } => (i64::from(*status), body.as_str()),
            Response::Streamed => (200, ""),
            Response::ClientGone => (499, ""),
        };
        let compared = if status != 200 {
            Compared {
                same: status_ours == status,
                ordered: true,
                logprobs_only: false,
                detail: format!("config B HTTP {status}, ours {status_ours}: {}", clip(body_ours, 200)),
            }
        } else if mode == "stream" || mode == "prod" {
            if let Some(dump) = &cli.dump {
                fs::write(dump.join(format!("{rid}.stream.ours.jsonl")), frames.concat())?;
            }
            compare_stream(&frames, &case.ref_frames)
        } else {
            if let Some(dump) = &cli.dump {
                fs::write(dump.join(format!("{rid}.nonstream.ours.json")), body_ours)?;
            }
            compare_body(body_ours, &case.ref_body)
        };
        let Compared {
            same,
            ordered,
            logprobs_only,
            detail,
        } = compared;
        n_total += 1;
        n_same += usize::from(same);
        order_diff += usize::from(same && !ordered);
        let kind = format!("{}/{mode}", rid.split('-').next().unwrap_or(rid));
        for (map, key) in [(&mut by_kind, kind), (&mut by_endpoint, format!("{endpoint}/{mode}"))] {
            let e = map.entry(key).or_default();
            e.0 += usize::from(same);
            e.1 += 1;
        }
        row.insert("identical".into(), same.into());
        row.insert("key_order_identical".into(), (same && ordered).into());
        if !same {
            row.insert("differs_only_in_logprobs".into(), logprobs_only.into());
        }
        row.insert("detail".into(), detail.clone().into());
        rows.push(Value::Object(row));
        if !same {
            println!("DIFF {rid} {mode}: {detail}");
        }
    }
    println!(
        "response parity: {n_same}/{n_total} identical (schema-equal), {n_skipped} skipped; of the identical, \
         {order_diff} differ in object key order only"
    );
    let tally = |m: &BTreeMap<String, (usize, usize)>| -> Value {
        Value::Object(
            m.iter()
                .map(|(k, (same, total))| (k.clone(), json!({"identical": same, "total": total})))
                .collect(),
        )
    };
    for (k, (same, total)) in &by_kind {
        println!("  {k}: {same}/{total}");
    }
    for (k, (same, total)) in &by_endpoint {
        println!("  by endpoint {k}: {same}/{total}");
    }
    let out = json!({"identical": n_same, "total": n_total, "skipped": n_skipped,
                     "key_order_differences": order_diff, "by_kind": tally(&by_kind),
                     "by_endpoint": tally(&by_endpoint), "rows": rows});
    fs::write(&cli.out, serde_json::to_string_pretty(&out)?)
        .with_context(|| format!("cannot write {}", cli.out.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_ids_of_return_tokens_as_token_ids() {
        assert_eq!(token_id_of("token_id:200006"), Some(200_006));
        assert_eq!(token_id_of("hello"), None);
    }

    #[test]
    fn normalize_drops_volatile_fields_and_masks_tool_ids() {
        let mut j = json!({"id": "x", "created": 1, "system_fingerprint": "f", "model": "m",
            "choices": [{"delta": {"tool_calls": [{"id": "chatcmpl-tool-0123456789abcdef", "index": 0}]}}]});
        normalize(&mut j);
        assert_eq!(
            j,
            json!({"model": "m", "choices": [{"delta": {"tool_calls": [{"id": "<ID>", "index": 0}]}}]})
        );
    }

    #[test]
    fn tool_call_ids_must_be_prefix_and_sixteen_hex_digits() {
        let with = |id: &str| json!({"choices": [{"message": {"tool_calls": [{"id": id}]}}]});
        assert!(tool_ids_well_formed(&with("chatcmpl-tool-0123456789abcdef")));
        assert!(!tool_ids_well_formed(&with("chatcmpl-tool-0123456789ABCDEF")));
        assert!(!tool_ids_well_formed(&with("call_1")));
    }

    #[test]
    fn bodies_equal_up_to_key_order_are_the_same_but_not_ordered() {
        let reference = json!({"object": "chat.completion", "choices": [{"index": 0}]});
        let ours = r#"{"choices":[{"index":0}],"object":"chat.completion","id":"x"}"#;
        let c = compare_body(ours, &reference);
        assert!(c.same && !c.ordered);
        let c = compare_body(
            r#"{"object":"chat.completion","choices":[{"index":0}],"created":5}"#,
            &reference,
        );
        assert!(c.same && c.ordered);
    }

    #[test]
    fn a_differing_logprobs_field_is_reported_as_logprobs_only() {
        let reference = json!({"choices": [{"text": "x", "logprobs": {"top": [1.5]}}]});
        let c = compare_body(r#"{"choices":[{"text":"x","logprobs":{"top":[2.5]}}]}"#, &reference);
        assert!(!c.same && c.logprobs_only);
        let c = compare_body(r#"{"choices":[{"text":"y","logprobs":{"top":[1.5]}}]}"#, &reference);
        assert!(!c.same && !c.logprobs_only);
    }
}
