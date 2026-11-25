//! Gate G-FE4, process level: the `ink-server` binary on an ephemeral loopback port, driven over real sockets.
//!
//! The run tests need the model's tokenizer files (kept out of git) and say so when `INK_MODEL_DIR` is unset:
//!
//! ```sh
//! # from engine/rust/
//! scripts/build.sh env INK_MODEL_DIR=/w/testdata-local/inkling cargo test -p ink-server --offline --locked
//! ```

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

const BIN: &str = env!("CARGO_BIN_EXE_ink-server");

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("INK_MODEL_DIR")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty());
    if dir.is_none() {
        eprintln!("SKIPPED: INK_MODEL_DIR is not set (see the module documentation of tests/server.rs)");
    }
    dir
}

/// A running server, killed when dropped.
struct Server {
    child: Child,
    addr: SocketAddr,
    log: std::thread::JoinHandle<String>,
}

impl Server {
    fn start(model_dir: &std::path::Path, args: &[&str]) -> Self {
        let mut child = Command::new(BIN)
            .arg("--model-dir")
            .arg(model_dir)
            .args(["--port", "0"])
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("ink-server starts");
        let mut lines = BufReader::new(child.stderr.take().expect("stderr"));
        let mut seen = String::new();
        let addr = loop {
            let mut line = String::new();
            assert!(
                lines.read_line(&mut line).expect("log line") > 0,
                "the server ended early:\n{seen}"
            );
            seen.push_str(&line);
            if let Some(rest) = line.split("listening on http://").nth(1) {
                break rest
                    .split_whitespace()
                    .next()
                    .expect("an address")
                    .parse()
                    .expect("a socket address");
            }
        };
        assert!(
            seen.contains("rust version="),
            "the first log line is the build line:\n{seen}"
        );
        // keep draining the log so the server never blocks on a full pipe
        let log = std::thread::spawn(move || {
            let mut rest = seen;
            lines.read_to_string(&mut rest).ok();
            rest
        });
        Self { child, addr, log }
    }

    fn get(&self, path: &str) -> (u16, String) {
        request(self.addr, &format!("GET {path} HTTP/1.0\r\n\r\n"), b"")
    }

    fn post(&self, path: &str, body: &str) -> (u16, String) {
        let head = format!(
            "POST {path} HTTP/1.0\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n",
            body.len()
        );
        request(self.addr, &head, body.as_bytes())
    }

    /// SIGTERM, then the exit status and the whole log.
    fn terminate(mut self) -> (std::process::ExitStatus, String) {
        let kill = Command::new("kill")
            .args(["-TERM", &self.child.id().to_string()])
            .status()
            .expect("kill runs");
        assert!(kill.success());
        let t0 = Instant::now();
        let status = loop {
            if let Some(s) = self.child.try_wait().expect("wait") {
                break s;
            }
            assert!(
                t0.elapsed() < Duration::from_secs(15),
                "the server did not stop on SIGTERM"
            );
            std::thread::sleep(Duration::from_millis(10));
        };
        let log = std::mem::replace(&mut self.log, std::thread::spawn(String::new))
            .join()
            .expect("log");
        (status, log)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn request(addr: SocketAddr, head: &str, body: &[u8]) -> (u16, String) {
    let mut s = TcpStream::connect(addr).expect("connect");
    s.set_read_timeout(Some(Duration::from_secs(30))).expect("timeout");
    s.write_all(head.as_bytes()).expect("send");
    s.write_all(body).expect("send");
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).expect("read");
    let text = String::from_utf8(raw).expect("UTF-8");
    let (head, body) = text.split_once("\r\n\r\n").expect("a header block");
    let status = head.split(' ').nth(1).and_then(|s| s.parse().ok()).expect("a status");
    (status, body.to_owned())
}

const CHAT: &str = r#"{"model":"inkling-2-mini","messages":[{"role":"user","content":"Hi"}],"max_tokens":64"#;

#[test]
fn version_flag_prints_the_build_line() {
    let out = Command::new(BIN).arg("--version").output().expect("runs");
    assert!(out.status.success());
    let text = String::from_utf8(out.stdout).expect("UTF-8");
    assert!(
        text.contains("rust version=") && text.contains(" commit=") && text.contains(" dirty="),
        "{text}"
    );
}

/// AGENTS hard rule 5: any non-loopback host is refused before anything is loaded or bound.
#[test]
fn a_non_loopback_host_is_refused() {
    for host in ["0.0.0.0", "::", "192.168.0.10", "example.com"] {
        let out = Command::new(BIN)
            .args(["--model-dir", "/nonexistent", "--port", "0", "--host", host])
            .output()
            .expect("runs");
        assert!(!out.status.success(), "{host}");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains("loopback only"), "{host}: {err}");
        assert!(!err.contains("listening"), "{host}: {err}");
    }
}

#[test]
fn mock_backend_serves_the_openai_routes_and_the_identity() {
    let Some(dir) = model_dir() else { return };
    let server = Server::start(&dir, &["--mock-delay-ms", "1"]);
    assert!(server.addr.ip().is_loopback());
    assert_eq!(server.get("/health"), (200, String::new()));

    let (status, body) = server.get("/version");
    assert_eq!(status, 200);
    let v: Value = serde_json::from_str(&body).expect("JSON");
    let id = &v["identity"];
    assert_eq!(
        (v["backend"].as_str(), id["engine"].as_str(), id["config"].as_str()),
        (Some("mock"), Some("rust"), Some("M1"))
    );
    let commit = id["commit"].as_str().expect("commit");
    assert_eq!(commit.len(), 40);
    assert!(commit.bytes().all(|b| b.is_ascii_hexdigit()));
    assert!(id["dirty"].is_boolean() && id["version"].is_string());
    assert_eq!(id["components"]["frontend"]["commit"], id["commit"]);
    assert!(
        v["version"]
            .as_str()
            .expect("line")
            .contains(&format!("commit={commit}")),
        "{v}"
    );

    let (status, body) = server.get("/v1/models");
    assert_eq!(status, 200);
    assert!(body.contains("inkling-2-mini") && body.contains("\"max_model_len\":262144"));

    let (status, body) = server.post("/v1/chat/completions", &format!("{CHAT},\"stream\":true}}"));
    assert_eq!(status, 200);
    assert!(body.ends_with("data: [DONE]\n\n"), "{body}");
    let text: String = body
        .split("\n\n")
        .filter_map(|f| f.strip_prefix("data: "))
        .filter_map(|d| serde_json::from_str::<Value>(d).ok())
        .filter_map(|j| j["choices"][0]["delta"]["content"].as_str().map(str::to_owned))
        .collect();
    assert!(
        text.starts_with("Prompt had ") && text.ends_with(" tokens."),
        "{text:?}"
    );

    let (status, body) = server.post("/v1/chat/completions", &format!("{CHAT}}}"));
    assert_eq!(status, 200);
    let j: Value = serde_json::from_str(&body).expect("JSON");
    assert_eq!(
        j["choices"][0]["message"]["reasoning"],
        "Mock back end: no model attached."
    );

    let (status, metrics) = server.get("/metrics");
    assert_eq!(status, 200);
    assert!(
        metrics.contains("ink:requests_total{model_name=\"inkling-2-mini\"} 2"),
        "{metrics}"
    );
}

/// `--backend fake`: the scheduler over `FakeRuntime`. The hash tokens are not text, so the request only has to
/// run to the length limit through scheduler, runtime and front end, with `/version` naming the scheduler.
#[test]
fn fake_backend_runs_requests_through_the_scheduler() {
    let Some(dir) = model_dir() else { return };
    let server = Server::start(&dir, &["--backend", "fake"]);
    let (_, body) = server.get("/version");
    let v: Value = serde_json::from_str(&body).expect("JSON");
    assert_eq!(v["backend"], "fake");
    assert!(v["identity"]["components"]["scheduler"].is_object(), "{v}");
    for stream in [true, false] {
        let extra = if stream { ",\"stream\":true" } else { "" };
        let (status, body) = server.post("/v1/chat/completions", &format!("{CHAT},\"min_tokens\":0{extra}}}"));
        assert_eq!(status, 200, "{body}");
        if stream {
            assert!(body.ends_with("data: [DONE]\n\n"), "{body}");
            assert!(!body.contains("InternalServerError"), "{body}");
        }
        assert!(body.contains("\"finish_reason\":\"length\""), "{body}");
    }
    let (status, _) = server.post("/v1/completions", r#"{"prompt":"Hello","max_tokens":5}"#);
    assert_eq!(status, 200);
    // the runtime's vocabulary bounds prompt token ids (200,058 for Inkling-2-Mini)
    let (status, body) = server.post("/v1/completions", r#"{"prompt":[1,200058],"max_tokens":5}"#);
    assert_eq!(status, 400, "{body}");
    assert!(body.contains("Token id 200058 is out of vocabulary"), "{body}");
    let (status, _) = server.post("/v1/completions", r#"{"prompt":[1,200057],"max_tokens":2}"#);
    assert_eq!(status, 200);
    let (status, body) = server.get("/health");
    assert_eq!((status, body.as_str()), (200, ""));
}

#[test]
fn sigterm_stops_the_server_cleanly() {
    let Some(dir) = model_dir() else { return };
    let server = Server::start(&dir, &["--backend", "fake"]);
    assert_eq!(server.get("/health").0, 200);
    let (status, log) = server.terminate();
    assert!(status.success(), "{status:?}\n{log}");
    assert!(log.contains("terminating"), "{log}");
}

/// `--backend engine` needs the pin file and refuses a missing pin file.
#[test]
fn engine_backend_needs_gemm_algos() {
    let Some(dir) = model_dir() else { return };
    let run = |args: &[&str]| {
        let out = Command::new(BIN)
            .arg("--model-dir")
            .arg(&dir)
            .args(["--port", "0", "--backend", "engine"])
            .args(args)
            .output()
            .expect("runs");
        assert!(!out.status.success(), "{args:?}");
        String::from_utf8_lossy(&out.stderr).into_owned()
    };
    assert!(run(&[]).contains("needs --gemm-algos"));
    let err = run(&["--gemm-algos", "/nonexistent/pins.json"]);
    assert!(
        err.contains("runtime construction failed") || err.contains("pin file"),
        "{err}"
    );
}
