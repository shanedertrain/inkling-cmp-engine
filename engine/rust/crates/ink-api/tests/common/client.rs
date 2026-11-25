//! A minimal HTTP/1.x client over a raw socket (the socket-level tests of gate G-FE4 want the bytes on the wire).

use std::net::SocketAddr;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// A complete response.
#[derive(Debug)]
pub struct Reply {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    /// The body with any chunked framing removed.
    pub body: Vec<u8>,
    /// The bytes after the header block exactly as they came.
    pub raw_body: Vec<u8>,
}

impl Reply {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    pub fn text(&self) -> String {
        String::from_utf8(self.body.clone()).expect("UTF-8 body")
    }
}

/// What to send.
pub struct Req<'a> {
    pub method: &'a str,
    pub path: &'a str,
    pub headers: Vec<(&'a str, String)>,
    pub body: &'a [u8],
    /// `Content-Length` to declare instead of the body's length (the body is then not sent).
    pub declared_length: Option<usize>,
    /// 0 for HTTP/1.0 (the body is delimited by the close), 1 for HTTP/1.1 (chunked).
    pub minor: u8,
}

impl<'a> Req<'a> {
    pub fn get(path: &'a str) -> Self {
        Self {
            method: "GET",
            path,
            headers: Vec::new(),
            body: b"",
            declared_length: None,
            minor: 1,
        }
    }

    pub fn post(path: &'a str, body: &'a [u8]) -> Self {
        Self {
            method: "POST",
            path,
            headers: vec![("content-type", "application/json".to_owned())],
            body,
            declared_length: None,
            minor: 1,
        }
    }

    pub fn header(mut self, name: &'a str, value: impl Into<String>) -> Self {
        self.headers.push((name, value.into()));
        self
    }

    pub fn http10(mut self) -> Self {
        self.minor = 0;
        self
    }

    fn head(&self, addr: SocketAddr) -> Vec<u8> {
        let mut head = format!(
            "{} {} HTTP/1.{}\r\nhost: {addr}\r\n",
            self.method, self.path, self.minor
        );
        if self.minor == 1 {
            head.push_str("connection: close\r\n");
        }
        for (k, v) in &self.headers {
            head.push_str(&format!("{k}: {v}\r\n"));
        }
        if self.method == "POST" {
            head.push_str(&format!(
                "content-length: {}\r\n",
                self.declared_length.unwrap_or(self.body.len())
            ));
        }
        head.push_str("\r\n");
        head.into_bytes()
    }
}

/// Opens a connection and sends the request (the body too, unless a length was declared).
pub async fn send(addr: SocketAddr, req: &Req<'_>) -> TcpStream {
    let mut s = TcpStream::connect(addr).await.expect("connect");
    s.write_all(&req.head(addr)).await.expect("send head");
    if req.declared_length.is_none() {
        s.write_all(req.body).await.expect("send body");
    }
    s
}

/// Sends a request and reads the whole response until the server closes the connection.
pub async fn fetch(addr: SocketAddr, req: Req<'_>) -> Reply {
    let mut s = send(addr, &req).await;
    let mut buf = Vec::new();
    tokio::time::timeout(Duration::from_secs(30), s.read_to_end(&mut buf))
        .await
        .expect("the server answers in time")
        .expect("read");
    parse(&buf)
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

pub fn parse(buf: &[u8]) -> Reply {
    let split = find(buf, b"\r\n\r\n").expect("a header block");
    let head = std::str::from_utf8(&buf[..split]).expect("ASCII head");
    let mut lines = head.split("\r\n");
    let status = lines
        .next()
        .and_then(|l| l.split(' ').nth(1))
        .and_then(|s| s.parse().ok())
        .expect("a status line");
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
        .collect();
    let raw_body = buf[split + 4..].to_vec();
    let chunked = headers
        .iter()
        .any(|(k, v)| k.eq_ignore_ascii_case("transfer-encoding") && v.eq_ignore_ascii_case("chunked"));
    let body = if chunked { dechunk(&raw_body) } else { raw_body.clone() };
    Reply {
        status,
        headers,
        body,
        raw_body,
    }
}

/// Strict de-chunking (bare hex sizes, CRLF after every chunk, nothing after the last one), so that it cannot hide a
/// framing error.
fn dechunk(mut raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let eol = find(raw, b"\r\n").expect("a chunk size line");
        let line = std::str::from_utf8(&raw[..eol]).expect("ASCII size line");
        assert!(
            !line.is_empty() && line.bytes().all(|b| b.is_ascii_hexdigit()),
            "chunk size line {line:?}"
        );
        let size = usize::from_str_radix(line, 16).expect("hex size");
        raw = &raw[eol + 2..];
        if size == 0 {
            assert_eq!(raw, b"\r\n", "only the final CRLF follows the last chunk");
            return out;
        }
        assert_eq!(&raw[size..size + 2], b"\r\n", "a chunk ends in CRLF");
        out.extend_from_slice(&raw[..size]);
        raw = &raw[size + 2..];
    }
}

/// Reads from `s` until `needle` occurred `n` times or `limit` passed; returns what was read.
pub async fn read_until_count(s: &mut TcpStream, needle: &str, n: usize, limit: Duration) -> Vec<u8> {
    let mut buf = Vec::new();
    let deadline = tokio::time::Instant::now() + limit;
    let mut chunk = [0u8; 4096];
    while String::from_utf8_lossy(&buf).matches(needle).count() < n {
        let left = deadline.saturating_duration_since(tokio::time::Instant::now());
        match tokio::time::timeout(left, s.read(&mut chunk)).await {
            Ok(Ok(0)) | Err(_) => break,
            Ok(Ok(k)) => buf.extend_from_slice(&chunk[..k]),
            Ok(Err(e)) => panic!("read failed: {e}"),
        }
    }
    buf
}
