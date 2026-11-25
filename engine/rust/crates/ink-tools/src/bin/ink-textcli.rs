// Ported from engine/server/tools/ink_textcli.cpp @ 4171493.
//! Text-side command line for parity checks: render chat requests to prompt ids, encode text.
//!
//! * `ink-textcli render MODEL_DIR REQUESTS.json OUT.jsonl` reads `{"requests":[{"id","endpoint","body"},..]}` (`-`
//!   reads standard input, so a `.zst` recording is passed through `zstd -dc`) and writes `{"id","ids"}` or
//!   `{"id","error"}` per chat request.
//! * `ink-textcli encode MODEL_DIR TEXT_FILE` prints the ids of the file's text, special tokens recognised, as one
//!   JSON line.
#![forbid(unsafe_code)]
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use ink_text::{ChatRenderer, Specials, Tokenizer, resolve_chat_params};
use serde_json::{Value, json};

#[derive(Parser)]
#[command(name = "ink-textcli", version = ink_record::BUILD_LINE, about = "Render and tokenise tool")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Render the chat requests of a request set to prompt token ids.
    Render {
        /// Directory with the model's `tokenizer.json`.
        model_dir: PathBuf,
        /// Request set (`{"requests": [...]}`); `-` reads standard input.
        requests: PathBuf,
        /// Output, one JSON line per chat request.
        out: PathBuf,
    },
    /// Print the token ids of a text file (special tokens recognised).
    Encode {
        /// Directory with the model's `tokenizer.json`.
        model_dir: PathBuf,
        /// Text file.
        text_file: PathBuf,
    },
}

fn read_input(path: &Path) -> Result<Vec<u8>> {
    if path == Path::new("-") {
        let mut bytes = Vec::new();
        io::stdin().read_to_end(&mut bytes).context("reading standard input")?;
        Ok(bytes)
    } else {
        fs::read(path).with_context(|| format!("cannot open {}", path.display()))
    }
}

fn render(model_dir: &Path, requests: &Path, out: &Path) -> Result<()> {
    let tok = Tokenizer::load(model_dir)?;
    let renderer = ChatRenderer::new(&tok)?;
    let all: Value = serde_json::from_slice(&read_input(requests)?).context("parsing the request set")?;
    let mut out = BufWriter::new(File::create(out).with_context(|| format!("cannot create {}", out.display()))?);
    let mut rendered = 0usize;
    for req in all["requests"]
        .as_array()
        .context("request set has no `requests` list")?
    {
        if req["endpoint"] != "chat" {
            continue;
        }
        let ids = resolve_chat_params(&req["body"]).and_then(|p| renderer.render(&p));
        let row = match ids {
            Ok(ids) => json!({"id": req["id"], "ids": ids}),
            Err(e) => json!({"id": req["id"], "error": e.message()}),
        };
        writeln!(out, "{row}")?;
        rendered += 1;
    }
    out.flush()?;
    eprintln!("{rendered} rendered");
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Render {
            model_dir,
            requests,
            out,
        } => render(&model_dir, &requests, &out),
        Command::Encode { model_dir, text_file } => {
            let tok = Tokenizer::load(model_dir)?;
            let text = String::from_utf8(read_input(&text_file)?).context("text file is not UTF-8")?;
            println!("{}", json!(tok.encode(&text, Specials::Recognize)?));
            Ok(())
        }
    }
}
