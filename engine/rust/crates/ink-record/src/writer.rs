use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

use crate::{ComponentName, Identity, IdentityError};

/// Schema name carried by every record.
pub const SCHEMA: &str = "ink-record/1";

/// What the writer does with an identity that is dirty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirtyPolicy {
    /// Refuse: a record of a dirty tree must not exist.
    Refuse,
    /// Write `<name>.dirty.jsonl`; such a record is for debugging and is never quoted.
    Allow,
}

/// The card a record was measured on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gpu {
    /// Index on the host.
    pub index: u32,
    /// Card UUID.
    pub uuid: String,
    /// vBIOS type: `6D` or `67`.
    pub vbios: String,
}

/// How a record was produced (AGENTS.md hard rule 7), beside the identity.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Context {
    /// Command line.
    pub argv: Vec<String>,
    /// Configuration of the run.
    pub config: serde_json::Value,
    /// Card, when the run used one.
    pub gpu: Option<Gpu>,
    /// Power cap in watts, when a card was used.
    pub power_cap_w: Option<u32>,
    /// What the numbers are compared with.
    pub baseline: String,
    /// Short name of the run.
    pub label: String,
}

/// A record that was not written.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RecordError {
    /// A component of the process is dirty and the policy is [`DirtyPolicy::Refuse`].
    #[error("refusing to write a record: {component:?} is built from a dirty tree")]
    Dirty {
        /// The first dirty component.
        component: ComponentName,
    },
    /// The identity is missing or malformed.
    #[error(transparent)]
    Identity(#[from] IdentityError),
    /// The record name is empty or not a plain file name.
    #[error("invalid record name {0:?}")]
    InvalidName(String),
    /// The file could not be created or written; an existing record is never overwritten.
    #[error("record file {path}: {source}")]
    Io {
        /// File that failed.
        path: PathBuf,
        /// Cause.
        source: std::io::Error,
    },
    /// The record could not be serialised.
    #[error("record json: {0}")]
    Json(#[from] serde_json::Error),
    /// The timestamp could not be formatted.
    #[error("record timestamp: {0}")]
    Time(#[from] time::error::Format),
}

#[derive(Serialize)]
struct Line<'a, T> {
    schema: &'static str,
    time: String,
    identity: &'a Identity,
    context: &'a Context,
    data: &'a T,
}

/// Appends records of one run to `<dir>/<name>.jsonl`, one JSON object per line.
#[derive(Debug)]
pub struct RecordWriter {
    path: PathBuf,
    file: File,
    identity: Identity,
    context: Context,
}

impl RecordWriter {
    /// Creates the record file; it must not exist yet.
    ///
    /// The file is `<name>.jsonl`, or `<name>.dirty.jsonl` when a component is dirty and `policy` allows it.
    ///
    /// # Errors
    ///
    /// [`RecordError::Identity`] when the identity is malformed, [`RecordError::Dirty`] when it is dirty and
    /// `policy` is [`DirtyPolicy::Refuse`], [`RecordError::InvalidName`] for a name with a path separator, and
    /// [`RecordError::Io`] when the file exists or cannot be created. No file is created on any error before
    /// the last.
    pub fn create(
        dir: &Path,
        name: &str,
        identity: Identity,
        context: Context,
        policy: DirtyPolicy,
    ) -> Result<Self, RecordError> {
        if name.is_empty() || name.starts_with('.') || name.contains(['/', '\\']) {
            return Err(RecordError::InvalidName(name.to_owned()));
        }
        identity.validate()?;
        let dirty = identity.dirty_components();
        let file_name = match (dirty.first(), policy) {
            (Some(&component), DirtyPolicy::Refuse) => return Err(RecordError::Dirty { component }),
            (Some(_), DirtyPolicy::Allow) => format!("{name}.dirty.jsonl"),
            (None, _) => format!("{name}.jsonl"),
        };
        let path = dir.join(file_name);
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|source| RecordError::Io {
                path: path.clone(),
                source,
            })?;
        Ok(Self {
            path,
            file,
            identity,
            context,
        })
    }

    /// File being written.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Writes one record carrying the identity, the context and `data`.
    ///
    /// # Errors
    ///
    /// [`RecordError::Json`] when `data` cannot be serialised, [`RecordError::Io`] when the write fails.
    pub fn append<T: Serialize>(&mut self, data: &T) -> Result<(), RecordError> {
        let line = Line {
            schema: SCHEMA,
            time: OffsetDateTime::now_utc().format(&Rfc3339)?,
            identity: &self.identity,
            context: &self.context,
            data,
        };
        let mut bytes = serde_json::to_vec(&line)?;
        bytes.push(b'\n');
        self.file
            .write_all(&bytes)
            .and_then(|()| self.file.flush())
            .map_err(|source| RecordError::Io {
                path: self.path.clone(),
                source,
            })
    }
}
