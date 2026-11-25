use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Which implementation produced the numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Engine {
    /// The C++ engine.
    Native,
    /// The Rust engine.
    Rust,
}

/// A part of the engine that is in the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComponentName {
    /// HTTP front end.
    Frontend,
    /// Scheduler.
    Scheduler,
    /// Tools and harnesses.
    Tools,
    /// Forward runtime; it produces the numbers, so it names the engine.
    Runtime,
    /// Drafter.
    Drafter,
    /// Kernel pack.
    Kpack,
}

/// Implementation, version and source tree of one component.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Source {
    /// Implementation language.
    #[serde(rename = "impl")]
    pub implementation: Engine,
    /// Semantic version.
    pub version: String,
    /// Git commit, 40 lowercase hex digits.
    pub commit: String,
    /// The tree under `engine/` differed from `commit`.
    pub dirty: bool,
    /// sha256 of the difference; present exactly when `dirty`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diff: Option<String>,
    /// C ABI version, for a component reached through `ink_rt.h`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abi: Option<u32>,
}

impl Source {
    /// The identity of this Rust build, from `BUILD_ID.json` at build time.
    #[must_use]
    pub fn this_build() -> Self {
        let dirty = env!("INK_BUILD_DIRTY") == "1";
        Self {
            implementation: Engine::Rust,
            version: env!("CARGO_PKG_VERSION").to_owned(),
            commit: env!("INK_BUILD_COMMIT").to_owned(),
            dirty,
            diff: dirty.then(|| env!("INK_BUILD_DIFF").to_owned()),
            abi: None,
        }
    }

    fn validate(&self) -> Result<(), String> {
        if !is_hex(&self.commit, 40) {
            return Err(format!("commit must be 40 lowercase hex digits, got {:?}", self.commit));
        }
        match (&self.diff, self.dirty) {
            (None, false) => Ok(()),
            (Some(d), true) if is_hex(d, 64) => Ok(()),
            _ => Err("diff must be a sha256 hex string when dirty and absent when clean".to_owned()),
        }
    }
}

/// One entry of [`Identity::components`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Component {
    /// A component with a source tree.
    Source(Source),
    /// The kernel pack, identified by its content hash (sha256, 64 hex digits).
    Kpack {
        /// Pack id.
        id: String,
    },
}

/// What produced a benchmark record: the runtime's identity at the top, every component of the process below.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Identity {
    /// Implementation of the runtime.
    pub engine: Engine,
    /// Version of the runtime.
    pub version: String,
    /// Commit of the runtime.
    pub commit: String,
    /// Whether the runtime's tree was dirty.
    pub dirty: bool,
    /// Diff hash of the runtime's tree, when dirty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diff: Option<String>,
    /// Mixed configuration name, `M0` to `M5`.
    pub config: String,
    /// sha256 of the `Cargo.lock` this build used, 64 lowercase hex digits (`BUILD_ID.json` `lock`).
    pub lock_sha256: String,
    /// Tree hash of `vendor.sha256` this build used (vendor tree, `Cargo.lock` and `.cargo/config.toml`), 64 hex
    /// digits (`BUILD_ID.json` `vendor`).
    pub vendor_sha256: String,
    /// Components in the process; always includes [`ComponentName::Runtime`].
    pub components: BTreeMap<ComponentName, Component>,
}

/// An identity that cannot go into a record.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum IdentityError {
    /// A field is missing, malformed or inconsistent.
    #[error("invalid identity of {component}: {reason}")]
    Invalid {
        /// Component name, or `engine` for the top-level fields.
        component: String,
        /// What is wrong.
        reason: String,
    },
}

impl Identity {
    /// Builds an identity from the runtime's source, the mixed configuration name and the other components.
    ///
    /// The runtime is inserted as the `runtime` component; an entry of that name in `others` is replaced.
    #[must_use]
    pub fn new(
        runtime: Source,
        config: impl Into<String>,
        others: impl IntoIterator<Item = (ComponentName, Component)>,
    ) -> Self {
        let mut components: BTreeMap<_, _> = others.into_iter().collect();
        components.insert(ComponentName::Runtime, Component::Source(runtime.clone()));
        Self {
            engine: runtime.implementation,
            version: runtime.version,
            commit: runtime.commit,
            dirty: runtime.dirty,
            diff: runtime.diff,
            config: config.into(),
            lock_sha256: env!("INK_BUILD_LOCK").to_owned(),
            vendor_sha256: env!("INK_BUILD_VENDOR").to_owned(),
            components,
        }
    }

    /// Names of the components whose tree is dirty, in component order.
    #[must_use]
    pub fn dirty_components(&self) -> Vec<ComponentName> {
        self.components
            .iter()
            .filter(|(_, c)| matches!(c, Component::Source(s) if s.dirty))
            .map(|(name, _)| *name)
            .collect()
    }

    /// Checks that every field is well formed and the top-level fields equal the `runtime` component.
    ///
    /// # Errors
    ///
    /// Returns [`IdentityError::Invalid`] naming the first component or field that is wrong.
    pub fn validate(&self) -> Result<(), IdentityError> {
        let top = Source {
            implementation: self.engine,
            version: self.version.clone(),
            commit: self.commit.clone(),
            dirty: self.dirty,
            diff: self.diff.clone(),
            abi: None,
        };
        invalid("engine", top.validate())?;
        if self.config.is_empty() {
            return invalid("config", Err("empty".to_owned()));
        }
        for (field, value) in [
            ("lock_sha256", &self.lock_sha256),
            ("vendor_sha256", &self.vendor_sha256),
        ] {
            if !is_hex(value, 64) {
                return invalid(
                    "engine",
                    Err(format!("{field} must be 64 lowercase hex digits, got {value:?}")),
                );
            }
        }
        match self.components.get(&ComponentName::Runtime) {
            Some(Component::Source(r))
                if r.implementation == top.implementation
                    && r.version == top.version
                    && r.commit == top.commit
                    && r.dirty == top.dirty
                    && r.diff == top.diff => {}
            _ => {
                return invalid(
                    "runtime",
                    Err("missing or different from the top-level identity".to_owned()),
                );
            }
        }
        for (name, component) in &self.components {
            let result = match component {
                Component::Source(s) => s.validate(),
                Component::Kpack { id } if is_hex(id, 64) => Ok(()),
                Component::Kpack { id } => Err(format!("kpack id must be 64 lowercase hex digits, got {id:?}")),
            };
            invalid(&format!("{name:?}").to_lowercase(), result)?;
        }
        Ok(())
    }
}

fn invalid(component: &str, result: Result<(), String>) -> Result<(), IdentityError> {
    result.map_err(|reason| IdentityError::Invalid {
        component: component.to_owned(),
        reason,
    })
}

fn is_hex(s: &str, len: usize) -> bool {
    s.len() == len && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}
