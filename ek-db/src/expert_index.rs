use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ek_base::error::EKResult;
use serde::{Deserialize, Serialize};

/// Per-expert entry in the index manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertEntry {
    /// Byte length of the pre-serialized SafeTensors blob in the cache dir.
    pub size_bytes: u64,
    /// Original shard files that contain this expert's tensors.
    /// Informational — reserved for future HTTP range-request support.
    pub shard_files: Vec<String>,
    /// Tensor names present in the blob (e.g. "...down_proj.weight").
    pub tensor_names: Vec<String>,
    /// Whether the pre-serialized blob is present in the OpenDAL cache dir.
    pub cached: bool,
}

/// Top-level expert index manifest.
/// Written to `{model_root}/ek-expert-index.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertIndex {
    pub version: u32,
    pub model_name: String,
    /// Key format matches `ExpertKey::as_object_key()`: `"{model}/l{layer}-e{expert}"`.
    pub entries: HashMap<String, ExpertEntry>,
}

impl ExpertIndex {
    pub const VERSION: u32 = 1;
    pub const FILE_NAME: &'static str = "ek-expert-index.json";

    pub fn new(model_name: String) -> Self {
        Self {
            version: Self::VERSION,
            model_name,
            entries: HashMap::new(),
        }
    }

    pub fn upsert(&mut self, key: String, entry: ExpertEntry) {
        self.entries.insert(key, entry);
    }

    /// Absolute path of the index file for the given model root.
    pub fn index_path(model_root: &Path) -> PathBuf {
        model_root.join(Self::FILE_NAME)
    }

    /// Write the manifest to `{model_root}/ek-expert-index.json`.
    pub fn save(&self, model_root: &Path) -> EKResult<()> {
        let path = Self::index_path(model_root);
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json)?;
        Ok(())
    }

    /// Load the manifest from `{model_root}/ek-expert-index.json`.
    /// Returns `None` (not an error) when the file does not exist.
    pub fn load(model_root: &Path) -> EKResult<Option<Self>> {
        let path = Self::index_path(model_root);
        if !path.exists() {
            return Ok(None);
        }
        let raw = std::fs::read_to_string(&path)?;
        let index: ExpertIndex = serde_json::from_str(&raw)?;
        Ok(Some(index))
    }
}
