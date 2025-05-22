use std::{collections::HashMap, path::PathBuf, time};

use once_cell::sync::OnceCell;

pub struct PerfTimer {
    start: time::Instant,
    name: String,
    point: HashMap<String, time::Instant>,
}

impl PerfTimer {
    pub fn new(name: &str) -> Self {
        let start = time::Instant::now();
        PerfTimer {
            start,
            name: name.into(),
            point: HashMap::new(),
        }
    }

    pub fn stop(&mut self, name: &str) {
        let now = time::Instant::now();
        self.point.insert(name.to_string(), now);
    }
}

impl Drop for PerfTimer {
    fn drop(&mut self) {
        let now = time::Instant::now();
        let elapsed = now.duration_since(self.start);
        log::debug!(
            "PerfTimer({}): total elapsed_ms={}",
            self.name,
            elapsed.as_millis()
        );
        for (name, point) in &self.point {
            let elapsed = point.duration_since(self.start);
            log::debug!(
                "PerfTimer({}/{}) elapsed_ms={}",
                self.name,
                name,
                elapsed.as_millis()
            );
        }
    }
}

pub fn workspace_root() -> PathBuf {
    let root = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(root.to_owned())
        .parent()
        .unwrap()
        .to_path_buf()
}

pub struct Defers {
    cb: Box<dyn Fn()>,
}

impl Defers {
    pub fn defer(f: Box<dyn Fn()>) -> Self {
        // self.stack.set(f);
        Self { cb: f }
    }
}

impl Drop for Defers {
    fn drop(&mut self) {
        let v = self.cb.as_mut();
        v()
    }
}


unsafe impl Send for Defers {}
unsafe impl Sync for Defers {}