use std::{
    collections::{HashMap, VecDeque},
    sync::{LazyLock, Mutex},
};

/// Sliding-window expert frequency tracker.
///
/// Maintains the last `window_size` poller ticks of per-expert request counts.
/// Each tick, worker heartbeats contribute counts via `add_counts()`; at the end
/// of the tick the `commit_tick()` call advances the window.
pub struct ExpertFrequencyTracker {
    window_size: usize,
    /// Counts accumulated from worker heartbeats for the *current* (not yet committed) tick
    pending: Mutex<HashMap<String, u64>>,
    /// Committed tick snapshots; index 0 = most recent, index window_size-1 = oldest
    window: Mutex<VecDeque<HashMap<String, u64>>>,
}

/// Global singleton — shared between the controller's service/state.rs (writes) and
/// the poller / dispatcher (reads).
pub static FREQ_TRACKER: LazyLock<ExpertFrequencyTracker> =
    LazyLock::new(|| ExpertFrequencyTracker::new(3));

pub fn get_freq_tracker() -> &'static ExpertFrequencyTracker {
    &FREQ_TRACKER
}

impl ExpertFrequencyTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            pending: Mutex::new(HashMap::new()),
            window: Mutex::new(VecDeque::new()),
        }
    }

    /// Merge per-expert counts from one worker's heartbeat into the pending bucket.
    pub fn add_counts(&self, counts: HashMap<String, u64>) {
        if counts.is_empty() {
            return;
        }
        let mut pending = self.pending.lock().unwrap();
        for (k, v) in counts {
            *pending.entry(k).or_default() += v;
        }
    }

    /// Close the current pending bucket and push it into the sliding window.
    /// Called once per poller tick (every 5 s), after all heartbeats for the tick have
    /// been processed.
    pub fn commit_tick(&self) {
        let snapshot = {
            let mut pending = self.pending.lock().unwrap();
            std::mem::take(&mut *pending)
        };
        let mut window = self.window.lock().unwrap();
        window.push_front(snapshot);
        if window.len() > self.window_size {
            window.pop_back();
        }
    }

    /// Returns `true` once at least one tick has been committed (i.e. we have real data).
    pub fn has_data(&self) -> bool {
        !self.window.lock().unwrap().is_empty()
    }

    /// Sum of requests for `expert_id` across all committed ticks in the window.
    pub fn rate_in_window(&self, expert_id: &str) -> u64 {
        self.window
            .lock()
            .unwrap()
            .iter()
            .map(|bucket| bucket.get(expert_id).copied().unwrap_or(0))
            .sum()
    }

    /// All experts that appeared in the window, sorted by descending total rate.
    pub fn all_sorted(&self) -> Vec<(String, u64)> {
        let window = self.window.lock().unwrap();
        let mut totals: HashMap<String, u64> = HashMap::new();
        for bucket in window.iter() {
            for (k, v) in bucket {
                *totals.entry(k.clone()).or_default() += v;
            }
        }
        drop(window);
        let mut sorted: Vec<(String, u64)> = totals.into_iter().filter(|(_, v)| *v > 0).collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted
    }

    /// Top `n` experts by rate in the window.
    pub fn top_n(&self, n: usize) -> Vec<(String, u64)> {
        let mut sorted = self.all_sorted();
        sorted.truncate(n);
        sorted
    }
}
