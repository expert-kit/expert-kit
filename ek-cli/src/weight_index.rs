use std::{collections::BTreeSet, path::PathBuf, time::Instant};

use clap::Subcommand;
use ek_base::error::EKResult;
use ek_db::{
    expert_index::{ExpertEntry, ExpertIndex},
    safetensor::transformer::{TransformerModelDesc, TransformerPretrained},
};
use indicatif::{ProgressBar, ProgressStyle};
use tokio::fs;

#[derive(Subcommand, Debug)]
pub enum WeightIndexCommand {
    /// Pre-extract all expert blobs and write ek-expert-index.json.
    ///
    /// The command is idempotent: experts already present in cache_dir are skipped.
    /// The index is checkpointed after every layer so a crashed run can be resumed.
    Build {
        /// Model checkpoint directory (must contain config.json and
        /// model.safetensors.index.json).
        #[arg(long)]
        model: PathBuf,

        /// OpenDAL Fs cache root — the `weight.cache.Fs.path` value from your
        /// EK config. Pre-extracted blobs are written to `{cache_dir}/{model_name}/`.
        #[arg(long)]
        cache_dir: PathBuf,

        /// Only index MoE layers >= this value (inclusive). Defaults to first MoE layer.
        #[arg(long)]
        layer_start: Option<usize>,

        /// Only index MoE layers < this value (exclusive). Defaults to last MoE layer.
        #[arg(long)]
        layer_end: Option<usize>,
    },

    /// Benchmark fast path (pre-extracted cache) vs slow path (mmap+serialize).
    ///
    /// Fetches `--samples` experts from each path and reports mean/p50/p99 latency.
    Bench {
        /// Model checkpoint directory.
        #[arg(long)]
        model: PathBuf,

        /// Cache directory containing pre-extracted blobs and ek-expert-index.json.
        #[arg(long)]
        cache_dir: PathBuf,

        /// Number of experts to sample per path (default: 64).
        #[arg(long, default_value_t = 64)]
        samples: usize,

        /// Random seed for reproducible expert selection (default: 42).
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
}

pub async fn execute_weight_index(cmd: WeightIndexCommand) -> EKResult<()> {
    match cmd {
        WeightIndexCommand::Build {
            model,
            cache_dir,
            layer_start,
            layer_end,
        } => build_index(model, cache_dir, layer_start, layer_end).await,
        WeightIndexCommand::Bench {
            model,
            cache_dir,
            samples,
            seed,
        } => bench(model, cache_dir, samples, seed).await,
    }
}

async fn build_index(
    model_root: PathBuf,
    cache_dir: PathBuf,
    layer_start: Option<usize>,
    layer_end: Option<usize>,
) -> EKResult<()> {
    // --- Load model metadata ---
    let desc = TransformerModelDesc {
        root: model_root.clone(),
        ..Default::default()
    };
    let pretrained = TransformerPretrained::try_from_desc(&desc)?;
    let model_name = model_root
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_owned();
    let vital = pretrained.config().normalized_vital()?;

    let (moe_start, moe_end) = vital.moe_layers;
    let layer_from = layer_start.unwrap_or(moe_start);
    let layer_to = layer_end.unwrap_or(moe_end);
    let num_experts = vital.routed_experts;
    let total = (layer_to - layer_from) * num_experts;

    log::info!(
        "indexing model={} layers=[{},{}) experts_per_layer={} total={}",
        model_name,
        layer_from,
        layer_to,
        num_experts,
        total
    );

    // --- Ensure per-model cache sub-directory exists ---
    let model_cache_dir = cache_dir.join(&model_name);
    fs::create_dir_all(&model_cache_dir).await?;

    // --- Load existing index or start fresh (for idempotent re-runs) ---
    let mut index = ExpertIndex::load(&model_root)?
        .unwrap_or_else(|| ExpertIndex::new(model_name.clone()));

    // --- Progress bar ---
    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap(),
    );

    // --- Main extraction loop ---
    for layer in layer_from..layer_to {
        for expert in 0..num_experts {
            let expert_key = format!("{}/l{}-e{}", model_name, layer, expert);
            let blob_path = model_cache_dir.join(format!("l{}-e{}", layer, expert));

            pb.set_message(format!("layer={layer} expert={expert}"));

            // Skip if already extracted and recorded (idempotent).
            if blob_path.exists() && index.entries.contains_key(&expert_key) {
                pb.inc(1);
                continue;
            }

            // Extract serialized SafeTensors bytes for this expert.
            let bytes = pretrained.get_expert(layer, expert).await?;
            let size_bytes = bytes.len() as u64;

            // Discover tensor names from the blob.
            let st = safetensors::SafeTensors::deserialize(&bytes)?;
            let tensor_names: Vec<String> =
                st.names().iter().map(|n| n.to_string()).collect();

            // Discover source shard files from weight_map.
            let shard_files =
                derive_shard_files(&desc, &tensor_names);

            // Write blob to cache.
            fs::write(&blob_path, &bytes).await?;

            index.upsert(
                expert_key,
                ExpertEntry {
                    size_bytes,
                    shard_files,
                    tensor_names,
                    cached: true,
                },
            );

            pb.inc(1);
        }

        // Checkpoint the index after each layer so a crash can be resumed.
        // Write to the model_cache_dir (not model_root which may be read-only).
        index.save(&model_cache_dir)?;
    }

    pb.finish_with_message("done");
    log::info!(
        "index written to {}",
        ExpertIndex::index_path(&model_cache_dir).display()
    );
    Ok(())
}

async fn bench(
    model_root: PathBuf,
    cache_dir: PathBuf,
    samples: usize,
    seed: u64,
) -> EKResult<()> {
    let desc = TransformerModelDesc {
        root: model_root.clone(),
        ..Default::default()
    };
    let pretrained = TransformerPretrained::try_from_desc(&desc)?;
    let vital = pretrained.config().normalized_vital()?;

    let model_name = model_root
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_owned();
    let model_cache_dir = cache_dir.join(&model_name);

    let index = ExpertIndex::load(&model_cache_dir)?
        .ok_or_else(|| ek_base::error::EKError::NotFound(
            format!("no ek-expert-index.json in {}; run `weight build` first", model_cache_dir.display())
        ))?;

    // Select `samples` experts using a simple LCG so selection is deterministic.
    let (moe_start, moe_end) = vital.moe_layers;
    let num_experts = vital.routed_experts;
    let total = (moe_end - moe_start) * num_experts;
    let sample_count = samples.min(total);

    let mut lcg = seed;
    let mut experts: Vec<(usize, usize)> = Vec::with_capacity(sample_count);
    let mut seen = std::collections::HashSet::new();
    while experts.len() < sample_count {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let idx = (lcg >> 33) as usize % total;
        if seen.insert(idx) {
            let layer = moe_start + idx / num_experts;
            let expert = idx % num_experts;
            experts.push((layer, expert));
        }
    }

    println!("Benchmarking {} experts — model={} samples={}", total, model_name, sample_count);
    println!("{:<12} {:>10} {:>10} {:>10} {:>12}", "path", "mean(ms)", "p50(ms)", "p99(ms)", "total(ms)");
    println!("{}", "-".repeat(58));

    // --- Fast path: read pre-extracted blob from cache dir ---
    let mut fast_times: Vec<f64> = Vec::with_capacity(sample_count);
    for &(layer, expert) in &experts {
        let blob_path = model_cache_dir.join(format!("l{}-e{}", layer, expert));
        let t = Instant::now();
        let _ = fs::read(&blob_path).await?;
        fast_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    print_stats("fast (index)", &mut fast_times);

    // --- Slow path: mmap shard + safetensors::serialize ---
    let mut slow_times: Vec<f64> = Vec::with_capacity(sample_count);
    for &(layer, expert) in &experts {
        let key = format!("{}/l{}-e{}", model_name, layer, expert);
        // Verify this expert is in the index so we compare the same set.
        if !index.entries.contains_key(&key) {
            continue;
        }
        let t = Instant::now();
        let _ = pretrained.get_expert(layer, expert).await?;
        slow_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    print_stats("slow (mmap)", &mut slow_times);

    Ok(())
}

fn print_stats(label: &str, times: &mut [f64]) {
    if times.is_empty() {
        println!("{:<12} {:>10}", label, "n/a");
        return;
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = times.len();
    let mean = times.iter().sum::<f64>() / n as f64;
    let p50 = times[n / 2];
    let p99 = times[(n as f64 * 0.99) as usize].min(*times.last().unwrap());
    let total: f64 = times.iter().sum();
    println!(
        "{:<12} {:>10.2} {:>10.2} {:>10.2} {:>12.1}",
        label, mean, p50, p99, total
    );
}

/// Look up the shard file(s) for a set of tensor names by reading
/// `model.safetensors.index.json` directly.
fn derive_shard_files(desc: &TransformerModelDesc, tensor_names: &[String]) -> Vec<String> {
    let index_path = desc.root.join(&desc.weight_map_name);
    let Ok(raw) = std::fs::read_to_string(&index_path) else {
        return vec![];
    };
    let Ok(val) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return vec![];
    };
    let Some(weight_map) = val.get("weight_map").and_then(|v| v.as_object()) else {
        return vec![];
    };
    let mut shards = BTreeSet::new();
    for name in tensor_names {
        if let Some(shard) = weight_map.get(name).and_then(|v| v.as_str()) {
            shards.insert(shard.to_owned());
        }
    }
    shards.into_iter().collect()
}
