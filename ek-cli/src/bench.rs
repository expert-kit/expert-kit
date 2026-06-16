use std::time::Instant;

use clap::Subcommand;
use ek_base::error::EKResult;
use ek_db::{safetensor::ExpertKey, weight_manager::LocalWeightManager};
use tokio::task::JoinSet;

#[derive(Subcommand, Debug)]
pub enum BenchCommand {
    /// Benchmark expert weight loading: parallel vs sequential, with per-tier stats.
    WeightLoad {
        /// Model name (e.g., "qwen3", "ds-tiny").
        #[arg(long)]
        model: String,

        /// Number of experts to load per layer.
        #[arg(long, default_value_t = 64)]
        experts: usize,

        /// Starting layer index.
        #[arg(long, default_value_t = 0)]
        layer: usize,

        /// Also run a sequential pass (after evicting the mem cache) for comparison.
        #[arg(long, default_value_t = false)]
        sequential: bool,
    },
}

pub async fn execute_bench(cmd: BenchCommand) -> EKResult<()> {
    match cmd {
        BenchCommand::WeightLoad {
            model,
            experts,
            layer,
            sequential,
        } => bench_weight_load(model, experts, layer, sequential).await,
    }
}

async fn bench_weight_load(
    model: String,
    experts: usize,
    layer: usize,
    run_sequential: bool,
) -> EKResult<()> {
    let wm = LocalWeightManager::new_shared();

    let keys: Vec<ExpertKey> = (0..experts)
        .map(|i| ExpertKey::new(model.clone(), layer, i))
        .collect();

    // --- Parallel pass ---
    log::info!("Starting parallel load of {experts} experts (model={model}, layer={layer})");
    let t0 = Instant::now();
    let mut js: JoinSet<EKResult<()>> = JoinSet::new();
    for key in &keys {
        let wm = wm.clone();
        let key = key.clone();
        js.spawn(async move {
            wm.get_expert(&key).await?;
            Ok(())
        });
    }
    while let Some(res) = js.join_next().await {
        res.map_err(|e| ek_base::error::EKError::InvalidInput(e.to_string()))??;
    }
    let parallel_ms = t0.elapsed().as_millis();

    // Print per-tier stats after parallel pass.
    print_stats(&wm, "parallel");

    // --- Sequential pass (optional) ---
    let seq_ms = if run_sequential {
        // Evict mem cache to force cold reads.
        wm.evict_all();
        log::info!("Starting sequential load of {experts} experts");
        let t1 = Instant::now();
        for key in &keys {
            wm.get_expert(key).await?;
        }
        let elapsed = t1.elapsed().as_millis();
        print_stats(&wm, "sequential");
        Some(elapsed)
    } else {
        None
    };

    // --- Summary ---
    println!("\n=== Benchmark Summary ===");
    println!("model={model}  layer={layer}  experts={experts}");
    println!("parallel_ms   : {parallel_ms}");
    if let Some(seq) = seq_ms {
        let speedup = seq as f64 / parallel_ms.max(1) as f64;
        println!("sequential_ms : {seq}");
        println!("speedup       : {speedup:.2}x");
    }

    Ok(())
}

fn print_stats(wm: &LocalWeightManager, label: &str) {
    let stats = wm.stats();
    println!("\n--- Per-tier stats [{label}] ---");
    println!(
        "{:<10} {:>8} {:>14}",
        "tier", "hits", "avg_latency_ms"
    );
    for (name, tier) in [
        ("mem", &stats.mem),
        ("disk", &stats.disk),
        ("peer", &stats.peer),
        ("central", &stats.central),
    ] {
        let (hits, total_ns) = tier.snapshot();
        let avg_ms = if hits > 0 {
            (total_ns as f64 / hits as f64) / 1_000_000.0
        } else {
            0.0
        };
        println!("{:<10} {:>8} {:>14.3}", name, hits, avg_ms);
    }
}
