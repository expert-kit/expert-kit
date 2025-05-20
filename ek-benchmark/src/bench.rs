use std::time::{Instant, SystemTime, UNIX_EPOCH};

use polars::frame::DataFrame;
use polars::frame::row::Row;
use polars::prelude::*;

use ek_computation::ffn::{Expert, ExpertBackend, ExpertShape, expert_ort::NDArrayTensor};

#[allow(dead_code)]
pub struct Benchmarker {
    run_iterations: usize,
    experiment_repeats: usize,
    experts: Vec<ExpertBenchmark>,
}

#[allow(dead_code)]
pub struct ExpertBenchmark(pub ExpertBackend);

impl ExpertBenchmark {
    fn backend(&self) -> std::string::String {
        match &self.0 {
            ExpertBackend::Torch(exp) => exp.backend(),
            ExpertBackend::Onnx(onnx_exp) => onnx_exp.backend(),
        }
    }
    
    fn shape(&self) -> ExpertShape {
        match &self.0 {
            ExpertBackend::Torch(exp) => exp.shape(),
            ExpertBackend::Onnx(onnx_exp) => onnx_exp.shape(),
        }
    }
    
    fn forward(&self, batch_size: usize) -> Instant {
        match &self.0 {
            ExpertBackend::Torch(exp) => {
                let input = exp.rand_input(batch_size);
                let start = Instant::now();
                let _ = exp.forward(&input);
                start
            }
            ExpertBackend::Onnx(onnx_exp) => {
                let input: NDArrayTensor<f32> = onnx_exp.rand_input(batch);

                let start = Instant::now();
                let _ = onnx_exp.forward(&input);
                start
            }
        }
    }
}

#[allow(dead_code)]
impl Benchmarker {
    pub fn new(experts: Vec<ExpertBenchmark>) -> Self {
        Benchmarker {
            experts,
            run_iterations: 10,
            experiment_repeats: 1,
        }
    }

    pub fn iterations(&mut self, iterations: usize) -> &mut Benchmarker {
        self.run_iterations = iterations;
        self
    }
    
    pub fn repeats(&mut self, repeats: usize) -> &mut Benchmarker {
        self.experiment_repeats = repeats;
        self
    }

    pub fn scan_batch(&self, batch_sizes: &[usize]) -> DataFrame {
        let mut rows: Vec<Row> = Vec::new();
        let system_info = sysinfo::System::new_all();
        let cpu_brand = system_info.cpus()[0].brand();
        info!("Running batch scan on CPU: {}", cpu_brand);
        
        // Warm up all experts
        for expert in self.experts.iter() {
            expert.forward(1);
        }

        // Run the experiments multiple times if requested
        for experiment_id in 0..self.experiment_repeats {
            for iter_idx in 0..self.run_iterations {
                for current_batch in batch_sizes.iter() {
                    for (expert_idx, expert) in self.experts.iter().enumerate() {
                        let start = expert.forward(*current_batch);
                        let shape = expert.shape();
                        let backend = expert.backend();
                        let row = vec![
                            AnyValue::UInt64(experiment_id as u64),
                            AnyValue::UInt64(iter_idx as u64),
                            AnyValue::UInt64(*current_batch as u64),
                            AnyValue::UInt64(expert_idx as u64),
                            AnyValue::UInt64(start.elapsed().as_micros() as u64),
                            AnyValue::UInt64(shape.dim as u64),
                            AnyValue::UInt64(shape.hidden as u64),
                            AnyValue::StringOwned(backend.into()),
                        ];
                        rows.push(Row(row));
                    }
                }
            }
        }
        
        let schema = Schema::from_iter(vec![
            Field::new("experiment_id".into(), DataType::UInt64),
            Field::new("iteration".into(), DataType::UInt64),
            Field::new("batch_size".into(), DataType::UInt64),
            Field::new("expert_id".into(), DataType::UInt64),
            Field::new("elapsed_micros".into(), DataType::UInt64),
            Field::new("input_dim".into(), DataType::UInt64),
            Field::new("hidden_dim".into(), DataType::UInt64),
            Field::new("backend".into(), DataType::String),
        ]);

        let df = DataFrame::from_rows_and_schema(rows.as_slice(), &schema).unwrap();

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let hardware_col = Series::new("hardware".into(), vec![cpu_brand; df.height()]);
        let timestamp_col = Series::new("timestamp".into(), vec![timestamp; df.height()]);

        df.hstack(&[hardware_col.into(), timestamp_col.into()]).unwrap()
    }
    
    pub fn calculate_summary(&self, df: &DataFrame) -> DataFrame {
        // Group by batch_size and experiment_id, then calculate average elapsed time
        df.clone().lazy()
            .group_by([col("batch_size"), col("experiment_id")])
            .agg([col("elapsed_micros").mean().alias("avg_elapsed_micros")])
            .collect()
            .unwrap()
    }
    
    pub fn calculate_final_summary(&self, df: &DataFrame) -> DataFrame {
        // Group by batch_size only, averaging across all experiments
        df.clone().lazy()
            .group_by([col("batch_size")])
            .agg([
                col("elapsed_micros").mean().alias("avg_elapsed_micros"),
                col("elapsed_micros").std(1).alias("std_elapsed_micros"),
                col("elapsed_micros").min().alias("min_elapsed_micros"),
                col("elapsed_micros").max().alias("max_elapsed_micros"),
            ])
            .sort_by_exprs([col("batch_size")], Default::default())
            .collect()
            .unwrap()
    }
}