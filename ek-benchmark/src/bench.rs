use std::time::{Instant, SystemTime, UNIX_EPOCH};

use polars::frame::DataFrame;
use polars::frame::row::Row;
use polars::prelude::*;

use crate::expert::{Expert, ExpertShape};
use crate::expert_ort::OnnxFFN;
use crate::expert_torch::TorchFFN;

pub enum GenericExpert {
    Torch(TorchFFN),
    Ort(OnnxFFN),
}
pub struct BenchmarkerImpl {
    iterations: usize,
    experts: Vec<GenericExpert>,
}

impl GenericExpert {
    fn backend(&self) -> std::string::String {
        match self {
            GenericExpert::Torch(exp) => exp.backend(),
            GenericExpert::Ort(exp) => exp.backend(),
        }
    }
    fn shape(&self) -> ExpertShape {
        match self {
            GenericExpert::Torch(exp) => exp.shape(),
            GenericExpert::Ort(exp) => exp.shape(),
        }
    }
    fn forward(&self, batch: usize) -> Instant {
        match self {
            GenericExpert::Torch(exp) => {
                let input = exp.rand_input(batch);

                let start = Instant::now();
                let _ = exp.forward(input);
                start
            }
            GenericExpert::Ort(exp) => {
                let input = exp.rand_input(batch);
                let start = Instant::now();
                let _ = exp.forward(input);
                start
            }
        }
    }
}

impl BenchmarkerImpl {
    pub fn new(experts: Vec<GenericExpert>) -> Self {
        BenchmarkerImpl {
            experts,
            iterations: 10,
        }
    }

    pub fn iterations(&mut self, iterations: usize) -> &mut BenchmarkerImpl {
        self.iterations = iterations;
        self
    }

    pub fn scan_batch(&self, size: &[usize]) -> DataFrame {
        let mut rows: Vec<Row> = Vec::new();
        let b = sysinfo::System::new_all();
        let brand = b.cpus()[0].brand();
        info!("scan batch in cpu: {}", brand);
        for expert in self.experts.iter() {
            // warm up
            expert.forward(1);
        }

        for iter in 0..self.iterations {
            for batch in size.iter() {
                let mut eidx = 0;
                for expert in self.experts.iter() {
                    let start = expert.forward(*batch);
                    let shape = expert.shape();
                    let backend = expert.backend();
                    let row = vec![
                        AnyValue::UInt64(iter as u64),
                        AnyValue::UInt64(*batch as u64),
                        AnyValue::UInt64(eidx as u64),
                        AnyValue::UInt64(start.elapsed().as_micros() as u64),
                        AnyValue::UInt64(shape.dim as u64),
                        AnyValue::UInt64(shape.hidden as u64),
                        AnyValue::StringOwned(backend.into()),
                    ];
                    rows.push(Row(row));
                    eidx += 1;
                }
            }
        }
        let schema = Schema::from_iter(vec![
            Field::new("iter".into(), DataType::UInt64),
            Field::new("batch".into(), DataType::Int32),
            Field::new("expert_id".into(), DataType::UInt64),
            Field::new("elapsed".into(), DataType::Float64),
            Field::new("input_dim".into(), DataType::UInt64),
            Field::new("hidden_dim".into(), DataType::UInt64),
            Field::new("backend".into(), DataType::String),
        ]);

        let df = DataFrame::from_rows_and_schema(rows.as_slice(), &schema).unwrap();

        let cur_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let hardware = Series::new("hardware".into(), vec![brand; df.height()]);
        let ts = Series::new("time".into(), vec![cur_ts; df.height()]);
        let df = df.hstack(&[hardware.into(), ts.into()]).unwrap();
        df
    }
}
