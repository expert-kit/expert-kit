pub mod ek {
    pub mod worker {
        pub mod v1 {
            tonic::include_proto!("ek.worker.v1");
        }
    }
    pub mod object {
        pub mod v1 {
            tonic::include_proto!("ek.object.v1");
        }
    }
}

pub mod pbonnx {
    tonic::include_proto!("onnx");
}
