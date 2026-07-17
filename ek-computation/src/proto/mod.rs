#[allow(clippy::all)]
mod inner {
    pub mod ek {
        pub mod worker {
            pub mod v2 {
                tonic::include_proto!("ek.worker.v2");
            }
        }
        pub mod object {
            pub mod v1 {
                tonic::include_proto!("ek.object.v1");
            }
        }

        pub mod control {
            pub mod v1 {
                tonic::include_proto!("ek.control.v1");
            }
            pub mod v2 {
                tonic::include_proto!("ek.control.v2");
            }
        }
    }
}

pub use inner::ek;
