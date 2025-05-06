use std::sync::Arc;

use crate::proto::ek::worker::v1::{self};
use once_cell::sync::OnceCell;
use tokio::{
    fs::{self},
    io::AsyncWriteExt,
    sync::Mutex,
};

pub struct DebugTracer {
    intra_egress_counter: usize,
    intra_ingress_counter: usize,
}

impl DebugTracer {
    pub fn get() -> Arc<Mutex<Self>> {
        static INSTANCE: OnceCell<Arc<Mutex<DebugTracer>>> = OnceCell::new();
        let inst = INSTANCE.get_or_init(|| {
            let instance = Arc::new(Mutex::new(DebugTracer {
                intra_egress_counter: 0,
                intra_ingress_counter: 0,
            }));
            instance.clone()
        });
        inst.clone()
    }

    pub async fn inter_input(&mut self, req: v1::ForwardReq, reqid: usize) {
        let root = std::path::PathBuf::from("/tmp/ektrace/inter_input");
        let meta_fp = format!("{}.req.meta", reqid);
        let ts_fp = format!("{}.safetensors", reqid);
        let mut ts = fs::File::create(root.join(ts_fp)).await.unwrap();
        let mut meta = fs::File::create(root.join(meta_fp)).await.unwrap();
        ts.write(&req.tensor).await.unwrap();
        for e in req.sequences.iter().enumerate() {
            meta.write(format!("seq-{}\n", e.0).as_bytes())
                .await
                .unwrap();
            for v in &e.1.experts {
                meta.write(format!("\t{}\n", v).as_bytes()).await.unwrap();
            }
        }
    }

    pub async fn inter_output(
        &mut self,
        reqid: usize,
        size: &[usize],
        input_tensor: &[u8],
        output_tensor: &[u8],
    ) {
        let root = std::path::PathBuf::from("/tmp/ektrace/inter_output");
        let meta_fp = format!("{}.resp.meta", reqid);
        let ts_inp_fp = format!("{}.inp.safetensors", reqid);
        let mut ts_inp = fs::File::create(root.join(ts_inp_fp)).await.unwrap();
        ts_inp.write(input_tensor).await.unwrap();
        let ts_out_fp = format!("{}.out.safetensors", reqid);
        let mut ts_out = fs::File::create(root.join(ts_out_fp)).await.unwrap();
        ts_out.write(output_tensor).await.unwrap();

        let mut meta = fs::File::create(root.join(meta_fp)).await.unwrap();
        meta.write("output size: ".as_bytes()).await.unwrap();
        for d in size.iter() {
            meta.write(format!("{},\t", *d).as_bytes()).await.unwrap();
        }
    }

    pub async fn intra_egress(&mut self, req: v1::ForwardReq) -> usize {
        self.intra_egress_counter += 1;
        let root = std::path::PathBuf::from("/tmp/ektrace/intra_egress");
        let tensor_file = format!("{}.req.safetensors", self.intra_egress_counter);
        let meta_file = format!("{}.req.meta", self.intra_egress_counter);
        let mut tensor = fs::File::create(root.join(tensor_file)).await.unwrap();
        tensor.write(req.tensor.clone().as_slice()).await.unwrap();
        let mut meta = fs::File::create(root.join(meta_file)).await.unwrap();
        for e in req.sequences.iter().enumerate() {
            meta.write(format!("seq-{}\n", e.0).as_bytes())
                .await
                .unwrap();
            for v in &e.1.experts {
                meta.write(format!("\t{}\n", v).as_bytes()).await.unwrap();
            }
        }
        self.intra_egress_counter
    }

    pub async fn intra_ingress(&mut self, resp: &[u8]) {
        self.intra_ingress_counter += 1;
        let root = std::path::PathBuf::from("/tmp/ektrace/intra_ingress");
        let tensor_fp = format!("{}.resp.safetensors", self.intra_ingress_counter);
        let mut ts = fs::File::create(root.join(tensor_fp)).await.unwrap();
        ts.write(resp).await.unwrap();
    }
}
