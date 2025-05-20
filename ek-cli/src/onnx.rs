use std::io::Write;

use clap::Subcommand;
use ek_base::error::EKResult;
use ek_computation::ffn::DType;
use ort::session::Session;
use prost::Message;

#[derive(Subcommand, Debug)]
pub enum OnnxCommand {
    Export {
        #[arg(long, default_value_t = ("onnx_model.onnx").to_string())]
        output: String,
        #[arg(long, default_value_t = 7168)]
        hidden_size: i64,
        #[arg(long, default_value_t = 2048)]
        inter_size: i64,
    },
}

pub async fn execute_onnx(cmd: OnnxCommand) -> EKResult<()> {
    match cmd {
        OnnxCommand::Export {
            output,
            hidden_size,
            inter_size,
        } => {
            let builder = ek_computation::onnx::exporter::ExpertOnnxBuilder {
                intermediate_size: inter_size,
                hidden_size,
                data_type: DType::Float,
            };
            let msg = builder.build();
            let mut file = std::fs::File::create(output).unwrap();
            let raw = msg.encode_to_vec();
            file.write_all(raw.as_slice())
                .expect("failed to write to file");
            let _session = Session::builder()
                .expect("failed to create session")
                .commit_from_memory(raw.as_slice())
                .unwrap();
            Ok(())
        }
    }
}
