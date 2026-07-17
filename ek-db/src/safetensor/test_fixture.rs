//! Build a small Qwen SafeTensors checkpoint for crate-local tests.

use std::{collections::BTreeMap, path::PathBuf, sync::OnceLock};

use safetensors::{Dtype, tensor::TensorView};

const MODEL_NAME: &str = "qwen-test";
const NUM_LAYERS: usize = 10;
const NUM_EXPERTS: usize = 256;
const HIDDEN_DIM: usize = 16;
const INTERMEDIATE_DIM: usize = 8;

pub(crate) fn synthetic_qwen_model() -> PathBuf {
    static MODEL_ROOT: OnceLock<PathBuf> = OnceLock::new();

    MODEL_ROOT
        .get_or_init(|| {
            let parent =
                std::env::temp_dir().join(format!("expert-kit-ek-db-tests-{}", std::process::id()));
            let model_root = parent.join(MODEL_NAME);
            if parent.exists() {
                std::fs::remove_dir_all(&parent).unwrap();
            }
            std::fs::create_dir_all(&model_root).unwrap();

            let config = serde_json::json!({
                "model_type": "qwen3_moe",
                "num_hidden_layers": NUM_LAYERS,
                "num_experts": NUM_EXPERTS,
                "hidden_size": HIDDEN_DIM,
                "moe_intermediate_size": INTERMEDIATE_DIM,
            });
            std::fs::write(
                model_root.join("config.json"),
                serde_json::to_vec_pretty(&config).unwrap(),
            )
            .unwrap();

            let tensor_data = vec![0_u8; HIDDEN_DIM * INTERMEDIATE_DIM * 2];
            let mut tensors = BTreeMap::new();
            let mut weight_map = serde_json::Map::new();
            for layer_id in 0..NUM_LAYERS {
                for expert_id in 0..NUM_EXPERTS {
                    for projection in ["gate_proj", "up_proj", "down_proj"] {
                        let name = format!(
                            "model.layers.{layer_id}.mlp.experts.{expert_id}.{projection}.weight"
                        );
                        let shape = if projection == "down_proj" {
                            vec![HIDDEN_DIM, INTERMEDIATE_DIM]
                        } else {
                            vec![INTERMEDIATE_DIM, HIDDEN_DIM]
                        };
                        tensors.insert(
                            name.clone(),
                            TensorView::new(Dtype::BF16, shape, &tensor_data).unwrap(),
                        );
                        weight_map.insert(
                            name,
                            serde_json::Value::String("model.safetensors".to_string()),
                        );
                    }
                }
            }

            safetensors::tensor::serialize_to_file(
                &tensors,
                &None,
                &model_root.join("model.safetensors"),
            )
            .unwrap();
            let index = serde_json::json!({
                "metadata": {},
                "weight_map": weight_map,
            });
            std::fs::write(
                model_root.join("model.safetensors.index.json"),
                serde_json::to_vec(&index).unwrap(),
            )
            .unwrap();

            model_root
        })
        .clone()
}
