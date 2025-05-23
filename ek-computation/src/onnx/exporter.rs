use prost::Message;

use crate::{backend::DType, proto::pbonnx};

#[allow(dead_code)]
enum ActFn {
    SiLU,
}
pub struct ExpertOnnxBuilder {
    pub intermediate_size: i64,
    pub hidden_size: i64,
    pub data_type: DType,
}

impl From<pbonnx::tensor_proto::DataType> for DType {
    fn from(value: pbonnx::tensor_proto::DataType) -> Self {
        match value {
            pbonnx::tensor_proto::DataType::Bfloat16 => DType::BFloat16,
            pbonnx::tensor_proto::DataType::Float => DType::Float,
            pbonnx::tensor_proto::DataType::Int8 => DType::Int8,
            pbonnx::tensor_proto::DataType::Int16 => DType::Int16,
            _ => unimplemented!(),
        }
    }
}

impl From<DType> for pbonnx::tensor_proto::DataType {
    fn from(val: DType) -> Self {
        match val {
            DType::BFloat16 => pbonnx::tensor_proto::DataType::Bfloat16,
            DType::Float => pbonnx::tensor_proto::DataType::Float,
            DType::Int8 => pbonnx::tensor_proto::DataType::Int8,
            DType::Int16 => pbonnx::tensor_proto::DataType::Int16,
            _ => unimplemented!(),
        }
    }
}

impl ExpertOnnxBuilder {
    fn tensor_pv(&self, p1: &str, v2: i64) -> pbonnx::type_proto::Value {
        pbonnx::type_proto::Value::TensorType(pbonnx::type_proto::Tensor {
            elem_type: Into::<pbonnx::tensor_proto::DataType>::into(self.data_type) as i32,
            shape: Some(pbonnx::TensorShapeProto {
                dim: vec![
                    pbonnx::tensor_shape_proto::Dimension {
                        value: Some(pbonnx::tensor_shape_proto::dimension::Value::DimParam(
                            p1.to_string(),
                        )),
                        ..Default::default()
                    },
                    pbonnx::tensor_shape_proto::Dimension {
                        value: Some(pbonnx::tensor_shape_proto::dimension::Value::DimValue(v2)),
                        ..Default::default()
                    },
                ],
            }),
        })
    }

    #[allow(unused)]
    fn tensor_vv(&self, v1: i64, v2: i64) -> pbonnx::type_proto::Value {
        pbonnx::type_proto::Value::TensorType(pbonnx::type_proto::Tensor {
            elem_type: Into::<pbonnx::tensor_proto::DataType>::into(self.data_type) as i32,
            shape: Some(pbonnx::TensorShapeProto {
                dim: vec![
                    pbonnx::tensor_shape_proto::Dimension {
                        value: Some(pbonnx::tensor_shape_proto::dimension::Value::DimValue(v1)),
                        ..Default::default()
                    },
                    pbonnx::tensor_shape_proto::Dimension {
                        value: Some(pbonnx::tensor_shape_proto::dimension::Value::DimValue(v2)),
                        ..Default::default()
                    },
                ],
            }),
        })
    }

    fn build_outputs(&self) -> Vec<pbonnx::ValueInfoProto> {
        let res = vec![pbonnx::ValueInfoProto {
            name: "output".to_string(),
            r#type: Some(pbonnx::TypeProto {
                value: Some(self.tensor_pv("batch_size", self.hidden_size)),
                ..Default::default()
            }),
            ..Default::default()
        }];
        res
    }

    fn build_inputs(&self) -> Vec<pbonnx::ValueInfoProto> {
        let inp = vec![pbonnx::ValueInfoProto {
            name: "input".to_string(),
            r#type: Some(pbonnx::TypeProto {
                value: Some(self.tensor_pv("batch_size", self.hidden_size)),
                ..Default::default()
            }),
            ..Default::default()
        }];

        inp
    }

    fn kv(&self, k: &str, v: &str) -> pbonnx::StringStringEntryProto {
        pbonnx::StringStringEntryProto {
            key: k.to_string(),
            value: v.to_string(),
        }
    }

    fn build_initializers(&self) -> Vec<pbonnx::TensorProto> {
        let init = vec![
            pbonnx::TensorProto {
                name: "onnx::MatMul_13".to_string(),
                data_type: Into::<pbonnx::tensor_proto::DataType>::into(self.data_type) as i32,
                dims: vec![self.hidden_size, self.intermediate_size],
                data_location: pbonnx::tensor_proto::DataLocation::External as i32,
                external_data: vec![self.kv("location", "#onnx::MatMul_13")],
                ..Default::default()
            },
            pbonnx::TensorProto {
                name: "onnx::MatMul_14".to_string(),
                data_type: Into::<pbonnx::tensor_proto::DataType>::into(self.data_type) as i32,
                data_location: pbonnx::tensor_proto::DataLocation::External as i32,
                dims: vec![self.hidden_size, self.intermediate_size],
                external_data: vec![self.kv("location", "#onnx::MatMul_14")],
                ..Default::default()
            },
            pbonnx::TensorProto {
                name: "onnx::MatMul_15".to_string(),
                data_type: Into::<pbonnx::tensor_proto::DataType>::into(self.data_type) as i32,
                data_location: pbonnx::tensor_proto::DataLocation::External as i32,
                external_data: vec![self.kv("location", "#onnx::MatMul_15")],
                dims: vec![self.intermediate_size, self.hidden_size],
                ..Default::default()
            },
        ];
        init
    }

    fn build_graph(&self) -> pbonnx::GraphProto {
        pbonnx::GraphProto {
            name: "main_graph".to_string(),
            node: self.build_nodes(),
            input: self.build_inputs(),
            output: self.build_outputs(),
            initializer: self.build_initializers(),
            ..Default::default()
        }
    }
    fn build_nodes(&self) -> Vec<pbonnx::NodeProto> {
        let node = vec![
            pbonnx::NodeProto {
                input: vec!["input".to_string(), "onnx::MatMul_13".to_string()],
                output: vec!["/gate_proj/MatMul_output_0".to_string()],
                name: "/gate_proj/MatMul".to_string(),
                op_type: "MatMul".to_string(),
                ..pbonnx::NodeProto::default()
            },
            pbonnx::NodeProto {
                input: vec!["/gate_proj/MatMul_output_0".to_string()],
                output: vec!["/act_fn/Sigmoid_output_0".to_string()],
                name: "/act_fn/Sigmoid".to_string(),
                op_type: "Sigmoid".to_string(),
                ..pbonnx::NodeProto::default()
            },
            pbonnx::NodeProto {
                input: vec![
                    "/gate_proj/MatMul_output_0".to_string(),
                    "/act_fn/Sigmoid_output_0".to_string(),
                ],
                output: vec!["/act_fn/Mul_output_0".to_string()],
                name: "/act_fn/Mul".to_string(),
                op_type: "Mul".to_string(),
                ..pbonnx::NodeProto::default()
            },
            pbonnx::NodeProto {
                input: vec!["input".to_string(), "onnx::MatMul_14".to_string()],
                output: vec!["/up_proj/MatMul_output_0".to_string()],
                name: "/up_proj/MatMul".to_string(),
                op_type: "MatMul".to_string(),
                ..pbonnx::NodeProto::default()
            },
            pbonnx::NodeProto {
                input: vec![
                    "/act_fn/Mul_output_0".to_string(),
                    "/up_proj/MatMul_output_0".to_string(),
                ],
                output: vec!["/Mul_output_0".to_string()],
                name: "/Mul".to_string(),
                op_type: "Mul".to_string(),
                ..pbonnx::NodeProto::default()
            },
            pbonnx::NodeProto {
                input: vec!["/Mul_output_0".to_string(), "onnx::MatMul_15".to_string()],
                output: vec!["output".to_string()],
                name: "/down_proj/MatMul".to_string(),
                op_type: "MatMul".to_string(),
                ..pbonnx::NodeProto::default()
            },
        ];
        node
    }
    pub fn build(&self) -> pbonnx::ModelProto {
        let graph = self.build_graph();

        pbonnx::ModelProto {
            ir_version: 9,
            opset_import: vec![pbonnx::OperatorSetIdProto {
                version: 20,
                ..Default::default()
            }],
            producer_name: "expert-kit".to_string(),
            producer_version: "dev".to_string(),
            doc_string: "expert onnx model file generated by expert-kit".to_string(),
            graph: Some(graph),
            ..Default::default()
        }
    }

    pub fn build_raw(&self) -> Vec<u8> {
        let msg = self.build();
        let mut buf = Vec::new();
        msg.encode(&mut buf).unwrap();
        buf
    }
}

#[cfg(test)]
mod test {

    use ort::session::Session;
    use prost::Message;

    use crate::{
        backend::{DType, Device, ort::NDArrayTensor},
        ffn::meta::ExpertWeight,
    };

    #[test]
    fn test_basic_export() {
        ort::init().commit().unwrap();
        let builder = super::ExpertOnnxBuilder {
            intermediate_size: 7168,
            hidden_size: 2048,
            data_type: DType::Float,
        };
        let model = builder.build();
        let raw = model.encode_to_vec();

        let rand_weight: ExpertWeight<NDArrayTensor<f32>> =
            ExpertWeight::from_rand_matmul(2048, 7168, DType::Float, Device::CPU);

        let session = Session::builder()
            .expect("failed to create session")
            .with_external_initializer("onnx::MatMul_13", rand_weight.up_w.into())
            .expect("should load up")
            .with_external_initializer("onnx::MatMul_14", rand_weight.gate_w.into())
            .expect("should load up")
            .with_external_initializer("onnx::MatMul_15", rand_weight.down_w.into())
            .expect("should load up")
            .commit_from_memory(raw.as_slice())
            .unwrap();

        assert!(session.inputs.len() == 1);
        assert!(session.outputs.len() == 1);
        dbg!("output", &session.outputs);
        let meta = session.metadata().expect("should have metadata");
        assert_eq!(meta.producer().expect("should have producer"), "expert-kit");
    }
}
