fn main() -> Result<(), Box<dyn std::error::Error>> {
    // tonic_build::("../ek-proto/ek")?;
    tonic_build::configure().build_server(true).compile_protos(
        &[
            "../ek-proto/ek/control/v1/control.proto",
            "../ek-proto/ek/worker/v1/expert.proto",
            "../ek-proto/ek/object/v1/object.proto",
            "../ek-proto/onnx/onnx.proto",
        ],
        &["../ek-proto"],
    )?;
    eprintln!("protobuf built");
    Ok(())
}
