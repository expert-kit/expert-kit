fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    // SAFETY: This single-threaded build script sets PROTOC before code generation starts.
    unsafe { std::env::set_var("PROTOC", protoc) };

    tonic_build::configure().build_server(true).compile_protos(
        &[
            "../ek-proto/ek/control/v1/control.proto",
            "../ek-proto/ek/worker/v2/common.proto",
            "../ek-proto/ek/worker/v2/computation.proto",
            "../ek-proto/ek/control/v2/lifecycle.proto",
            "../ek-proto/ek/control/v2/weight_control.proto",
            "../ek-proto/ek/object/v1/object.proto",
        ],
        &["../ek-proto"],
    )?;
    eprintln!("protobuf built");
    Ok(())
}
