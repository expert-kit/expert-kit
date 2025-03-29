fn main() -> Result<(), Box<dyn std::error::Error>> {
    // tonic_build::("../ek-proto/ek")?;
    tonic_build::configure()
        .build_server(true)
        .compile_protos(&["../ek-proto/ek/worker/v1/expert.proto"], &["../ek-proto"])?;
    eprintln!("protobuf built");
    Ok(())
}
