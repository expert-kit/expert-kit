fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::compile_protos("../ek-proto/expert.proto")?;
    eprintln!("protobuf built");
    Ok(())
}
