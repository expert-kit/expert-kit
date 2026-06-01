fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Path to ek-proto directory
    let proto_root = "../../ek-proto";

    // Compile all required protos together (to handle dependencies)
    tonic_build::configure()
        .build_server(false)  // Client only, no server code
        .compile(
            &[
                format!("{}/ek/object/v1/object.proto", proto_root),
                format!("{}/ek/worker/v1/expert.proto", proto_root),
                format!("{}/ek/control/v1/routing.proto", proto_root),
            ],
            &[proto_root],
        )?;

    Ok(())
}
