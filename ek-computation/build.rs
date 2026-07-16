fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    // SAFETY: This single-threaded build script sets PROTOC before code generation starts.
    unsafe { std::env::set_var("PROTOC", protoc) };

    // tonic_build::("../ek-proto/ek")?;
    let os = std::env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    match os.as_str() {
        "linux" | "windows" => {
            if let Some(lib_path) = std::env::var_os("DEP_TCH_LIBTORCH_LIB") {
                println!(
                    "cargo:rustc-link-arg=-Wl,-rpath={}",
                    lib_path.to_string_lossy()
                );
            }
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
            // println!("cargo:rustc-link-arg=-Wl,--copy-dt-needed-entries");
            println!("cargo:rustc-link-arg=-ltorch");
        }
        _ => {}
    }
    tonic_build::configure().build_server(true).compile_protos(
        &[
            "../ek-proto/ek/control/v1/control.proto",
            "../ek-proto/ek/control/v1/routing.proto",
            "../ek-proto/ek/worker/v1/expert.proto",
            "../ek-proto/ek/worker/v2/common.proto",
            "../ek-proto/ek/worker/v2/computation.proto",
            "../ek-proto/ek/control/v2/lifecycle.proto",
            "../ek-proto/ek/control/v2/weight_control.proto",
            "../ek-proto/ek/object/v1/object.proto",
            "../ek-proto/onnx/onnx.proto",
        ],
        &["../ek-proto"],
    )?;
    eprintln!("protobuf built");
    Ok(())
}
