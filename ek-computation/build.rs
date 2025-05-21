use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // tonic_build::("../ek-proto/ek")?;
    tonic_build::configure().build_server(true).compile_protos(
        &[
            "../ek-proto/ek/worker/v1/expert.proto",
            "../ek-proto/ek/object/v1/object.proto",
        ],
        &["../ek-proto"],
    )?;
    eprintln!("protobuf built");

    if cfg!(feature = "npu") {
        config_cann_bindings()
    }

    Ok(())
}

fn config_cann_bindings() {
        // Look for ASCEND_PATH environment variable
        let ascend_path = if let Ok(path) = env::var("ASCEND_CUSTOM_PATH") {
            path
        } else {
            "/usr/local/Ascend/ascend-toolkit/latest".to_string()
        };

        // Link directories
        println!("cargo:rustc-link-search=native={}/lib64", ascend_path);
        
        // Link required libraries
        println!("cargo:rustc-link-lib=ascendcl");
        println!("cargo:rustc-link-lib=nnopbase");
        println!("cargo:rustc-link-lib=opapi");
        
        // Specify header location for bindgen if needed
        println!("cargo:include={}/include", ascend_path);
        println!("cargo:include={}/include/aclnn", ascend_path);
}