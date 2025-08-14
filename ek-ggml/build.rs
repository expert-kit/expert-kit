use std::{env, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=ggml");

    let dst = cmake::Config::new("ggml")
        .profile("Release")
        // .define("GGML_STATIC", "ON")
        .define("GGML_LLAMAFILE", "ON")
        .define("GGML_CUDA", "ON")
        .build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-lib=dylib=ggml-base");
    println!("cargo:rustc-link-lib=dylib=ggml-cpu");
    println!("cargo:rustc-link-lib=dylib=ggml-cuda");

    let mut bindings = bindgen::Builder::default();

    for header in dst.join("include").read_dir()? {
        if let Ok(header) = header
            && ["ggml.h", "ggml-cpu.h", "ggml-cuda.h"]
                .contains(&header.file_name().to_str().unwrap())
        {
            bindings = bindings.header(header.path().to_string_lossy());
        }
    }

    let bindings = bindings.generate()?;

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("bindings.rs"))?;

    Ok(())
}
