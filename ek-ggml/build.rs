use std::{env, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=ggml");

    let dst = cmake::Config::new("ggml")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("GGML_STATIC", "ON")
        .define("GGML_LLAMAFILE", "ON")
        .build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=dylib=gomp");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    let mut bindings = bindgen::Builder::default();

    for header in dst.join("include").read_dir()? {
        if let Ok(header) = header
            && ["ggml-cpu.h"].contains(&header.file_name().to_str().unwrap())
        {
            bindings = bindings.header(header.path().to_string_lossy());
        }
    }

    let bindings = bindings.generate()?;

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("bindings.rs"))?;

    Ok(())
}
