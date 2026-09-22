fn main() {
    let moonshine = std::env::var_os("CARGO_FEATURE_MOONSHINE").is_some();
    let linux = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux");

    println!("cargo:rerun-if-changed=src/isoc23_compat.c");
    println!("cargo:rerun-if-changed=build.rs");

    if moonshine && linux {
        cc::Build::new()
            .file("src/isoc23_compat.c")
            .compile("fotonvoice_isoc23_compat");
    }
}
