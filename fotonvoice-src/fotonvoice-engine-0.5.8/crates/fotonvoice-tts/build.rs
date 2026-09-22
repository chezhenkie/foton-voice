fn main() {
    let inflect = std::env::var_os("CARGO_FEATURE_INFLECT_MICRO").is_some();
    let linux = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux");

    println!("cargo:rerun-if-changed=src/isoc23_compat.c");
    println!("cargo:rerun-if-changed=build.rs");

    if inflect && linux {
        cc::Build::new()
            .file("src/isoc23_compat.c")
            .compile("fotonvoice_tts_isoc23_compat");
    }
}
