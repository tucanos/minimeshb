fn main() {
    let mut rpath = Vec::new();
    // See https://github.com/xgarnaud/libmeshb-sys#using
    if let Ok(meshb_rpath) = std::env::var("DEP_MESHB.7_RPATH") {
        rpath.push(meshb_rpath);
    }
    println!("cargo:rerun-if-env-changed=DEP_MESHB.7_RPATH");

    for p in &rpath {
        // Needed to build the tests
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        println!("cargo:rustc-link-arg=-Wl,-rpath,{p}");
    }

    if !rpath.is_empty() {
        // non standard key
        // see https://doc.rust-lang.org/cargo/reference/build-script-examples.html#linking-to-system-libraries
        // and https://github.com/rust-lang/cargo/issues/5077
        println!("cargo:rpath={}", rpath.join(":"));
    }
}
