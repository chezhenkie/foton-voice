#!/bin/bash
source /home/user1linux/.cargo/env
export CC_x86_64_pc_windows_gnu=x86_64-w64-mingw32-gcc
export CXX_x86_64_pc_windows_gnu=x86_64-w64-mingw32-g++
export CARGO_TARGET_DIR=/home/user1linux/vc-probe
cd /mnt/c/Users/lap1user/ocdev/fotonvoice-engine/fotonvoice-src/ort-probe
echo "PROBE-START $(date +%T)"
cargo build --release --target x86_64-pc-windows-gnu
echo "PROBE-END EXIT=$? $(date +%T)"
