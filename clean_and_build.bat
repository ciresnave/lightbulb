@echo off
cd lightbulb
cargo clean
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cargo build --example test_awq_inference --features cuda --release -j 1
