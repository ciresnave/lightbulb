@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd lightbulb
cargo build --example test_awq_inference --features cuda --release -vv -j 1
