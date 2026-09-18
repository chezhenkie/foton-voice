@echo off
echo MARK-START %date% %time%
call C:\apps\build-tools\msvc\MSVC\setup_x64.bat
set CUDA_HOME=C:\apps\build-tools\cuda
set PATH=C:\apps\build-tools\cuda\bin;C:\apps\build-tools\cmake-4.4.3-windows-x86_64\bin;%USERPROFILE%\.cargo\bin;%PATH%
set INCLUDE=C:\apps\build-tools\cuda\include;%INCLUDE%
set LIB=C:\apps\build-tools\cuda\lib\x64;%LIB%
set LIBCLANG_PATH=C:\apps\build-tools\llvm\bin
set CMAKE_GENERATOR=NMake Makefiles
set CMAKE_MAKE_PROGRAM=C:\apps\build-tools\msvc\MSVC\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64\nmake.exe
set CARGO_TARGET_DIR=C:\Users\lap1user\ocdev\fotonvoice-engine\fotonvoice-src\target-msvc
cd /d C:\Users\lap1user\ocdev\fotonvoice-engine\fotonvoice-src\fotonvoice-engine-0.5.8
cargo build --release -p fotonvoice-app --features parakeet-webgpu,moonshine-webgpu
echo MARK-END EXIT=%errorlevel%
