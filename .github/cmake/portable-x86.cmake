# Portable but fast ggml CPU configuration.
#
# ggml decides its x86 kernel set from two variables in ggml/CMakeLists.txt:
#
#   if (GGML_NATIVE OR NOT GGML_NATIVE_DEFAULT)
#       set(INS_ENB OFF)
#   else()
#       set(INS_ENB ON)
#   endif()
#   option(GGML_SSE42 ... ${INS_ENB})
#   option(GGML_AVX  ... ${INS_ENB})
#   option(GGML_AVX2 ... ${INS_ENB})
#   option(GGML_BMI2 ... ${INS_ENB})
#
# GGML_NATIVE=ON is what passes -march=native, which on an AVX-512 build host
# bakes EVEX into the artifact and kills it on CPUs without AVX-512. But simply
# turning GGML_NATIVE off is not enough, because that also lands INS_ENB on the
# OFF branch and leaves the CPU backend with no SIMD at all: bare x86-64
# baseline. Measured cost of that on an Intel Core Ultra 5 226V, large-v3-turbo
# Q5_0, 8 threads: 3630 ms of audio took 128235 ms to transcribe, a real-time
# factor of 0.028, which stalled the whole machine.
#
# INS_ENB is ON only when GGML_NATIVE is OFF while GGML_NATIVE_DEFAULT is still
# ON. GGML_NATIVE_DEFAULT is derived from whether SOURCE_DATE_EPOCH is defined,
# so it must NOT be set; GGML_NATIVE is forced off here instead. CMake reads
# CMAKE_TOOLCHAIN_FILE from the environment and runs this file before the
# project, early enough to seed the cache, so option() will not override it.
#
# Result: /arch:AVX2 with FMA and F16C on MSVC (ggml-cpu/CMakeLists.txt:290),
# and no -march=native. The AVX-512 options are forced off as well; on MSVC they
# are a separate, mutually exclusive branch, so this is belt and braces.
#
# Tradeoff worth stating plainly: this raises the CPU floor from baseline x86-64
# to AVX2, i.e. Haswell (2013) or newer. That is safe for any machine with a
# Vulkan-capable GPU and it is what makes the CPU path usable.

set(GGML_NATIVE        OFF CACHE BOOL "" FORCE)
set(GGML_AVX512        OFF CACHE BOOL "" FORCE)
set(GGML_AVX512_VBMI   OFF CACHE BOOL "" FORCE)
set(GGML_AVX512_VNNI   OFF CACHE BOOL "" FORCE)
set(GGML_AVX512_BF16   OFF CACHE BOOL "" FORCE)
set(GGML_AVX_VNNI      OFF CACHE BOOL "" FORCE)

# Fail loudly rather than silently producing a slow build if ggml ever renames
# these, which is what happened the first time this was guessed at.
foreach(_req GGML_NATIVE GGML_AVX512)
  if(NOT DEFINED ${_req})
    message(FATAL_ERROR "portable-x86.cmake: ${_req} was not honoured")
  endif()
endforeach()

message(STATUS "portable-x86: GGML_NATIVE=${GGML_NATIVE} GGML_AVX512=${GGML_AVX512}")