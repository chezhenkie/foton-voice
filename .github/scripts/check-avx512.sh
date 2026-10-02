#!/bin/bash
# Verifies a built .exe contains no AVX-512 instructions.
#
# A green build on the runner says nothing about the CPU of the machine that
# runs the artifact: the runner has AVX-512, so the toolchain happily emits
# EVEX instructions that abort with 0xc000001d on any target without it.
# This was observed as 5092 zmm instructions in an artifact built for an
# Intel Core Ultra 5 226V (Lunar Lake), which has no AVX-512.
#
# Usage: check-avx512.sh <path-to-exe>
set -u

EXE="${1:-}"
if [ -z "$EXE" ] || [ ! -f "$EXE" ]; then
    echo "usage: check-avx512.sh <path-to-exe>" >&2
    exit 2
fi

if ! command -v objdump >/dev/null 2>&1; then
    echo "objdump not found; cannot verify" >&2
    exit 2
fi

echo "scanning $EXE for EVEX (AVX-512) instructions"

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

objdump -d --section=.text "$EXE" >"$tmp" 2>/dev/null
if [ ! -s "$tmp" ]; then
    echo "objdump produced no output; cannot verify (treating as FAIL)" >&2
    exit 1
fi

count=$(grep -c 'zmm' "$tmp" || true)
size=$(wc -c <"$tmp")

echo "disassembly bytes: $size"
echo "AVX-512 (zmm) instruction references: $count"

if [ "$count" -ne 0 ]; then
    echo ""
    echo "FAIL: $count AVX-512 instructions found. First sites:"
    grep -n 'zmm' "$tmp" | head -10
    echo ""
    echo "This artifact will crash with STATUS_ILLEGAL_INSTRUCTION (0xc000001d)"
    echo "on any CPU without AVX-512. Check RUSTFLAGS has the -avx512* disables."
    exit 1
fi

echo "OK: no AVX-512 instructions. The binary is portable to AVX-512-less CPUs."
exit 0