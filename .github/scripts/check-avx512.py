#!/usr/bin/env python3
"""Fail the build if a Windows exe contains AVX-512 instructions.

Why this exists
---------------
A green build on a GitHub runner proves nothing about the CPU of the machine
that will run the artifact. The runner has AVX-512, so the toolchain can emit
EVEX instructions into a binary meant for CPUs that do not have them. Such an
artifact dies on first model load with STATUS_ILLEGAL_INSTRUCTION
(0xc000001d). Observed: 5092 EVEX instructions in an artifact built for an
Intel Core Ultra 5 226V (Lunar Lake), which has no AVX-512.

How it works
------------
Delegates to a real disassembler (llvm-objdump, or GNU objdump as a fallback)
and counts DECODED instructions that use a zmm register or a k-mask register.

The previous version of this script scanned raw bytes for the EVEX prefix 0x62
without any notion of instruction boundaries. On one known-bad artifact it
reported 95242 sites where objdump's real disassembly found 5092, so ~95% of
its hits were data or misaligned bytes. It could not distinguish a broken
artifact from a clean one, and it failed good builds for the wrong reason.
Real disassembly removes that entire class of error.

Usage: check-avx512.py <path-to-exe> [--objdump PATH]
Exit:  0 clean, 1 AVX-512 found, 2 could not verify.
"""

import os
import re
import subprocess
import sys

# A decoded instruction line looks like:
#   1425211: 62 f1 7c 48 10 05 75 dc 6f 00   vmovups 0x6fdc75(%rip),%zmm0
# AVX-512 only shows up as a zmm register, or as an opmask k0-k7 operand.
# Both spellings are matched; both are impossible on pre-AVX-512 hardware.
AVX512_OPERAND = re.compile(r"%zmm[0-9]|(?<![\w])%k[0-7]")

INSN_LINE = re.compile(r"^\s*([0-9a-fA-F]+):\s")

# Only decode code. --disassemble on the whole file is slower and would pick up
# data blobs that merely look like code.
DISASM_ARGS = ["--disassemble", "--no-show-raw-insn"]

# Searched in order. GH-hosted Windows runners put LLVM on PATH, and ship GNU
# objdump inside the Git for Windows tree.
OBJDUMP_CANDIDATES = [
    "llvm-objdump.exe",
    "llvm-objdump",
    "objdump.exe",
    "objdump",
]


def find_objdump(explicit=None):
    if explicit:
        return explicit

    # Env override, so this can be validated against a known-answer vector on
    # a machine that has neither LLVM nor a matching objdump.
    env = os.environ.get("AVX512_OBJDUMP")
    if env:
        return env

    from shutil import which
    for name in OBJDUMP_CANDIDATES:
        found = which(name)
        if found:
            return found

    # Common install locations that are not always on PATH.
    fixed = [
        r"C:\Program Files\LLVM\bin\llvm-objdump.exe",
        r"C:\Program Files\Git\mingw64\bin\objdump.exe",
        r"C:\Program Files\Git\usr\bin\objdump.exe",
    ]
    for path in fixed:
        if os.path.isfile(path):
            return path
    return None


def count_avx512(tool, binary):
    """Return (count, sample_lines) from a real disassembly of `binary`.

    Raises RuntimeError if the tool produced no disassembly at all, because
    "no output" must never be mistaken for "clean".
    """
    proc = subprocess.run(
        [tool] + DISASM_ARGS + [binary],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        errors="replace",
    )

    count = 0
    sample = []
    insn_lines = 0

    for line in proc.stdout.splitlines():
        if not INSN_LINE.match(line):
            continue
        insn_lines += 1
        if not AVX512_OPERAND.search(line):
            continue
        count += 1
        if len(sample) < 10:
            sample.append(line.rstrip())

    if insn_lines == 0:
        raise RuntimeError(
            "%s produced no disassembly for %s (exit %d). stderr: %s"
            % (tool, binary, proc.returncode, proc.stderr.strip()[:400])
        )
    return count, sample, insn_lines


def main(argv):
    explicit = None
    args = []
    i = 1
    while i < len(argv):
        if argv[i] == "--objdump" and i + 1 < len(argv):
            explicit = argv[i + 1]
            i += 2
            continue
        args.append(argv[i])
        i += 1

    if len(args) != 1:
        print(__doc__)
        return 2

    binary = args[0]
    if not os.path.isfile(binary):
        print("ERROR: no such file: %s" % binary, file=sys.stderr)
        return 2

    tool = find_objdump(explicit)
    if not tool:
        print("ERROR: no disassembler found; cannot verify.", file=sys.stderr)
        print("Tried: %s" % ", ".join(OBJDUMP_CANDIDATES), file=sys.stderr)
        return 2

    print("disassembler : %s" % tool)
    print("scanning     : %s" % binary)

    try:
        count, sample, insn_lines = count_avx512(tool, binary)
    except (RuntimeError, OSError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2

    print("instructions decoded : %d" % insn_lines)
    print("AVX-512 instructions : %d" % count)

    if count:
        print("")
        print("FAIL: this artifact contains AVX-512 instructions.")
        for line in sample:
            print("  %s" % line)
        if count > len(sample):
            print("  ... and %d more" % (count - len(sample)))
        print("")
        print("It will crash with STATUS_ILLEGAL_INSTRUCTION (0xc000001d) on any")
        print("CPU without AVX-512. Check that GGML_NATIVE is off (ggml passes")
        print("-march=native when it is on, so SOURCE_DATE_EPOCH must be set),")
        print("and that no prebuilt AVX-512 objects are being linked in.")
        return 1

    print("")
    print("PASS: no AVX-512 instructions in decoded code.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))