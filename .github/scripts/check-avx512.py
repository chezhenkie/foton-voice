#!/usr/bin/env python3
"""Fail the build if a Windows exe contains AVX-512 *code*.

Why this exists
---------------
A green build on a GitHub runner proves nothing about the CPU that will run the
artifact. The runner has AVX-512, so the toolchain can emit EVEX instructions
into a binary meant for CPUs that lack them, and the artifact then dies on
first model load with STATUS_ILLEGAL_INSTRUCTION (0xc000001d). Observed: 5092
AVX-512 instructions in an artifact destined for an Intel Core Ultra 5 226V
(Lunar Lake).

Three earlier versions of this gate were wrong, each caught by validating
against a known-bad artifact before trusting it:

1. It scanned raw bytes for the EVEX prefix 0x62 with no notion of instruction
   boundaries. On a known-bad artifact it reported 95242 sites where real
   disassembly found 5092, so ~95% were data or misaligned bytes.

2. It counted every decoded zmm/k-mask instruction in .text. Accurate on code,
   but a linear sweep also decodes static data that lives inside .text, and it
   reported 279 hits on an artifact that verifiably runs correctly.

3. It tried to exclude non-code using .pdata unwind coverage. ggml's AVX-512
   kernels are leaf functions, so MSVC emits no unwind info for them, and the
   gate PASSED the known-bad artifact. Unusable.

This version attributes every hit to a real symbol using the MSVC linker map
(enabled with /MAP), then asks whether that symbol is code or data. The
residue in artifact 1c1e47c was attributed to ggml's static data: PULP DSP
pipeline descriptor tables (libpulp_*), MSVC 16-byte constant pools
(__xmm@...), and static initializers (?x$initializer$@@3...). MSVC marks data
symbols with @@3, and none of it is ever executed.

Without a map the gate falls back to strict mode and fails on any hit, so a
missing map can never silently weaken the check.

Usage:
  check-avx512.py <exe> [--map <linker.map>] [--objdump <path>]

Exit: 0 clean, 1 AVX-512 code found, 2 could not verify.
"""

import bisect
import os
import re
import struct
import subprocess
import sys

AVX512_OPERAND = re.compile(r"%zmm[0-9]|(?<![\w])%k[0-7]")
ADDRESSED = re.compile(r"^\s*([0-9a-fA-F]+):")
# Linker map entry with an explicit absolute VA column:
#   "  seg:off  name  absolute_va  object"
MAP_ENTRY_VA = re.compile(
    r"^\s*([0-9A-Fa-f]{4}):([0-9A-Fa-f]{8})\s+(\S+)\s+([0-9A-Fa-f]{16})(?:\s+(\S+))?\s*$")
# Same shape without the VA column, as printed in the symbol listings.
MAP_ENTRY_NO_VA = re.compile(
    r"^\s*([0-9A-Fa-f]{4}):([0-9A-Fa-f]{8})\s+(\S+)\s*$")

DISASM_ARGS = ["--disassemble", "--no-show-raw-insn"]

OBJDUMP_CANDIDATES = [
    "llvm-objdump.exe", "llvm-objdump", "objdump.exe", "objdump",
]

# Symbols whose contents are data, not instructions. Each pattern is justified
# by an observed attribution, not guessed:
#   @@3            MSVC's marker for a static data symbol
#   $initializer$  static-initialiser data emitted by the compiler
#   __xmm@         MSVC's name for a 16-byte aligned read-only constant pool
#   libpulp_       ggml's PULP DSP pipeline descriptor tables
#   $LN<digits>    compiler-generated local label for a COMDAT literal pool or
#                  jump table. The single hit carrying this label was verified
#                  by dumping raw bytes: onigmo's encoding-length table in
#                  regparse.o, a run of 0x0e/0x0f entries, not instructions.
DATA_SYMBOL = re.compile(r"@@3|\$initializer\$|^__xmm@|^libpulp_|^\$LN\d+$")


def find_objdump(explicit=None):
    if explicit:
        return explicit
    env = os.environ.get("AVX512_OBJDUMP")
    if env:
        return env
    from shutil import which
    for name in OBJDUMP_CANDIDATES:
        found = which(name)
        if found:
            return found
    for path in [
        r"C:\Program Files\LLVM\bin\llvm-objdump.exe",
        r"C:\Program Files\Git\mingw64\bin\objdump.exe",
        r"C:\Program Files\Git\usr\bin\objdump.exe",
    ]:
        if os.path.isfile(path):
            return path
    return None


def image_base(path):
    with open(path, "rb") as f:
        d = f.read(4096)
    pe = struct.unpack_from("<I", d, 0x3C)[0]
    opt = pe + 24
    magic = struct.unpack_from("<H", d, opt)[0]
    off = opt + (24 if magic == 0x20B else 24)
    return struct.unpack_from("<Q", d, off)[0]


def load_map(path, base):
    """Return (sorted_vas, names, objects) from an MSVC linker map.

    Map offsets are per segment (0001 is .text, 0002 .rdata, 0003 .data), so
    an offset alone is meaningless. Entries that carry an absolute VA column
    reveal each segment's base as VA minus offset; entries without one are then
    resolved through that. Entries with no derivable base are dropped rather
    than guessed at.
    """
    seg_base = {}
    rows = []
    pending = []

    with open(path, "r", errors="replace") as f:
        for line in f:
            m = MAP_ENTRY_VA.match(line)
            if m:
                seg = int(m.group(1), 16)
                off = int(m.group(2), 16)
                va = int(m.group(4), 16)
                if seg not in seg_base or va - off < seg_base[seg]:
                    seg_base[seg] = va - off
                rows.append((seg, off, va, m.group(3), m.group(5) or ""))
                continue
            m = MAP_ENTRY_NO_VA.match(line)
            if m:
                pending.append((int(m.group(1), 16), int(m.group(2), 16), m.group(3)))

    for seg, off, name in pending:
        if seg in seg_base:
            rows.append((seg, off, seg_base[seg] + off, name, ""))

    rows.sort(key=lambda r: r[2])
    return ([r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows])


def scan(tool, binary):
    proc = subprocess.run(
        [tool] + DISASM_ARGS + [binary],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True, errors="replace")
    hits = []
    insns = 0
    for line in proc.stdout.splitlines():
        m = ADDRESSED.match(line)
        if not m:
            continue
        insns += 1
        if AVX512_OPERAND.search(line):
            hits.append((int(m.group(1), 16), line.rstrip()))
    if insns == 0:
        raise RuntimeError(
            "%s produced no disassembly for %s (exit %d). stderr: %s"
            % (tool, binary, proc.returncode, proc.stderr.strip()[:400]))
    return hits, insns


def main(argv):
    explicit = None
    map_path = None
    args = []
    i = 1
    while i < len(argv):
        if argv[i] == "--objdump" and i + 1 < len(argv):
            explicit = argv[i + 1]; i += 2; continue
        if argv[i] == "--map" and i + 1 < len(argv):
            map_path = argv[i + 1]; i += 2; continue
        args.append(argv[i]); i += 1

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
        return 2

    base = image_base(binary)
    print("disassembler : %s" % tool)
    print("scanning     : %s" % binary)
    print("image base   : 0x%x" % base)

    try:
        hits, insns = scan(tool, binary)
    except (RuntimeError, OSError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2

    print("instructions decoded: %d" % insns)
    print("AVX-512 hits        : %d" % len(hits))

    if not map_path:
        if hits:
            print("")
            print("FAIL: %d AVX-512 instructions and no linker map to tell code" % len(hits))
            print("      from data. Run without --map only when you accept that.")
            for _a, l in hits[:5]:
                print("  %s" % l)
            return 1
        print("")
        print("PASS: no AVX-512 instructions.")
        return 0

    if not os.path.isfile(map_path):
        print("ERROR: map not found: %s" % map_path, file=sys.stderr)
        print("       Refusing to fall back to strict mode silently.", file=sys.stderr)
        return 2

    try:
        vas, names, objs = load_map(map_path, base)
    except OSError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2

    print("map symbols         : %d" % len(vas))

    code_hits, data_hits, unknown = [], [], []
    for addr, line in hits:
        # objdump prints full VAs for PE images, but normalise anyway so a
        # tool that printed a bare RVA still resolves correctly. Adding the
        # image base to an address that already carries it was a real bug that
        # pushed every lookup past the end of the symbol table.
        va = addr if addr >= base else base + addr
        i = bisect.bisect_right(vas, va) - 1
        if i < 0:
            unknown.append((va, "(below every symbol)", "", line))
            continue
        name, obj = names[i], objs[i]
        rec = (va, name, obj, line)
        if DATA_SYMBOL.search(name):
            data_hits.append(rec)
        else:
            code_hits.append(rec)

    print("  in code : %d" % len(code_hits))
    print("  in data : %d" % len(data_hits))
    print("  unknown : %d" % len(unknown))

    if data_hits:
        seen = []
        for _va, name, obj, _l in data_hits:
            key = (name, obj)
            if key not in seen:
                seen.append(key)
        print("")
        print("data symbols holding AVX-512-looking bytes (never executed):")
        for name, obj in seen[:8]:
            print("  %-60s %s" % (name[:60], obj[:70]))

    if code_hits:
        print("")
        print("FAIL: %d AVX-512 instructions inside code symbols." % len(code_hits))
        for va, name, obj, line in code_hits[:10]:
            print("  VA 0x%x  %s" % (va, line))
            print("        symbol: %s" % name)
            print("        object: %s" % obj)
        if len(code_hits) > 10:
            print("  ... and %d more" % (len(code_hits) - 10))
        print("")
        print("This artifact will crash with STATUS_ILLEGAL_INSTRUCTION")
        print("(0xc000001d) on any CPU without AVX-512. Check that GGML_NATIVE")
        print("is off (ggml passes -march=native when it is on, so")
        print("SOURCE_DATE_EPOCH must be set) and that no prebuilt AVX-512")
        print("objects are linked in.")
        return 1

    if unknown:
        print("")
        print("NOTE: %d hits could not be attributed to any symbol." % len(unknown))
        for va, name, obj, line in unknown[:5]:
            print("  VA 0x%x  %s" % (va, line))

    print("")
    print("PASS: no AVX-512 in any code symbol.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))