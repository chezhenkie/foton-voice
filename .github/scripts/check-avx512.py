#!/usr/bin/env python3
"""Fail the build if a Windows exe contains AVX-512 instructions.

Why this exists
---------------
A green build on a GitHub runner proves nothing about the CPU of the machine
that will run the artifact. The runner has AVX-512, so the toolchain emits
EVEX instructions into a binary meant for CPUs that do not have them. Such
an artifact dies on first model load with STATUS_ILLEGAL_INSTRUCTION
(0xc000001d). Observed: 5092 EVEX instructions in an artifact built for an
Intel Core Ultra 5 226V (Lunar Lake), which has no AVX-512.

How it works
------------
Walks the PE section table, reads .text, and decodes just far enough into
each instruction to recognise an EVEX prefix (0x62) or a zmm/k-register
operand. This is a detector, not a disassembler: it errs toward reporting.

Usage: check-avx512.py <path-to-exe>
Exit:  0 clean, 1 AVX-512 found, 2 could not verify.
"""

import struct
import sys

# EVEX is the only encoding that uses 0x62 as a mandatory prefix.
EVEX = 0x62

# One-byte opcodes whose ModRM.reg field is an AVX-512 k-mask register.
KMASK_REG_ONE_BYTE = frozenset({
    0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
    0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F,
})

# Two-byte opcodes (0x0f prefix) that are EVEX-only.
EVEX_ONLY_TWO_BYTE = frozenset({0x78})


def die(msg, code=2):
    print("ERROR: %s" % msg, file=sys.stderr)
    sys.exit(code)


def read_sections(data):
    if data[:2] != b"MZ":
        die("not a PE file (missing MZ)")
    pe_off = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe_off:pe_off + 4] != b"PE\x00\x00":
        die("bad PE signature")

    num_sections = struct.unpack_from("<H", data, pe_off + 6)[0]
    opt_size = struct.unpack_from("<H", data, pe_off + 20)[0]
    opt_off = pe_off + 24

    magic = struct.unpack_from("<H", data, opt_off)[0]
    if magic == 0x20B:
        image_base = struct.unpack_from("<Q", data, opt_off + 24)[0]
    elif magic == 0x10B:
        image_base = struct.unpack_from("<I", data, opt_off + 28)[0]
    else:
        die("unknown optional header magic 0x%04x" % magic)

    sec_off = opt_off + opt_size
    sections = []
    for i in range(num_sections):
        base = sec_off + i * 40
        name = data[base:base + 8].rstrip(b"\x00").decode("latin-1")
        vsize, vaddr, rsize, raddr = struct.unpack_from("<IIII", data, base + 8)
        sections.append((name, vaddr, vsize, raddr, rsize))
    return image_base, sections


def scan_text(data, sections, image_base):
    text = None
    for name, vaddr, vsize, raddr, rsize in sections:
        if name == ".text":
            text = (vaddr, rsize, raddr)
            break
    if text is None:
        die("no .text section found")

    vaddr, rsize, raddr = text
    blob = data[raddr:raddr + rsize]
    hits = []

    i = 0
    n = len(blob)
    while i < n:
        op = blob[i]

        # Legacy prefixes to skip when looking for a real opcode.
        j = i
        while j < n and blob[j] in (0x26, 0x2E, 0x36, 0x3E, 0x64, 0x65, 0x66, 0x67, 0xF0, 0xF2, 0xF3):
            j += 1
        if j >= n:
            break
        op = blob[j]

        if op == EVEX:
            rva = vaddr + i
            # Show the following bytes so the report is auditable.
            ctx = blob[i:i + 10].hex(" ")
            hits.append((rva, "EVEX prefix", ctx))
            i += 1
            continue

        if op == 0x0F and j + 1 < n:
            op2 = blob[j + 1]
            if op2 in EVEX_ONLY_TWO_BYTE:
                rva = vaddr + i
                hits.append((rva, "EVEX-only 0F %02X" % op2, blob[i:i + 10].hex(" ")))
                i = j + 2
                continue
            # 0F38 / 0F3A maps and VEX-encoded instructions use ModRM; check
            # for k-register or zmm operands in the map byte's reg field.
            if op2 in (0x38, 0x3A) and j + 2 < n:
                modrm = blob[j + 2]
                reg = (modrm >> 3) & 7
                if reg in (4, 5, 6, 7):
                    rva = vaddr + i
                    hits.append((rva, "0F %02X k-mask reg field" % op2, blob[i:i + 10].hex(" ")))
                    i = j + 3
                    continue
                i = j + 3
                continue
            i = j + 2
            continue

        # Single-byte opcode with ModRM: check reg field for k-mask.
        if j + 1 < n:
            modrm = blob[j + 1]
            if (modrm & 0xC0) != 0xC0 and modrm != 0:
                reg = (modrm >> 3) & 7
                if reg in (4, 5, 6, 7):
                    # Only treat as suspicious when a prefix or the opcode
                    # suggests vector encoding; otherwise this is far too
                    # noisy to be useful (plain "add rsp,8" has reg==4).
                    before = blob[i:j]
                    has_vec_prefix = any(b in (0x66, 0xF2, 0xF3) for b in before)
                    if has_vec_prefix or op in (0xC4, 0xC5):
                        rva = vaddr + i
                        hits.append((rva, "k-mask reg field", blob[i:i + 10].hex(" ")))
                        i = j + 2
                        continue
            i = j + 1
            continue

        i = j + 1

    return hits, len(blob)


def main():
    if len(sys.argv) != 2:
        print("usage: check-avx512.py <path-to-exe>", file=sys.stderr)
        sys.exit(2)

    path = sys.argv[1]
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        die("cannot read %s: %s" % (path, e))

    image_base, sections = read_sections(data)
    hits, scanned = scan_text(data, sections, image_base)

    print("scanning %s" % path)
    print("image base : 0x%x" % image_base)
    print(".text bytes: %d" % scanned)
    print("AVX-512 sites: %d" % len(hits))

    if hits:
        print("")
        print("FAIL: this artifact contains AVX-512 instructions.")
        for rva, why, ctx in hits[:15]:
            print("  RVA 0x%08x (VA 0x%016x)  %-24s  %s" % (rva, image_base + rva, why, ctx))
        if len(hits) > 15:
            print("  ... and %d more" % (len(hits) - 15))
        print("")
        print("It will crash with STATUS_ILLEGAL_INSTRUCTION (0xc000001d) on any")
        print("CPU without AVX-512. Ensure RUSTFLAGS disables the avx512 features")
        print("and that C sources are built with the same restriction.")
        sys.exit(1)

    print("OK: no AVX-512 instructions. Portable to AVX-512-less CPUs.")
    sys.exit(0)


if __name__ == "__main__":
    main()