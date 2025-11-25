import triton
import triton.language as tl
import torch
import subprocess
import tempfile
import os
import re


# ============================================================================
# Helpers
# ============================================================================
def _flatten_compiled_entries(cache_obj):
    found = []
    if hasattr(cache_obj, "asm") and isinstance(getattr(cache_obj, "asm"), dict):
        found.append(cache_obj)
        return found

    if isinstance(cache_obj, dict):
        for v in cache_obj.values():
            found.extend(_flatten_compiled_entries(v))
    elif isinstance(cache_obj, (list, tuple)):
        for v in cache_obj:
            found.extend(_flatten_compiled_entries(v))

    return found


def _pick_best_for_device(entries):
    for e in entries:
        asm = getattr(e, "asm", {})
        if asm.get("cubin") is not None:
            return e
    return entries[0] if entries else None


# ============================================================================
# SASS Annotation
# ============================================================================
ANNOTATION_MAP = {
    r"LDG\.E": "// load (tl.load)",
    r"STG\.E": "// store (tl.store)",
    r"FADD": "// FP add  (a = x + y)",
    r"FMUL": "// FP multiply (b = x * y)",
    r"FFMA": "// fused multiply-add (c = a*y + x)",
    r"FMNMX": "// max(a,b)",
    r"FSEL": "// select (tl.where)",
    r"FSETP": "// compare predicate (a > b)",
    r"MUFU\.EX2": "// exp2 / exp approximation",
    r"HADD2": "// FP16 add",
    r"HFMA2": "// FP16 fma",
    r"F2FP": "// FP32 → FP16 convert",
    r"IMAD": "// integer multiply-add (index calc)",
    r"LOP3": "// bitwise logic (LOP3)",
    r"SHF": "// shift",
    r"SHR": "// shift-right",
    r"LEA": "// address calc",
    r"S2R": "// read special register (thread id / ctaid)",
    r"MOV": "// move / constant load",
}


def annotate_sass(sass_text):
    annotated = []
    for line in sass_text.splitlines():
        added_comment = False
        for pattern, comment in ANNOTATION_MAP.items():
            if re.search(pattern, line):
                annotated.append(f"{line:<80} {comment}")
                added_comment = True
                break
        if not added_comment:
            annotated.append(line)
    return "\n".join(annotated)


# ============================================================================
# Kernel
# ============================================================================
@triton.jit
def instruction_sampler_kernel(
    x_ptr, y_ptr, out_ptr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)

    # Loads
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)

    # FP ops
    a = x + y
    b = x * y
    c = a * y + x
    d = tl.maximum(a, b)
    e = tl.where(a > b, a, b)
    f = tl.exp(a)

    # FP16
    x16 = x.to(tl.float16)
    y16 = y.to(tl.float16)
    f16 = x16 * y16 + x16

    # INT ops
    xi = x.to(tl.int32)
    yi = y.to(tl.int32)

    addi = xi + yi
    add3 = xi + yi + 1
    muli = xi * yi
    imnmx = tl.maximum(xi, yi)
    shl = xi << 2
    shr = xi >> 1

    # POPCOUNT SWAR pattern
    t = xi - ((xi >> 1) & 0x55555555)
    t = (t & 0x33333333) + ((t >> 2) & 0x33333333)
    t = (t + (t >> 4)) & 0x0F0F0F0F
    popc = (t * 0x01010101) >> 24

    # LOP3-like
    lop3 = (xi & yi) ^ (~xi)

    # Conversions
    xf2i = x.to(tl.int32)
    xi2f = xi.to(tl.float32)
    f2f = x.to(tl.float16).to(tl.float32)

    # dot-like reduction
    dot_val = tl.sum(x16 * y16)

    # predication
    pred = xi > yi
    pred_sel = tl.where(pred, xi, yi)

    # store
    out = (a + b + c + d + e + f) + f16.to(tl.float32)
    tl.store(out_ptr + offs, out)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    BLOCK = 128
    N = BLOCK

    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    out = torch.zeros(N, device="cuda")

    print("Running Triton 3.2 kernel…")
    instruction_sampler_kernel[(1,)](x, y, out, BLOCK=BLOCK)
    torch.cuda.synchronize()
    print("Kernel executed.\n")

    print("Extracting PTX + CUBIN...")

    entries = _flatten_compiled_entries(instruction_sampler_kernel.cache)
    compiled = _pick_best_for_device(entries)

    asm = compiled.asm
    ptx = asm.get("ptx")
    cubin = asm.get("cubin")

    # --- Write PTX ---
    with open("instruction_sampler.ptx", "w") as f:
        f.write(ptx or "")
    print("✓ Wrote instruction_sampler.ptx")

    sass_raw = ""
    if cubin is not None:
        # --- Write raw cubin ---
        with open("instruction_sampler.cubin", "wb") as f:
            f.write(cubin)
        print("✓ Wrote instruction_sampler.cubin")

        # --- Disassemble ---
        sass_raw = subprocess.check_output(["nvdisasm", "instruction_sampler.cubin"], text=True)
        with open("instruction_sampler.sass", "w") as f:
            f.write(sass_raw)
        print("✓ Wrote instruction_sampler.sass (raw)")

        # --- Annotate SASS ---
        annotated = annotate_sass(sass_raw)
        with open("instruction_sampler_annotated.sass", "w") as f:
            f.write(annotated)
        print("✓ Wrote instruction_sampler_annotated.sass (annotated!)")

    print("\nDone.")
