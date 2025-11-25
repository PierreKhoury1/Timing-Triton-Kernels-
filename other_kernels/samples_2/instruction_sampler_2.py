import triton
import triton.language as tl
import torch
import subprocess
import re

# Helper functions
def flatten(cache):
    out = []
    if hasattr(cache, "asm"): return [cache]
    if isinstance(cache, dict):
        for v in cache.values(): out.extend(flatten(v))
    elif isinstance(cache, (list, tuple)):
        for v in cache: out.extend(flatten(v))
    return out

def pick(entries):
    for e in entries:
        if "cubin" in e.asm: return e
    return entries[0]

# Annotation
MAP = {
    r"POPC": "// popcount",
    r"BREV": "// bit reverse",
    r"IMAD": "// integer mad (SASS)",
    r"MUFU": "// special func",
    r"FFMA": "// fp32 fma",
    r"FADD": "// fp32 add",
    r"FMUL": "// fp32 mul",
    r"HMMA": "// tensor core mma"
}

def annotate(text):
    out = []
    for line in text.splitlines():
        added = False
        for pat, tag in MAP.items():
            if re.search(pat, line):
                out.append(f"{line:<120} {tag}")
                added = True
                break
        if not added:
            out.append(line)
    return "\n".join(out)

# ======================================================================
# Triton Kernel (PTX-correct inline assembly)
# ======================================================================
@triton.jit
def instruction_sampler(x_ptr, y_ptr, out_ptr,
                        BLOCK: tl.constexpr, REPS: tl.constexpr):

    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    x = tl.load(x_ptr + offs)
    xi = x.to(tl.int32)

    # ========== POPC chain ==========
    for _ in range(REPS):
        (xi,) = tl.inline_asm_elementwise(
            asm="popc.b32 $0, $1;",
            constraints="=r,r",
            args=[xi],
            dtype=(tl.int32,),
            is_pure=False,
            pack=1
        )

    # ========== BREV ==========
    for _ in range(REPS):
        (xi,) = tl.inline_asm_elementwise(
            asm="brev.b32 $0, $1;",
            constraints="=r,r",
            args=[xi],
            dtype=(tl.int32,),
            is_pure=False,
            pack=1
        )

    # ========== IMAD (PTX mad.lo.s32) ==========
    yi = (x * 13).to(tl.int32)
    for _ in range(REPS):
        (xi,) = tl.inline_asm_elementwise(
            asm="mad.lo.s32 $0, $1, 7, $2;",
            constraints="=r,r,r",
            args=[xi, yi],
            dtype=(tl.int32,),
            is_pure=False,
            pack=1
        )

    # ========== MUFU (PTX special functions) ==========
    xf = x
    for _ in range(REPS):
        (xf,) = tl.inline_asm_elementwise(
            asm="lg2.approx.f32 $0, $1;",
            constraints="=f,f",
            args=[xf],
            dtype=(tl.float32,),
            is_pure=False,
            pack=1
        )

    for _ in range(REPS):
        (xf,) = tl.inline_asm_elementwise(
            asm="rcp.approx.f32 $0, $1;",
            constraints="=f,f",
            args=[xf],
            dtype=(tl.float32,),
            is_pure=False,
            pack=1
        )

    for _ in range(REPS):
        (xf,) = tl.inline_asm_elementwise(
            asm="sqrt.approx.f32 $0, $1;",
            constraints="=f,f",
            args=[xf],
            dtype=(tl.float32,),
            is_pure=False,
            pack=1
        )

    tl.store(out_ptr + offs, xf + xi)


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    BLOCK = 128
    REPS = 4
    x = torch.randint(1, 200, (BLOCK,), device="cuda", dtype=torch.float32)
    y = torch.zeros_like(x)
    out = torch.zeros_like(x)

    print("Running kernel...")
    instruction_sampler[(1,)](
        x, y, out,
        BLOCK=BLOCK, REPS=REPS,
        num_warps=1, num_stages=1
    )
    torch.cuda.synchronize()

    # extract ASM
    entries = flatten(instruction_sampler.cache)
    obj = pick(entries)
    asm = obj.asm

    open("i.ptx", "w").write(asm["ptx"])
    open("i.cubin", "wb").write(asm["cubin"])

    sass = subprocess.check_output(["nvdisasm", "i.cubin"], text=True)
    open("i.sass", "w").write(sass)
    open("i_annotated.sass", "w").write(annotate(sass))

    print("✓ PTX, CUBIN, SASS, annotated SASS written.")
