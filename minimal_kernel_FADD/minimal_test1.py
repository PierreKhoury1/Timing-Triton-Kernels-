import os
import numpy as np
import torch
import triton
import triton.language as tl

# -----------------------------
# Helper: read device-wide global timer
# -----------------------------
@triton.jit
def _read_globaltimer():
    (t,) = tl.inline_asm_elementwise(
        asm="mov.u64 $0, %globaltimer;",
        constraints="=l",
        args=[],
        dtype=(tl.uint64,),
        is_pure=False,  # prevents reordering
        pack=1,
    )
    return t

# -----------------------------
# Simple kernel under test
# -----------------------------
@triton.jit
def toy_kernel(x_ptr, dt_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N

    # --- start timing ---
    t0 = _read_globaltimer()

    # region under test
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    y = x + 1.0
    tl.store(x_ptr + offs, y, mask=m)
    # end region

    # --- end timing ---
    t1 = _read_globaltimer()
    tl.store(dt_ptr + pid, (t1 - t0).to(tl.uint64))

# -----------------------------
# Cache utilities
# -----------------------------
def _flatten_compiled_entries(cache_obj):
    """
    Recursively walk Triton kernel cache and collect compiled entries that
    expose an `.asm` dict with 'ptx' and optionally 'cubin'.
    Works across Triton versions.
    """
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

def _pick_best_for_device(compiled_entries):
    """
    If multiple compiled variants exist (e.g., different SMs/specializations),
    pick the first one that has a CUBIN; otherwise fall back to the first entry.
    """
    if not compiled_entries:
        return None
    for e in compiled_entries:
        asm = getattr(e, "asm", {})
        if isinstance(asm, dict) and asm.get("cubin") is not None:
            return e
    return compiled_entries[0]

# -----------------------------
# Main
# -----------------------------
def main():
    torch.cuda.set_device(0)
    N = 1024
    BLOCK = 32
    grid = ((N + BLOCK - 1) // BLOCK,)

    x = torch.randn(N, device="cuda", dtype=torch.float32)
    dt = torch.empty(grid, device="cuda", dtype=torch.uint64)

    # Run once to JIT-compile + execute
    toy_kernel[grid](x, dt, N=N, BLOCK=BLOCK, num_warps=1, num_stages=1)
    torch.cuda.synchronize()

    # Print timing summary
    dtns = dt.cpu().numpy()
    dtns = dtns[dtns > 0]
    if dtns.size == 0:
        print("⚠️  No non-zero samples — likely because Nsight Compute replayed the kernel.")
    else:
        print(f"per-block Δt (ns): min/med/max = {int(dtns.min())} {int(np.median(dtns))} {int(dtns.max())}")

    # ==========================================================
    # Export PTX & CUBIN from the JIT cache (no triton.compiler.compile needed)
    # ==========================================================
    print("\n--- Exporting PTX & CUBIN ---")
    dev = torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(dev)
    sm = f"sm_{cc[0]}{cc[1]}"
    print(f"GPU compute capability: {sm} on device {dev} ({torch.cuda.get_device_name(dev)})")

    # Gather compiled artifacts from the kernel's cache
    compiled_entries = _flatten_compiled_entries(toy_kernel.cache)
    if not compiled_entries:
        raise RuntimeError(
            "Could not find compiled entries in Triton cache. "
            "Ensure the kernel was launched at least once."
        )

    compiled = _pick_best_for_device(compiled_entries)
    asm = getattr(compiled, "asm", {})
    ptx = asm.get("ptx", None)
    cubin = asm.get("cubin", None)

    if not ptx:
        raise RuntimeError("PTX not found in compiled artifact. Triton may have changed its internals.")

    with open("toy_kernel.ptx", "w") as f:
        f.write(ptx)
    if cubin is not None:
        with open("toy_kernel.cubin", "wb") as f:
            f.write(cubin)

    print("✅ Wrote toy_kernel.ptx" + (" and toy_kernel.cubin" if cubin is not None else ""))

if __name__ == "__main__":
    main()
