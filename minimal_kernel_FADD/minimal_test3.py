import os
import numpy as np
import torch
import triton
import triton.language as tl

# -----------------------------
# Helper: read device-wide global timer (ns)
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
# Kernel under test (adds +1 REPS times)
# -----------------------------
@triton.jit
def toy_kernel(x_ptr, dt_ptr, N: tl.constexpr, BLOCK: tl.constexpr, REPS: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N

    # --- start timing ---
    t0 = _read_globaltimer()

    # ===== region under test (amplified) =====
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    for _ in tl.static_range(REPS):
        x = x + 1.0
    tl.store(x_ptr + offs, x, mask=m)
    # ===== end region =====

    # --- end timing ---
    t1 = _read_globaltimer()

    # One u64 timing sample per program (block): scalar store, no mask
    tl.store(dt_ptr + pid, (t1 - t0).to(tl.uint64))


# -----------------------------
# Cache utilities
# -----------------------------
def _flatten_compiled_entries(cache_obj):
    """Recursively collect all compiled Triton kernel entries from the cache."""
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
    """Pick the compiled entry that contains PTX and CUBIN for the current device."""
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
    REPS = 1000  # adjust up/down for measurable duration
    grid = (1,) 
    #((N + BLOCK - 1) // BLOCK,)

    print(f"Launching kernel with:")
    print(f"  N={N}, BLOCK={BLOCK}, REPS={REPS}, grid={grid}\n")

    x = torch.randn(N, device="cuda", dtype=torch.float32)
    dt = torch.empty(grid, device="cuda", dtype=torch.uint64)

    # Run once to trigger JIT compilation + execution
    toy_kernel[grid](x, dt, N=N, BLOCK=BLOCK, REPS=REPS, num_warps=1, num_stages=1)
    torch.cuda.synchronize()

    # ------------------------------------------
    # Print verbose timing information
    # ------------------------------------------
    dtns = dt.cpu().numpy()
    dtns = dtns[dtns > 0]

    print("\n--- Timing Results ---")
    if dtns.size == 0:
        print("⚠️  No non-zero samples — region may be too short or profiler interfered.")
    else:
        print(f"Total blocks measured: {len(dtns)}")
        print(f"Globaltimer deltas (ns) per block:")
        for i, v in enumerate(dtns):
            print(f"  block {i:3d} → {v:8d} ns")

        print("\nSummary statistics:")
        print(f"  min   = {int(dtns.min()):>10d} ns")
        print(f"  median= {int(np.median(dtns)):>10d} ns")
        print(f"  max   = {int(dtns.max()):>10d} ns")
        print(f"  mean  = {dtns.mean():>10.2f} ns")
        print(f"  std   = {dtns.std():>10.2f} ns")
        print(f"  (per iteration ≈ {dtns.mean()/REPS:.4f} ns per add)\n")

    # ==========================================================
    # Export PTX & CUBIN from the JIT cache
    # ==========================================================
    print("\n--- Exporting PTX & CUBIN ---")
    dev = torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(dev)
    sm = f"sm_{cc[0]}{cc[1]}"
    print(f"GPU compute capability: {sm} on device {dev} ({torch.cuda.get_device_name(dev)})")

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
    print("   You can now inspect them via nvdisasm or grep for %globaltimer to verify placement.\n")


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    main()
