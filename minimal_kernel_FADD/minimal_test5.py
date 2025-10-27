import os
import numpy as np
import torch
import triton
import triton.language as tl

# ==========================================================
# Helper: read device-wide global timer (ns)
# ==========================================================
@triton.jit
def _read_globaltimer():
    (t,) = tl.inline_asm_elementwise(
        asm="mov.u64 $0, %globaltimer;",
        constraints="=l",
        args=[],
        dtype=(tl.uint64,),
        is_pure=False,
        pack=1,
    )
    return t


# ==========================================================
# Debug kernel: capture per-lane start/end globaltimer values
# ==========================================================
@triton.jit
def debug_timer_kernel(x_ptr, t0_ptr, t1_ptr, N: tl.constexpr, BLOCK: tl.constexpr, REPS: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N

    # --- start timing (per lane) ---
    t0 = _read_globaltimer()

    # ===== region under test =====
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    for _ in tl.static_range(REPS):
        x = x + 1.0
    tl.store(x_ptr + offs, x, mask=m)
    # ===== end region =====

    # --- end timing (per lane) ---
    t1 = _read_globaltimer()

    # store each lane's t0 and t1 individually
    tl.store(t0_ptr + offs, t0)
    tl.store(t1_ptr + offs, t1)


# ==========================================================
# Utility: flatten Triton JIT cache to get PTX/CUBIN
# ==========================================================
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


def _pick_best_for_device(compiled_entries):
    if not compiled_entries:
        return None
    for e in compiled_entries:
        asm = getattr(e, "asm", {})
        if isinstance(asm, dict) and asm.get("cubin") is not None:
            return e
    return compiled_entries[0]


# ==========================================================
# Main
# ==========================================================
def main():
    torch.cuda.set_device(0)
    dev = torch.cuda.current_device()
    name = torch.cuda.get_device_name(dev)
    cc = torch.cuda.get_device_capability(dev)
    sm = f"sm_{cc[0]}{cc[1]}"

    ns_per_cycle = 1 / 1.680  # assuming 1.68 GHz base clock

    N = 32  # one warp
    BLOCK = 32
    REPS = 1000
    grid = (1,)

    print(f"\nLaunching debug_timer_kernel:")
    print(f"  GPU: {name}")
    print(f"  Compute Capability: {sm}")
    print(f"  N={N}, BLOCK={BLOCK}, REPS={REPS}, grid={grid}\n")

    # allocate buffers
    x = torch.zeros(N, device="cuda", dtype=torch.float32)
    t0 = torch.empty(N, device="cuda", dtype=torch.uint64)
    t1 = torch.empty(N, device="cuda", dtype=torch.uint64)

    # launch kernel
    debug_timer_kernel[grid](x, t0, t1, N=N, BLOCK=BLOCK, REPS=REPS, num_warps=1, num_stages=1)
    torch.cuda.synchronize()

    # move to host and compute differences
    t0_cpu = t0.cpu().numpy()
    t1_cpu = t1.cpu().numpy()
    dt = t1_cpu - t0_cpu

    print("Per-lane %globaltimer readings (in ns):")
    print(" lane |          t0 (start)          |          t1 (end)            |   Δt (ns)")
    print("------+------------------------------+------------------------------+-----------")
    for lane in range(N):
        print(f" {lane:3d}  | {t0_cpu[lane]:>26d} | {t1_cpu[lane]:>26d} | {dt[lane]:>8d}")

    mean_dt = dt.mean()
    print("\nSummary:")
    print(f"  mean duration: {mean_dt:.2f} ns")
    print(f"  mean per iteration: {mean_dt / REPS:.4f} ns/add")
    print(f"  ≈ {(mean_dt / REPS) / ns_per_cycle:.2f} cycles/add")
    print(f"  quantization check (Δt mod 32): {int(mean_dt) % 32} ns\n")

    # ==========================================================
    # Export PTX & CUBIN from the JIT cache
    # ==========================================================
    print("--- Exporting PTX & CUBIN ---")
    compiled_entries = _flatten_compiled_entries(debug_timer_kernel.cache)
    if not compiled_entries:
        raise RuntimeError("Could not find compiled entries in Triton cache.")
    compiled = _pick_best_for_device(compiled_entries)
    asm = getattr(compiled, "asm", {})
    ptx = asm.get("ptx", None)
    cubin = asm.get("cubin", None)

    if ptx:
        with open("debug_timer_kernel.ptx", "w") as f:
            f.write(ptx)
    if cubin is not None:
        with open("debug_timer_kernel.cubin", "wb") as f:
            f.write(cubin)

    n_lines = ptx.count("\n") if ptx else 0
    print(f"✅ Wrote debug_timer_kernel.ptx ({n_lines} lines)"
          + (" and debug_timer_kernel.cubin" if cubin is not None else ""))
    print("   Inspect via `grep -n globaltimer debug_timer_kernel.ptx` "
          "or `nvdisasm --print-line-info debug_timer_kernel.cubin`\n")


# ==========================================================
# Entrypoint
# ==========================================================
if __name__ == "__main__":
    main()
