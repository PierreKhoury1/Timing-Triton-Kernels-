import os
import math
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
        is_pure=False,   # don't let the compiler CSE/reorder
        pack=1,
    )
    return t


# ==========================================================
# Debug kernel:
#   - capture per-lane start/end globaltimer values (t0/t1)
#   - perform a short back-to-back read loop to probe tick quantum
# ==========================================================
@triton.jit
def debug_timer_kernel(
    x_ptr, t0_ptr, t1_ptr, tick_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr, REPS: tl.constexpr, K_TICK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N

    # --- tick probe FIRST: back-to-back reads to find minimum non-zero Δt ---
    t_prev = _read_globaltimer()
    min_d = tl.full([BLOCK], (1 << 63) - 1, tl.uint64)
    for _ in tl.static_range(K_TICK):
        t_now = _read_globaltimer()
        d = (t_now - t_prev).to(tl.uint64)
        t_prev = t_now
        nz = d > 0
        min_d = tl.where(nz & (d < min_d), d, min_d)

    # store tick probe result per lane
    tl.store(tick_ptr + offs, min_d, mask=m)

    # ===== region under test (arithmetic only) =====
    # 1) Load BEFORE timing
    x = tl.load(x_ptr + offs, mask=m, other=0.0)

    # 2) START timer after the load
    t0 = _read_globaltimer()

    # 3) Dependent FADD chain
    for _ in tl.static_range(REPS):
        x = x + 1.0

    # 4) END timer before the store
    t1 = _read_globaltimer()

    # 5) Store AFTER timing
    tl.store(x_ptr + offs, x, mask=m)
    # ===== end region =====

    # store per-lane t0/t1
    tl.store(t0_ptr + offs, t0, mask=m)
    tl.store(t1_ptr + offs, t1, mask=m)


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
# Pretty printers
# ==========================================================
def _summarize_tick(dt_ns: np.ndarray, label: str):
    dt_ns = dt_ns[(dt_ns > 0)]
    if dt_ns.size == 0:
        print(f"{label}: no non-zero samples")
        return
    vals, counts = np.unique(dt_ns, return_counts=True)
    mode_val = int(vals[counts.argmax()])
    gcd_core = int(np.gcd.reduce(vals[: min(50, vals.size)].astype(np.int64)))
    print(f"{label}:")
    print(f"  samples: {dt_ns.size}")
    print(f"  min Δt: {int(dt_ns.min())} ns")
    print(f"  mode Δt: {mode_val} ns")
    print(f"  GCD-based quantum: {gcd_core} ns")
    status = "HIGH-RES" if gcd_core <= 64 else "DEFAULT"
    print(f"  status: {status}\n")


# ==========================================================
# Main
# ==========================================================
def main():
    torch.cuda.set_device(0)
    dev = torch.cuda.current_device()
    name = torch.cuda.get_device_name(dev)
    cc = torch.cuda.get_device_capability(dev)
    sm = f"sm_{cc[0]}{cc[1]}"

    # purely for cycles/add estimate
    ns_per_cycle = 1 / 1.680  # assume ~1.68 GHz (adjust to your board if needed)

    # config
    N = 32    # one warp
    BLOCK = 32
    REPS = 50
    K_TICK = 64  # short tick probe
    grid = (1,)

    print(f"\nLaunching debug_timer_kernel:")
    print(f"  GPU: {name}")
    print(f"  Compute Capability: {sm}")
    print(f"  N={N}, BLOCK={BLOCK}, REPS={REPS}, K_TICK={K_TICK}, grid={grid}\n")

    # buffers
    x = torch.zeros(N, device="cuda", dtype=torch.float32)
    t0 = torch.empty(N, device="cuda", dtype=torch.uint64)
    t1 = torch.empty(N, device="cuda", dtype=torch.uint64)
    tick = torch.empty(N, device="cuda", dtype=torch.uint64)

    # launch
    debug_timer_kernel[grid](
        x, t0, t1, tick,
        N=N, BLOCK=BLOCK, REPS=REPS, K_TICK=K_TICK,
        num_warps=1, num_stages=1
    )
    torch.cuda.synchronize()

    # host-side analysis
    t0_cpu = t0.cpu().numpy()
    t1_cpu = t1.cpu().numpy()
    dt = (t1_cpu - t0_cpu).astype(np.int64)
    tick_min = tick.cpu().numpy().astype(np.int64)

    # Per-lane table
    print("Per-lane %globaltimer readings (in ns):")
    print(" lane |          t0 (start)          |          t1 (end)            |   Δt (ns) | minΔ (tick)")
    print("------+------------------------------+------------------------------+-----------+------------")
    for lane in range(N):
        print(f" {lane:3d}  | {t0_cpu[lane]:>26d} | {t1_cpu[lane]:>26d} | {dt[lane]:>8d} | {tick_min[lane]:>10d}")

    # Summary for region-under-test duration
    mean_dt = float(dt.mean())
    print("\nSummary (region under test):")
    print(f"  mean duration: {mean_dt:.2f} ns")
    print(f"  mean per iteration: {mean_dt / REPS:.4f} ns/add")
    print(f"  ≈ {(mean_dt / REPS) / ns_per_cycle:.2f} cycles/add")
    print(f"  quantization check (Δt mod 32): {int(mean_dt) % 32} ns\n")

    # Tick quantum summary (this is the key part to classify DEFAULT vs HIGH-RES)
    _summarize_tick(tick_min, "Back-to-back read tick probe")

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
        with open("debug_timer_kernel2.ptx", "w") as f:
            f.write(ptx)
    if cubin is not None:
        with open("debug_timer_kernel2.cubin", "wb") as f:
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
