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
        is_pure=False,
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

    # ===== region under test =====
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    for _ in tl.static_range(REPS):
        x = x + 1.0
    tl.store(x_ptr + offs, x, mask=m)
    # ===== end region =====

    # --- end timing ---
    t1 = _read_globaltimer()
    tl.store(dt_ptr + pid, (t1 - t0).to(tl.uint64))

# -----------------------------
# Cache utilities
# -----------------------------
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

# -----------------------------
# Main
# -----------------------------
def main():
    torch.cuda.set_device(0)
    dev = torch.cuda.current_device()
    name = torch.cuda.get_device_name(dev)
    cc = torch.cuda.get_device_capability(dev)
    sm = f"sm_{cc[0]}{cc[1]}"
    #props = torch.cuda.get_device_properties(dev)
    # Handle both older and newer PyTorch attribute names
    #if hasattr(props, "clock_rate"):
   #     freq_ghz = props.clock_rate * 1e-6  # kHz → GHz
   # elif hasattr(props, "clockRate"):
  #      freq_ghz = props.clockRate * 1e-6  # kHz → GHz
  #  else: 
  #      raise AttributeError("Could not find clock rate attribute on device properties.")

    ns_per_cycle = 1e3 / 1.680
#
    N = 1024
    BLOCK = 32
    REPS = 1000
    grid = (1,)

    print(f"Launching kernel:")
    print(f"  GPU: {name}")
    print(f"  Compute Capability: {sm}")
    #print(f"  Base Clock: {freq_ghz:.3f} GHz  →  {ns_per_cycle:.3f} ns/cycle")
    print(f"  N={N}, BLOCK={BLOCK}, REPS={REPS}, grid={grid}\n")

    x = torch.randn(N, device="cuda", dtype=torch.float32)
    dt = torch.empty(grid, device="cuda", dtype=torch.uint64)

    toy_kernel[grid](x, dt, N=N, BLOCK=BLOCK, REPS=REPS, num_warps=1, num_stages=1)
    torch.cuda.synchronize()

    dtns = dt.cpu().numpy()
    dtns = dtns[dtns > 0]

    print("\n--- Timing Results ---")
    if dtns.size == 0:
        print("⚠️  No non-zero samples — region may be too short or profiler interfered.")
    else:
        total = len(dtns)
        print(f"Total blocks measured: {total}")
        for i, v in enumerate(dtns):
            print(f"  block {i:3d} → {v:8d} ns")
        mean_ns = dtns.mean()
        per_op = mean_ns / REPS
        per_cycle = per_op / ns_per_cycle

        print("\nSummary statistics:")
        print(f"  min    = {int(dtns.min()):>10d} ns")
        print(f"  median = {int(np.median(dtns)):>10d} ns")
        print(f"  max    = {int(dtns.max()):>10d} ns")
        print(f"  mean   = {mean_ns:>10.2f} ns")
        print(f"  std    = {dtns.std():>10.2f} ns")

        print(f"\nDerived metrics:")
        print(f"  (per iteration ≈ {per_op:.4f} ns per add)")
        print(f"  ≈ {per_cycle:.2f} GPU cycles per add")
        print(f"  Timer quantization check: Δt mod 32 = {int(mean_ns) % 32} ns\n")

    # ==========================================================
    # Export PTX & CUBIN from the JIT cache
    # ==========================================================
    print("\n--- Exporting PTX & CUBIN ---")
    compiled_entries = _flatten_compiled_entries(toy_kernel.cache)
    if not compiled_entries:
        raise RuntimeError("Could not find compiled entries in Triton cache.")
    compiled = _pick_best_for_device(compiled_entries)
    asm = getattr(compiled, "asm", {})
    ptx = asm.get("ptx", None)
    cubin = asm.get("cubin", None)

    with open("toy_kernel.ptx", "w") as f:
        f.write(ptx)
    if cubin is not None:
        with open("toy_kernel.cubin", "wb") as f:
            f.write(cubin)

    n_lines = ptx.count("\n")
    print(f"✅ Wrote toy_kernel.ptx ({n_lines} lines)" +
          (" and toy_kernel.cubin" if cubin is not None else ""))
    print("   Inspect via `grep -n globaltimer toy_kernel.ptx` or `nvdisasm --print-line-info toy_kernel.cubin`\n")

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    main()
