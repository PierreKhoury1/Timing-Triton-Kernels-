import os
import math
import numpy as np
import torch
import triton
import triton.language as tl

# ============================
# Helpers
# ============================
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

@triton.jit
def _read_clock64():
    (c,) = tl.inline_asm_elementwise(
        asm="mov.u64 $0, %clock64;",
        constraints="=l",
        args=[],
        dtype=(tl.uint64,),
        is_pure=False,
        pack=1,
    )
    return c

# ============================
# Kernel A: %globaltimer timing (ns), measuring mad.wide.u32
# ============================
@triton.jit
def debug_timer_globaltimer_kernel(
    x_ptr, t0_ptr, t1_ptr, tick_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr, REPS: tl.constexpr, K_TICK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N

    # 1) back-to-back %globaltimer probe
    t_prev = _read_globaltimer()
    min_d = tl.full([BLOCK], (1 << 63) - 1, tl.uint64)
    for _ in tl.static_range(K_TICK):
        t_now = _read_globaltimer()
        d = (t_now - t_prev).to(tl.uint64)
        t_prev = t_now
        nz = d > 0
        min_d = tl.where(nz & (d < min_d), d, min_d)
    tl.store(tick_ptr + offs, min_d, mask=m)

    # 2) arithmetic region: dependent mad.wide.u32 per lane
    # load a 64-bit accumulator per lane
    acc = tl.load(x_ptr + offs, mask=m, other=0).to(tl.uint64)

    # make 32-bit multiplicands (broadcasted)
    a = (offs + 1).to(tl.uint32)             # per-lane a
    b = tl.full([BLOCK], 17, tl.uint32)      # same b for all lanes

    t0 = _read_globaltimer()
    for _ in tl.static_range(REPS):
        # mad.wide.u32 dst64, src32_a, src32_b, src64_c;
        acc = tl.inline_asm_elementwise(
            "{ mad.wide.u32 $0, $1, $2, $3; }",
            constraints="=l,r,r,l",
            args=[a, b, acc],
            dtype=tl.uint64,
            is_pure=False,
            pack=1,
        )
    t1 = _read_globaltimer()

    tl.store(x_ptr + offs, acc, mask=m)
    tl.store(t0_ptr + offs, t0, mask=m)
    tl.store(t1_ptr + offs, t1, mask=m)


# ============================
# Kernel B: clock64() timing (cycles), with back-to-back probe
# ============================
@triton.jit
def debug_timer_clock64_kernel(
    x_ptr, c0_ptr, c1_ptr, clk_tick_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr, REPS: tl.constexpr, K_TICK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N

    # Back-to-back clock64() probe
    c_prev = _read_clock64()
    min_dc = tl.full([BLOCK], (1 << 63) - 1, tl.uint64)
    for _ in tl.static_range(K_TICK):
        c_now = _read_clock64()
        d = (c_now - c_prev).to(tl.uint64)
        c_prev = c_now
        nz = d > 0
        min_dc = tl.where(nz & (d < min_dc), d, min_dc)
    tl.store(clk_tick_ptr + offs, min_dc, mask=m)

    # Arithmetic-only bracket (keep your old fp32 add)
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    c0 = _read_clock64()
    for _ in tl.static_range(REPS):
        x = x + 1.0
    c1 = _read_clock64()
    tl.store(x_ptr + offs, x, mask=m)

    tl.store(c0_ptr + offs, c0, mask=m)
    tl.store(c1_ptr + offs, c1, mask=m)


# ============================
# Utils
# ============================
def _flatten_compiled_entries(cache_obj):
    found = []
    if hasattr(cache_obj, "asm") and isinstance(getattr(cache_obj, "asm"), dict):
        found.append(cache_obj); return found
    if isinstance(cache_obj, dict):
        for v in cache_obj.values(): found.extend(_flatten_compiled_entries(v))
    elif isinstance(cache_obj, (list, tuple)):
        for v in cache_obj: found.extend(_flatten_compiled_entries(v))
    return found

def _pick_best_for_device(compiled_entries):
    if not compiled_entries: return None
    for e in compiled_entries:
        asm = getattr(e, "asm", {})
        if isinstance(asm, dict) and asm.get("cubin") is not None:
            return e
    return compiled_entries[0]

def _summarize_tick(dt_ns: np.ndarray, label: str):
    dt_ns = dt_ns[(dt_ns > 0)]
    if dt_ns.size == 0:
        print(f"{label}: no non-zero samples"); return
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


# ============================
# Main
# ============================
def main():
    torch.cuda.set_device(0)
    dev = torch.cuda.current_device()
    name = torch.cuda.get_device_name(dev)
    cc = torch.cuda.get_device_capability(dev)
    sm = f"sm_{cc[0]}{cc[1]}"

    GHz = 1.680
    ns_per_cycle = 1.0 / GHz

    N = 32; BLOCK = 32
    REPS = 500
    K_TICK = 64
    grid = (1,)

    print(f"\nLaunching timers on: {name} ({sm})")
    print(f"Config: N={N}, BLOCK={BLOCK}, REPS={REPS}, K_TICK={K_TICK}, grid={grid}\n")

    # ---------- A) %globaltimer + mad.wide ----------
    # NOTE: xA must be uint64 because we store the 64-bit accumulator
    xA = torch.zeros(N, device="cuda", dtype=torch.uint64)
    t0 = torch.empty(N, device="cuda", dtype=torch.uint64)
    t1 = torch.empty(N, device="cuda", dtype=torch.uint64)
    tick_ns = torch.empty(N, device="cuda", dtype=torch.uint64)

    debug_timer_globaltimer_kernel[grid](
        xA, t0, t1, tick_ns,
        N=N, BLOCK=BLOCK, REPS=REPS, K_TICK=K_TICK,
        num_warps=1, num_stages=1
    )
    torch.cuda.synchronize()

    t0_cpu = t0.cpu().numpy()
    t1_cpu = t1.cpu().numpy()
    dt_ns = (t1_cpu - t0_cpu).astype(np.int64)
    tick_ns_cpu = tick_ns.cpu().numpy().astype(np.int64)

    print("=== %globaltimer (ns) — mad.wide.u32 ===")
    print(" lane |             t0 (ns)          |             t1 (ns)          |   Δt (ns) | tick(min)")
    print("------+------------------------------+------------------------------+-----------+----------")
    for lane in range(N):
        print(f" {lane:3d}  | {t0_cpu[lane]:>26d} | {t1_cpu[lane]:>26d} | {dt_ns[lane]:>8d} | {tick_ns_cpu[lane]:>8d}")

    mean_dt_ns = float(dt_ns.mean())
    print("\nSummary (%globaltimer region-under-test):")
    print(f"  mean duration: {mean_dt_ns:.2f} ns")
    print(f"  mean per iteration: {mean_dt_ns / REPS:.4f} ns/madwide")
    print(f"  quantization check (Δt mod 32): {int(mean_dt_ns) % 32} ns\n")
    _summarize_tick(tick_ns_cpu, "Back-to-back read tick probe (%globaltimer)")

    # export PTX/cubin if you want to disasm
    compiled_entries_A = _flatten_compiled_entries(debug_timer_globaltimer_kernel.cache)
    if compiled_entries_A:
        compiled_A = _pick_best_for_device(compiled_entries_A)
        asmA = getattr(compiled_A, "asm", {})
        ptxA = asmA.get("ptx", None)
        cubA = asmA.get("cubin", None)
        if ptxA:
            with open("debug_timer_globaltimer.ptx", "w") as f: f.write(ptxA)
        if cubA is not None:
            with open("debug_timer_globaltimer.cubin", "wb") as f: f.write(cubA)

    # ---------- B) clock64 test (your original) ----------
    xB = torch.zeros(N, device="cuda", dtype=torch.float32)
    c0 = torch.empty(N, device="cuda", dtype=torch.uint64)
    c1 = torch.empty(N, device="cuda", dtype=torch.uint64)
    tick_cyc = torch.empty(N, device="cuda", dtype=torch.uint64)

    debug_timer_clock64_kernel[grid](
        xB, c0, c1, tick_cyc,
        N=N, BLOCK=BLOCK, REPS=REPS, K_TICK=K_TICK,
        num_warps=1, num_stages=1
    )
    torch.cuda.synchronize()

    c0_cpu = c0.cpu().numpy().astype(np.int64)
    c1_cpu = c1.cpu().numpy().astype(np.int64)
    dc = (c1_cpu - c0_cpu).astype(np.int64)
    tick_cyc_cpu = tick_cyc.cpu().numpy().astype(np.int64)

    print("=== clock64() (cycles) ===")
    print(" lane |            c0 (cyc)          |            c1 (cyc)          |  Δcycles  | tick(min)")
    print("------+------------------------------+------------------------------+-----------+----------")
    for lane in range(N):
        print(f" {lane:3d}  | {c0_cpu[lane]:>26d} | {c1_cpu[lane]:>26d} | {dc[lane]:>8d} | {tick_cyc_cpu[lane]:>8d}")

    mean_dc = float(dc.mean())
    print("\nSummary (clock64 region-under-test):")
    print(f"  mean cycles: {mean_dc:.2f} cyc")
    print(f"  mean per iteration: {mean_dc / REPS:.4f} cyc/add")
    print(f"  (assuming {GHz:.3f} GHz) ⇒ {((mean_dc / REPS) * ns_per_cycle):.4f} ns/add\n")


if __name__ == "__main__":
    main()

