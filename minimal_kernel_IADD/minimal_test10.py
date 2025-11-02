import csv
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
        is_pure=False,    # side-effecting: don't hoist/CSE
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
        is_pure=False,    # side-effecting for placement stability
        pack=1,
    )
    return c

# ============================
# Kernel A: %globaltimer timing (ns), with tick probe
# ============================
@triton.jit
def debug_timer_globaltimer_kernel(
    x_ptr, t0_ptr, t1_ptr, tick_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr, REPS: tl.constexpr, K_TICK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N

    # Tick probe (back-to-back %globaltimer reads)
    t_prev = _read_globaltimer()
    min_d = tl.full([BLOCK], (1 << 63) - 1, tl.uint64)
    for _ in tl.static_range(K_TICK):
        t_now = _read_globaltimer()
        d = (t_now - t_prev).to(tl.uint64)
        t_prev = t_now
        nz = d > 0
        min_d = tl.where(nz & (d < min_d), d, min_d)
    tl.store(tick_ptr + offs, min_d, mask=m)

    # Arithmetic-only bracket (dependent adds)
    x = tl.load(x_ptr + offs, mask=m, other=0).to(tl.int32)
    t0 = _read_globaltimer()
    for _ in tl.static_range(REPS):
        (x,) = tl.inline_asm_elementwise(
            asm="add.cc.u32 $0, $1, 7;",
            constraints="=r,r",
            args=[x],
            dtype=(tl.int32,),
            is_pure=False,
            pack=1,
        )
        (x,) = tl.inline_asm_elementwise(
            asm="addc.u32 $0, $1, 0;",
            constraints="=r,r",
            args=[x],
            dtype=(tl.int32,),
            is_pure=False,
            pack=1,
        )
    t1 = _read_globaltimer()
    tl.store(x_ptr + offs, x, mask=m)

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

    # Back-to-back clock64() probe (min non-zero cycles)
    c_prev = _read_clock64()
    min_dc = tl.full([BLOCK], (1 << 63) - 1, tl.uint64)
    for _ in tl.static_range(K_TICK):
        c_now = _read_clock64()
        d = (c_now - c_prev).to(tl.uint64)
        c_prev = c_now
        nz = d > 0
        min_dc = tl.where(nz & (d < min_dc), d, min_dc)
    tl.store(clk_tick_ptr + offs, min_dc, mask=m)

    # Arithmetic-only bracket (dependent adds)
    x = tl.load(x_ptr + offs, mask=m, other=0).to(tl.int32)
    c0 = _read_clock64()
    for _ in tl.static_range(REPS):
        (x,) = tl.inline_asm_elementwise(
            asm="add.cc.u32 $0, $1, 7;",
            constraints="=r,r",
            args=[x],
            dtype=(tl.int32,),
            is_pure=False,
            pack=1,
        )
        (x,) = tl.inline_asm_elementwise(
            asm="addc.u32 $0, $1, 0;",
            constraints="=r,r",
            args=[x],
            dtype=(tl.int32,),
            is_pure=False,
            pack=1,
        )
    c1 = _read_clock64()
    tl.store(x_ptr + offs, x, mask=m)

    tl.store(c0_ptr + offs, c0, mask=m)
    tl.store(c1_ptr + offs, c1, mask=m)

# ============================
# Utilities
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

# ============================
# Sweep runner
# ============================
def sweep_repetitions_to_csv(
    reps_start=0,
    reps_end=250,
    reps_step=1,
    out_csv="sweep_results.csv",
    GHz=1.680,            # adjust if you pin clocks
    N=32,
    BLOCK=32,
    K_TICK=64,
    K_TICK_CLOCK64=64,
    grid=(1,),
    num_warps=1,
    num_stages=1,
    export_ptx_cubin_once=False,    # export kernels after warm-up
):
    """
    For reps in [reps_start, reps_end] with step reps_step:
      - Launch debug_timer_globaltimer_kernel(REPS=reps)
      - Launch debug_timer_clock64_kernel(REPS=reps)
      - Collect summaries and write as rows in CSV.
    """
    assert reps_end >= reps_start and reps_step >= 1
    ns_per_cycle = 1.0 / GHz

    # Device buffers (reused across runs)
    xA = torch.zeros(N, device="cuda", dtype=torch.int32)
    t0 = torch.empty(N, device="cuda", dtype=torch.uint64)
    t1 = torch.empty(N, device="cuda", dtype=torch.uint64)
    tick_ns = torch.empty(N, device="cuda", dtype=torch.uint64)

    xB = torch.zeros(N, device="cuda", dtype=torch.int32)
    c0 = torch.empty(N, device="cuda", dtype=torch.uint64)
    c1 = torch.empty(N, device="cuda", dtype=torch.uint64)
    tick_cyc = torch.empty(N, device="cuda", dtype=torch.uint64)

    # Warm-up (avoid JIT in measurements)
    debug_timer_globaltimer_kernel[grid](
        xA, t0, t1, tick_ns,
        N=N, BLOCK=BLOCK, REPS=1, K_TICK=1,
        num_warps=num_warps, num_stages=num_stages
    )
    debug_timer_clock64_kernel[grid](
        xB, c0, c1, tick_cyc,
        N=N, BLOCK=BLOCK, REPS=1, K_TICK=1,
        num_warps=num_warps, num_stages=num_stages
    )
    torch.cuda.synchronize()

    # Optional export of PTX/CUBIN once (for disassembly)
    if export_ptx_cubin_once:
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

        compiled_entries_B = _flatten_compiled_entries(debug_timer_clock64_kernel.cache)
        if compiled_entries_B:
            compiled_B = _pick_best_for_device(compiled_entries_B)
            asmB = getattr(compiled_B, "asm", {})
            ptxB = asmB.get("ptx", None)
            cubB = asmB.get("cubin", None)
            if ptxB:
                with open("debug_timer_clock64.ptx", "w") as f: f.write(ptxB)
            if cubB is not None:
                with open("debug_timer_clock64.cubin", "wb") as f: f.write(cubB)

    # CSV header
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "reps",
            # %globaltimer
            "gt_mean_ns", "gt_std_ns", "gt_min_ns", "gt_max_ns",
            "gt_per_iter_ns", "gt_tick_min_ns", "gt_tick_mode_ns", "gt_tick_gcd_ns",
            # clock64
            "clk_mean_cycles", "clk_std_cycles", "clk_min_cycles", "clk_max_cycles",
            "clk_per_iter_cycles", "clk_per_iter_ns_at_GHz",
            "clk_tick_min_cycles"
        ])

        # Sweep
        reps_range = range(reps_start, reps_end + 1, reps_step)
        for reps in reps_range:
            # --- A) %globaltimer kernel ---
            debug_timer_globaltimer_kernel[grid](
                xA, t0, t1, tick_ns,
                N=N, BLOCK=BLOCK, REPS=reps, K_TICK=K_TICK,
                num_warps=num_warps, num_stages=num_stages
            )
            # --- B) clock64 kernel ---
            debug_timer_clock64_kernel[grid](
                xB, c0, c1, tick_cyc,
                N=N, BLOCK=BLOCK, REPS=reps, K_TICK=K_TICK_CLOCK64,
                num_warps=num_warps, num_stages=num_stages
            )
            torch.cuda.synchronize()

            # Pull back results
            t0_cpu = t0.cpu().numpy()
            t1_cpu = t1.cpu().numpy()
            dt_ns = (t1_cpu - t0_cpu).astype(np.int64)
            tick_ns_cpu = tick_ns.cpu().numpy().astype(np.int64)

            c0_cpu = c0.cpu().numpy().astype(np.int64)
            c1_cpu = c1.cpu().numpy().astype(np.int64)
            dc = (c1_cpu - c0_cpu).astype(np.int64)
            tick_cyc_cpu = tick_cyc.cpu().numpy().astype(np.int64)

            # %globaltimer stats
            gt_mean = float(dt_ns.mean())
            gt_std  = float(dt_ns.std(ddof=0))
            gt_min  = int(dt_ns.min())
            gt_max  = int(dt_ns.max())

            tick_ns_nz = tick_ns_cpu[tick_ns_cpu > 0]
            if tick_ns_nz.size:
                vals, counts = np.unique(tick_ns_nz, return_counts=True)
                gt_tick_min  = int(tick_ns_nz.min())
                gt_tick_mode = int(vals[counts.argmax()])
                gt_tick_gcd  = int(np.gcd.reduce(vals[: min(50, vals.size)].astype(np.int64)))
            else:
                gt_tick_min = gt_tick_mode = gt_tick_gcd = 0

            gt_per_iter_ns = (gt_mean / reps) if reps > 0 else float("nan")

            # clock64 stats
            clk_mean = float(dc.mean())
            clk_std  = float(dc.std(ddof=0))
            clk_min  = int(dc.min())
            clk_max  = int(dc.max())
            clk_per_iter_cycles = (clk_mean / reps) if reps > 0 else float("nan")
            clk_per_iter_ns     = (clk_per_iter_cycles * ns_per_cycle) if reps > 0 else float("nan")
            clk_tick_min = int(tick_cyc_cpu[tick_cyc_cpu > 0].min()) if (tick_cyc_cpu > 0).any() else 0

            # Write row
            w.writerow([
                reps,
                gt_mean, gt_std, gt_min, gt_max,
                gt_per_iter_ns, gt_tick_min, gt_tick_mode, gt_tick_gcd,
                clk_mean, clk_std, clk_min, clk_max,
                clk_per_iter_cycles, clk_per_iter_ns,
                clk_tick_min
            ])

    print(f"✅ Wrote sweep results to '{out_csv}'")

# ============================
# Main (single entry point)
# ============================
if __name__ == "__main__":
    torch.cuda.set_device(0)
    dev = torch.cuda.current_device()
    print(f"GPU: {torch.cuda.get_device_name(dev)} (sm_{torch.cuda.get_device_capability(dev)[0]}{torch.cuda.get_device_capability(dev)[1]})")

    # Configure your sweep here:
    sweep_repetitions_to_csv(
        reps_start=0,
        reps_end=500,          # 0..500 inclusive => 501 runs
        reps_step=1,
        out_csv="sweep_results.csv",
        GHz=1.680,
        N=32, BLOCK=32,
        K_TICK=64, K_TICK_CLOCK64=64,
        grid=(1,),
        num_warps=1, num_stages=1,
        export_ptx_cubin_once=False
    )
