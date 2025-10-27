# measure_globaltimer_tick.py
import numpy as np
import torch, triton, triton.language as tl

@triton.jit
def _read_globaltimer():
    (t,) = tl.inline_asm_elementwise(
        asm="mov.u64 $0, %globaltimer;",
        constraints="=l",
        args=[],
        dtype=(tl.uint64,),
        is_pure=False,   # IMPORTANT: don't let the compiler move/merge reads
        pack=1,
    )
    return t

@triton.jit
def gtick_kernel(out, iters: tl.constexpr):
    # single program so we stay on one SM
    t_prev = _read_globaltimer()
    for i in range(iters):
        t_now = _read_globaltimer()
        dt = (t_now - t_prev).to(tl.uint64)
        tl.store(out + i, dt)
        t_prev = t_now

def measure_ticks(iters=20000):
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    out = torch.empty(iters, device="cuda", dtype=torch.uint64)

    # launch 1 program, 1-dim grid
    gtick_kernel[(1,)](out, iters=iters, num_warps=1, num_stages=1)
    torch.cuda.synchronize()
    dt = out.cpu().numpy()

    # discard zeros (back-to-back reads can occasionally be same tick)
    dt_ns = dt[dt > 0].astype(np.int64)

    # Convert to nanoseconds assumption: %globaltimer reports nanoseconds
    # We only need relative steps, so units don't matter for finding the quantum.
    # But we’ll label them "ns" below for readability.
    # Compute basic stats
    min_nonzero = dt_ns.min() if dt_ns.size else 0
    # Mode of small integers:
    vals, counts = np.unique(dt_ns, return_counts=True)
    mode_val = vals[counts.argmax()] if vals.size else 0

    # tiny histogram of the first few distinct bins
    # (helpful if there is a mixture like 32 ns and 64 ns)
    pairs = sorted(zip(vals, counts), key=lambda x: x[0])[:10]

    print("Samples (non-zero):", dt_ns.size)
    print("min Δt:", min_nonzero, "ns")
    print("mode Δt:", mode_val, "ns")
    print("first few bins:")
    for v,c in pairs:
        print(f"  {v:>6d} ns : {c}")

if __name__ == "__main__":
    measure_ticks(20000)
