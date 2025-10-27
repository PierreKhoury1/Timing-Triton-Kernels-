# run_flashattn_3060.py
import os, math, time
import torch
import triton
import triton.language as tl
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")



# --------- pin to your RTX 3060 Ti (GPU 0) ----------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
torch.cuda.set_device(0)
assert "3060" in torch.cuda.get_device_name(0).lower(), "GPU 0 is not a 3060/3060 Ti"

# (optional) sanity: Ampere (8.x)
major, minor = torch.cuda.get_device_capability(0)
assert major == 8, f"Expected Ampere (8.x); got {major}.{minor}"

@triton.jit
def flash_attn_fwd_kernel(
    Q, K, V, O,
    B, N, H,
    s_qb, s_qn, s_qh, s_qd,
    s_kb, s_kn, s_kh, s_kd,
    s_vb, s_vn, s_vh, s_vd,
    s_ob, s_on, s_oh, s_od,
    scale,                           # host-computed 1/sqrt(D_HEAD)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,            # constexpr so tl.arange(0, D_HEAD) is legal
):
    pid_bh = tl.program_id(0)        # which (batch, head)
    pid_m  = tl.program_id(1)        # which block of Q rows

    b = pid_bh // H
    h = pid_bh %  H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [M]
    offs_n = tl.arange(0, BLOCK_N)                     # [Nstep]
    offs_d = tl.arange(0, D_HEAD)                      # [D]

    # Q/O tiles [M, D]
    q_ptrs = Q + b*s_qb + offs_m[:, None]*s_qn + h*s_qh + offs_d[None, :]*s_qd
    o_ptrs = O + b*s_ob + offs_m[:, None]*s_on + h*s_oh + offs_d[None, :]*s_od

    q = tl.load(q_ptrs, mask=(offs_m[:, None] < N), other=0.0)

    # online softmax state
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)

    # stream over K/V along sequence
    for start_n in range(0, N, BLOCK_N):
        n_ids = start_n + offs_n

        k_ptrs = K + b*s_kb + n_ids[:, None]*s_kn + h*s_kh + offs_d[None, :]*s_kd
        v_ptrs = V + b*s_vb + n_ids[:, None]*s_vn + h*s_vh + offs_d[None, :]*s_vd

        k = tl.load(k_ptrs, mask=(n_ids[:, None] < N), other=0.0)
        v = tl.load(v_ptrs, mask=(n_ids[:, None] < N), other=0.0)

        # scores [M, Nstep] = [M, D] x [D, Nstep]
        qk = tl.dot(q, tl.trans(k)) * scale

        # Tail mask: ensure padding gets -inf so softmax gives zero prob
        col_mask = n_ids[None, :] < N
        qk = tl.where(col_mask, qk, float("-inf"))

        # online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p    = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.exp(m_i - m_ij)

        # accumulate output
        acc = acc * alpha[:, None] + tl.dot(p, v)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # finalize
    o = acc / l_i[:, None]
    tl.store(o_ptrs, o, mask=(offs_m[:, None] < N))


def flash_attn_fwd(Q, K, V, block_m=64, block_n=64, repeat=10):
    """
    Q,K,V: float32, cuda:0, shape [B, N, H, D], contiguous.
    Returns (O, avg_ms) with avg over `repeat` CUDA-event timings.
    """
    assert all(t.is_cuda and t.device.index == 0 for t in (Q, K, V))
    assert Q.dtype == K.dtype == V.dtype == torch.float32
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()

    B, N, H, D = Q.shape
    O = torch.empty_like(Q)

    grid = (B * H, triton.cdiv(N, block_m))
    scale = 1.0 / math.sqrt(D)

    # warmup compile + one run
    flash_attn_fwd_kernel[grid](
        Q, K, V, O,
        B, N, H,
        *Q.stride(), *K.stride(), *V.stride(), *O.stride(),
        scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        D_HEAD=D,
        num_warps=4,
        num_stages=2,
    )
    torch.cuda.synchronize()

    # CUDA event timing (robust)
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(repeat):
        start.record()
        flash_attn_fwd_kernel[grid](
            Q, K, V, O,
            B, N, H,
            *Q.stride(), *K.stride(), *V.stride(), *O.stride(),
            scale,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            D_HEAD=D,
            num_warps=4,
            num_stages=2,
        )
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))  # ms

    avg_ms = sum(times) / len(times)
    print(f"Kernel time (CUDA events): avg={avg_ms:.4f} ms  min={min(times):.4f}  max={max(times):.4f}")
    return O, avg_ms


if __name__ == "__main__":
    torch.manual_seed(0)

    # Force true FP32 for fair comparison (disable TF32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    # Choose an N not divisible by BLOCK_N to exercise tail masking
    B, N, H, D = 1, 257, 8, 64
    device = "cuda:0"

    Q = torch.randn(B, N, H, D, device=device, dtype=torch.float32).contiguous()
    K = torch.randn(B, N, H, D, device=device, dtype=torch.float32).contiguous()
    V = torch.randn(B, N, H, D, device=device, dtype=torch.float32).contiguous()

    O, _ = flash_attn_fwd(Q, K, V, block_m=64, block_n=64, repeat=10)

    # Reference check
    with torch.no_grad():
        q = Q.permute(0,2,1,3)                       # [B,H,N,D]
        k = K.permute(0,2,1,3)
        v = V.permute(0,2,1,3)
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D)
        ref = torch.softmax(attn, dim=-1) @ v        # [B,H,N,D]
        ref = ref.permute(0,2,1,3).contiguous()      # [B,N,H,D]
        print("max |diff|:", (O - ref).abs().max().item())

    # Optional PTX dump (after first compile)
    try:
        ptx = list(flash_attn_fwd_kernel.cache.values())[0].asm["ptx"]
        with open("flash_attn_fwd_kernel.ptx", "w") as f:
            f.write(ptx)
        print("Wrote PTX -> flash_attn_fwd_kernel.ptx")
    except Exception as e:
        print("PTX dump skipped:", e)
