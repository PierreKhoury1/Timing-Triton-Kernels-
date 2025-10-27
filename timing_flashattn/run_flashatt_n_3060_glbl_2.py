# run_flashattn_3060_double.py
import os, math, numpy as np
import torch
import triton
import triton.language as tl

# ---- FP32 reference (no TF32) ----
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

# ---- pin to your RTX 3060 Ti (GPU 0) ----
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
torch.cuda.set_device(0)
#print(torch.cuda.get_device_name(0))
assert "3060" in torch.cuda.get_device_name(0).lower()
major, minor = torch.cuda.get_device_capability(0)
assert major == 8, f"Expected Ampere (8.x), got {major}.{minor}"

# ---- %globaltimer helper (works with CUDA 12.4 ptxas + Triton 3.1) ----
@triton.jit
def _read_globaltimer():
    (t,) = tl.inline_asm_elementwise(
        asm="mov.u64 $0, %globaltimer;",  # single '%'
        constraints="=l",
        args=[],
        dtype=(tl.uint64,),
        is_pure=False,
        pack=1,
    )
    return t

# ========== baseline kernel (1× workload) ==========
@triton.jit
def flash_attn_fwd_kernel_1x(
    Q, K, V, O,
    B, N, H,
    s_qb, s_qn, s_qh, s_qd,
    s_kb, s_kn, s_kh, s_kd,
    s_vb, s_vn, s_vh, s_vd,
    s_ob, s_on, s_oh, s_od,
    scale,
    t_total,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m  = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh %  H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_HEAD)

    q_ptrs = Q + b*s_qb + offs_m[:, None]*s_qn + h*s_qh + offs_d[None, :]*s_qd
    o_ptrs = O + b*s_ob + offs_m[:, None]*s_on + h*s_oh + offs_d[None, :]*s_od
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < N), other=0.0)

    # timing start
    t0 = _read_globaltimer()

    # online softmax state
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        n_ids = start_n + offs_n
        k_ptrs = K + b*s_kb + n_ids[:, None]*s_kn + h*s_kh + offs_d[None, :]*s_kd
        v_ptrs = V + b*s_vb + n_ids[:, None]*s_vn + h*s_vh + offs_d[None, :]*s_vd
        k = tl.load(k_ptrs, mask=(n_ids[:, None] < N), other=0.0)
        v = tl.load(v_ptrs, mask=(n_ids[:, None] < N), other=0.0)

        qk = tl.dot(q, tl.trans(k)) * scale
        col_mask = n_ids[None, :] < N
        qk = tl.where(col_mask, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p    = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.exp(m_i - m_ij)

        acc = acc * alpha[:, None] + tl.dot(p, v)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    o = acc / l_i[:, None]
    tl.store(o_ptrs, o, mask=(offs_m[:, None] < N))

    # timing end + store
    t1 = _read_globaltimer()
    dt = (t1 - t0).to(tl.uint64)
    grid1 = tl.cdiv(N, BLOCK_M)
    idx = pid_bh * grid1 + pid_m
    one = tl.arange(0, 1)
    tl.store(t_total + idx + one, dt)

# ========== double-work kernel (2× workload) ==========
@triton.jit
def flash_attn_fwd_kernel_2x(
    Q, K, V, O,
    B, N, H,
    s_qb, s_qn, s_qh, s_qd,
    s_kb, s_kn, s_kh, s_kd,
    s_vb, s_vn, s_vh, s_vd,
    s_ob, s_on, s_oh, s_od,
    scale,
    t_total,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m  = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh %  H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_HEAD)

    q_ptrs = Q + b*s_qb + offs_m[:, None]*s_qn + h*s_qh + offs_d[None, :]*s_qd
    o_ptrs = O + b*s_ob + offs_m[:, None]*s_on + h*s_oh + offs_d[None, :]*s_od
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < N), other=0.0)

    # timing start
    t0 = _read_globaltimer()

    # do the whole attention twice
    o = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    for _ in range(2):
        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)

        for start_n in range(0, N, BLOCK_N):
            n_ids = start_n + offs_n
            k_ptrs = K + b*s_kb + n_ids[:, None]*s_kn + h*s_kh + offs_d[None, :]*s_kd
            v_ptrs = V + b*s_vb + n_ids[:, None]*s_vn + h*s_vh + offs_d[None, :]*s_vd
            k = tl.load(k_ptrs, mask=(n_ids[:, None] < N), other=0.0)
            v = tl.load(v_ptrs, mask=(n_ids[:, None] < N), other=0.0)

            qk = tl.dot(q, tl.trans(k)) * scale
            col_mask = n_ids[None, :] < N
            qk = tl.where(col_mask, qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p    = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            alpha = tl.exp(m_i - m_ij)

            acc = acc * alpha[:, None] + tl.dot(p, v)
            l_i = l_i * alpha + l_ij
            m_i = m_ij

        o = acc / l_i[:, None]  # keep last repeat's output

    tl.store(o_ptrs, o, mask=(offs_m[:, None] < N))

    # timing end + store
    t1 = _read_globaltimer()
    dt = (t1 - t0).to(tl.uint64)
    grid1 = tl.cdiv(N, BLOCK_M)
    idx = pid_bh * grid1 + pid_m
    one = tl.arange(0, 1)
    tl.store(t_total + idx + one, dt)

# ---------- helpers ----------
def run_kernel_and_time(which, Q, K, V, O, block_m, block_n):
    B, N, H, D = Q.shape
    grid = (B * H, triton.cdiv(N, block_m))
    scale = 1.0 / math.sqrt(D)
    t_total = torch.empty(grid[0]*grid[1], device=Q.device, dtype=torch.uint64)

    # warmup compile
    which[grid](
        Q, K, V, O,
        B, N, H,
        *Q.stride(), *K.stride(), *V.stride(), *O.stride(),
        scale, t_total,
        BLOCK_M=block_m, BLOCK_N=block_n, D_HEAD=D,
        num_warps=4, num_stages=2,
    )
    torch.cuda.synchronize()

    # CUDA events whole-kernel timing
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    which[grid](
        Q, K, V, O,
        B, N, H,
        *Q.stride(), *K.stride(), *V.stride(), *O.stride(),
        scale, t_total,
        BLOCK_M=block_m, BLOCK_N=block_n, D_HEAD=D,
        num_warps=4, num_stages=2,
    )
    end.record(); end.synchronize()

    ms_event = start.elapsed_time(end)
    ms_prog  = (t_total.detach().cpu().numpy() / 1e6)   # per-program
    return ms_event, ms_prog

if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, H, D = 1, 257, 8, 64
    Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.float32).contiguous()
    K = torch.randn(B, N, H, D, device="cuda", dtype=torch.float32).contiguous()
    V = torch.randn(B, N, H, D, device="cuda", dtype=torch.float32).contiguous()
    O = torch.empty_like(Q)

    bm, bn = 64, 64

    # --- 1× workload ---
    ms_event_1x, ms_prog_1x = run_kernel_and_time(flash_attn_fwd_kernel_1x, Q, K, V, O, bm, bn)
    print(f"[1x] whole-kernel (CUDA events): {ms_event_1x:.6f} ms")
    print(f"[1x] per-program (globaltimer): min={ms_prog_1x.min():.6f}  med={np.median(ms_prog_1x):.6f}  max={ms_prog_1x.max():.6f} ms")

    # --- 2× workload ---
    ms_event_2x, ms_prog_2x = run_kernel_and_time(flash_attn_fwd_kernel_2x, Q, K, V, O, bm, bn)
    print(f"[2x] whole-kernel (CUDA events): {ms_event_2x:.6f} ms")
    print(f"[2x] per-program (globaltimer): min={ms_prog_2x.min():.6f}  med={np.median(ms_prog_2x):.6f}  max={ms_prog_2x.max():.6f} ms")

    # Ratios
    print(f"Event ratio (2x/1x): {ms_event_2x / ms_event_1x:.2f}×")
    print(f"Program median ratio (2x/1x): {np.median(ms_prog_2x) / np.median(ms_prog_1x):.2f}×")

    # Quick correctness check (optional)
    with torch.no_grad():
        q = Q.permute(0,2,1,3); k = K.permute(0,2,1,3); v = V.permute(0,2,1,3)
        ref = (torch.softmax(torch.matmul(q, k.transpose(-1, -2))/math.sqrt(D), dim=-1) @ v).permute(0,2,1,3).contiguous()
        print("max |diff| (1x last O):", (O - ref).abs().max().item())
