"""SageAttention2-style FP8 prefill attention for sm120 (FlashRT Blackwell kernel, built from source on first use).

INT8 Q/K (per-warp blocks) x FP8 V (per-channel, transposed/padded layout), FP8 probabilities; causal with a prefix
offset (chunked prefill) and an LSE output so attention sinks can be folded afterwards.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import math
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

_OPS = None
_NS = "flashrt_sage2_blackwell"
SRC = Path(os.environ.get("FLASHRT_SAGE2_SRC", str(Path(__file__).resolve().parents[3] / "sage2")))
_LN2 = math.log(2.0)
_S_FP8_OFFSET = 8.807


def _preload_cublaslt():
    for parent in Path(torch.__file__).resolve().parents:
        cand = parent / "nvidia" / "cublas" / "lib" / "libcublasLt.so.12"
        if cand.exists():
            ctypes.CDLL(str(cand), mode=ctypes.RTLD_GLOBAL)
            return
    lib = ctypes.util.find_library("cublasLt")
    if lib:
        ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)


def ops():
    global _OPS
    if _OPS is None:
        from torch.utils.cpp_extension import load
        _preload_cublaslt()
        major, minor = torch.cuda.get_device_capability(0)
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}{'a' if major >= 12 else ''}")
        reg = os.environ.get("FLASHRT_SAGE2_REG_INCLUDE")
        inc = [str(SRC / "csrc")] + ([reg] if reg else [])
        load(name=_NS, sources=[str(SRC / "torch-ext" / "torch_binding.cpp"), str(SRC / "csrc" / "sage2_blackwell.cu")],
             extra_include_paths=inc, extra_cflags=["-O3", "-DCUDA_KERNEL"],
             extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-DCUDA_KERNEL", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                                "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__"],
             verbose=False)
        _OPS = getattr(torch.ops, _NS)
    return _OPS


def padded_k64(n: int) -> int:
    return ((n + 63) // 64) * 64


def attention_fp8(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sinks: torch.Tensor | None, sm_scale: float,
                  causal: bool = True) -> torch.Tensor:
    """q [Tq, Hq, 128] bf16; k, v [Tk, Hk, 128] bf16 (full sequence, the last Tq positions are the queries);
    returns [Tq, Hq, 128] bf16 with the sink folded (softmax over keys plus one logit sinks[h] per head)."""
    o = ops()
    Tq, Hq, D = q.shape
    Tk, Hk, _ = k.shape
    assert D == 128 and Tk >= Tq and Hq % Hk == 0
    dev = q.device
    qb = q.reshape(1, Tq, Hq, D).contiguous(); kb = k.reshape(1, Tk, Hk, D).contiguous(); vb = v.reshape(1, Tk, Hk, D).contiguous()
    q_i8 = torch.empty_like(qb, dtype=torch.int8); q_s = torch.empty((Hq * ((Tq + 31) // 32),), dtype=torch.float32, device=dev)
    k_i8 = torch.empty_like(kb, dtype=torch.int8); k_s = torch.empty((Hk * ((Tk + 63) // 64),), dtype=torch.float32, device=dev)
    v8 = torch.empty((1, D, Hk, padded_k64(Tk)), dtype=torch.int8, device=dev); v_s = torch.empty((Hk * D,), dtype=torch.float32, device=dev)
    o.quantize_q_bf16_d128(qb, q_i8, q_s); o.quantize_k_bf16_d128(kb, k_i8, k_s); o.quantize_v_fp8_bf16_d128(vb, v8, v_s)
    out = torch.empty_like(qb); lse = torch.empty((1, Hq, Tq), dtype=torch.float32, device=dev)
    o.sage2_qk_int8_sv_f8_lse_bf16_d128(q_i8, k_i8, v8, q_s, k_s, v_s, float(sm_scale), bool(causal), int(Tk - Tq), out, lse)
    out = out.reshape(Tq, Hq, D)
    if sinks is not None:
        # the kernel stores log2(sum exp) minus S_FP8_OFFSET (8.807 = log2 448, the FP8 probability scale)
        f = torch.sigmoid((lse[0].transpose(0, 1) + _S_FP8_OFFSET) * _LN2 - sinks[None, :].float())   # [Tq, Hq]
        out = (out.float() * f[:, :, None]).to(out.dtype)
    return out


@triton.jit
def _gather_pages_kernel(cache_ptr, bt_ptr, out_ptr, L, s_blk, s_tok, s_head, page, scale, H: tl.constexpr,
                         D: tl.constexpr, BLOCK_T: tl.constexpr):
    # fp8 (E4M3 bytes) paged K or V [blocks][...] -> bf16 rows [L, H, D] for tokens 0..L-1 of one sequence
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    m = t < L
    blk = tl.load(bt_ptr + t // page, mask=m, other=0).to(tl.int64)
    tok = (t % page).to(tl.int64)
    d = tl.arange(0, D)
    for h in tl.static_range(H):
        src = cache_ptr + blk[:, None] * s_blk + tok[:, None] * s_tok + h * s_head + d[None, :]
        v = tl.load(src, mask=m[:, None], other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale
        tl.store(out_ptr + (t.to(tl.int64)[:, None] * H + h) * D + d[None, :], v.to(tl.bfloat16), mask=m[:, None])


def gather_pages_bf16(cache: torch.Tensor, block_table: torch.Tensor, L: int, hnd: bool, scale: float, H: int, D: int) -> torch.Tensor:
    """One pass: fp8 pages (uint8 or float8 view; (blocks, H, page, D) if hnd else (blocks, page, H, D)) -> bf16 [L, H, D]."""
    c = cache if cache.dtype == torch.uint8 else cache.view(torch.uint8)
    page = c.shape[2] if hnd else c.shape[1]
    s_blk = c.stride(0); s_head = c.stride(1) if hnd else c.stride(2); s_tok = c.stride(2) if hnd else c.stride(1)
    out = torch.empty((L, H, D), dtype=torch.bfloat16, device=c.device)
    _gather_pages_kernel[(triton.cdiv(L, 64),)](c, block_table, out, L, s_blk, s_tok, s_head, page, float(scale),
                                                H=H, D=D, BLOCK_T=64, num_warps=4)
    return out
