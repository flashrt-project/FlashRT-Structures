"""NVFP4 value shadow cache for Flint decode attention (opt-in, FLASHRT_NATIVE_KV_V4=1).

The vLLM KV cache stays as allocated (fp8 K and V, read by the host prefill attention); alongside it, every V row
written to the cache is also quantized to E2M1 codes with one E4M3 scale per 16 dims into a shadow pair
(codes [blocks, block_size, kv_heads, 64] uint8, scales [blocks, block_size, kv_heads, 8] uint8) that the native
decode attention reads (kv_mode 2: 200 bytes per token and head instead of 256)."""

import torch
import triton
import triton.language as tl

_BLOCK_T = 32


@triton.jit(do_not_specialize=["num_tokens"])
def _v_fp4_write_kernel(
    v_ptr,
    v_stride,
    slot_ptr,
    num_tokens,
    codes_ptr,
    scales_ptr,
    c_blk,
    c_tok,
    c_head,
    s_blk,
    s_tok,
    s_head,
    block_size,
    NUM_KV_HEADS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    head = tl.program_id(1)
    tokens = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = tokens < num_tokens
    tok64 = tokens.to(tl.int64)
    d = tl.arange(0, 128)
    v = tl.load(
        v_ptr + tok64[:, None] * v_stride + head * 128 + d[None, :],
        mask=t_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    v3 = tl.reshape(v, (BLOCK_T, 8, 16))
    amax = tl.max(tl.abs(v3), axis=2)  # [T, 8]
    sc = (amax / 6.0).to(tl.float8e4nv)  # E4M3 scale per 16 dims
    scf = sc.to(tl.float32)
    q = tl.abs(v3) / tl.where(scf > 0, scf, 1.0)[:, :, None]
    lvl = tl.where(
        q < 0.25,
        0,
        tl.where(
            q < 0.75,
            1,
            tl.where(
                q < 1.25,
                2,
                tl.where(
                    q < 1.75,
                    3,
                    tl.where(q < 2.5, 4, tl.where(q < 3.5, 5, tl.where(q < 5.0, 6, 7))),
                ),
            ),
        ),
    )
    code = (lvl + tl.where(v3 < 0, 8, 0)).to(tl.uint8)  # [T, 8, 16]
    c2 = tl.reshape(code, (BLOCK_T, 64, 2))
    lo, hi = tl.split(c2)  # [T, 64] each
    packed = lo | (hi << 4)
    slot = tl.load(slot_ptr + tokens, mask=t_mask, other=-1).to(tl.int64)
    ok = (slot >= 0) & t_mask
    blk = slot // block_size
    tok = slot % block_size
    cdst = codes_ptr + blk * c_blk + tok * c_tok + head * c_head
    sdst = scales_ptr + blk * s_blk + tok * s_tok + head * s_head
    tl.store(cdst[:, None] + tl.arange(0, 64)[None, :], packed, mask=ok[:, None])
    tl.store(
        sdst[:, None] + tl.arange(0, 8)[None, :],
        sc.to(tl.uint8, bitcast=True),
        mask=ok[:, None],
    )


def v_fp4_write(
    v: torch.Tensor,
    v_stride: int,
    slot_mapping: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    block_size: int,
    num_kv_heads: int = 4,
):
    """v: rows of [num_kv_heads * 128] bf16 (row stride v_stride elements); slot_mapping int32/int64 [num_tokens]."""
    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return
    grid = (triton.cdiv(num_tokens, _BLOCK_T), num_kv_heads)
    _v_fp4_write_kernel[grid](
        v,
        v_stride,
        slot_mapping,
        num_tokens,
        codes,
        scales,
        codes.stride(0),
        codes.stride(1),
        codes.stride(2),
        scales.stride(0),
        scales.stride(1),
        scales.stride(2),
        block_size,
        NUM_KV_HEADS=num_kv_heads,
        BLOCK_T=_BLOCK_T,
        num_warps=4,
    )


class V4Shadow:
    """Per-layer shadow pair, allocated lazily from the fp8 cache's shape; keyed by the K cache's data pointer."""

    def __init__(self):
        self.by_cache = {}

    def get(self, k_cache: torch.Tensor):
        key = k_cache.data_ptr()
        pair = self.by_cache.get(key)
        if pair is None:
            blocks, block_size, heads, _ = k_cache.shape
            codes = torch.zeros(
                (blocks, block_size, heads, 64),
                dtype=torch.uint8,
                device=k_cache.device,
            )
            scales = torch.zeros(
                (blocks, block_size, heads, 8), dtype=torch.uint8, device=k_cache.device
            )
            pair = (codes, scales)
            self.by_cache[key] = pair
        return pair
