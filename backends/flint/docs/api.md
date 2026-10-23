# Flint Tensor operators on SM120

This optional producer exposes prequantized NVFP4 projections, dense SwiGLU,
routed/shared FFNs and paged decode attention as Torch Tensor operations.
[FlashRT Structures](https://github.com/flashrt-project/FlashRT-Structures)
owns host discovery, weight residency, attachment and rollback; vLLM retains
request scheduling and KV allocation. The existing
[standalone Flint frontend](https://github.com/flashrt-project/FlashRT/blob/main/docs/flint_usage.md) remains a separate consumer.

## Build and install

Follow the independent [backend build instructions](../README.md). The wheel
build compiles both `_flashrt_flint_ops` and `_flashrt_flint_sage2` for SM120.
It needs a matching CUDA-enabled PyTorch, CUDA compiler, Triton, setuptools,
wheel and Ninja; it does not need a FlashRT checkout, CUTLASS checkout or
compiled FlashRT module. Core Structures installation does not trigger it.
No import, attachment or forward compiles kernels. Missing or stale binaries
refuse clearly before execution.

## API and contracts

```python
from flashrt_structures_flint.nvfp4 import PackedMatrix, Linear
matrix = PackedMatrix(codes, scales, global_scale)
projection = Linear(matrix)  # resolves ABI/device/symbols and validates weights
y = projection(x)
```

| Surface | Contract |
|---|---|
| `PackedMatrix` | CUDA contiguous uint8 codes `[N,K/2]`, adjacent E2M1 nibbles; uint8 E4M3 scale bytes `[N,K/16]`; finite positive global scale; N multiple of 128, K multiple of 64 |
| `Linear` | BF16 contiguous `[M,K]` → BF16 `[M,N]` |
| `DenseFfn` | Interleaved gate/up (groups of gate8/up8), precise SiLU, then down; shape relationships validated at bind |
| `RoutedSharedFfn` | Coupled sigmoid/top6 router, 256 experts, hidden 2048, expert intermediate 512, ungated shared intermediate 1024; capacity 1–8192 rows |
| `LmHead` | Packed NVFP4 vocabulary projection with FP32 logits |
| `sage2.attention_fp8` | BF16 Q `[Tq,Hq,128]`, K/V `[Tk,Hk,128]`, `Tk>=Tq`, `Hq % Hk == 0`; causal prefix offset `Tk-Tq`, optional per-head sink; BF16 result |

ABI 1 and the hardware/profile metadata live in `_flashrt_flint_ops.TENSOR_API_METADATA`.
The Python producer validates all required symbols before the first execution.
`Linear`, `DenseFfn`, `LmHead` and `RoutedSharedFfn` reject calls unless the
activation and every weight tensor share the same CUDA device. Routed/shared
per-expert global scales and scratch must also be on that device. Device
mismatches raise `ValueError` before entering CUDA execution.
Input owners, packed weights and graph buffers must outlive every replay.
Sharing scratch supports one sequential execution lane; independent concurrent
lanes require separate scratch. Packing and dynamic activation quantization
belong to the producer, not to a host adapter's calibration traversal.

## Execution forms

The default W4A16 path retains checkpoint NVFP4 codes and scales. Optional
W4A8 uses int8 Tensor Cores and dynamic per-32 activation scales; the M=1
path remains W4A16. Dense FP8 and grouped FP4 prefill forms introduce further
rounding and require the matching vLLM binary operators. They are not described
as lossless transformations. FP8 KV uses the host's unit-scale cache contract.

The supported serving recipe and measured batch 1–8 cell are in the companion
[Flint/vLLM guide](https://github.com/flashrt-project/FlashRT-Structures/blob/feat/flint-w4a8-lm-head/docs/flint_vllm.md).
Configuration is fixed before importing this module. The entire vLLM route is
opt-in; importing `flash_rt` or leaving the plugin disabled does not select it.

| Producer switch | Default | Meaning |
|---|---:|---|
| `FLASHRT_NATIVE_A8` | 0 | Enable W4A8 where the row/shape policy admits it |
| `FLASHRT_NATIVE_A8_MIN_M` / `A8_DENSE_MIN_M` | 2 / 192 | Minimum W4A8 activation/dense rows |
| `FLASHRT_NATIVE_A8_GU_MIN_M` | 1000000000 | Grouped gate/up W4A8 threshold; effectively disabled until selected |
| `FLASHRT_NATIVE_FP4_MOE_MIN_M` | 1000000000 | Grouped FP4 prefill threshold |
| `FLASHRT_NATIVE_DENSE_FP8_MIN_M` | 1000000000 | Dense FP8 prefill threshold |
| `FLASHRT_NATIVE_FP4_SF_CACHE` | 1 | Cache packed weight scale-factor layouts at bind; disable for the measured memory-constrained recipe |
| `FLASHRT_NATIVE_GEMV_MMA_MAX` | 16 | Exact few-row W4A16 MMA band |
| `FLASHRT_NATIVE_RESIDENT` | 1 | Exact resident-activation policy for qualified 9–16-row shapes |
| `FLASHRT_NATIVE_SIDE_STREAM` | 1 | Overlap the shared expert on a producer-owned stream |

Other tuning and fake-quantization switches in `nvfp4.py` are experimental,
not part of the recommended serving recipe. In particular `STREAM_MAX_M`,
`SPLITK`, `FAKE_A4_*`, `FAKE_DENSE_FP8_MIN_M`, `A8_PER_ROW` and diagnostic
dump/check flags remain disabled unless explicitly selected. Experimental
packed K-FP8/V-FP4 cache bindings are not qualified by the FP8-KV measurements.

## Validation

Run from the Structures repository root, with the backend wheel installed:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q backends/flint/tests
# Check GPU occupancy first; then run arithmetic/capture tests on SM120.
python -m pytest -q backends/flint/tests
```

Tests cover stale ABI/symbol/device refusal, W4A16/W4A8 numerical references,
changed-input graph replay, Sage2 causal-prefix/sink attention and vLLM routing.
CUDA tests skip explicitly when the qualified hardware is absent. Validate the
existing standalone frontend separately in FlashRT; its tests and canonical
CUDA remain in that repository. Real-host quality and paired decode/prefill
recipes are in the [integration guide](../../../docs/flint_vllm.md).

## Sources and credit

- [RecursiveSI Flint](https://huggingface.co/RecursiveSI/flint) and the
  [official NVFP4 checkpoint](https://huggingface.co/RecursiveSI/flint-nvfp4-gguf): model and weights.
- [FlashRT](https://github.com/flashrt-project/FlashRT) and
  [FlashRT Structures](https://github.com/flashrt-project/FlashRT-Structures): producers and host integration.
- [SageAttention](https://github.com/thu-ml/SageAttention) and
  [FlashRT-HF-kernels](https://github.com/flashrt-project/FlashRT-HF-kernels/tree/059d28393239c8fec04699bccaed6f4c88d25f24/sageattention2-blackwell): adapted attention cores; see [vendor notices](../flashrt_structures_flint/csrc/sage2/VENDOR.md).
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer),
  [CUTLASS](https://github.com/NVIDIA/cutlass), [Triton](https://github.com/triton-lang/triton)
  and [vLLM](https://github.com/vllm-project/vllm): attention helpers, GEMM primitives, Tensor producers and serving.
