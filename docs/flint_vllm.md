# Flint: vLLM + FlashRT on RTX 5090

This experimental, opt-in integration runs RecursiveSI Flint's official NVFP4
checkpoint inside vLLM using separately built FlashRT producers. It covers
batch-one latency and small decode batches (the reported range is 1–8).
vLLM retains its scheduler, graph policy, KV allocation and request protocol.
It is separate from FlashRT's standalone Flint frontend and from the normal
Hub-backed Structures route.

## Tested stack and ownership

| Component | Tested cell |
|---|---|
| GPU | RTX 5090, SM120, 32 GB; driver 580.159.03 |
| Runtime | Linux x86-64, Python 3.12.3, CUDA 13.0, torch 2.13.0+cu130, Triton 3.7.1 |
| vLLM build | vLLM Flint source `aed08d76a682f1475957c0730ea1f5ecb150fda0`, matching vLLM 0.32.0 CUDA binary payload, plus the checked-in host patch |
| Dependencies | FlashInfer 0.7.2, Transformers 5.20.2, tokenizers 0.23.3 |
| Model | `RecursiveSI/flint-nvfp4-gguf`; BF16 activations, FP8 KV with unit host scales |
| Producer | FlashRT `flash_rt_flint` plus the optional prebuilt Sage2 module |
| Consumer | This package's explicit `flashrt_native` vLLM plugin; default OFF |

This source/binary combination is the measured development cell, not a claim
that a released vLLM wheel alone supports Flint or that arbitrary dependency
versions are interchangeable. In particular, this source cell uses
Transformers 5.20.2, beyond the 0.32.0 wheel's declared `<5.20` bound.
Re-qualify a different combination rather than silently changing the host.

The model digest is
`97cc8009ff9b132d5cf3c488e6767962bb5a5dc158a90ed661bbbddf876b8bde`.
No new structure specification or default host policy is selected by this
plugin. The coupled routed/shared FFN wrapper is a provisional host-stage
composition of the FlashRT operator profile, rather than a registered general catalog structure;
cross-family topology recognition does not certify another model's GPU path.

| FlashRT region | Integration | Qualification / retained boundary |
|---|---|---|
| QKV/O projection | `linear_proj/native_w4a16` | packed NVFP4 weights; row-dependent producer policy |
| Dense SwiGLU | `decoder_ffn/native_w4a16` | precise SiLU; optional dense FP8 prefill |
| Router + routed/shared FFN | `routed_ffn/native_w4a16` | explicit sigmoid/top6 profile; grouped FP4 prefill |
| Decode norm/attention/layer chain | explicit Flint host stages | paged attention plus producer composition; recorded separately from guarded seams |
| Full-attention prefill | Sage2 producer via host patch | INT8 Q/K, FP8 V/probabilities, causal-prefix offset and LSE sink fold |
| Sliding-window prefill | retained host stage | host attention remains responsible |
| Embedding, allocation, scheduling, sampling | retained host stages | vLLM owns these stages |

## Install

First build and install the paired
[FlashRT producer](https://github.com/flashrt-project/FlashRT/blob/feat/flint-w4a8-sm120/docs/native_nvfp4_api.md).
Build Sage2 explicitly before creating a serving engine; the adapter does not
search a source checkout or compile kernels at inference time.

The host patch supplies FP8 KV writes, the LSE/sink prefill path and its
Sage2 dispatch. It is restricted to Flint's model/attention files. Apply it to
the pinned vLLM build in both the vLLM and vLLM + FlashRT configurations:

```bash
export VLLM_SRC="<vllm-source-checkout>"
git -C "$VLLM_SRC" checkout --detach aed08d76a682f1475957c0730ea1f5ecb150fda0
git -C "$VLLM_SRC" apply --check "$(pwd)/patches/vllm-flint-sm120.patch"
git -C "$VLLM_SRC" apply "$(pwd)/patches/vllm-flint-sm120.patch"
# Install with the matching precompiled wheel, following vLLM's source-install workflow.
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION="<matching-vllm-wheel>" \
  uv pip install --no-deps --no-build-isolation -e "$VLLM_SRC"
pip install -e . --no-deps
```

The default Hub installation and `vllm_engine.install_load_hook()` behavior
remain available. The plugin entry point is loaded in each worker; without
`FLASHRT_STRUCTURES_NATIVE=1`, it leaves FlashRT acceleration disabled.

## Recommended measured configuration

Set the following before importing the producer or constructing an engine:

```bash
export FLASHRT_NATIVE_A8=1 FLASHRT_NATIVE_LM_HEAD=1
export FLASHRT_NATIVE_ATTN=1 FLASHRT_NATIVE_FUSE=1
export FLASHRT_NATIVE_ATTN_MAX=8 FLASHRT_NATIVE_ATTN_MAX_FP8=8
export FLASHRT_NATIVE_A8_GU_MIN_M=1024
export FLASHRT_NATIVE_FP4_MOE_MIN_M=1024 FLASHRT_NATIVE_FP4_SF_CACHE=0
export FLASHRT_NATIVE_DENSE_FP8_MIN_M=1024 FLASHRT_SAGE_PREFILL=1
export VLLM_DISABLE_COMPILE_CACHE=1
```

The switches above select FlashRT operators for **vLLM + FlashRT**.
For the **vLLM** baseline, set `FLASHRT_STRUCTURES_NATIVE=0` and
`FLASHRT_SAGE_PREFILL=0`.
All scheduling flags and input tokens must match. Prefix caching is disabled
for the reported timings.

```bash
export MODEL="<official-nvfp4.gguf>" CONFIG="<model-config-and-tokenizer>"
FLASHRT_STRUCTURES_NATIVE=1 VLLM_PLUGINS=flashrt_native \
  python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --hf-config-path "$CONFIG" --tokenizer "$CONFIG" \
  --host 127.0.0.1 --port 8000 --served-model-name flint \
  --max-model-len 110592 --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.85 --no-enable-prefix-caching
```

The serving recipe uses one tensor/pipeline lane, no expert parallelism,
no speculative/resident loop and one sequential execution lane. Do not enable
diagnostics when reproducing the FP8-KV tables. Experimental NVFP4 shadow or
packed KV caches, fake attention/KV quantization and per-32 weight requantization
are not included in this integration; their old switches explicitly refuse.
Quantization, packing and compute kernels belong to the FlashRT producer.
Dynamic/non-unit KV scales are unsupported here.

## Historical batch-8 decode result

An earlier paired measurement on RTX 5090 recorded **up to 21% higher batch-8
decode throughput (1.21×)** with FlashRT:

| Prompt tokens/request | Batch | vLLM (tok/s) | vLLM + FlashRT (tok/s) | Speedup | Throughput gain |
|---:|---:|---:|---:|---:|---:|
| 1,022 | 8 | 1417 | **1715** | **1.21×** | +21.0% |

Historical record: 2026-10-23, BF16 KV, GPU memory utilization 0.94, 256 output
tokens per request, maximum 64 sequences. Each speedup uses its paired vLLM
baseline. This observation used an earlier development build on a different
RTX 5090 system from the current-build measurements below. It is a historical
result, not a remeasurement or performance qualification of the current PR
head; results vary across systems and runs.

## Current-build performance protocol

Measured 2026-10-24 (UTC+8), using the same built artifacts and identical
distinct public llama.cpp token windows within each pair. Each arm uses a fresh
process, one warmup and two measured passes: 128 output tokens for 1K prompts and 64 for long prompts,
greedy with EOS ignored. Tables take the median of the two measured pass
values (per-pass steady-step P50 for decode). Graph policy, KV dtype, memory fraction, batch budget
and input windows match within each pair. Short-context capacity is 8192;
long-context capacity is 110592; the scheduling budget is 8192 tokens.

Decode is `active batch / median steady engine-step wall time`, excluding all
prefill/first-token steps and partial-batch tails. Prefill is engine-step wall
time through the first generated token (through all first tokens for a batch).
This includes the final prompt chunk. Tokenization, loading and graph setup
are excluded. HTTP and request transport are excluded.
Concurrency does not promise a constant execution batch throughout a request;
only full steady decode steps contribute to the decode metric.

**vLLM** and **vLLM + FlashRT** use the same pinned/patched vLLM build;
only the FlashRT acceleration switches differ.

### Decode

| Prompt tokens/request | Batch | vLLM (tok/s) | vLLM + FlashRT (tok/s) | Speedup | Throughput gain |
|---:|---:|---:|---:|---:|---:|
| 1,022 | 1 | 275.7 | **530.7** | **1.93×** | +92.5% |
| 32,768 | 1 | 253.6 | **408.9** | **1.61×** | +61.2% |
| 24,998–32,768 | 4 | 728.1 | **799.4** | **1.10×** | +9.8% |
| 99,986 | 1 | 217.3 | **332.0** | **1.53×** | +52.7% |

<details>
<summary>Current-build batch-8 comparison (FP8 KV)</summary>

| Prompt tokens/request | Batch | vLLM (tok/s) | vLLM + FlashRT (tok/s) | Speedup | Throughput gain |
|---:|---:|---:|---:|---:|---:|
| 1,022 | 8 | 1460.9 | **1613.1** | **1.10×** | +10.4% |

This is the current-build result under the FP8 KV protocol stated above;
it is separate from the historical BF16 KV measurement.

</details>

### Prefill

| Prompt tokens/request | Batch | vLLM (ms) | vLLM + FlashRT (ms) | Speedup | Latency reduction |
|---:|---:|---:|---:|---:|---:|
| 1,022 | 1 | 44.40 | **41.69** | **1.07×** | 6.1% |
| 1,022 | 8 | 237.52 | **152.95** | **1.55×** | 35.6% |
| 32,768 | 1 | 1,552.31 | **839.95** | **1.85×** | 45.9% |
| 24,998–32,768 | 4 | 5,771.01 | **3,161.68** | **1.83×** | 45.2% |
| 99,986 | 1 | 8,776.40 | **4,125.84** | **2.13×** | 53.0% |

The four-request cell uses three 32,768-token windows and one 24,998-token window in both arms. The measured pass-P50 decode ranges span at most 0.8% within a cell.


### Quality

| Configuration | Mean KL to vLLM BF16 logits | Top-1 agreement |
|---|---:|---:|
| vLLM, FP8 KV | 0.0048 | 98.8% |
| vLLM + FlashRT, FP8 KV | 0.0045 | 98.0% |

Teacher-forced comparison uses four real coding prompts and 63 continuation
positions each (252 total); the initial prompt-logit row is excluded.
This is a focused logit comparison to the vLLM BF16 reference, not an FP32-reference
score or a broad held-out task evaluation. Low-precision FlashRT operators are
not expected to reproduce vLLM's entire greedy continuation token for token.

## Reproduce

The input recipe stores only small templates, public source locations and
digests. It reconstructs the eight original token sequences exactly from
llama.cpp `1b44e30775dcb2f93952116b05f89e5bd797fb00` and the official Flint
tokenizer. It does not ship generated token arrays, private prompts or logs.

```bash
git clone https://github.com/ggml-org/llama.cpp.git build/llama.cpp
python benchmarks/make_flint_inputs.py --llama-dir build/llama.cpp \
  --gguf "$MODEL" --out build/flint-inputs.json
# Check GPU occupancy before starting. Alternate vLLM / vLLM + FlashRT order across pairs.
FLASHRT_SAGE_PREFILL=0 python benchmarks/flint_vllm.py --arm host \
  --model "$MODEL" --config "$CONFIG" --tokens build/flint-inputs.json \
  --concurrency 1 --in-len 1022 --max-len 8192 --nout 128 --reps 2 --out build/host.json
python benchmarks/flint_vllm.py --arm native \
  --model "$MODEL" --config "$CONFIG" --tokens build/flint-inputs.json \
  --concurrency 1 --in-len 1022 --max-len 8192 --nout 128 --reps 2 --out build/native.json
```

Repeat with concurrency 8 for the short-context pair. For long-context pairs,
use input lengths 32768 or 99986, capacity 110592 and 64 outputs. The
four-request long-context pair uses short final windows; reproduce it with
`--allow-ragged --concurrency 4 --in-len 32768`, and inspect `input_lengths`
instead of assuming all requests have the requested length. The script emits
actual lengths, token digests, per-step measurements, FlashRT execution records
and the ordinary guarded-seam ledger. It refuses a missing FlashRT hook or
unexpected guarded fallback.

## Lifecycle and limits

Bind happens after host weight loading and before tracing. Canonical packed
bytes are retained before Marlin conversion; replaced host weights move into
the ordinary WeightStore. Region seams, the optional LM head, attention
methods and layer composition restore on failed attachment or pre-capture
detach. Once graphs capture device pointers, destroy the engine/graphs before
detaching. Hot swapping, overlapping attachment and multiple execution lanes
sharing scratch are unsupported.

Seam calls/fallbacks are in `handle.report()` / `handle.summary()`. Explicit
host-stage activation is in `handle.records["native_regions"]`; it is not
misrepresented as guarded-seam coverage. Graph replays do not increment Python
call counters. Standalone kernel/capture tests and checkpoint quality qualify
the selected path; no cross-model, SGLang or other-GPU performance is claimed.

## References

- [RecursiveSI Flint](https://huggingface.co/RecursiveSI/flint) and the
  [official NVFP4 GGUF](https://huggingface.co/RecursiveSI/flint-nvfp4-gguf): model, tokenizer and checkpoint.
- [FlashRT](https://github.com/flashrt-project/FlashRT): compute operators and catalog;
  [FlashRT Structures](https://github.com/flashrt-project/FlashRT-Structures): host integration.
- [vLLM](https://github.com/vllm-project/vllm): host engine and pinned Flint model implementation;
  [source installation](https://docs.vllm.ai/en/latest/contributing/code/) documents precompiled-wheel builds.
- [SageAttention](https://github.com/thu-ml/SageAttention),
  [FlashInfer](https://github.com/flashinfer-ai/flashinfer),
  [CUTLASS](https://github.com/NVIDIA/cutlass) and
  [Triton](https://github.com/triton-lang/triton): attention and quantized compute primitives.
  Adapted kernel sources retain their original notices in FlashRT's `csrc/sage2`.
- [llama.cpp](https://github.com/ggml-org/llama.cpp): public MIT-licensed code corpus for input fixtures;
  it is not the executor baseline in these vLLM measurements.
