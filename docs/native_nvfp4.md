# FlashRT prequantized W4A16 integration (development)

This integration uses separately built FlashRT Tensor operators in vLLM.
It does not use the HF kernel client or inject a kernel source checkout into
the host. Build and qualify `flash_rt.native_kernels.nvfp4` first, install
its Python package and native binary into the same environment, then install
this Structures branch. See the producer's `docs/native_nvfp4_api.md` for the
artifact ABI, capabilities, packing and standalone qualification.

## Coverage and ownership

| Region | Form | State/coverage boundary |
|---|---|---|
| Projection | `linear_proj/native_w4a16` | contiguous BF16 inputs, original checkpoint quantization |
| Dense FFN | `decoder_ffn/native_w4a16` | precise-SiLU gate/up plus down, losslessly interleaved at bind |
| Coupled sparse FFN | `routed_ffn/native_w4a16` | router + routed bank + ungated shared expert; artifact-qualified sigmoid/top6 profile |
| Norm/fusion host stage | optional | explicit decode-layer fusion; restored with the attachment |
| Attention and KV | optional paged attention | host keeps KV allocation; see the pinned [Flint cell](flint_vllm.md) |
| Token selection/scheduling | host retained | vLLM owns these stages |

`routed_ffn` describes a larger dataflow than `moe_experts`: router scores,
selection, expert reduction and shared contribution belong to its boundary.
Recognition uses slots and tensor relations. Model IDs and concrete module
class names do not select it. Its topology occurs across sparse-decoder
families; the first native numerical/shape profile remains separately scoped.
Topology recognition and CPU reference tests are not cross-model GPU speed
qualification. Unmeasured profiles remain unqualified.

This first binding is provisional: the reference boundary also occurs in
DeepSeek-V2-style routed/shared FFNs, but that family's addressing and GPU
performance have not been qualified by this route. It is not selected by an
automatic certified-fast release policy.

## Lifecycle

The existing `vllm_engine.install_load_hook` accepts optional `attach_fn` and
`load_context` callbacks. Default calls keep the existing attachment behavior.
The context covers load and attachment; `attach_fn(model, **kwargs)` runs after
load and before tracing. Aliases and delegated load calls attach once. A
failed load unwinds the context and does not mark the runner active forever.

The FlashRT adapter uses this window to retain canonical packed bytes in CPU
memory before Marlin conversion. It offloads replaced host weights through the
existing `WeightStore` before uploading each replacement, then commits with
`swap.attach(..., store=store)` and `handle.consume(store)`. This avoids a second resident GPU
expert bank. CPU restore tickets consume host memory and are part of the
setup budget. Sharing scratch through `workspace.lease` does not share outputs
or KV state. The qualified execution lane is single-stream and sequential;
concurrent lanes need independent scratch.

Every form subclasses `GuardedSeam`. This explicit configuration is strict:
an unsupported dtype, shape or layout refuses. It does not silently dispatch
the active region back to the host. `handle.report()` and `handle.summary()`
are the ordinary ledger; `handle.records` adds region and offload receipts.
Counts cover eager/capture execution, not graph replays. Verify captured paths
with a profiler or corresponding replay parity tests.

The standalone compile contract is Dynamo full-graph tracing with the eager
backend, plus changed-input CUDA Graph replay. This does not certify Inductor
fusion or performance. Use the host's supported graph policy identically in
both comparison arms.

`handle.detach()` restores original modules and stored parameter bytes before
graph capture. Once the engine has captured references to those buffers,
destroy the engine/graphs before detach or freeing owners. Online migration,
reload, concurrent attachment and hot swapping captured buffers are unsupported.
Detach before serializing consumed host weights. Do not claim original vLLM
parameter-subclass metadata survives its generic off-device restore tickets;
recreate an engine before using host-specific checkpoint reload machinery.

## Optional host stages

The explicit Flint profile may also replace the vocabulary head, paged decode
attention and layer-level decode composition. These are model host stages,
not new general structures. They are recorded in `handle.records["native_regions"]`
outside the per-seam ledger. Original methods and head weights are restored
on attach failure or pre-capture detach through the same handle and WeightStore.
The optional Sage2 producer must already be built; no source lookup or kernel
compilation occurs in this adapter.

## Opt-in serving

The `vllm.general_plugins` entry installs independently in every worker. With
the environment variable absent, it makes no changes to the engine.

```bash
# Set model/config/tokenizer paths to your local copies. These are examples.
export MODEL="<official-nvfp4.gguf>"
export CONFIG="<config-and-tokenizer>"

FLASHRT_STRUCTURES_NATIVE=1 VLLM_PLUGINS=flashrt_native \
  python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --hf-config-path "$CONFIG" --tokenizer "$CONFIG" \
  --host 127.0.0.1 --port 8000 --served-model-name model \
  --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.94 --no-enable-prefix-caching
```

Use `FLASHRT_STRUCTURES_NATIVE=0` with identical arguments for the control.
Keep the same compile-cache policy in both arms. Add the model's required
reasoning/tool-parser options in both. The first adapter cell requires one
tensor/pipeline lane with no expert parallelism. It qualifies only the
prequantized W4A16 interfaces present in the tested vLLM build, not arbitrary
vLLM versions or packed formats. Request concurrency is not a guarantee of a
fixed execution batch size.

## Acceptance

Record the producer revision/binary digest, consumer revision, model hash,
host source/binary revisions, runtime versions, hardware and precise boundary.
Require local arithmetic and real-checkpoint boundary parity, negative controls,
guards and restore, compile/capture, host repeat/null checks and real-input
model quality before claiming a completed adaptation. Timings use paired final
execution forms and include memory/KV/scheduling costs. Kernel smokes and
synthetic capacity tests are not held-out model-quality evidence.

Keep raw receipts, checkpoints, machine paths, private prompts and profiles
outside both source repositories. No artifact is published or certified merely
because it imported or one request completed.

For the tested W4A8/FP8/FP4 serving recipe, host version and independent
prefill/decode measurements, use [Flint on vLLM](flint_vllm.md).
