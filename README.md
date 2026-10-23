# FlashRT Structures

Attach [FlashRT](https://github.com/flashrt-project/FlashRT)'s verified
acceleration structures to an **unmodified** PyTorch host: `lerobot`,
Isaac-GR00T, `openpi`, `transformers`, `diffusers`, and inside vLLM /
SGLang. No fork, no edit to the host's source. Kernels arrive from the
Hugging Face kernel hub at bind time.
The experimental Flint integration below additionally requires a pinned host patch.

```python
import flashrt_structures as structures

plan = structures.attach(model, forward)   # discover → calibrate → gate → activate
print(structures.explain(plan))            # bound / routed / kept-at-host / refused, with reasons

loop = structures.decode_loop(model, max_len=4096)      # serving door
out  = loop.generate(input_ids, max_new_tokens=256)
```

Measured on VLA, VLM, LLM and video models across RTX 5090 and Jetson AGX
Thor; the films are in the
[walkthrough](https://huggingface.co/spaces/liangsu9988/fast-kernels-are-not-fast-pipelines).

## What this package is, and what it is not

A *structure* is a versioned specification of one model region: boundary
tensors, framework-neutral weight slots, calibration points, gates, and a
plain-torch reference that is the gate's ground truth. Those
specifications, and the per-host *bindings* that say where the positions
sit on a concrete host, live in the FlashRT repository as
`flash_rt.catalog` and ship in the pure-Python `flash-rt` wheel.

This package is everything that acts on them for a PyTorch host:

| layer | here |
|---|---|
| `impls/` | executable forms per structure family: FP8 / NVFP4 / weight-only projections and FFNs, fused norm producers, attention cores, fused decoder and vision towers, the whole-step decode loop, graph-lowering pins |
| `adapters/` | host adapters: `transformers`, `diffusers`, Gemma / Qwen attention, gated-delta, and the vLLM / SGLang engine hooks |
| `discover.py`, `autobuild.py`, `points.py`, `schemes.py`, `gates.py` | structural discovery, the one-pass discover → calibrate → bind assembly, calibration collection, quantisation schemes, accuracy judgment |
| `guard.py`, `swap.py`, `frontdoor.py`, `stages.py`, `recipe.py` | the runtime contract and ledger, attach / detach, the one-call door, graph capture with declared swap windows, recipes |
| `examples/` | the explicit pipeline: seat tables, calibration hooks and binder calls written out by hand for GR00T N1.7 and π0.5 |

Structure specs, references and bindings are **not** here. A new structure
or a new host binding is a change to `flash_rt/catalog/` in FlashRT; a new
executable form, adapter or door is a change here.

## Install

```bash
pip install flashrt-structures            # pulls flash-rt (pure Python) for the catalog
pip install "flashrt-structures[hub]"     # + the kernel hub client, needed to bind
```

`flash-rt` must be a version that ships the catalog as `flash_rt.catalog`:
FlashRT `main` after the structures split, or a wheel newer than 0.1.0
(the 0.1.0 wheel on PyPI predates the split and still carries the layer
inside `flash_rt.structures`). An older flash-rt makes `import
flashrt_structures` fail with a message that says exactly this.

Bring your own torch. Which kernels exist is decided by the torch version:
the hub's published face is thickest at `torch 2.11 / cu128` on x86-64 and
much thinner at the newest release. If binds refuse on a fresh install,
check the torch version first.

Three boundaries produce refusals that look like bugs and are not:

- **Kernel availability follows the hub's build matrix, not this
  package's.** The wheel installs on any torch; the kernels do not exist
  for every torch.
- **`HF_HUB_OFFLINE=1` makes every kernel unavailable, even with a fully
  warm cache**, because a version specifier has to resolve refs online.
  Air-gapped deployments should stage packages and point at them with
  `LOCAL_KERNELS=<repo>=<path>` rather than switching the hub offline.
- **aarch64 (Jetson Thor, sm_110) is not covered by the current
  qualification pass.** Nothing in the wheel is architecture-bound, but
  the engine adapters were last verified against vLLM 0.26 on Thor in
  August 2026, not in the release run.

## Where to read next

- [`docs/hosts.md`](docs/hosts.md) — which door your host takes, how to
  tell a seated run from a refused one from a door that never fired, how
  to read a refusal, and what has been measured where.
- [`docs/structures.md`](docs/structures.md) — the norm: what a structure
  is, the three-layer split, calibration reuse, accuracy bands, the
  runtime contract and ledger, and the norms that came from being wrong.
- [`docs/serving_engines.md`](docs/serving_engines.md) — attaching inside
  vLLM or SGLang without forking either.
- [`docs/adopt_in_20_lines.md`](docs/adopt_in_20_lines.md) — a measured
  Qwen3-VL-8B adoption, twenty lines end to end.
- [`examples/README.md`](examples/README.md) — the explicit pipeline
  against the automatic one, measured on two GR00T N1.7 hosts.
- [`docs/structure_release_qualification.md`](docs/structure_release_qualification.md)
  — how a structure moves from a reusable boundary to a released hardware
  route.
- [`AGENTS.md`](AGENTS.md) — the operating procedure for producing
  structures; [`docs/structure_contributing.md`](docs/structure_contributing.md)
  — the contribution boundary and PR self-review checklist.

## Flint on vLLM (experimental, SM120)

An explicit native route attaches FlashRT's NVFP4/W4A8 projections, FFNs,
paged decode attention and quantized prefill to Flint in vLLM. The engine
keeps scheduling and KV allocation. This route requires separately built
FlashRT artifacts and a pinned Flint host patch; the general Hub route above
keeps its existing installation and host behavior.

RTX 5090, matched FP8 KV and graph policy; steady engine-step wall timing.

| 1,022-token prompt | Host decode | Native decode | Speedup |
|---|---:|---:|---:|
| Batch 1 | 275.7 tok/s | **530.7 tok/s** | **1.93×** |
| Batch 8 | 1460.9 tok/s | **1613.1 tok/s** | **1.10×** |

[Setup, batch 1–8 measurements and limits](docs/flint_vllm.md) ·
[Native operator ownership](docs/native_nvfp4.md).

## Tests

The experimental [native prequantized W4A16 route](docs/native_nvfp4.md) uses
an independently built and installed FlashRT Tensor artifact. It shares the
engine lifecycle, guards, weight store and ledger with the other forms; its
support and qualification limits are documented separately.

```bash
pip install -e . pytest
pytest tests            # the CPU set; two *_gpu tests need a CUDA device and hub access
```

## Sources and credit

[FlashRT](https://github.com/flashrt-project/FlashRT) provides the structure
catalog and native producers; this package owns the host adapters and
attachment lifecycle. Model and serving ownership remain with
[RecursiveSI Flint](https://huggingface.co/RecursiveSI/flint) and
[vLLM](https://github.com/vllm-project/vllm). The optional prefill route adapts
[SageAttention](https://github.com/thu-ml/SageAttention) through FlashRT, with
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[CUTLASS](https://github.com/NVIDIA/cutlass) and
[Triton](https://github.com/triton-lang/triton) primitives. Native source and
license notices are maintained in the
[producer's vendor record](https://github.com/flashrt-project/FlashRT/blob/feat/flint-w4a8-sm120/csrc/sage2/VENDOR.md).
The Flint benchmark corpus comes from [llama.cpp](https://github.com/ggml-org/llama.cpp);
vLLM is the executor baseline for those measurements. For FlashRT's paper
citation, see the [upstream citation section](https://github.com/flashrt-project/FlashRT#citation).

## License

Apache-2.0, same as FlashRT.
