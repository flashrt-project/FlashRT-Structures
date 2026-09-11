# FlashRT Structures

Attach [FlashRT](https://github.com/flashrt-project/FlashRT)'s verified
acceleration structures to an **unmodified** PyTorch host: `lerobot`,
Isaac-GR00T, `openpi`, `transformers`, `diffusers`, and inside vLLM /
SGLang. No fork, no edit to the host's source. Kernels arrive from the
Hugging Face kernel hub at bind time.

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

## Tests

```bash
pip install -e . pytest
pytest tests            # the CPU set; two *_gpu tests need a CUDA device and hub access
```

## License

Apache-2.0, same as FlashRT.
