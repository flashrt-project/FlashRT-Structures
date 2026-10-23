# Contributing

Read [`AGENTS.md`](AGENTS.md) for the operating procedure of this layer
and [`docs/structure_contributing.md`](docs/structure_contributing.md)
for the contribution boundary and the copyable PR self-review checklist.

Structure specifications, their plain-torch references and the per-host
bindings live in the FlashRT repository under `flash_rt/catalog/`; a new
structure or binding is a change there, an executable form, adapter or
door is a change here.

## Optional operator backend ownership

The kernel-free rule applies to the core `flashrt_structures/` consumer.
This repository also maintains the explicitly separate producer distribution
[`flashrt-structures-flint`](backends/flint/README.md) in `backends/flint/`.
That package owns its Tensor API, CUDA/Triton kernels, quantization, packing,
shape dispatch, Sage2 and Flint/vLLM execution composition. This is an explicit
exception to the previous repository-wide prohibition on kernel source.

It has its own build metadata, native modules, operator namespaces, provenance
and tests. Core installation must neither build nor require it. Consumers
resolve the installed artifact at bind time and retain discovery, guards,
WeightStore, lifecycle, ledger and rollback. No private source injection or
inference-time compilation is allowed. Hardware support belongs to backend
metadata; no duplicate architecture policy is added to core implementations.
Missing artifacts refuse clearly. Existing standalone Flint CUDA kernels stay
in FlashRT; shared backend support is an attributed, pinned source snapshot.

Backend changes require independent source-package/wheel builds, CPU ABI/device
refusals, GPU arithmetic and changed-input capture tests, and relevant real-host
correctness. Changes to algorithms, precision or dispatch require paired
performance/quality qualification. A migration must compare the old and new
artifact at the same inputs and preserve standalone frontend regressions.
This exception does not license kernels in core adapters, impls or catalog
specifications, nor bypass other producer packages' ownership or release gates.
