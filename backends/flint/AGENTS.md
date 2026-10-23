# Flint operator backend

This subtree is the explicitly authorized operator producer exception in the
root AGENTS.md. It is a separate distribution, not part of the core wheel.

- Own Tensor interfaces, CUDA/Triton, packing/quantization, dispatch and
  vLLM execution composition here. Keep discovery and lifecycle in core.
- Build source distributions and wheels independently; never reach into a
  FlashRT checkout or binary. Never compile at import or inference time.
- Maintain package/ABI metadata, device refusals and operator namespaces.
  Optional imports must stay optional. Resolve consumer capabilities at bind.
- Preserve source licenses, NOTICE and exact origin revisions in sources.json.
  Canonical standalone kernels stay in FlashRT. Update support snapshots
  explicitly and requalify them; do not edit the canonical producer by proxy.
- Check GPU occupancy and use an exclusive experiment lock before GPU runs.
- Require CPU refusals, GPU numerics, changed-input capture and real-host
  quality gates. Use paired, matched decode/prefill timings for performance.
- Commit source and small public recipes only; no binaries, checkpoints,
  machine paths, credentials, private prompts or generated traces.
