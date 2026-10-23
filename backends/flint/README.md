# Flint operator backend

`flashrt-structures-flint` is the independently built, optional SM120 operator
backend for Flint in vLLM. The import package is `flashrt_structures_flint`.
Core `flashrt-structures` does not build or install it by default.

It owns the Tensor API, dynamic quantization and packing, CUDA/Triton kernels,
row/shape dispatch, Sage2 and Flint/vLLM execution scheduling. Discovery,
Guard/WeightStore binding and attachment rollback remain in the core package.
It does not import FlashRT or Structures and never compiles during import,
model load or forward. Existing standalone Flint kernels remain in FlashRT.

## Build and install

Linux, RTX 5090 / SM120, CUDA 13.0, a matching CUDA-enabled PyTorch, Triton,
setuptools, wheel and Ninja are required. The qualified stack is PyTorch
2.13.0+cu130. Bring your own matching framework; no dependencies are downloaded
by the commands below.

```bash
TORCH_CUDA_ARCH_LIST=12.0a MAX_JOBS=2 \
  python -m pip wheel --no-build-isolation --no-deps ./backends/flint -w dist
python -m pip install --no-deps dist/flashrt_structures_flint-*.whl
python -m pip install --no-deps -e .
```

Both `_flashrt_flint_ops` and `_flashrt_flint_sage2` are compiled into this wheel.
They use their own extension and Torch operator namespaces. No FlashRT checkout,
compiled FlashRT module, source-root option or inference-time compiler is used.
`nvfp4.load()` and `sage2.load()` resolve installed artifacts and fail clearly
when the binary, ABI, hardware or symbols are missing. Loading the core package
with this backend absent preserves the default Hub/host behavior.

FP4 MoE and dense FP8 prefill optionally call the matching vLLM CUDA operators;
their tested vLLM source/binary pair is in [the integration guide](../../docs/flint_vllm.md).
Other models, GPU architectures and host versions are not qualified by this cell.

## Validation

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q backends/flint/tests
# Check GPU occupancy before enabling GPU tests.
python -m pytest -q backends/flint/tests
```

The package has separate ABI/device refusals, Tensor numerics, changed-input
graph capture and Sage2 tests. Core integration/lifecycle tests remain under
the repository's `tests/`. See `sources.json`, `NOTICE` and the source notices
for provenance and licenses. Compiler fast-math remains disabled.
