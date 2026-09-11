"""The runtime contract across the split: a captured stage exported through
FlashRT's provider (``flash_rt.runtime.provider``) is adopted and ticked by
a native consumer (FlashRT-Nexus), and the boundary windows carry the
right values.

Needs a CUDA device, FlashRT's built ``exec/`` and ``runtime/`` (on
``PYTHONPATH`` or beside the flash_rt checkout) and a Nexus library named
by ``FLASHRT_NEXUS_LIB``; skips otherwise.
"""
import importlib.util
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device")


def _native_available():
    try:
        from flash_rt.runtime.exec import _import_native
        _import_native()
        import flash_rt.runtime.export  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _native_available(), reason="flash_rt exec/runtime not built")
@pytest.mark.skipif(importlib.util.find_spec("flashrt_nexus") is None, reason="flashrt_nexus")
@pytest.mark.skipif(not os.environ.get("FLASHRT_NEXUS_LIB"), reason="FLASHRT_NEXUS_LIB unset")
def test_captured_stage_exports_and_a_native_consumer_ticks_it():
    import flashrt_structures as s
    from flashrt_nexus import AdoptedRuntime

    torch.manual_seed(0)
    x = torch.zeros(4, 8, device="cuda")
    w = torch.randn(8, 8, device="cuda")
    out = torch.zeros(4, 8, device="cuda")

    def fn():
        out.copy_(torch.tanh(x @ w))
        return out

    stage = s.capture(fn, windows={"x": x, "out": out}, reference=None,
                      gate_cos=0, min_speedup=0, verbose=False)
    ports = [
        dict(name="x", modality="tensor", dtype="f32", layout="flat",
             direction="in", update="swap", required=True, shape=(4, 8), window="x"),
        dict(name="out", modality="tensor", dtype="f32", layout="flat",
             direction="out", update="swap", shape=(4, 8), window="out"),
    ]
    export = stage.export(ports=ports, identity={"producer": "structures-test"})
    runtime = AdoptedRuntime(export, owners=(stage,),
                             nexus_lib=os.environ["FLASHRT_NEXUS_LIB"])
    try:
        for _ in range(3):
            xi = torch.randn(4, 8, device="cuda")
            x.copy_(xi)
            torch.cuda.synchronize()
            runtime.step()
            want = torch.tanh(xi @ w)
            assert (out - want).abs().max().item() < 1e-5
    finally:
        runtime.close()
        export.release()
