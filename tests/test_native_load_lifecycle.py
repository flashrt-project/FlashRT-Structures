"""The native path extends the existing hook; its default remains compatible."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("custom", [False, True])
def test_hook_default_and_scoped_preload_are_single_attachment(monkeypatch, custom):
    import importlib

    from flashrt_structures.adapters import vllm_engine as engine

    events = []

    class Runner:
        def load_model(self):
            events.append("load")
            self.model = object()

    real_import = importlib.import_module

    def import_module(name, *args, **kwargs):
        if name in [
            "vllm.v1.worker.gpu.model_runner",
            "vllm.v1.worker.gpu_model_runner",
        ]:
            return SimpleNamespace(GPUModelRunner=Runner)
        if name == "vllm.v2.worker.gpu_model_runner":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setattr(engine, "_patch_would_not_survive", lambda: None)

    def attach(model, **kwargs):
        events.append("attach")
        return object()

    monkeypatch.setattr(engine, "attach_engine", attach)

    @contextmanager
    def scope(runner):
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    before = len(engine._ATTACHED)
    kwargs = {"attach_fn": attach, "load_context": scope} if custom else {}
    patched = engine.install_load_hook(**kwargs)
    assert len(patched) == 1  # two module paths can export the same class
    runner = Runner()
    runner.load_model()
    assert events == (
        ["enter", "load", "attach", "exit"] if custom else ["load", "attach"]
    )
    assert len(engine._ATTACHED) == before + 1
    assert runner._frt_load_hook_active is False


def test_weight_capture_failure_leaves_runner_reenterable(monkeypatch):
    import importlib

    from flashrt_structures.adapters import vllm_engine as engine

    events = []

    class Runner:
        def load_model(self):
            raise RuntimeError("failed load")

    real_import = importlib.import_module

    def import_module(name, *args, **kwargs):
        if name == "vllm.v1.worker.gpu.model_runner":
            return SimpleNamespace(GPUModelRunner=Runner)
        if name.startswith("vllm."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setattr(engine, "_patch_would_not_survive", lambda: None)

    @contextmanager
    def scope(runner):
        try:
            yield
        finally:
            events.append("exit")

    engine.install_load_hook(load_context=scope)
    runner = Runner()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="failed load"):
            runner.load_model()
        assert runner._frt_load_hook_active is False
    assert events == ["exit", "exit"]
