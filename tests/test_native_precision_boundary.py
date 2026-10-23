"""Consumer integration refuses undeclared precision instead of implementing it."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
from flashrt_structures.adapters import vllm_native as adapter


_MODES = [
    ("FLASHRT_NATIVE_KV_V4", "1"),
    ("FLASHRT_NATIVE_PER32", "dense"),
    ("FLASHRT_NATIVE_PER32", "all"),
    ("FLASHRT_NATIVE_FAKE_ATTN_FP8", "1"),
    ("FLASHRT_NATIVE_FAKE_ATTN_FP8", "3"),
    ("FLASHRT_KV_FAKE_NVFP4", "k"),
    ("FLASHRT_KV_FAKE_NVFP4", "v"),
]


@pytest.fixture(autouse=True)
def clean_options(monkeypatch):
    for option, _ in _MODES:
        monkeypatch.delenv(option, raising=False)
    monkeypatch.setattr(adapter, "_INSTALLED", False)


@pytest.mark.parametrize("option,value", _MODES)
@pytest.mark.parametrize("entry", ["register", "attach", "capture"])
def test_removed_precision_mode_refuses_before_loading_producer(
    monkeypatch, option, value, entry
):
    monkeypatch.setenv(option, value)
    monkeypatch.setenv("FLASHRT_STRUCTURES_NATIVE", "1")
    with pytest.raises(ValueError, match=option):
        if entry == "register":
            adapter.register()
        elif entry == "attach":
            adapter.attach_native(None)
        else:
            with adapter.capture_weights(SimpleNamespace()):
                pytest.fail("capture entered with an unsupported precision mode")
    assert adapter._ACTIVE.get() is None
    assert not adapter._LOAD_LOCK.locked()
    assert not adapter._INSTALLED


@pytest.mark.parametrize("dtype", ["auto", "bfloat16", "fp8", "fp8_e4m3"])
def test_recommended_cache_modes_remain_admitted(dtype):
    adapter._check_native_options(
        SimpleNamespace(cache_config=SimpleNamespace(cache_dtype=dtype))
    )


def test_unpublished_packed_cache_refuses_before_capture():
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(cache_dtype="flashrt_kfp8_vfp4")
        )
    )
    with pytest.raises(ValueError, match="packed K-FP8/V-FP4"):
        with adapter.capture_weights(runner):
            pytest.fail("capture entered with an unsupported packed cache")
    assert adapter._ACTIVE.get() is None
    assert not adapter._LOAD_LOCK.locked()


def test_disabled_plugin_preserves_default_host_route(monkeypatch):
    monkeypatch.setenv("FLASHRT_STRUCTURES_NATIVE", "0")
    for option, value in _MODES:
        monkeypatch.setenv(option, value)
    assert adapter.register() is None
    assert not adapter._INSTALLED
