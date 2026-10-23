"""The optional producer is resolved at bind and never required by default."""

from types import SimpleNamespace

import pytest

from flashrt_structures.backends import flint


@pytest.mark.parametrize("component", ["nvfp4", "sage2", "vllm"])
def test_absent_backend_refuses_with_install_remedy(monkeypatch, component):
    def absent(name):
        assert name == "flashrt_structures_flint." + component
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(flint.importlib, "import_module", absent)
    with pytest.raises(ValueError, match="separately built flashrt-structures-flint"):
        flint.load_module(component)


@pytest.mark.parametrize("component", ["nvfp4", "sage2", "vllm"])
def test_bind_resolves_installed_component(monkeypatch, component):
    sentinel = SimpleNamespace()
    names = []

    def installed(name):
        names.append(name)
        return sentinel

    monkeypatch.setattr(flint.importlib, "import_module", installed)
    assert flint.load_module(component) is sentinel
    assert names == ["flashrt_structures_flint." + component]


def test_disabled_plugin_does_not_resolve_optional_backend(monkeypatch):
    from flashrt_structures.adapters import vllm_native

    monkeypatch.delenv("FLASHRT_STRUCTURES_NATIVE", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("disabled plugin attempted backend resolution")

    monkeypatch.setattr(flint, "load_module", forbidden)
    assert vllm_native.register() is None
