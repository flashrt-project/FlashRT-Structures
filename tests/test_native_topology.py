"""Cross-family topology recognition and negative controls, without vLLM."""

import pytest

torch = pytest.importorskip("torch")
from flashrt_structures.adapters.vllm_native import _check_silu, routed_sites


def host(hidden, inter, experts):
    root = torch.nn.Module()
    root.gate = torch.nn.Linear(hidden, experts, bias=False)
    root.shared_expert = torch.nn.Module()
    root.shared_expert.gate_up_proj = torch.nn.Linear(hidden, 2 * inter, bias=False)
    root.shared_expert.down_proj = torch.nn.Linear(inter, hidden, bias=False)
    root.experts = torch.nn.Module()
    root.experts.bank = torch.nn.Module()
    root.experts.bank.w13_weight = torch.nn.Parameter(
        torch.zeros(experts, 2 * inter, hidden)
    )
    root.experts.bank.w2_weight = torch.nn.Parameter(
        torch.zeros(experts, hidden, inter)
    )
    return root


def test_topology_independent_of_shape_and_class_name():
    for dims in [(32, 16, 8), (48, 24, 12)]:
        module = host(*dims)
        found = list(routed_sites(module))
        assert len(found) == 1 and found[0][1] is module
        # Recognition does not imply this narrower native artifact admits it.
        del module.shared_expert.down_proj
        assert list(routed_sites(module)) == []
    assert list(routed_sites(torch.nn.Linear(32, 32))) == []


def test_observed_activation_dataflow_and_negative_control():
    class Projection(torch.nn.Module):
        input_size_per_partition = 32
        output_size_per_partition = 64

        def forward(self, x):
            raise AssertionError("real projection must not execute in activation probe")

    class Ffn(torch.nn.Module):
        def __init__(self, gelu=False):
            super().__init__()
            self.gate_up_proj = Projection()
            self.down_proj = Projection()
            self.gelu = gelu

        def forward(self, x):
            gu, _ = self.gate_up_proj(x)
            gate, up = gu.chunk(2, dim=-1)
            fn = torch.nn.functional.gelu if self.gelu else torch.nn.functional.silu
            act = (fn(gate.float()) * up.float()).to(gu.dtype)
            return self.down_proj(act)[0]

    for gelu in [False, True]:
        module = Ffn(gelu)
        original = (module.gate_up_proj, module.down_proj)
        if gelu:
            with pytest.raises(ValueError, match="dataflow"):
                _check_silu(module, torch.device("cpu"))
        else:
            _check_silu(module, torch.device("cpu"))
        assert (module.gate_up_proj, module.down_proj) == original


def test_backend_parameter_alias_follows_existing_store_and_restore():
    from types import SimpleNamespace

    from flashrt_structures.adapters.vllm_native import _parameter_alias_refresh
    from flashrt_structures.storage import WeightStore

    bank = torch.nn.Module()
    bank.register_parameter(
        "scale", torch.nn.Parameter(torch.arange(8.0), requires_grad=False)
    )
    expected = bank.scale.detach().clone()
    desc = SimpleNamespace(scale=bank.scale, alpha_or_gscale=None, zp=None, bias=None)
    config = SimpleNamespace(_a1=None, _a2=None, _w1=desc, _w2=desc)
    bank.quant_method = SimpleNamespace(moe_quant_config=config)
    refresh = _parameter_alias_refresh(bank)
    store = WeightStore()
    store.stash_module("bank", bank)
    refresh()
    assert desc.scale is bank.scale and desc.scale.is_meta
    store.restore_module(bank)
    refresh()
    assert desc.scale is bank.scale and torch.equal(desc.scale, expected)
    desc.scale = torch.ones(3)
    with pytest.raises(ValueError, match="restorable parameter alias"):
        _parameter_alias_refresh(bank)
