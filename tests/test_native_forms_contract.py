"""CPU contracts for native forms, using independent Tensor-facing fixtures."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from flashrt_structures import swap
from flashrt_structures.guard import GuardRefused
from flashrt_structures.impls.linear_proj.native_w4a16 import LinearProjNative


def test_native_projection_guard_consume_restore_and_compilation():
    torch.manual_seed(1)
    host = torch.nn.Linear(32, 64, bias=False, dtype=torch.bfloat16)
    state = {k: v.clone() for k, v in host.state_dict().items()}
    weight = host.weight.detach().clone()

    class Bound:
        matrix = SimpleNamespace(
            k=32, n=64, codes=torch.empty(64, 16, dtype=torch.uint8)
        )

        def __call__(self, x):
            return torch.nn.functional.linear(x, weight)

    seam = LinearProjNative(Bound(), host, capacity=4)
    model = torch.nn.Sequential(host).eval()
    handle = swap.attach(model, {"0": seam}, on_guard_fail="raise")
    try:
        x = torch.randn(2, 32, dtype=torch.bfloat16)
        assert torch.equal(model(x), torch.nn.functional.linear(x, weight))
        assert handle.summary()["guarded_calls"] > 0
        assert handle.summary()["fallbacks"] == 0
        assert torch.equal(seam.state_dict()["weight"], state["weight"])
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        assert torch.equal(compiled(x), model(x))
        with pytest.raises(GuardRefused, match="capacity"):
            model(torch.ones(5, 32, dtype=torch.bfloat16))
        with pytest.raises(GuardRefused, match="contiguous"):
            model(torch.ones(32, 2, dtype=torch.bfloat16).T)
        with pytest.raises(GuardRefused, match="rank"):
            model(torch.ones(32, dtype=torch.bfloat16))
        with pytest.raises(GuardRefused, match="migrate"):
            model.to(torch.float32)
        # This uses the existing off-device store rather than a second weight
        # residency mechanism. Compiled callers are discarded before detach.
        del compiled
        handle.consume()
        assert host.weight.is_meta
        assert torch.equal(model(x), torch.nn.functional.linear(x, weight))
    finally:
        handle.detach()
    assert model[0] is host and torch.equal(host.weight, state["weight"])
    handle.detach()
