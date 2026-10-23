"""Host-stage mutations use the ordinary attachment rollback lifecycle."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from flashrt_structures.adapters.vllm_native import _AttributeChanges


def test_instance_mutations_restore_methods_fields_and_reverse_order():
    class Host(torch.nn.Module):
        def forward(self, value):
            return value + 1

    host, other = Host(), Host()
    changes = _AttributeChanges()
    events = []
    changes.undo.append(lambda: events.append("weights"))
    changes.set(host, "forward", lambda value: value * 2)
    changes.set(host, "_native_qw", torch.tensor([1.0]))
    changes.set(host, "forward", lambda value: value * 3)
    assert host(2) == 6 and other(2) == 3
    changes.restore()
    assert host(2) == 3 and not hasattr(host, "_native_qw")
    assert "forward" not in vars(host) and events == ["weights"]
    changes.restore()
    assert events == ["weights"]


def test_quant_method_and_weight_restore_share_one_revert_chain():
    host = torch.nn.Linear(2, 2, bias=False)
    original = host.weight.detach().clone()
    method = object()
    host.quant_method = method
    from flashrt_structures.storage import WeightStore

    store, changes = WeightStore(), _AttributeChanges()
    store.stash_module("lm_head", host)
    changes.undo.append(lambda: store.restore_module(host))
    changes.set(host, "quant_method", object())
    assert host.weight.is_meta
    changes.restore()
    assert host.quant_method is method
    assert torch.equal(host.weight, original)


def test_prefill_timing_includes_first_token_step():
    path = Path(__file__).parents[1] / "benchmarks/flint_vllm.py"
    spec = importlib.util.spec_from_file_location("flint_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Two prompt chunks: the second emits the first token and is still prefill.
    result = module.summarize_steps(
        [(2.0, 0, False), (3.0, 1, False), (0.5, 1, True), (0.5, 1, True)], 1
    )
    assert result["prefill_s"] == 5.0
    assert result["decode_tok_s"] == 2.0
    assert result["n_decode_steps"] == 2
