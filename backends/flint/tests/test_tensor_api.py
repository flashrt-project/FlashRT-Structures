"""CPU validation and opt-in GPU numerics for the separate Flint Tensor API."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
from flashrt_structures_flint import nvfp4 as api
from flashrt_structures_flint import sage2


def test_tensor_artifact_refuses_stale_abi_and_missing_symbols(monkeypatch):
    import flashrt_structures_flint

    monkeypatch.setattr(api, "_EXTENSION", None)
    fake = SimpleNamespace(TENSOR_API_METADATA={"abi": 0})
    monkeypatch.setattr(
        flashrt_structures_flint, "_flashrt_flint_ops", fake, raising=False
    )
    with pytest.raises(ValueError, match="ABI 1"):
        api.load()
    fake.TENSOR_API_METADATA = {"abi": 1, "capabilities": [(12, 0)]}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))
    with pytest.raises(ValueError, match="missing required symbols"):
        api.load()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))
    with pytest.raises(ValueError, match="capability"):
        api.load()


def test_sage2_missing_binary_refuses_without_compilation(monkeypatch):
    monkeypatch.setattr(sage2, "_OPS", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))

    def absent(name):
        raise ImportError(name)

    monkeypatch.setattr(sage2.importlib, "import_module", absent)
    with pytest.raises(ValueError, match="build it explicitly"):
        sage2.load()


def test_packed_shape_and_cpu_tensor_are_refused():
    with pytest.raises(ValueError, match="K divisible"):
        api.PackedMatrix(
            torch.zeros((128, 3), dtype=torch.uint8),
            torch.zeros((128, 1), dtype=torch.uint8),
            1.0,
        ).validate()
    with pytest.raises(ValueError, match="contiguous tensor"):
        api.PackedMatrix(
            torch.zeros((128, 32), dtype=torch.uint8),
            torch.zeros((128, 4), dtype=torch.uint8),
            1.0,
        ).validate()


def _cuda():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120 and the optional Flint binaries")
    pytest.importorskip("flashrt_structures_flint._flashrt_flint_ops")


@pytest.mark.parametrize("rows,a8", [(1, False), (8, False), (8, True), (256, True)])
def test_linear_tensor_api_numerics_and_changed_input_capture(monkeypatch, rows, a8):
    _cuda()
    monkeypatch.setattr(api, "_A8", a8)
    monkeypatch.setattr(api, "_A8_DENSE_MIN_M", 2)
    torch.manual_seed(7)
    n, k = 256, 2048
    codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    scales = torch.full((n, k // 16), 0x38, dtype=torch.uint8, device="cuda")
    levels = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device="cuda",
    )
    w = levels[torch.stack([codes & 15, codes >> 4], -1).long()].reshape(n, k) * 0.125
    linear = api.Linear(api.PackedMatrix(codes, scales, 0.125))
    x = torch.randn((rows, k), dtype=torch.bfloat16, device="cuda")
    linear(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            linear(x)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = linear(x)
    for shift in [0.0, 0.25]:
        x.add_(shift)
        graph.replay()
        expected = x.float() @ w.T
        error = (result.float() - expected).norm() / expected.norm()
        assert error.item() < (0.025 if a8 else 0.006)


@pytest.mark.parametrize("prefix", [0, 71])
def test_sage2_causal_prefix_and_sink_matches_fp32_attention(prefix):
    _cuda()
    pytest.importorskip("flashrt_structures_flint._flashrt_flint_sage2")
    torch.manual_seed(11)
    nq, nk, hq, hk, d = 65, 65 + prefix, 8, 2, 128
    q = torch.randn((nq, hq, d), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((nk, hk, d), dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    sinks = torch.randn(hq, device="cuda")
    result = sage2.attention_fp8(q, k, v, sinks, d**-0.5)
    keys = k.float().repeat_interleave(hq // hk, 1)
    values = v.float().repeat_interleave(hq // hk, 1)
    logits = torch.einsum("qhd,khd->hqk", q.float(), keys) * d**-0.5
    mask = (
        torch.arange(nk, device="cuda")[None]
        > torch.arange(nq, device="cuda")[:, None] + prefix
    )
    logits.masked_fill_(mask[None], float("-inf"))
    with_sink = torch.cat([logits, sinks[:, None, None].expand(hq, nq, 1)], -1)
    probs = with_sink.softmax(-1)[..., :nk]
    expected = torch.einsum("hqk,khd->qhd", probs, values)
    assert ((result.float() - expected).norm() / expected.norm()).item() < 0.05


def _packed_reference(n, k, global_scale):
    codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    scales = torch.full((n, k // 16), 0x38, dtype=torch.uint8, device="cuda")
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda"
    )
    weight = levels[torch.stack([codes & 15, codes >> 4], -1).long()].reshape(n, k)
    return api.PackedMatrix(codes, scales, global_scale), weight * global_scale


@pytest.mark.parametrize("rows,a8", [(1, False), (8, False), (1, True), (8, True)])
def test_dense_ffn_and_head_changed_input_graph_reference(monkeypatch, rows, a8):
    _cuda()
    monkeypatch.setattr(api, "_A8", a8)
    torch.manual_seed(17)
    gu, wgu = _packed_reference(2048, 2048, 0.03125)
    down, wd = _packed_reference(2048, 1024, 0.03125)
    head, wh = _packed_reference(256, 2048, 1.0)
    dense, lm = api.DenseFfn(gu, down), api.LmHead(head)
    x = torch.randn((rows, 2048), dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            dense(x)
            lm(x)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual_dense, actual_head = dense(x), lm(x)
    for shift in [0.0, 0.25]:
        x.add_(shift)
        graph.replay()
        gu_ref = (x.float() @ wgu.T).reshape(rows, -1, 16)
        gate, up = gu_ref[..., :8].reshape(rows, -1), gu_ref[..., 8:].reshape(rows, -1)
        expected_dense = (torch.nn.functional.silu(gate) * up).to(
            torch.bfloat16
        ).float() @ wd.T
        expected_head = x.float() @ wh.T
        for actual, expected in [
            (actual_dense, expected_dense),
            (actual_head, expected_head),
        ]:
            assert ((actual.float() - expected).norm() / expected.norm()).item() < (
                0.04 if a8 else 0.015
            )


@pytest.mark.parametrize("fp8,window", [(False, 0), (False, 32), (True, 0), (True, 32)])
def test_paged_attention_matches_fp32_reference(fp8, window):
    _cuda()
    api.load()
    torch.manual_seed(23)
    q = torch.randn((8, 32 * 128), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((4, 16, 4, 128), dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    if fp8:
        k, v = k.to(torch.float8_e4m3fn), v.to(torch.float8_e4m3fn)
    keys = k.float().reshape(64, 4, 128).repeat_interleave(8, 1)
    values = v.float().reshape(64, 4, 128).repeat_interleave(8, 1)
    tables = torch.arange(4, dtype=torch.int32, device="cuda").repeat(8, 1)
    lengths = torch.full((8,), 64, dtype=torch.int32, device="cuda")
    sinks = torch.randn(32, device="cuda")
    actual = api._decode_attention(
        q,
        k.view(torch.uint8) if fp8 else k,
        v.view(torch.uint8) if fp8 else v,
        tables,
        lengths,
        sinks,
        window,
        128**-0.5,
        64,
    )
    logits = (
        torch.einsum("bhd,khd->bhk", q.float().reshape(8, 32, 128), keys) * 128**-0.5
    )
    if window:
        logits[..., : 64 - window] = float("-inf")
    probs = torch.cat([logits, sinks[None, :, None].expand(8, 32, 1)], -1).softmax(-1)[
        ..., :64
    ]
    expected = torch.einsum("bhk,khd->bhd", probs, values).reshape(8, -1)
    assert ((actual.float() - expected).norm() / expected.norm()).item() < 0.01


def test_backend_imports_without_flashrt_or_structures():
    import subprocess
    import sys

    code = """
import importlib.abc
import sys

class BlockConsumers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'flash_rt', 'flashrt_structures'}:
            raise ImportError('consumer import forbidden: ' + fullname)

sys.meta_path.insert(0, BlockConsumers())
from flashrt_structures_flint import nvfp4, sage2, fp4_moe, dense_fp8, vllm
assert not any(n.split('.')[0] in {'flash_rt', 'flashrt_structures'} for n in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
