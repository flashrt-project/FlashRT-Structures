"""Attention binding preserves producer/host routing without running GPU kernels."""

import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from flashrt_structures.adapters import vllm_native as adapter


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize(
    "phase,tokens,max_len",
    [("decode", 8, None), ("decode", 9, 1024), ("prefill", 8, None)],
)
def test_attention_routes_without_consumer_quantization(
    monkeypatch, fused, phase, tokens, max_len
):
    events = []

    class Metadata:
        pass

    metadata = Metadata()
    metadata.num_prefills = 1 if phase == "prefill" else 0
    metadata.num_decodes = 8 if phase == "decode" else 0
    metadata.num_decode_tokens = metadata.num_decodes
    metadata.decode = SimpleNamespace(
        block_tables=torch.zeros((8, 1), dtype=torch.int32),
        seq_lens=torch.ones(8, dtype=torch.int32),
        max_seq_len=max_len,
        q_len_per_req=1,
    )

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = SimpleNamespace(layer_name="attention")
            self.q_norm = SimpleNamespace(weight=torch.ones(2), variance_epsilon=1e-5)
            self.k_norm = SimpleNamespace(weight=torch.ones(2))
            self.softmax_offset = torch.zeros(2)
            self.impl = SimpleNamespace(window_left=-1)
            self.head_dim = 2
            self.num_heads = self.num_kv_heads = 2
            self.q_size = self.kv_size = 4
            self.cache = torch.zeros((1, 16, 2, 2), dtype=torch.uint8)

        def _cache_views(self):
            return self.cache, self.cache.clone(), torch.arange(tokens)

        def _cos_sin(self):
            return None

    def qk_cache(qkv, *args):
        events.append("qk_cache")
        return qkv[:, :4]

    def host(q, k, v, out, layer_name):
        events.append("host")
        assert layer_name == "attention"
        out.fill_(3)

    def producer(kind):
        def call(x, *args):
            events.append(kind)
            assert args[-1] == (max_len or 0)
            return torch.full((x.shape[0], 4), 2, dtype=x.dtype)

        return call

    modules = {
        "vllm.forward_context": SimpleNamespace(
            get_forward_context=lambda: SimpleNamespace(
                attn_metadata={"attention": metadata}
            )
        ),
        "vllm.model_executor.layers.attention.attention": SimpleNamespace(
            unified_attention_with_output=host
        ),
        "vllm.models.flint.model": SimpleNamespace(FlintAttention=Attention),
        "vllm.models.flint.ops.qk_norm_rope": SimpleNamespace(
            qk_norm_rope_cache=qk_cache
        ),
        "vllm.v1.attention.backends.flashinfer": SimpleNamespace(
            FlashInferMetadata=Metadata
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("FLASHRT_NATIVE_ATTN_FUSE", str(int(fused)))
    monkeypatch.setenv("FLASHRT_NATIVE_ATTN_MAX_FP8", "8")
    model = torch.nn.Module()
    model.attention = Attention()
    changes = adapter._AttributeChanges()
    api = SimpleNamespace(
        _decode_attention_qkv=producer("fused"), _decode_attention=producer("decode")
    )
    adapter._install_native_decode_attention(model, api, changes)
    output = model.attention._native_attention_core(
        torch.arange(tokens), torch.zeros((tokens, 12), dtype=torch.bfloat16)
    )
    assert output.shape == (tokens, 4)
    assert events == (
        ["qk_cache", "host"]
        if phase == "prefill"
        else (["fused"] if fused else ["qk_cache", "decode"])
    )
    if phase == "decode" and tokens > 8:
        assert torch.equal(output[8:], torch.zeros_like(output[8:]))
    changes.restore()
    assert not hasattr(model.attention, "_native_attention_core")
    assert not hasattr(model.attention, "_native_decode_plan")
