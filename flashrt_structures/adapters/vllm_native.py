"""Explicit prequantized native route over the standard engine lifecycle.

Capture lossless canonical bytes before the host's layout transform, bind
between load and trace, and retire replaced host weights through WeightStore.
No model ID, source path, environment name or checkpoint path is recognized.
"""

import logging
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from weakref import WeakKeyDictionary

import torch

from ..guard import GuardRefused

from .. import swap
from ..impls.decoder_ffn import native_w4a16 as dense_impl
from ..impls.linear_proj import native_w4a16 as linear_impl
from ..impls.routed_ffn import native_w4a16 as routed_impl
from ..storage import WeightStore

_ACTIVE = ContextVar("flashrt_native_weight_capture", default=None)
_LOAD_LOCK = threading.Lock()
_INSTALLED = False
_LOG = logging.getLogger(__name__)


def _check_native_options(config=None):
    """Refuse precision experiments that are outside this consumer integration."""
    for option in (
        "FLASHRT_NATIVE_KV_V4",
        "FLASHRT_NATIVE_PER32",
        "FLASHRT_NATIVE_FAKE_ATTN_FP8",
        "FLASHRT_KV_FAKE_NVFP4",
    ):
        if os.environ.get(option, "0") not in ("", "0"):
            raise ValueError(
                f"{option}: experimental precision mode is not provided by this integration"
            )
    cache = getattr(config, "cache_config", None)
    if getattr(cache, "cache_dtype", None) == "flashrt_kfp8_vfp4":
        raise ValueError("packed K-FP8/V-FP4 cache is not provided by this integration")


class _AttributeChanges:
    """Restore exact instance attributes after a failed attach or before capture."""

    def __init__(self):
        self.undo = []

    def set(self, obj, name, value):
        owned = name in vars(obj)
        previous = getattr(obj, name, None)

        def restore():
            if owned:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)

        self.undo.append(restore)
        setattr(obj, name, value)

    def restore(self):
        while self.undo:
            self.undo.pop()()


def _cpu(tensor):
    return tensor.detach().to(device="cpu", copy=True).contiguous()


class WeightCapture:
    def __init__(self, runner=None, profile=None):
        self.linear = WeakKeyDictionary()
        self.experts = WeakKeyDictionary()
        self.runner = runner
        self.profile = profile

    def projection(self, layer):
        if not getattr(layer, "is_w4a16_nvfp4", False):
            return
        if (
            layer.weight.dtype != torch.uint8
            or layer.weight_scale.dtype != torch.float8_e4m3fn
        ):
            return
        if (
            layer.weight_global_scale.numel() != 1
            or getattr(layer, "bias", None) is not None
        ):
            return
        self.linear[layer] = {
            "codes": _cpu(layer.weight),
            "scales": _cpu(layer.weight_scale.view(torch.uint8)),
            "gs": float(layer.weight_global_scale.item()),
        }

    def expert_bank(self, method, layer):
        if not method.use_a16:
            return
        # Only retain a bank that could serve the artifact's coupled profile.
        # Shape qualification is not identity-based discovery.
        if self.profile is None:
            raise ValueError(
                "native artifact did not declare a routed/shared shape profile"
            )
        h, e, _top, i, _shared = self.profile
        if tuple(layer.w13_weight.shape) != (e, 2 * i, h // 2) or tuple(
            layer.w2_weight.shape
        ) != (e, h, i // 2):
            return
        if (
            layer.w13_weight_scale.dtype != torch.float8_e4m3fn
            or layer.w2_weight_scale.dtype != torch.float8_e4m3fn
        ):
            return
        gs = layer.w13_weight_scale_2
        if gs.shape != (e, 2) or not torch.equal(gs[:, 0], gs[:, 1]):
            raise ValueError(
                "native routed FFN requires exactly equal gate/up global scales"
            )
        self.experts[layer] = {
            "gu_codes": _cpu(layer.w13_weight),
            "gu_scales": _cpu(layer.w13_weight_scale.view(torch.uint8)),
            "down_codes": _cpu(layer.w2_weight),
            "down_scales": _cpu(layer.w2_weight_scale.view(torch.uint8)),
            "gu_global": _cpu(gs[:, 0]),
            "down_global": _cpu(layer.w2_weight_scale_2),
        }


@contextmanager
def capture_weights(runner):
    _check_native_options(getattr(runner, "vllm_config", None))
    from flash_rt.native_kernels.nvfp4 import load
    from vllm.model_executor.kernels.linear.nvfp4.marlin import MarlinNvFp4LinearKernel
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE

    profile = tuple(load().TENSOR_API_METADATA["routed_shared_profile"])
    linear_load = MarlinNvFp4LinearKernel.process_weights_after_loading
    expert_load = ModelOptNvFp4FusedMoE.process_weights_after_loading
    if not _LOAD_LOCK.acquire(blocking=False):
        raise RuntimeError(
            "concurrent native weight loads require separate worker processes"
        )
    capture = WeightCapture(runner, profile)
    token = _ACTIVE.set(capture)

    def projection(self, layer):
        capture.projection(layer)
        return linear_load(self, layer)

    def experts(self, layer):
        capture.expert_bank(self, layer)
        return expert_load(self, layer)

    MarlinNvFp4LinearKernel.process_weights_after_loading = projection
    ModelOptNvFp4FusedMoE.process_weights_after_loading = experts
    try:
        yield capture
    finally:
        MarlinNvFp4LinearKernel.process_weights_after_loading = linear_load
        ModelOptNvFp4FusedMoE.process_weights_after_loading = expert_load
        capture.linear.clear()
        capture.experts.clear()
        _ACTIVE.reset(token)
        _LOAD_LOCK.release()


def _holder(module):
    return next(
        (
            m
            for m in module.modules()
            if hasattr(m, "w13_weight") and hasattr(m, "w2_weight")
        ),
        None,
    )


def routed_sites(model):
    """Discover the coupled topology; capability qualification follows later."""
    for path, module in model.named_modules():
        gate = getattr(module, "gate", None)
        shared = getattr(module, "shared_expert", None)
        experts = getattr(module, "experts", None)
        if gate is None or shared is None or experts is None:
            continue
        weight = getattr(gate, "weight", None)
        if not torch.is_tensor(weight) or weight.ndim != 2:
            continue
        bank = _holder(experts)
        if (
            bank is None
            or not hasattr(shared, "gate_up_proj")
            or not hasattr(shared, "down_proj")
        ):
            continue
        yield path, module, bank


def _parameter_alias_refresh(bank):
    """Keep backend scale descriptors on the WeightStore-owned parameters.

    vLLM's quant descriptors retain parameter objects outside nn.Module's
    parameter registry. Rebinding those aliases after stash releases their
    old device storage; after restore it points them at the restored truth.
    No tensor copy or independent restore store is introduced here.
    """
    config = getattr(getattr(bank, "quant_method", None), "moe_quant_config", None)
    parameters = dict(bank.named_parameters())
    aliases = []
    if config is not None:
        for slot in ("_a1", "_a2", "_w1", "_w2"):
            desc = getattr(config, slot, None)
            if desc is None:
                continue
            for field in ("scale", "alpha_or_gscale", "zp", "bias"):
                value = getattr(desc, field, None)
                if not torch.is_tensor(value):
                    continue
                name = next(
                    (
                        name
                        for name, p in parameters.items()
                        if p.device == value.device
                        and p.dtype == value.dtype
                        and p.shape == value.shape
                        and p.stride() == value.stride()
                        and p.data_ptr() == value.data_ptr()
                    ),
                    None,
                )
                if name is None:
                    raise ValueError(
                        "backend descriptor is not a restorable parameter alias"
                    )
                aliases.append((desc, field, name))

    def refresh():
        for desc, field, name in aliases:
            setattr(desc, field, bank.get_parameter(name))

    return refresh


class ParallelProjection(torch.nn.Module):
    """Preserve the host projection's (output, bias) protocol."""

    def __init__(self, seam):
        super().__init__()
        self.seam = seam

    def forward(self, x, *args, **kwargs):
        if args or kwargs:
            raise ValueError("native projection does not admit extra host arguments")
        return self.seam(x), None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "seam":
                raise
            return getattr(super().__getattr__("seam").host_module, name)


def _check_silu(module, device):
    # Observe the host's actual dataflow, including hosts whose activation is
    # a function in forward rather than an act_fn attribute. Temporary probe
    # projections are restored before any capture or inference can occur.
    gu, dn = module.gate_up_proj, module.down_proj
    h = getattr(gu, "input_size_per_partition", None)
    two_i = getattr(gu, "output_size_per_partition", None)
    if h is None or two_i is None:
        raise ValueError("FFN projection dimensions are not explicit in this host")
    packed = (
        torch.linspace(-2, 2, two_i, device=device, dtype=torch.float32)
        .to(torch.bfloat16)
        .reshape(1, -1)
    )
    x = torch.ones((1, h), device=device, dtype=torch.bfloat16)
    calls = []

    class GateProbe(torch.nn.Module):
        flint_gemv_silu_mul = False

        def forward(self, value):
            calls.append(("gate", value is x))
            return packed, None

    class DownProbe(torch.nn.Module):
        def forward(self, value):
            calls.append(("down", True))
            return value, None

    try:
        module.gate_up_proj, module.down_proj = GateProbe(), DownProbe()
        got = module(x)
    finally:
        module.gate_up_proj, module.down_proj = gu, dn
    gate, up = packed.chunk(2, dim=-1)
    expected = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)
    if calls != [("gate", True), ("down", True)] or not torch.equal(got, expected):
        raise ValueError("FFN dataflow is outside the precise-SiLU numerical profile")


def attach_native(model, *, capture=None, capacity=8192):
    """Bind a complete supported native cell, then return the ordinary handle.

    Replaced originals are off-device restore tickets, never a second GPU
    bank. The explicit route refuses unsupported forms instead of hot fallback.
    Attention, KV, token control and unqualified fusions remain host-owned.
    """
    _check_native_options()
    from flash_rt.native_kernels import nvfp4 as api

    api.load()
    capture = capture or _ACTIVE.get()
    if capture is None:
        raise RuntimeError(
            "native attachment requires pre-transform packed-weight capture"
        )
    if capture.runner is not None:
        from vllm.distributed import get_tensor_model_parallel_world_size

        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError(
                "native engine profile currently requires tensor parallel size one"
            )
        config = getattr(capture.runner, "vllm_config", None)
        if config is not None:
            _check_native_options(config)
            p = config.parallel_config
            if p.pipeline_parallel_size != 1 or p.enable_expert_parallel:
                raise ValueError(
                    "native engine profile requires one pipeline/expert lane"
                )
            if config.scheduler_config.max_num_batched_tokens > capacity:
                raise ValueError(
                    "host token budget exceeds the native scratch capacity"
                )
    modules = dict(model.named_modules())
    occupied = []
    alias_refreshers = []
    swaps = {}
    originals = []
    store = WeightStore()
    counts = {"routed_ffn": 0, "decoder_ffn": 0, "linear_proj": 0}
    handle = None
    changes = _AttributeChanges()
    alias_refreshers.append(changes.restore)

    def covered(path):
        return any(path == p or path.startswith(p + ".") for p in occupied)

    def upload(payload, interleaved=False):
        c, s = payload["codes"], payload["scales"]
        if interleaved:
            c, s = api.interleave_gate_up(c, s)
        c, s = c.to(device), s.to(device)
        return api.PackedMatrix(c, s, payload["gs"])

    def take(path, module, builder, kind, refresh=None):
        # Stash before uploading the replacement. Failure restores the host
        # from the same store and releases prior replacements sequentially.
        store.stash_module(path, module)
        originals.append((path, module))
        if refresh is not None:
            alias_refreshers.append(refresh)
            refresh()
        seam = builder()
        swaps[path] = seam
        occupied.append(path)
        counts[kind] += 1

    try:
        if os.environ.get("FLASHRT_SAGE_PREFILL", "0") == "1":
            from .sage2 import ops

            ops()  # installed producer must resolve before any engine trace
        for path, module, bank in routed_sites(model):
            if bank not in capture.experts:
                raise ValueError(
                    f"{path}: coupled FFN has no admitted canonical packed bank"
                )
            if (
                getattr(bank, "scoring_func", None) != "sigmoid"
                or not getattr(bank, "renormalize", False)
                or getattr(module, "top_k", None) != capture.profile[2]
                or getattr(bank, "use_grouped_topk", False)
                or getattr(bank, "e_score_correction_bias", None) is not None
                or getattr(bank, "routed_scaling_factor", 1.0) != 1.0
                or getattr(getattr(bank, "activation", None), "value", None) != "silu"
                or getattr(module, "shared_expert_gate", None) is not None
            ):
                raise ValueError(
                    f"{path}: coupled routing outside the declared native profile"
                )
            device = module.gate.weight.device
            shared = module.shared_expert
            _check_silu(shared, device)
            g = capture.linear[shared.gate_up_proj]
            d = capture.linear[shared.down_proj]
            bank_data = capture.experts[bank]
            router = module.gate.weight.detach()

            def builder(
                module=module, g=g, d=d, b=bank_data, router=router, device=device
            ):
                gc, gs = api.interleave_gate_up(b["gu_codes"], b["gu_scales"])
                gc, gs = gc.to(device), gs.to(device)
                dc = b["down_codes"].reshape(-1, b["down_codes"].shape[-1]).to(device)
                ds = b["down_scales"].reshape(-1, b["down_scales"].shape[-1]).to(device)
                gu = api.PackedMatrix(gc, gs, 1.0)
                dn = api.PackedMatrix(dc, ds, 1.0)
                return routed_impl.bind_packed(
                    router,
                    gu,
                    dn,
                    b["gu_global"].to(device),
                    b["down_global"].to(device),
                    upload(g, True),
                    upload(d),
                    module,
                    capacity=capacity,
                )

            take(path, module, builder, "routed_ffn", _parameter_alias_refresh(bank))
        for path, module in modules.items():
            if not path or covered(path):
                continue
            gu = getattr(module, "gate_up_proj", None)
            dn = getattr(module, "down_proj", None)
            if gu not in capture.linear or dn not in capture.linear:
                continue
            device = gu.weight.device
            _check_silu(module, device)
            g, d = capture.linear[gu], capture.linear[dn]
            take(
                path,
                module,
                lambda m=module, g=g, d=d: dense_impl.bind_packed(
                    upload(g, True), upload(d), m, capacity=capacity
                ),
                "decoder_ffn",
            )
        for path, module in modules.items():
            if not path or covered(path) or module not in capture.linear:
                continue
            device = module.weight.device
            payload = capture.linear[module]
            take(
                path,
                module,
                lambda m=module, p=payload: ParallelProjection(
                    linear_impl.bind_packed(upload(p), m, capacity=capacity)
                ),
                "linear_proj",
            )
        lm = getattr(model, "lm_head", None)
        if (
            os.environ.get("FLASHRT_NATIVE_LM_HEAD", "0") == "1"
            and lm is not None
            and getattr(model, "gguf_path", None)
        ):
            # Experimental: NVFP4 vocabulary projection (fp32 logits) replacing the
            # host's dequantized BF16 copy; recorded as a host stage outside the guarded-seam ledger.
            import gguf
            from vllm.models.flint.gguf_nvfp4 import _as_tensor, _repack_nvfp4

            reader = gguf.GGUFReader(model.gguf_path)
            tensor = {t.name: t for t in reader.tensors}["output.weight"]
            device = lm.weight.device
            codes, scales = _repack_nvfp4(
                _as_tensor(tensor.data).to(device), int(tensor.shape[0])
            )
            codes, scales = codes.contiguous(), scales.view(torch.uint8).contiguous()
            head = api.LmHead(api.PackedMatrix(codes, scales, 1.0))

            class _NativeHead:
                def apply(self, layer, x, bias=None):
                    if bias is not None:
                        raise GuardRefused("native lm_head takes no bias")
                    return head(x)

            store.stash_module("lm_head", lm)
            originals.append(("lm_head", lm))
            changes.undo.append(lambda: store.restore_module(lm))
            changes.set(lm, "quant_method", _NativeHead())
            # WeightStore leaves a same-shape meta parameter for introspection and rollback.
            counts["lm_head"] = 1
            _LOG.info("Native lm_head attached (NVFP4, fp32 logits)")
        if os.environ.get("FLASHRT_NATIVE_ATTN", "0") == "1":
            _install_native_decode_attention(model, api, changes)
            counts["decode_attention"] = 1
            if os.environ.get("FLASHRT_NATIVE_FUSE", "0") == "1":
                _install_native_layer_fusion(model, api, swaps, changes)
                counts["decode_layer_fusion"] = 1
        if not swaps:
            raise ValueError("native route found no qualifying prequantized regions")
        model.eval()
        handle = swap.attach(
            model,
            swaps,
            on_guard_fail="raise",
            store=store,
            revert=alias_refreshers,
            records={
                "native_regions": counts,
                "prebind_offloaded_bytes": store.stats["freed_bytes"],
            },
        )
        handle.consume(store)
        _LOG.info(
            "Native regions attached: %s; off-device restore bytes=%d",
            counts,
            store.stats["freed_bytes"],
        )
        return handle
    except Exception:
        # No graph can exist yet. Drop native owners before restoring each
        # original so rollback never requires a full second GPU bank.
        if handle is not None:
            swaps.clear()
            handle.detach()
        else:
            for path, module in reversed(originals):
                swaps.pop(path, None)
                store.restore_module(module)
            for refresh in reversed(alias_refreshers):
                refresh()
        raise


def _install_native_decode_attention(model, api, changes):
    """Experimental: pure single-token decode batches of up to FLASHRT_NATIVE_ATTN_MAX
    sequences run the native paged-KV split-KV attention (sinks, sliding window);
    everything else keeps the host attention. Bypasses the guarded-seam ledger."""
    import torch
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.attention.attention import (
        unified_attention_with_output,
    )
    from vllm.models.flint.model import FlintAttention
    from vllm.models.flint.ops.qk_norm_rope import qk_norm_rope_cache
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadata

    max_seqs = int(os.environ.get("FLASHRT_NATIVE_ATTN_MAX", "8"))
    max_seqs_fp8 = int(os.environ.get("FLASHRT_NATIVE_ATTN_MAX_FP8", "2"))

    def plan(self, k_cache=None):
        md = get_forward_context().attn_metadata
        md = md.get(self.attn.layer_name) if isinstance(md, dict) else md
        limit = (
            max_seqs_fp8
            if k_cache is not None
            and k_cache.dtype in (torch.uint8, torch.int8, torch.float8_e4m3fn)
            else max_seqs
        )
        if (
            not isinstance(md, FlashInferMetadata)
            or md.num_prefills != 0
            or (not 0 < md.num_decodes <= limit)
            or (md.num_decode_tokens != md.num_decodes)
            or (md.decode is None)
            or (getattr(md.decode, "q_len_per_req", 1) != 1)
        ):
            return None
        return (md.decode, md.num_decodes)

    fuse_rope = os.environ.get("FLASHRT_NATIVE_ATTN_FUSE", "1") == "1"

    def attention_core(self, positions, qkv):
        k_cache, v_cache, layer_slots = self._cache_views()
        if fuse_rope and k_cache is not None:
            p = plan(self, k_cache)
            if p is not None:
                decode, n = p
                num_tokens = qkv.shape[0]
                left = self.impl.window_left
                window = left + 1 if left >= 0 else 0
                cs = self._cos_sin()
                mlen = int(getattr(decode, "max_seq_len", 0) or 0)
                attn_output = api._decode_attention_qkv(
                    qkv[:n].contiguous(),
                    positions[:n].contiguous(),
                    self._native_qw,
                    self._native_kw,
                    cs if cs is not None else self._native_qw,
                    cs is not None,
                    float(self.q_norm.variance_epsilon),
                    k_cache,
                    v_cache,
                    decode.block_tables,
                    decode.seq_lens,
                    self.softmax_offset,
                    window,
                    float(self.head_dim ** (-0.5)),
                    mlen,
                )
                if n != num_tokens:
                    attn_output = torch.cat(
                        [
                            attn_output,
                            torch.zeros(
                                (num_tokens - n, attn_output.shape[1]),
                                dtype=attn_output.dtype,
                                device=attn_output.device,
                            ),
                        ],
                        0,
                    )
                return attn_output
        q = qk_norm_rope_cache(
            qkv,
            positions,
            self.q_norm.weight,
            self.k_norm.weight,
            self.q_norm.variance_epsilon,
            self._cos_sin(),
            k_cache,
            v_cache,
            layer_slots,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )
        num_tokens = qkv.shape[0]
        p = plan(self, k_cache) if k_cache is not None else None
        if p is not None:
            decode, n = p
            left = self.impl.window_left
            window = left + 1 if left >= 0 else 0
            mlen = int(getattr(decode, "max_seq_len", 0) or 0)
            attn_output = api._decode_attention(
                q[:n].contiguous(),
                k_cache,
                v_cache,
                decode.block_tables,
                decode.seq_lens,
                self.softmax_offset,
                window,
                float(self.head_dim ** (-0.5)),
                mlen,
            )
            if n != num_tokens:
                attn_output = torch.cat([attn_output, torch.zeros_like(q[n:])], 0)
        else:
            attn_output = torch.empty_like(q)
            _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            unified_attention_with_output(
                q.view(num_tokens, self.num_heads, self.head_dim),
                k.view(num_tokens, self.num_kv_heads, self.head_dim),
                v.view(num_tokens, self.num_kv_heads, self.head_dim),
                attn_output.view(num_tokens, self.num_heads, self.head_dim),
                self.attn.layer_name,
            )
        return attn_output

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        output, _ = self.o_proj(attention_core(self, positions, qkv))
        return output

    for module in model.modules():
        if isinstance(module, FlintAttention):
            changes.set(module, "forward", forward.__get__(module, FlintAttention))
            changes.set(
                module,
                "_native_attention_core",
                attention_core.__get__(module, FlintAttention),
            )
            changes.set(
                module, "_native_qw", module.q_norm.weight.detach().float().contiguous()
            )
            changes.set(
                module, "_native_kw", module.k_norm.weight.detach().float().contiguous()
            )
            changes.set(
                module, "_native_decode_plan", plan.__get__(module, FlintAttention)
            )
    _LOG.info(
        "Native decode attention attached (paged KV, up to %d sequences)", max_seqs
    )


def _install_native_layer_fusion(model, api, swaps, changes):
    """Experimental: single-token pure-decode steps run the frozen frontend's fused PDL chain per
    layer (residual add + RMSNorm fused into the QKV GEMV, PDL o_proj, norm + router + routed/shared
    experts, dense FFN with fused norm). Other shapes keep the per-region path."""
    import torch
    from vllm.models.flint.model import FlintDecoderLayer

    # 2..fuse_rows rows run the batched chain (our norm kernel instead of the Triton one, so every boundary can
    # early-trigger); 0 disables it
    fuse_rows = int(os.environ.get("FLASHRT_NATIVE_FUSE_ROWS", "8"))
    for name, layer in model.named_modules():
        if not isinstance(layer, FlintDecoderLayer):
            continue
        attn = layer.self_attn
        # the seams are not attached yet: take them from the pending swap table
        try:
            qkv_m = swaps[f"{name}.self_attn.qkv_proj"].seam.bound.matrix
            o_m = swaps[f"{name}.self_attn.o_proj"].seam.bound.matrix
            mlp = swaps[f"{name}.mlp"]
        except KeyError:
            continue
        in_w = layer.input_layernorm.weight.detach().float().contiguous()
        post_w = layer.post_attention_layernorm.weight.detach().float().contiguous()
        in_eps = float(layer.input_layernorm.variance_epsilon)
        post_eps = float(layer.post_attention_layernorm.variance_epsilon)
        routed_args = getattr(getattr(mlp, "bound", None), "args", None)
        dense_ffn = getattr(mlp, "bound", None) if routed_args is None else None
        if routed_args is None and (
            dense_ffn is None or not hasattr(dense_ffn, "gate_up")
        ):
            continue
        host_forward = layer.forward

        def forward(
            positions,
            hidden_states,
            residual,
            *,
            layer=layer,
            attn=attn,
            qkv_m=qkv_m,
            o_m=o_m,
            in_w=in_w,
            post_w=post_w,
            in_eps=in_eps,
            post_eps=post_eps,
            routed_args=routed_args,
            dense_ffn=dense_ffn,
            host_forward=host_forward,
        ):
            m = hidden_states.shape[0]
            if 2 <= m <= fuse_rows and attn._native_decode_plan() is not None:
                # batched chain: both norms on our PDL kernel, the swapped attention / MLP modules do the rest
                xn, residual = api.rows_norm(hidden_states, residual, in_w, in_eps)
                hidden = attn(positions, xn)
                xn2, residual = api.rows_norm(hidden, residual, post_w, post_eps)
                return layer.mlp(xn2), residual
            if m != 1 or attn._native_decode_plan() is None:
                return host_forward(positions, hidden_states, residual)
            qkv, residual = api.fused_m1_attn_in(
                qkv_m, hidden_states, residual, in_w, in_eps
            )
            attn_out = attn._native_attention_core(positions, qkv)
            hidden = api.fused_m1_linear(o_m, attn_out, 0)
            if routed_args is not None:
                return api.fused_m1_moe(routed_args, hidden, residual, post_w, post_eps)
            return api.fused_m1_dense(dense_ffn, hidden, residual, post_w, post_eps)

        changes.set(layer, "forward", forward)
    _LOG.info("Native single-token layer fusion attached")


def register():
    """vLLM general plugin; explicit opt-in, installed independently per worker."""
    global _INSTALLED, _LOG
    if os.environ.get("FLASHRT_STRUCTURES_NATIVE") != "1" or _INSTALLED:
        return
    _check_native_options()
    from vllm.logger import init_logger

    from .vllm_engine import install_load_hook

    _LOG = init_logger("vllm.flashrt_structures.native")
    install_load_hook(
        attach_fn=attach_native, load_context=capture_weights, allow_spawn=True
    )
    _INSTALLED = True
