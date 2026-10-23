"""Flint/vLLM scheduling for the optional backend; no Structures imports."""

import logging
import os

_LOG = logging.getLogger(__name__)


def install_decode_attention(model, api, changes):
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


def install_layer_fusion(model, api, swaps, changes):
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
