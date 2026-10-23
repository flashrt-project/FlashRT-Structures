"""Precise-SiLU native decoder_ffn over prequantized interleaved weights."""

import torch

from ...guard import PROCEED, GuardedSeam, GuardRefused


class DecoderFfnNative(GuardedSeam, torch.nn.Module):
    _frt_host_attr = "host_module"
    _frt_can_fallback = False

    def __init__(self, bound, original, *, capacity=8192):
        super().__init__()
        self.bound, self.host_module = bound, original
        self._k = bound.gate_up.k
        self._frt_arm(
            dtypes={torch.bfloat16},
            device=bound.gate_up.codes.device,
            k=self._k,
            row_capacity=capacity,
        )
        self._frt_guard.notes.update(
            provider="native",
            activation="precise_silu_bf16_inputs",
            dispatch="all admitted rows; no host fallback",
        )

    def forward(self, x):
        if x.ndim < 2:
            raise GuardRefused("native FFN requires rank >= 2")
        admitted = self._frt_admit(x)
        if admitted is not PROCEED:
            return admitted
        if not x.is_contiguous():
            self._frt_guard.last_reason = "native FFN requires contiguous rank >= 2"
            raise GuardRefused(self._frt_guard.last_reason)
        return self.bound(x.reshape(-1, self._k)).reshape(x.shape)


def bind_packed(gate_up, down, original, *, capacity=8192):
    from ...backends.flint import load_module

    api = load_module()
    DenseFfn = api.DenseFfn

    bound = DenseFfn(gate_up, down)
    bound(
        torch.zeros((1, gate_up.k), dtype=torch.bfloat16, device=gate_up.codes.device)
    )
    return DecoderFfnNative(bound, original, capacity=capacity)
