"""Coupled routed/shared executable form; no model identity predicates.

The native artifact supplies its shape/numerical profile. This form uses the
same guard, scratch ledger and transactional swap lifecycle as other forms.
It is a larger boundary than an external-routing expert bank.
"""

import torch

from ... import workspace
from ...guard import PROCEED, GuardedSeam, GuardRefused


class RoutedFfnNative(GuardedSeam, torch.nn.Module):
    _frt_host_attr = "host_module"
    _frt_can_fallback = False

    def __init__(self, bound, original):
        super().__init__()
        self.bound, self.host_module = bound, original
        router = bound.args[0]
        self._k = router.shape[1]
        self._frt_arm(
            dtypes={torch.bfloat16},
            device=router.device,
            k=self._k,
            row_capacity=bound.capacity,
        )
        self._frt_guard.notes.update(
            provider="native",
            routing="sigmoid_top6_normalized_bf16",
            shared="ungated",
            dispatch="all admitted rows; no host fallback",
        )

    def forward(self, x):
        if x.ndim != 2:
            raise GuardRefused("native routed FFN requires rank two")
        admitted = self._frt_admit(x)
        if admitted is not PROCEED:
            return admitted
        if not x.is_contiguous():
            self._frt_guard.last_reason = "native routed FFN requires contiguous [M,H]"
            raise GuardRefused(self._frt_guard.last_reason)
        return self.bound(x)


def bind_packed(
    router,
    expert_gu,
    expert_down,
    gu_global,
    down_global,
    shared_gu,
    shared_down,
    original,
    *,
    capacity=8192,
):
    from flash_rt.native_kernels.nvfp4 import RoutedSharedFfn, workspace_shapes

    # Scratch is shared only across sequential layers. Returned model outputs
    # are allocated by the op and never alias this pool. Counter buffers are
    # zeroed on first lease; the artifact resets them after each invocation.
    scratch = [
        workspace.lease(shape, dtype, router.device, tag=f"native_routed_ffn_{i}")
        for i, (shape, dtype) in enumerate(workspace_shapes(capacity))
    ]
    bound = RoutedSharedFfn(
        router,
        expert_gu,
        expert_down,
        gu_global,
        down_global,
        shared_gu,
        shared_down,
        scratch,
        capacity,
    )
    bound(torch.zeros((1, router.shape[1]), dtype=torch.bfloat16, device=router.device))
    return RoutedFfnNative(bound, original)
