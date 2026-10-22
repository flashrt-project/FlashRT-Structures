"""Already-packed native W4A16 projection, over the existing linear boundary."""

import torch

from ...guard import PROCEED, GuardedSeam, GuardRefused


class LinearProjNative(GuardedSeam, torch.nn.Module):
    """Strict native form; original is kept for state/rollback, not hot fallback.

    Packing and architecture qualification belong to the installed native
    artifact. No checkpoint conversion, source loading or calibration occurs.
    """

    _frt_host_attr = "host_module"
    _frt_can_fallback = False

    def __init__(self, bound, original, *, capacity=8192):
        super().__init__()
        self.bound = bound
        self.host_module = original
        matrix = bound.matrix
        self._k, self._n = matrix.k, matrix.n
        self._frt_arm(
            dtypes={torch.bfloat16},
            device=matrix.codes.device,
            k=self._k,
            row_capacity=capacity,
        )
        self._frt_guard.notes.update(
            provider="native",
            layout="adjacent_e2m1_block16_e4m3",
            dispatch="all admitted rows; no host fallback",
        )

    def forward(self, x):
        if x.ndim < 2:
            raise GuardRefused("native projection requires rank >= 2")
        admitted = self._frt_admit(x)
        if admitted is not PROCEED:
            return admitted
        if not x.is_contiguous():
            self._frt_guard.last_reason = (
                "native projection requires contiguous rank >= 2"
            )
            raise GuardRefused(self._frt_guard.last_reason)
        return self.bound(x.reshape(-1, self._k)).reshape(*x.shape[:-1], self._n)


def bind_packed(matrix, original, *, capacity=8192):
    from flash_rt.native_kernels.nvfp4 import Linear

    bound = Linear(matrix)
    bound(torch.zeros((1, matrix.k), dtype=torch.bfloat16, device=matrix.codes.device))
    return LinearProjNative(bound, original, capacity=capacity)
