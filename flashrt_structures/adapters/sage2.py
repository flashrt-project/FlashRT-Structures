"""Host-facing aliases for the separately built FlashRT Sage2 Tensor artifact.

The native producer owns build, device qualification and computation. This
consumer never locates sources or compiles kernels inside the serving engine.
"""

from flash_rt.native_kernels.sage2 import attention_fp8, gather_pages_bf16, load as ops

__all__ = ["attention_fp8", "gather_pages_bf16", "ops"]
