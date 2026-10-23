"""Resolve Sage2 once when the optional adapter is explicitly attached."""

from ..backends.flint import load_module

_backend = load_module("sage2")
attention_fp8 = _backend.attention_fp8
gather_pages_bf16 = _backend.gather_pages_bf16
ops = _backend.load
