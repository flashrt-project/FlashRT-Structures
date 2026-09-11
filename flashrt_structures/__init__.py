"""FlashRT structures — verified, composable model sub-blocks.

A structure is a versioned specification of one model region: boundary
tensors, framework-neutral weight slots, a plain reference implementation
used as ground truth, and qualification gates. This package hosts the
structure catalog and its registry. Implementations, host adapters, and
the qualification harness build on top of these specifications.
"""

try:
    from flash_rt.catalog.binding import (
        BindingSpec,
        CoverageSegment,
        list_bindings,
        load_binding,
    )
    from flash_rt.catalog.registry import StructureSpec, list_structures, load
except ImportError as _e:  # pragma: no cover - depends on the installed flash-rt
    raise ImportError(
        "flashrt-structures needs a flash-rt that ships the structure "
        "catalog as flash_rt.catalog (FlashRT main after the structures "
        "split; the 0.1.0 wheel on PyPI predates it and still carries the "
        "layer as flash_rt.structures). Install FlashRT from "
        "https://github.com/flashrt-project/FlashRT or upgrade flash-rt."
    ) from _e


def get(name):
    """Explicit door: pull one structure and bind it yourself.

    Mirrors ``kernels.get_kernel``: ``get("decoder_ffn").bind(module,
    calibration=[...])`` returns a gated drop-in replacement you plug in
    where you choose. See :mod:`flashrt_structures.handle`.
    """
    from flashrt_structures.handle import get as _get

    return _get(name)


def capture(fn, **kwargs):
    """Capture door: graph a hot stage with declared swap windows.

    See :func:`flashrt_structures.stages.capture`.
    """
    from flashrt_structures.stages import capture as _capture

    return _capture(fn, **kwargs)


def auto_swaps(model, forward, **kwargs):
    """Distribution layer: discover, calibrate, bind — one pass, no
    per-seam scaffolding. Returns an :class:`AutoPlan` of swaps.

    See :func:`flashrt_structures.autobuild.auto_swaps`.
    """
    from flashrt_structures.autobuild import auto_swaps as _auto

    return _auto(model, forward, **kwargs)


def run_recipe(recipe, model, ctx=None, **kwargs):
    """Recipe door: assemble declared levers, audit same-process on the
    graph, certify or refuse — one call, one receipt.

    See :mod:`flashrt_structures.recipe` for ``Recipe``/``Lever``/
    ``Gates`` and the switch lifecycle.
    """
    from flashrt_structures.recipe import run_recipe as _run

    return _run(recipe, model, ctx, **kwargs)


def attach(model, forward, **kwargs):
    """One-call front door: discover, calibrate, gate, activate.

    See :func:`flashrt_structures.frontdoor.attach`. Imported lazily so
    that spec-only consumers do not pay for torch-side machinery.
    """
    from flashrt_structures.frontdoor import attach as _attach

    return _attach(model, forward, **kwargs)


def adopt_prequantized(model, fmt="ct_nvfp4", **kwargs):
    """Checkpoint door: adopt an already-quantized checkpoint by
    converting its packed projections into structure impls.

    See :func:`flashrt_structures.prequantized.adopt_prequantized`.
    """
    from flashrt_structures.prequantized import (
        adopt_prequantized as _adopt)

    return _adopt(model, fmt, **kwargs)


def quantize_on_adopt(model, fmt="moe_experts_nvfp4", **kwargs):
    """Checkpoint door: quantize a full-precision checkpoint that cannot
    fit the card, converting its dominant structure family (a sparse-MoE
    expert bank) into structure impls at load time.

    See :func:`flashrt_structures.quantize_on_adopt.quantize_on_adopt`.
    """
    from flashrt_structures.quantize_on_adopt import (
        quantize_on_adopt as _adopt)

    return _adopt(model, fmt, **kwargs)


def explain(plan):
    """Coverage table for one plan: bound / routed / kept / refused,
    each with its reason. See :mod:`flashrt_structures.explain`."""
    from flashrt_structures.explain import explain as _explain

    return _explain(plan)


def decode_loop(model, *, max_len, compile_step=True,
                compile_prefill=True, kv_band=None):
    """Serving door: the whole-loop decode form (static cache + compiled
    step + whole-step CUDA graph) over whatever structures are attached.

    See :mod:`flashrt_structures.impls.decode_loop.whole_step`.
    """
    from flashrt_structures.impls.decode_loop.whole_step import (
        build_decode_loop)

    return build_decode_loop(model, max_len=max_len,
                             compile_step=compile_step,
                             compile_prefill=compile_prefill,
                             kv_band=kv_band)


def aot_package(module, args=(), kwargs=None,
                package_path="module_aot.pt2", **opts):
    """Whole-graph door: export the swapped module and AOT-compile it
    into a reusable package (graph breaks are defects, not fallbacks).

    See :mod:`flashrt_structures.aot`.
    """
    from flashrt_structures.aot import aot_package as _pkg

    return _pkg(module, args=args, kwargs=kwargs,
                package_path=package_path, **opts)


def aot_load(package_path, weights=None):
    """Load an AOT package back as a callable graph."""
    from flashrt_structures.aot import aot_load as _load

    return _load(package_path, weights=weights)


from . import schemes  # noqa: E402  (registry: quantisation schemes)

__all__ = [
    "BindingSpec",
    "CoverageSegment",
    "StructureSpec",
    "adopt_prequantized",
    "aot_load",
    "aot_package",
    "decode_loop",
    "explain",
    "attach",
    "capture",
    "get",
    "list_bindings",
    "list_structures",
    "load",
    "load_binding",
    "quantize_on_adopt",
    "run_recipe",
    "schemes",
]
