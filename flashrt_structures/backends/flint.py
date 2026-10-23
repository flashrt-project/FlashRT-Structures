"""Resolve an explicitly installed Flint backend at bind time."""

import importlib


def load_module(component="nvfp4"):
    try:
        return importlib.import_module("flashrt_structures_flint." + component)
    except ImportError as exc:
        raise ValueError(
            "optional Flint operator backend unavailable; install the separately "
            "built flashrt-structures-flint package (see backends/flint/README.md)"
        ) from exc
