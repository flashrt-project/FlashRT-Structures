"""Explicit CUDA wheel build; never used by the core Structures installation."""

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "flashrt_structures_flint" / "csrc"
requested = os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
if requested != "12.0a":
    raise RuntimeError("Flint backend builds require TORCH_CUDA_ARCH_LIST=12.0a")
nvcc = ["-O3", "--expt-relaxed-constexpr", "-DCUDA_KERNEL"]
for macro in (
    "__CUDA_NO_HALF_OPERATORS__",
    "__CUDA_NO_HALF_CONVERSIONS__",
    "__CUDA_NO_HALF2_OPERATORS__",
    "__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "__CUDA_NO_BFLOAT16_OPERATORS__",
    "__CUDA_NO_BFLOAT162_OPERATORS__",
):
    nvcc.append("-U" + macro)


def relative(path):
    return str(path.relative_to(ROOT))


extensions = [
    CUDAExtension(
        "flashrt_structures_flint._flashrt_flint_ops",
        sources=[relative(SRC / "bindings.cpp")]
        + [relative(p) for p in sorted((SRC / "flint").glob("*.cu"))],
        include_dirs=[str(SRC)],
        extra_compile_args={"cxx": ["-O3"], "nvcc": nvcc},
    ),
    CUDAExtension(
        "flashrt_structures_flint._flashrt_flint_sage2",
        sources=[
            relative(SRC / "sage2/torch-ext/torch_binding.cpp"),
            relative(SRC / "sage2/csrc/sage2_blackwell.cu"),
        ],
        include_dirs=[str(SRC / "sage2/csrc")],
        extra_compile_args={"cxx": ["-O3", "-DCUDA_KERNEL"], "nvcc": nvcc},
    ),
]
setup(ext_modules=extensions, cmdclass={"build_ext": BuildExtension})
