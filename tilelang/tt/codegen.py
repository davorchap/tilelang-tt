"""
TileLang Tenstorrent Backend Code Generation (Artifact Generation stage)

This module provides Python bindings for TT kernel codegen and artifact generation.
"""

from typing import Dict
import tvm
from tvm import IRModule


def emit_tt_artifacts(mod: IRModule, target: str = "tenstorrent") -> Dict[str, str]:
    """
    Generate all TT artifacts from an IRModule.

    This is the main codegen entry point that produces:
    - compute.cpp: Persistent compute kernel with MAIN() function
    - tt.plan.json: Scheduling metadata (grid, cores, assignments)

    Args:
        mod: TVM IRModule with TT Defaults stage-3 transforms applied (schedule + sharding metadata)
        target: Target string (default: "tenstorrent")

    Returns:
        Dictionary mapping artifact names to their string contents:
        {
            "compute.cpp": "...",
            "tt.plan.json": "..."
        }

    Example:
        >>> mod = create_tt_gemm(M=256, N=256, K=256)
        >>> mod = apply_ws1_ws2_ws3(mod)
        >>> artifacts = emit_tt_artifacts(mod)
        >>> print(artifacts["compute.cpp"][:100])
        // Generated TT Compute Kernel...
    """
    codegen_func = tvm.ffi.get_global_func("tl.codegen.EmitTTArtifacts")
    result = codegen_func(mod, target)

    # Convert TVM Map to Python dict
    return dict(result.items())


def write_artifacts_to_disk(artifacts: Dict[str, str], output_dir: str = "."):
    """
    Write generated artifacts to disk.

    Args:
        artifacts: Dictionary of artifact names to contents
        output_dir: Directory to write files (default: current directory)

    Example:
        >>> artifacts = emit_tt_artifacts(mod)
        >>> write_artifacts_to_disk(artifacts, output_dir="./build/tt_kernels")
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for filename, content in artifacts.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            f.write(content)
        print(f"✓ Wrote {filepath}")


__all__ = [
    "emit_tt_artifacts",
    "write_artifacts_to_disk",
]
