"""
Tenstorrent Codegen Module (v5 Specification)
Version: 5.0
Date: 2025-10-15

Python-based code generation for Tenstorrent backend.
Generates reader.cpp, compute.cpp, writer.cpp, and main.cpp from v5 TIR.
"""

from .codegen_tt import CodegenTT
import os
from typing import Dict
import json


def emit_tt_artifacts(ir_module) -> Dict[str, str]:
    """
    Generate Tenstorrent artifacts from an IR module.

    Args:
        ir_module: TVM IRModule with TT-lowered functions

    Returns:
        Dictionary of generated artifacts (filename -> content)
    """
    codegen = CodegenTT()
    return codegen.generate(ir_module)


def write_artifacts_to_disk(artifacts: Dict[str, str], output_dir: str = "./tt_output"):
    """
    Write generated artifacts to disk.

    Args:
        artifacts: Dictionary of filename -> content
        output_dir: Directory to write files to
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename, content in artifacts.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            f.write(content)

    # Also write a metadata file
    metadata = {
        "files": list(artifacts.keys()),
        "version": "v5.0",
        "generator": "tilelang.tenstorrent.codegen"
    }

    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)


__all__ = ["CodegenTT", "emit_tt_artifacts", "write_artifacts_to_disk"]
