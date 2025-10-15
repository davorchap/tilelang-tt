"""
Tenstorrent Codegen Module (v5 Specification)
Version: 5.0
Date: 2025-10-15

Python-based code generation for Tenstorrent backend.
Generates reader.cpp, compute.cpp, writer.cpp, and main.cpp from v5 TIR.
"""

from .codegen_tt import CodegenTT

__all__ = ["CodegenTT"]