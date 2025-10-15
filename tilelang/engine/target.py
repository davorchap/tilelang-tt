"""
Target module for TileLang engine.
This provides the target selection and backend switching functionality.
"""

from typing import Union
from tilelang import tvm as tvm
from tvm.target import Target
from tilelang.utils.target import (
    TENSTORRENT_TARGET,
    determine_target as _determine_target,
    AVALIABLE_TARGETS,
    check_cuda_availability,
    check_hip_availability,
)

# Export from utils.target for backward compatibility
__all__ = [
    'TENSTORRENT_TARGET',
    'determine_target',
    'use_tt_backend',
    'AVALIABLE_TARGETS',
    'check_cuda_availability',
    'check_hip_availability',
]

# Global flag for TT backend
_USE_TT_BACKEND = False


def determine_target(target: Union[str, Target] = "auto") -> Union[str, Target]:
    """
    Determine the appropriate target, with TT backend override support.

    If TT backend is enabled via use_tt_backend(True), this will return
    the Tenstorrent target regardless of the input.
    """
    global _USE_TT_BACKEND

    if _USE_TT_BACKEND:
        return TENSTORRENT_TARGET

    return _determine_target(target)


def use_tt_backend(enable: bool = True):
    """
    Enable or disable the Tenstorrent backend globally.

    When enabled, all compilation will target the Tenstorrent backend
    regardless of the specified target.
    """
    global _USE_TT_BACKEND
    _USE_TT_BACKEND = enable

    if enable:
        print(f"Tenstorrent backend enabled - all compilation will target '{TENSTORRENT_TARGET}'")
    else:
        print("Tenstorrent backend disabled - using default target selection")
