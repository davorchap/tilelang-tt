"""Tenstorrent-specific TileLang utilities and helpers.

This package provides Tenstorrent-specific functionality including:
- Default annotation helpers for schedule and sharding (WS1)
- Schedule and sharding inference passes (WS2)
- TT-specific transforms and utilities
"""

from .target import apply_tt_defaults
from .passes import (
    infer_default_tt_schedule,
    infer_default_tt_shard,
    apply_ws2_passes,
)

__all__ = [
    "apply_tt_defaults",
    "infer_default_tt_schedule",
    "infer_default_tt_shard",
    "apply_ws2_passes",
]
