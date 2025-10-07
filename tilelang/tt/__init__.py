"""Tenstorrent-specific TileLang utilities and helpers.

This package provides Tenstorrent-specific functionality including:
- Default annotation helpers for schedule and sharding
- TT-specific transforms and utilities
"""

from .target import apply_tt_defaults

__all__ = ["apply_tt_defaults"]
