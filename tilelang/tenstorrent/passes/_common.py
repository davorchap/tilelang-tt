"""Utility helpers shared across Tenstorrent Python passes."""

from __future__ import annotations

from typing import Any, Dict, Optional

from tilelang import tvm as tvm
from tvm.tir import FloatImm, IntImm


def dtype_num_bytes(dtype_str: str) -> int:
    """Return the byte width of a dtype string."""
    dt = tvm.runtime.DataType(dtype_str)
    if dt.bits % 8 != 0:
        raise ValueError(f"Unsupported dtype bit-width for {dtype_str}")
    return (dt.bits // 8) * dt.lanes


def infer_data_format(dtype_str: str) -> str:
    """Map TileLang dtype strings to TT data format enums."""
    mapping = {
        "float16": "Float16_b",
        "float32": "Float32",
        "bfloat16": "BFloat16_b",
        "int8": "Int8",
        "uint8": "UInt8",
    }
    return mapping.get(dtype_str, dtype_str)


def to_int(value: Any, default: int = 0) -> int:
    """Convert TVM numeric objects or Python primitives to int."""
    if value is None:
        return default
    if isinstance(value, tvm.tir.IntImm):
        return int(value.value)
    return int(value)


def array_to_int_list(array_obj: Any) -> Optional[list[int]]:
    """Convert a TVM Array of ints to a Python list."""
    if array_obj is None:
        return None
    if isinstance(array_obj, tvm.ir.Array):
        return [int(x) for x in array_obj]
    return None


def map_to_python(map_obj: Any) -> Dict[str, Any]:
    """Convert a TVM Map with string keys into a Python dict."""
    if map_obj is None or not isinstance(map_obj, tvm.ir.Map):
        return {}
    return {str(key): value for key, value in map_obj.items()}


def map_of_maps_to_python(map_obj: Any) -> Dict[str, Dict[str, Any]]:
    """Convert nested TVM Map objects into nested Python dictionaries."""
    if map_obj is None or not isinstance(map_obj, tvm.ir.Map):
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for key, value in map_obj.items():
        if isinstance(value, tvm.ir.Map):
            result[str(key)] = {
                str(inner_key): convert_to_python(inner_value)
                for inner_key, inner_value in value.items()
            }
        else:
            result[str(key)] = {"value": convert_to_python(value)}
    return result


def convert_to_python(obj: Any) -> Any:
    """Recursively convert TVM container objects to native Python types."""
    if isinstance(obj, tvm.ir.Map):
        return {str(k): convert_to_python(v) for k, v in obj.items()}
    if isinstance(obj, tvm.ir.Array):
        return [convert_to_python(x) for x in obj]
    if isinstance(obj, IntImm):
        return int(obj)
    if isinstance(obj, FloatImm):
        return float(obj)
    if isinstance(obj, tvm.tir.PrimExpr):
        if isinstance(obj, IntImm):
            return int(obj)
        if isinstance(obj, FloatImm):
            return float(obj)
        return obj
    return obj


def get_attr(attrs: Optional[tvm.ir.Map], key: str, default: Any = None) -> Any:
    """Small helper to read from an attribute map."""
    if attrs is None:
        return default
    if key in attrs:
        return attrs[key]
    return default


def convert_dict_for_ffi(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Python primitives into FFI-friendly TVM objects."""
    result: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, bool):
            result[key] = tvm.runtime.const(int(value), "int32")
        elif isinstance(value, int):
            result[key] = tvm.runtime.const(value, "int32")
        elif isinstance(value, (list, tuple)):
            converted_list = []
            for elem in value:
                if isinstance(elem, bool):
                    converted_list.append(tvm.runtime.const(int(elem), "int32"))
                elif isinstance(elem, int):
                    converted_list.append(tvm.runtime.const(elem, "int32"))
                else:
                    converted_list.append(elem)
            result[key] = converted_list
        elif isinstance(value, dict):
            result[key] = convert_dict_for_ffi(value)
        else:
            result[key] = value
    return result


__all__ = [
    "array_to_int_list",
    "convert_dict_for_ffi",
    "convert_to_python",
    "dtype_num_bytes",
    "get_attr",
    "infer_data_format",
    "map_of_maps_to_python",
    "map_to_python",
    "to_int",
]
