"""Python implementation of the PropagateTTLayout pass."""

from __future__ import annotations

from tilelang import tvm as tvm

from ._common import convert_dict_for_ffi, convert_to_python, dtype_num_bytes, get_attr, infer_data_format


def propagate_tt_layout(mod: tvm.IRModule) -> tvm.IRModule:
    """Propagate buffer layout metadata into `tt.cb.*` circular-buffer attributes."""

    def transform(func: tvm.tir.PrimFunc, *_):
        if not isinstance(func, tvm.tir.PrimFunc):
            return func

        new_func = func
        for _, buffer in func.buffer_map.items():
            buffer_name = buffer.name
            buffer_attr_key = f"tt.buffer.{buffer_name}"
            buffer_meta_obj = get_attr(func.attrs, buffer_attr_key)
            if buffer_meta_obj is None:
                continue

            buffer_meta = convert_to_python(buffer_meta_obj)

            dtype_str = str(buffer_meta.get("dtype", buffer.dtype))
            tile_shape = buffer_meta.get("tile_shape")
            if not isinstance(tile_shape, (list, tuple)) or len(tile_shape) != 2:
                tile_shape = [32, 32]
            tile_shape = [int(tile_shape[0]), int(tile_shape[1])]

            try:
                element_bytes = dtype_num_bytes(dtype_str)
            except ValueError:
                element_bytes = 0

            page_size = tile_shape[0] * tile_shape[1] * element_bytes
            depth = int(buffer_meta.get("cb_depth", 2))
            data_format = buffer_meta.get("data_format", infer_data_format(dtype_str))

            cb_entry = {
                "page_size": page_size,
                "depth": depth,
                "data_format": str(data_format),
            }

            cb_entry_ffi = convert_dict_for_ffi(cb_entry)
            new_func = new_func.with_attr(f"tt.cb.{buffer_name}", cb_entry_ffi)

        return new_func

    pass_obj = tvm.tir.transform.prim_func_pass(transform, opt_level=0, name="tl.PropagateTTLayout")
    return pass_obj(mod)


__all__ = ["propagate_tt_layout"]
