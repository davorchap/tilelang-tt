"""
Wrapper for LowerSharedToCB_v5 pass to work with modules
"""

import tvm
from tvm import tir
from lower_shared_to_cb_v5 import LowerSharedToCB_v5 as LowerSharedToCB_v5_func


def LowerSharedToCB_v5(mod):
    """
    Apply LowerSharedToCB_v5 pass to all functions in a module.

    Args:
        mod: TVM IR Module

    Returns:
        Transformed module
    """
    if isinstance(mod, tir.PrimFunc):
        # If it's a function, wrap it in a module
        temp_mod = tvm.IRModule({"main": mod})
        pass_obj = LowerSharedToCB_v5_func
        result = pass_obj(temp_mod)
        return result["main"]
    else:
        # Apply pass to module
        pass_obj = LowerSharedToCB_v5_func
        return pass_obj(mod)