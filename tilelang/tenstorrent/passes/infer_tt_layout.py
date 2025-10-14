"""
InferTTLayout: stamps layout/shard metadata onto PrimFuncs.
This pass ensures all buffers have layout descriptors, applying defaults where needed.
"""
from __future__ import annotations
from typing import Optional, Dict, Any
import logging

try:
    import tvm
    from tvm import tir, IRModule
except ImportError:  # pragma: no cover
    tvm = None
    tir = None
    IRModule = object

from ..attrs import TT_LAYOUT_DESC

logger = logging.getLogger(__name__)

class InferTTLayout:
    """
    Pass to infer and attach layout descriptors to PrimFuncs.
    
    This pass ensures every buffer has a layout descriptor, applying
    sensible defaults based on buffer usage patterns.
    """
    
    def __init__(self, defaults: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the pass with optional default layout settings.
        
        Args:
            defaults: Default layout descriptors to apply when none exist.
                     Example: {"shard": "DRAM", "interleave": True}
        """
        self.defaults = defaults or {"shard": "DRAM", "interleave": False}
    
    def __call__(self, mod: IRModule) -> IRModule:
        """Apply the pass to an IRModule."""
        if tvm is None:
            return mod
        
        new_funcs = {}
        for gvar, func in mod.functions_items():
            if not isinstance(func, tir.PrimFunc):
                new_funcs[gvar] = func
                continue
            
            # Get existing layout descriptors or initialize empty
            layouts = func.attrs.get(TT_LAYOUT_DESC, None) if func.attrs else None
            
            if layouts is None:
                layouts = {}
                
            # Ensure all buffers have layout descriptors
            for param in func.params:
                buffer = func.buffer_map.get(param, None)
                if buffer and buffer.name not in layouts:
                    # Apply defaults for this buffer
                    layout = dict(self.defaults)
                    
                    # Infer better defaults based on buffer properties
                    if "output" in buffer.name.lower() or buffer.name == "C":
                        # Output buffers often go to L1 for fast writeback
                        layout["shard"] = "L1"
                        layout["tile_id_order"] = "row_major"
                    elif "weight" in buffer.name.lower() or buffer.name == "B":
                        # Weights might benefit from interleaving
                        layout["interleave"] = True
                    
                    layouts[buffer.name] = layout
                    logger.debug(f"Inferred layout for buffer {buffer.name}: {layout}")
            
            # Attach the layouts to the function
            if layouts:
                func = func.with_attr(TT_LAYOUT_DESC, tvm.runtime.convert(layouts))
                logger.info(f"Attached layout descriptors to function {gvar}")
            
            new_funcs[gvar] = func
        
        return tvm.IRModule(new_funcs)