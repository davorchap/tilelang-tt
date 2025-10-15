"""Tenstorrent-specific kernel adapter that bypasses compilation.

The TT backend generates its own complete artifacts (reader/compute/writer/host)
and doesn't need the standard wrapper/compilation infrastructure.
"""

import json
import logging
import torch
from typing import List, Optional, Union, Callable, Dict, Any

from tilelang import tvm as tvm
from tvm.target import Target
from tilelang.engine.param import KernelParam
from tvm import tir
from tilelang.jit.adapter.base import BaseKernelAdapter
from tilelang.utils.language import retrieve_func_from_module
from tilelang.utils.tensor import map_torch_type

logger = logging.getLogger(__name__)


class TTKernelAdapter(BaseKernelAdapter):
    """Adapter for Tenstorrent backend.

    This adapter is much simpler than the Cython adapter because:
    1. TT codegen already produces complete artifacts (reader/compute/writer/host)
    2. No compilation is needed - artifacts are generated ready-to-run
    3. Actual execution requires TT hardware or simulator
    """

    def __init__(self,
                 params: List[KernelParam],
                 result_idx: List[int],
                 target: Union[str, Target],
                 func_or_mod: Union[tir.PrimFunc, tvm.IRModule],
                 host_mod: Optional[tvm.IRModule] = None,
                 device_mod: Optional[tvm.IRModule] = None,
                 kernel_global_source: Optional[str] = None,
                 verbose: bool = False,
                 pass_configs: Optional[Dict[str, Any]] = None,
                 compile_flags: Optional[List[str]] = None):
        """Initialize TT adapter.

        Args:
            params: List of tensor types for inputs/outputs
            result_idx: Indices of output tensors
            target: Target platform (should be 'tenstorrent')
            func_or_mod: TIR function or module
            kernel_global_source: JSON string containing TT artifacts
            verbose: Enable verbose logging
        """
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self.kernel_global_source = kernel_global_source
        self.verbose = verbose

        if isinstance(func_or_mod, tir.PrimFunc):
            self.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod

        self.target = target if isinstance(target, Target) else Target(target)

        # Parse TT artifacts from JSON
        self.artifacts = {}
        if kernel_global_source:
            try:
                self.artifacts = json.loads(kernel_global_source)
                if verbose:
                    logger.info(f"Loaded TT artifacts: {list(self.artifacts.keys())}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse TT artifacts: {e}")

        # Process buffer information for runtime
        self.buffer_dtype_map = self._process_buffer_dtype()
        self.buffer_device_map = self._process_buffer_device()

        # Add libpath attribute to prevent cache error
        # For TT, we don't have a compiled library, but we need this for compatibility
        self.libpath = None  # TT artifacts are not compiled into a .so file

        self._post_init()

    def _process_buffer_dtype(self) -> Dict[str, torch.dtype]:
        """Extract buffer dtype information."""
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        buffer_dtype_map = {}

        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                buffer_dtype_map[buffer.name] = (i, map_torch_type(buffer.dtype))

        return buffer_dtype_map

    def _process_buffer_device(self) -> Dict[str, torch.device]:
        """Extract buffer device information."""
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        buffer_device_map = {}

        # TT backend uses CPU for host-side tensor management
        device = torch.device("cpu")

        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                buffer_device_map[buffer.name] = (i, device)

        return buffer_device_map

    def _convert_torch_func(self) -> Callable:
        """Returns a PyTorch-compatible function wrapper for the kernel.

        For TT backend, this is a simulation stub since actual execution
        requires TT hardware or simulator.
        """

        def lambda_forward(*args, stream: int = -1, skip_tensor_validation: bool = False):
            """Simulation stub for TT kernel execution.

            Args:
                args: List of input tensors
                stream: Unused for TT backend
                skip_tensor_validation: Whether to skip validation

            Returns:
                Output tensors (currently just returns inputs for simulation)
            """
            if self.verbose:
                logger.info("TT kernel execution (simulation mode)")
                logger.info(f"  Input tensors: {len(args)}")
                logger.info(f"  Artifacts available: {list(self.artifacts.keys())}")

            # In a real implementation, this would:
            # 1. Transfer tensors to TT device
            # 2. Execute reader kernel
            # 3. Execute compute kernel
            # 4. Execute writer kernel
            # 5. Transfer results back

            # For now, just return a copy of the last tensor as output
            if self.result_idx and args:
                outputs = []
                for idx in self.result_idx:
                    if idx < len(args):
                        # Return a copy to simulate computation
                        outputs.append(args[idx].clone())
                    else:
                        # Create a dummy output tensor
                        shape = args[0].shape if args else (1,)
                        dtype = args[0].dtype if args else torch.float32
                        outputs.append(torch.zeros(shape, dtype=dtype))
                return outputs[0] if len(outputs) == 1 else outputs

            return None

        return lambda_forward

    @property
    def prim_func(self) -> tir.PrimFunc:
        """Returns the primary TIR function from the IR module."""
        return retrieve_func_from_module(self.ir_module)

    @property
    def is_dynamic(self):
        """Indicates whether the kernel handles dynamic shapes."""
        # TT backend doesn't support dynamic shapes yet
        return False

    def get_kernel_source(self, kernel_only: bool = False):
        """Returns the source code of the kernel (TT artifacts as JSON)."""
        return self.kernel_global_source

    def get_artifact(self, name: str) -> Optional[str]:
        """Get a specific TT artifact by name.

        Args:
            name: Artifact name (e.g., 'reader.cpp', 'compute.cpp', 'writer.cpp', 'main.cpp')

        Returns:
            The artifact content or None if not found
        """
        return self.artifacts.get(name)

    @classmethod
    def from_database(cls,
                      params: List[KernelParam],
                      result_idx: List[int],
                      target: Union[str, Target],
                      func_or_mod: Union[tir.PrimFunc, tvm.IRModule],
                      kernel_global_source: str,
                      pass_configs: Optional[Dict[str, Any]] = None,
                      compile_flags: Optional[List[str]] = None):
        """Create adapter from cached data (used when loading from disk cache).

        Args:
            params: List of tensor types for inputs/outputs
            result_idx: Indices of output tensors
            target: Target platform (should be 'tenstorrent')
            func_or_mod: TIR function or module
            kernel_global_source: JSON string containing TT artifacts
            pass_configs: Optional pass configuration
            compile_flags: Optional compile flags

        Returns:
            TTKernelAdapter instance created from cached data
        """
        # For TT, we don't have compiled libraries, just artifacts
        # Create the adapter with the loaded artifacts
        return cls(
            params=params,
            result_idx=result_idx,
            target=target,
            func_or_mod=func_or_mod,
            kernel_global_source=kernel_global_source,
            verbose=False,  # Don't be verbose when loading from cache
            pass_configs=pass_configs,
            compile_flags=compile_flags,
        )
