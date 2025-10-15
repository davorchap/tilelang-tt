#!/bin/bash
# Run specific failing tests
export LD_LIBRARY_PATH=build/tvm:$LD_LIBRARY_PATH

echo "Testing jit decorator tests..."
python3 testing/python/tenstorrent/test_jit_decorator.py

echo ""
echo "Testing TT intrinsic lowering tests..."
python3 testing/python/tenstorrent/test_lower_gemm_to_tt_intrinsics.py