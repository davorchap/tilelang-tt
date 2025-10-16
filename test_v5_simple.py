#!/usr/bin/env python3
"""
Simple test to debug v5 pass issues
"""

import sys
import os
sys.path.append("tilelang/tenstorrent/passes")

# Import through tilelang
try:
    import tilelang
    print("✓ Tilelang imported successfully")
    from tilelang import tvm
    print("✓ TVM imported through tilelang")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Try to import TVM script
try:
    from tvm.script import tir as T
    import tvm.script
    print("✓ TVM script imported successfully")
except Exception as e:
    print(f"✗ Failed to import TVM script: {e}")
    sys.exit(1)

# Try to create a simple module
try:
    @tvm.script.ir_module
    class TestModule:
        @T.prim_func
        def func(A: T.Buffer((256, 256), "float16")):
            A_shared = T.alloc_buffer((32, 32), "float16", scope="shared")
            T.evaluate(0)

    print("✓ Created test module successfully")
    func = TestModule["func"]
    print(f"  Function type: {type(func)}")
    print(f"  Module type: {type(TestModule)}")
except Exception as e:
    print(f"✗ Failed to create test module: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Try to import and apply our pass
try:
    # Update imports to use tilelang's tvm
    # Need to update the pass file imports too
    import importlib.util
    spec = importlib.util.spec_from_file_location("lower_shared_to_cb_v5",
                                                  "tilelang/tenstorrent/passes/lower_shared_to_cb_v5.py")
    module = importlib.util.module_from_spec(spec)

    # Inject tilelang's tvm into the module's namespace
    module.tvm = tvm
    module.tir = tvm.tir

    spec.loader.exec_module(module)
    LowerSharedToCB_v5 = module.LowerSharedToCB_v5

    print("✓ Imported LowerSharedToCB_v5 successfully")
    print(f"  Pass type: {type(LowerSharedToCB_v5)}")

    # Try to apply the pass
    result = LowerSharedToCB_v5(TestModule)
    print("✓ Applied pass successfully")
    print(f"  Result type: {type(result)}")

    # Check if we got the function back
    if "func" in result:
        result_func = result["func"]
        print(f"  Result function type: {type(result_func)}")

        # Check for attributes
        if hasattr(result_func, 'attrs'):
            print(f"  Attributes: {list(result_func.attrs.keys())}")
            if "tt.conceptual_cbs" in result_func.attrs:
                print("  ✓ Found tt.conceptual_cbs attribute")
            else:
                print("  ✗ Missing tt.conceptual_cbs attribute")
    else:
        print("  ✗ 'func' not found in result")

except Exception as e:
    print(f"✗ Failed with pass: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Done ===")