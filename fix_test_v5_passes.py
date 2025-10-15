#!/usr/bin/env python3
"""Fix test_v5_passes.py to properly call the v5 passes"""

import re

# Read the test file
with open("testing/python/tenstorrent/test_v5_passes.py", "r") as f:
    content = f.read()

# Pattern 1: Fix LowerSharedToCB_v5 calls
pattern1 = r'func = Before\["func"\]\s+transformed = LowerSharedToCB_v5\(func, Before, None\)'
replacement1 = '''# Apply the pass using TVM's pass infrastructure
        transformed_mod = LowerSharedToCB_v5(Before)
        transformed = transformed_mod["func"]'''

content = re.sub(pattern1, replacement1, content)

# Pattern 2: Fix LowerTTTileIntrinsics_v5 calls
pattern2 = r'transformed = LowerTTTileIntrinsics_v5\(func, Before, None\)'
replacement2 = '''# Apply the pass
        Before["func"] = func
        transformed_mod = LowerTTTileIntrinsics_v5(Before)
        transformed = transformed_mod["func"]'''

content = re.sub(pattern2, replacement2, content)

# Pattern 3: Fix GridToCoreGrid_v5 calls
pattern3 = r'transformed = GridToCoreGrid_v5\(func, Before, None\)'
replacement3 = '''# Apply the pass
        Before["func"] = func
        transformed_mod = GridToCoreGrid_v5(Before)
        transformed = transformed_mod["func"]'''

content = re.sub(pattern3, replacement3, content)

# Pattern 4: Fix the integration test passes
content = re.sub(
    r'func = GridToCoreGrid_v5\(func, Original, None\)',
    '''# Apply B2 pass
        Original["gemm"] = func
        result_mod = GridToCoreGrid_v5(Original)
        func = result_mod["gemm"]''',
    content
)

content = re.sub(
    r'func = LowerSharedToCB_v5\(func, Original, None\)',
    '''# Apply C1 pass
        result_mod["gemm"] = func
        result_mod = LowerSharedToCB_v5(result_mod)
        func = result_mod["gemm"]''',
    content
)

content = re.sub(
    r'func = LowerTTTileIntrinsics_v5\(func, Original, None\)',
    '''# Apply C2 pass
        result_mod["gemm"] = func
        result_mod = LowerTTTileIntrinsics_v5(result_mod)
        func = result_mod["gemm"]''',
    content
)

# Write the fixed content
with open("testing/python/tenstorrent/test_v5_passes.py", "w") as f:
    f.write(content)

print("Fixed test_v5_passes.py")