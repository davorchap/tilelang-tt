#!/bin/bash
# Script to fix TVM imports in all test files

cd testing/python/tenstorrent

FILES=(
    "test_a3_attach_tensor_accessor.py"
    "test_a3_simple.py"
    "test_block_transformer.py"
    "test_codegen_compute_visitor.py"
    "test_codegen_pipeline.py"
    "test_codegen_reader_visitor.py"
    "test_codegen_visitor_base.py"
    "test_codegen_writer_visitor.py"
    "test_end_to_end_pipeline.py"
    "test_host_program_pipeline.py"
    "test_ir_to_codegen_integration.py"
    "test_lower_gemm_to_tt_intrinsics.py"
    "test_memory_space_lower_tt.py"
    "test_reader_writer_pipeline.py"
    "test_tile_pad_tt.py"
    "test_tt_tiles_to_core_map.py"
    "test_v5_metadata_passes.py"
    "test_verify_tt_ir.py"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "Fixing $file..."
        # Create temp file with fixed imports
        python3 << 'PYTHON_SCRIPT'
import sys
import re

file = sys.argv[1]
with open(file, 'r') as f:
    content = f.read()

# Pattern 1: Replace standalone "import tvm" at start of line
# Add tilelang import before it
if re.search(r'^import tvm$', content, re.MULTILINE):
    # Find the line and replace
    lines = content.split('\n')
    new_lines = []
    tvm_imported = False

    for i, line in enumerate(lines):
        if line.strip() == 'import tvm' and not tvm_imported:
            # Add tilelang imports
            new_lines.append('# Import tilelang first to get proper TVM')
            new_lines.append('import tilelang')
            new_lines.append('from tilelang import tvm')
            tvm_imported = True
        elif line.strip().startswith('import tvm') and 'tilelang' not in line and not tvm_imported:
            # This is something like "import tvm.something"
            # We need to add tilelang import before this line
            new_lines.append('# Import tilelang first to get proper TVM')
            new_lines.append('import tilelang')
            new_lines.append('from tilelang import tvm')
            new_lines.append(line)
            tvm_imported = True
        else:
            new_lines.append(line)

    content = '\n'.join(new_lines)

with open(file, 'w') as f:
    f.write(content)

print(f"Fixed {file}")
PYTHON_SCRIPT
        python3 -c "
import sys
file = '$file'
exec(open('/dev/stdin').read())
" < <(cat << 'EOF'
import re

with open(file, 'r') as f:
    content = f.read()

# Pattern 1: Replace standalone "import tvm" at start of line
# Add tilelang import before it
if re.search(r'^import tvm$', content, re.MULTILINE):
    # Find the line and replace
    lines = content.split('\n')
    new_lines = []
    tvm_imported = False

    for i, line in enumerate(lines):
        if line.strip() == 'import tvm' and not tvm_imported:
            # Add tilelang imports
            new_lines.append('# Import tilelang first to get proper TVM')
            new_lines.append('import tilelang')
            new_lines.append('from tilelang import tvm')
            tvm_imported = True
        elif line.strip().startswith('import tvm') and 'tilelang' not in line and not tvm_imported:
            # Skip the old import tvm line
            continue
        else:
            new_lines.append(line)

    content = '\n'.join(new_lines)

    with open(file, 'w') as f:
        f.write(content)

    print(f"Fixed {file}")
else:
    print(f"No 'import tvm' found in {file}")
EOF
)
    fi
done

echo "Done fixing all files!"
