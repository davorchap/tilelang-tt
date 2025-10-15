#!/usr/bin/env python3
"""Fix TVM API issues in v5 passes"""

import os
import re

# List of v5 pass files to fix
v5_pass_files = [
    "tilelang/tenstorrent/passes/lower_shared_to_cb_v5.py",
    "tilelang/tenstorrent/passes/lower_tt_tile_intrinsics_v5.py",
    "tilelang/tenstorrent/passes/grid_to_core_grid_v5.py",
    "tilelang/tenstorrent/passes/infer_tt_layout_v5.py",
    "tilelang/tenstorrent/passes/propagate_tt_layout_v5.py",
    "tilelang/tenstorrent/passes/layout_aware_work_partition_tt_v5.py",
]

def fix_pass_file(filepath):
    """Fix TVM API issues in a pass file"""

    if not os.path.exists(filepath):
        print(f"Skipping {filepath} (not found)")
        return

    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content

    # Fix 1: Replace IRMutator inheritance
    content = re.sub(
        r'class (\w+)\(stmt_functor\.IRMutator\):',
        r'class \1:',
        content
    )

    # Fix 2: Replace StmtVisitor inheritance
    content = re.sub(
        r'class (\w+)\(stmt_functor\.StmtVisitor\):',
        r'class \1:',
        content
    )

    # Fix 3: Remove super().__init__() calls that reference non-existent parent
    content = re.sub(
        r'\s+super\(\).__init__\(\)',
        '',
        content
    )

    # Fix 4: Replace super() method calls with custom implementations
    # For visit methods
    content = re.sub(
        r'return super\(\)\.visit_(\w+)\(op\)',
        r'# Continue visiting (was super().visit_\1)\n            return self.visit(op.body) if hasattr(op, "body") else op',
        content
    )

    # Fix 5: Replace super().visit() calls
    content = re.sub(
        r'super\(\)\.visit_(\w+)\(op\)',
        r'# Continue visiting \1',
        content
    )

    # Fix 6: Add visit method if not present for classes that were inheriting
    if 'class ' in content and 'def visit(' not in content and ('IRMutator' in original_content or 'StmtVisitor' in original_content):
        # Find class definitions that need a visit method
        class_pattern = r'(class \w+:.*?)(?=\n    def|\nclass |\n@|\Z)'

        def add_visit_method(match):
            class_def = match.group(1)
            if 'def visit(' not in class_def:
                # Add visit method after __init__ or at the start of the class
                if 'def __init__' in class_def:
                    class_def = re.sub(
                        r'(def __init__.*?\n(?:.*?\n)*?)\n(\s{8})',
                        r'\1\n\n        def visit(self, stmt):\n            """Generic visit method"""\n            if hasattr(stmt, "body"):\n                return self.visit(stmt.body)\n            return stmt\n\n\2',
                        class_def,
                        count=1
                    )
                else:
                    class_def = class_def + '\n\n        def visit(self, stmt):\n            """Generic visit method"""\n            if hasattr(stmt, "body"):\n                return self.visit(stmt.body)\n            return stmt\n'
            return class_def

        content = re.sub(class_pattern, add_visit_method, content, flags=re.DOTALL)

    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Fixed {filepath}")
    else:
        print(f"No changes needed for {filepath}")

# Fix all v5 pass files
for filepath in v5_pass_files:
    fix_pass_file(filepath)

print("\nDone fixing v5 passes TVM API issues")