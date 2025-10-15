#!/usr/bin/env python3
"""Comprehensive fix for v5 passes visitor patterns"""

import os
import re

# Fix lower_tt_tile_intrinsics_v5.py
filepath = "tilelang/tenstorrent/passes/lower_tt_tile_intrinsics_v5.py"

if os.path.exists(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Fix the TileIntrinsicLowerer visit method
    content = re.sub(
        r'def visit\(self, stmt\):\s*"""Generic visit method"""\s*if hasattr\(stmt, "body"\):\s*return self\.visit\(stmt\.body\)\s*return stmt',
        '''def visit(self, stmt):
            """Generic visit method that dispatches to specific visit methods"""
            if stmt is None:
                return None

            # Dispatch to specific visit methods based on node type
            if isinstance(stmt, tir.For):
                return self.visit_for(stmt)
            elif isinstance(stmt, tir.Evaluate):
                return self.visit_evaluate(stmt)
            elif isinstance(stmt, tir.BufferStore):
                return self.visit_buffer_store(stmt)
            elif isinstance(stmt, tir.SeqStmt):
                new_seq = []
                for s in stmt.seq:
                    new_s = self.visit(s)
                    if new_s is not None:
                        new_seq.append(new_s)
                return tir.SeqStmt(new_seq) if new_seq else None
            elif hasattr(stmt, "body"):
                new_body = self.visit(stmt.body)
                if new_body != stmt.body:
                    return stmt.with_body(new_body) if hasattr(stmt, 'with_body') else stmt
                return stmt
            else:
                return stmt''',
        content,
        flags=re.DOTALL
    )

    # Fix ReductionChecker to have proper visit method
    content = re.sub(
        r'class ReductionChecker:\s+def __init__\(self, var\):\s+self\.var = var\s+self\.has_reduction = False\s+def visit_buffer_store',
        '''class ReductionChecker:
                def __init__(self, var):
                    self.var = var
                    self.has_reduction = False

                def visit(self, stmt):
                    """Visit method for ReductionChecker"""
                    if isinstance(stmt, tir.BufferStore):
                        self.visit_buffer_store(stmt)
                    elif hasattr(stmt, "body"):
                        self.visit(stmt.body)

                def visit_buffer_store''',
        content
    )

    # Fix HeuristicChecker to have proper visit method
    content = re.sub(
        r'class HeuristicChecker:\s+def __init__\(self\):\s+self\.has_heuristics = False\s+self\.issues = \[\]\s+def visit_evaluate',
        '''class HeuristicChecker:
        def __init__(self):
            self.has_heuristics = False
            self.issues = []

        def visit(self, stmt):
            """Visit and check for heuristics"""
            if isinstance(stmt, tir.Evaluate):
                self.visit_evaluate(stmt)
            elif isinstance(stmt, tir.SeqStmt):
                for s in stmt.seq:
                    self.visit(s)
            elif hasattr(stmt, "body"):
                self.visit(stmt.body)

        def visit_evaluate''',
        content
    )

    # Remove super() calls that no longer exist
    content = re.sub(r'return super\(\)\.visit_buffer_store\(buffer_store\)',
                     'return buffer_store', content)

    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Fixed {filepath}")

# Fix grid_to_core_grid_v5.py
filepath = "tilelang/tenstorrent/passes/grid_to_core_grid_v5.py"

if os.path.exists(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Fix the CoreGridTransformer visit method
    content = re.sub(
        r'def visit\(self, stmt\):\s*"""Generic visit method"""\s*if hasattr\(stmt, "body"\):\s*return self\.visit\(stmt\.body\)\s*return stmt',
        '''def visit(self, stmt):
            """Generic visit method that dispatches to specific visit methods"""
            if stmt is None:
                return None

            # Dispatch to specific visit methods based on node type
            if isinstance(stmt, tir.For):
                return self.visit_for(stmt)
            elif isinstance(stmt, tir.BufferStore):
                return self.visit_buffer_store(stmt)
            elif isinstance(stmt, tir.BufferLoad):
                return self.visit_buffer_load(stmt)
            elif isinstance(stmt, tir.SeqStmt):
                new_seq = []
                for s in stmt.seq:
                    new_s = self.visit(s)
                    if new_s is not None:
                        new_seq.append(new_s)
                return tir.SeqStmt(new_seq) if new_seq else None
            elif hasattr(stmt, "body"):
                new_body = self.visit(stmt.body)
                if new_body != stmt.body:
                    return stmt.with_body(new_body) if hasattr(stmt, 'with_body') else stmt
                return stmt
            else:
                return stmt''',
        content,
        flags=re.DOTALL
    )

    # Fix ValidationChecker
    content = re.sub(
        r'class ValidationChecker:\s+def __init__\(self\):\s+self\.has_gpu = False\s+self\.gpu_patterns = \[\]\s+def visit_for',
        '''class ValidationChecker:
        def __init__(self):
            self.has_gpu = False
            self.gpu_patterns = []

        def visit(self, stmt):
            """Visit and check for GPU patterns"""
            if isinstance(stmt, tir.For):
                self.visit_for(stmt)
            elif isinstance(stmt, tir.SeqStmt):
                for s in stmt.seq:
                    self.visit(s)
            elif hasattr(stmt, "body"):
                self.visit(stmt.body)

        def visit_for''',
        content
    )

    # Remove super() calls
    content = re.sub(r'# Continue visiting \(was super\(\)\.visit_for\)', '', content)
    content = re.sub(r'# Continue visiting \(was super\(\)\.visit_buffer_store\)', '', content)
    content = re.sub(r'# Continue visiting \(was super\(\)\.visit_buffer_load\)', '', content)

    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Fixed {filepath}")

print("\nDone with comprehensive v5 fixes")