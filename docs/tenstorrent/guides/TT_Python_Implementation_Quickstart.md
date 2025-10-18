# TT Backend Python Implementation Quickstart

**Version:** 1.0
**Date:** 2025-10-15
**Purpose:** Get developers productive in <30 minutes

---

## 🚀 Quick Start (5 minutes)

### Step 1: Create Your First Pass

```python
# File: tilelang/tenstorrent/passes/my_first_pass.py

import tvm
from tvm import tir

@tvm.tir.transform.prim_func_pass(opt_level=0)
def MyFirstPass(func, mod, ctx):
    """Your pass description here"""

    # Add a simple attribute to verify it works
    func = func.with_attr("tt.my_pass_ran", True)
    return func
```

### Step 2: Test It

```python
# File: testing/python/tenstorrent/test_my_first_pass.py

import tvm
from tvm import tir
from tilelang.tenstorrent.passes.my_first_pass import MyFirstPass

def test_basic():
    # Create a simple function
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((256, 256), "float32")):
            T.evaluate(0)  # Dummy body

    # Apply your pass
    result = MyFirstPass(Module["main"], Module, None)

    # Verify it worked
    assert result.attrs["tt.my_pass_ran"] == True
    print("✅ Pass works!")

if __name__ == "__main__":
    test_basic()
```

### Step 3: Run It

```bash
cd tilelang-tt
python testing/python/tenstorrent/test_my_first_pass.py
# Output: ✅ Pass works!
```

---

## 📚 Pass Implementation Templates

### Template 1: Simple Attribute Pass

```python
@tvm.tir.transform.prim_func_pass(opt_level=0)
def AddMetadataPass(func, mod, ctx):
    """Add metadata to function or buffers"""

    # Add function-level metadata
    func = func.with_attr("tt.some_metadata", {"key": "value"})

    # Add buffer-level metadata
    for param in func.params:
        if param in func.buffer_map:
            buffer = func.buffer_map[param]
            # Create new buffer with attribute
            new_buffer = tir.decl_buffer(
                buffer.shape,
                buffer.dtype,
                buffer.name,
                buffer_attrs={"tt.layout": "interleaved"}
            )
            func.buffer_map[param] = new_buffer

    return func
```

### Template 2: IR Visitor (Analysis)

```python
@tvm.tir.transform.prim_func_pass(opt_level=0)
def AnalysisPass(func, mod, ctx):
    """Analyze IR without modification"""

    class Analyzer(tir.stmt_functor.StmtVisitor):
        def __init__(self):
            super().__init__()
            self.stats = {"calls": 0, "loops": 0}

        def visit_evaluate(self, op):
            if isinstance(op.value, tir.Call):
                self.stats["calls"] += 1
            super().visit_evaluate(op)

        def visit_for(self, op):
            self.stats["loops"] += 1
            super().visit_for(op)

    analyzer = Analyzer()
    analyzer.visit(func.body)

    # Store analysis results
    func = func.with_attr("tt.analysis", analyzer.stats)
    return func
```

### Template 3: IR Mutator (Transformation)

```python
@tvm.tir.transform.prim_func_pass(opt_level=0)
def TransformPass(func, mod, ctx):
    """Transform IR structure"""

    class Transformer(tir.stmt_functor.IRMutator):
        def visit_evaluate(self, op):
            # Replace pattern
            if self._matches_pattern(op):
                return self._create_replacement(op)
            return super().visit_evaluate(op)

        def _matches_pattern(self, op):
            if isinstance(op.value, tir.Call):
                return op.value.op.name == "old_intrinsic"
            return False

        def _create_replacement(self, op):
            # Create new IR nodes
            new_call = tir.call_extern(
                "void", "new_intrinsic",
                op.value.args[0]  # Reuse arguments
            )
            return tir.Evaluate(new_call)

    transformer = Transformer()
    new_body = transformer.visit(func.body)
    return func.with_body(new_body)
```

### Template 4: Module Pass (Creates Multiple Functions)

```python
@tvm.tir.transform.module_pass(opt_level=0)
def SplitPass(mod, ctx):
    """Split one function into multiple"""

    new_funcs = {}

    for gvar, func in mod.functions.items():
        if not isinstance(func, tir.PrimFunc):
            new_funcs[gvar] = func
            continue

        # Create multiple functions
        func1 = create_variant1(func)
        func2 = create_variant2(func)

        # Add with new names
        gvar1 = tvm.ir.GlobalVar(f"{gvar.name_hint}_part1")
        gvar2 = tvm.ir.GlobalVar(f"{gvar.name_hint}_part2")

        new_funcs[gvar1] = func1
        new_funcs[gvar2] = func2

    return tvm.ir.IRModule(new_funcs)
```

---

## 🔍 Common IR Patterns

### Pattern 1: Finding Intrinsic Calls

```python
def find_intrinsic_calls(func, intrinsic_name):
    """Find all calls to a specific intrinsic"""
    calls = []

    class CallFinder(tir.stmt_functor.StmtVisitor):
        def visit_evaluate(self, op):
            if isinstance(op.value, tir.Call):
                if op.value.op.name == intrinsic_name:
                    calls.append(op)
            super().visit_evaluate(op)

    finder = CallFinder()
    finder.visit(func.body)
    return calls
```

### Pattern 2: Building Statement Sequences

```python
def build_protocol_sequence(cb_name, tile_id):
    """Build a sequence of protocol calls"""
    stmts = []

    # Create IR nodes
    stmts.append(tir.Evaluate(
        tir.call_extern("void", "cb_reserve_back", cb_name, 1)
    ))

    stmts.append(tir.Evaluate(
        tir.call_extern("void", "noc_async_read", tile_id)
    ))

    stmts.append(tir.Evaluate(
        tir.call_extern("void", "cb_push_back", cb_name, 1)
    ))

    # Combine into sequence
    return tir.SeqStmt(stmts)
```

### Pattern 3: Extracting Buffer Information

```python
def get_buffer_info(func, buffer_name):
    """Extract buffer metadata"""
    for param in func.params:
        if param in func.buffer_map:
            buffer = func.buffer_map[param]
            if buffer.name == buffer_name:
                return {
                    "shape": buffer.shape,
                    "dtype": buffer.dtype,
                    "strides": buffer.strides,
                    "attrs": buffer.attrs if hasattr(buffer, 'attrs') else {}
                }
    return None
```

### Pattern 4: Loop Analysis

```python
def analyze_loops(func):
    """Analyze loop structure"""

    class LoopAnalyzer(tir.stmt_functor.StmtVisitor):
        def __init__(self):
            super().__init__()
            self.loops = []
            self.current_depth = 0

        def visit_for(self, op):
            self.loops.append({
                "var": op.loop_var.name,
                "extent": op.extent,
                "depth": self.current_depth,
                "kind": op.kind
            })
            self.current_depth += 1
            super().visit_for(op)
            self.current_depth -= 1

    analyzer = LoopAnalyzer()
    analyzer.visit(func.body)
    return analyzer.loops
```

---

## 🛠 Debugging Techniques

### Interactive Debugging

```python
# Add breakpoint in your pass
import ipdb

@tvm.tir.transform.prim_func_pass(opt_level=0)
def DebuggablePass(func, mod, ctx):
    ipdb.set_trace()  # Drops into debugger here

    # Explore the IR
    print(func)  # Print entire function
    print(func.attrs)  # Print attributes
    print(func.buffer_map)  # Print buffers

    return func
```

### IR Pretty Printing

```python
def debug_print_ir(stmt, indent=0):
    """Pretty print IR structure"""
    prefix = "  " * indent

    if isinstance(stmt, tir.SeqStmt):
        print(f"{prefix}SeqStmt:")
        for s in stmt.seq:
            debug_print_ir(s, indent + 1)
    elif isinstance(stmt, tir.Evaluate):
        print(f"{prefix}Evaluate: {stmt.value}")
    elif isinstance(stmt, tir.For):
        print(f"{prefix}For {stmt.loop_var} in {stmt.extent}:")
        debug_print_ir(stmt.body, indent + 1)
    else:
        print(f"{prefix}{type(stmt).__name__}")
```

### Validation Helpers

```python
def validate_cb_ids(func):
    """Validate CB IDs are within range"""
    cb_ids = set()

    class CBCollector(tir.stmt_functor.StmtVisitor):
        def visit_evaluate(self, op):
            if isinstance(op.value, tir.Call):
                # Extract CB ID from various intrinsics
                if "cb_" in op.value.op.name:
                    cb_name = op.value.args[0]
                    if isinstance(cb_name, tir.StringImm):
                        # Extract number from "cb_in0", "cb_out1", etc.
                        import re
                        match = re.search(r'cb_[a-z]+(\d+)', cb_name.value)
                        if match:
                            cb_ids.add(int(match.group(1)))

    collector = CBCollector()
    collector.visit(func.body)

    assert max(cb_ids, default=0) < 32, f"CB IDs exceed limit: {cb_ids}"
    return True
```

---

## 📊 Testing Strategies

### Unit Test Structure

```python
import pytest
import tvm
from tvm import tir

class TestMyPass:
    def setup_method(self):
        """Setup test fixtures"""
        self.simple_kernel = self.create_simple_kernel()
        self.complex_kernel = self.create_complex_kernel()

    def create_simple_kernel(self):
        @tvm.script.ir_module
        class Module:
            @T.prim_func
            def main(A: T.Buffer((256, 256), "float32")):
                for i, j in T.grid(256, 256):
                    A[i, j] = 0.0
        return Module["main"]

    def test_basic_functionality(self):
        """Test basic pass functionality"""
        result = MyPass()(self.simple_kernel)
        assert validate_output(result)

    @pytest.mark.parametrize("size", [128, 256, 512])
    def test_different_sizes(self, size):
        """Test with different tensor sizes"""
        kernel = create_kernel_with_size(size)
        result = MyPass()(kernel)
        assert validate_output(result)

    def test_error_handling(self):
        """Test error conditions"""
        with pytest.raises(ValueError):
            MyPass()(invalid_kernel)
```

### Integration Testing

```python
def test_pass_integration():
    """Test pass in pipeline context"""

    # Create pipeline
    pipeline = [
        Pass1(),
        Pass2(),
        MyPass(),
        Pass3()
    ]

    # Apply sequentially with validation
    kernel = create_test_kernel()
    for i, pass_func in enumerate(pipeline):
        print(f"Applying {pass_func.__class__.__name__}")
        kernel = pass_func(kernel)

        # Validate intermediate state
        assert is_valid_ir(kernel), f"Invalid after pass {i}"

    # Validate final output
    assert final_validation(kernel)
```

---

## 🚨 Common Pitfalls & Solutions

### Pitfall 1: Mutating Shared Objects

```python
# ❌ Wrong - modifies shared object
def bad_pass(func, mod, ctx):
    func.attrs["new_key"] = "value"  # DON'T DO THIS
    return func

# ✅ Correct - creates new function
def good_pass(func, mod, ctx):
    func = func.with_attr("new_key", "value")
    return func
```

### Pitfall 2: Forgetting to Visit Children

```python
# ❌ Wrong - doesn't visit loop body
class BadVisitor(tir.stmt_functor.StmtVisitor):
    def visit_for(self, op):
        # Process loop but forget to visit body
        self.loop_count += 1

# ✅ Correct - visits children
class GoodVisitor(tir.stmt_functor.StmtVisitor):
    def visit_for(self, op):
        self.loop_count += 1
        super().visit_for(op)  # Visit body
```

### Pitfall 3: Type Confusion

```python
# Common type checks
def process_value(val):
    if isinstance(val, tir.IntImm):
        return val.value  # Extract Python int
    elif isinstance(val, tir.StringImm):
        return val.value  # Extract Python string
    elif isinstance(val, tir.Var):
        return val.name  # Get variable name
    else:
        return str(val)  # Fallback
```

---

## 📝 Pass Documentation Template

```python
"""
Pass Name: InsertDSTManagementTT
Purpose: Insert DST register lifecycle management for compute kernels
Author: Your Name
Date: 2025-10-15

Input Expectations:
- Compute kernels with tt.mm.mma intrinsics
- Kernel role == "compute"

Output Guarantees:
- DST acquire/release around compute
- CB synchronization added
- Pack operations inserted

Example:
    Before:
        for k in range(8):
            tt.mm.mma(cb_a, cb_b, dst=0)

    After:
        tt.dst.acquire()
        for k in range(8):
            cb_wait_front(cb_a, 1)
            tt.mm.mma(cb_a, cb_b, dst=0)
            cb_pop_front(cb_a, 1)
        tt.dst.commit()
        pack_tile(0, cb_out)
        tt.dst.release()

Testing:
    pytest testing/python/tenstorrent/test_dst_management.py
"""
```

---

## 🏃 Quick Development Workflow

### 1. Create Pass File
```bash
touch tilelang/tenstorrent/passes/my_new_pass.py
```

### 2. Write Minimal Implementation
```python
# Start with simplest possible version
@tvm.tir.transform.prim_func_pass(opt_level=0)
def MyNewPass(func, mod, ctx):
    print(f"Processing: {func.attrs}")
    return func
```

### 3. Test Immediately
```bash
# Interactive test
python -c "
from tilelang.tenstorrent.passes.my_new_pass import MyNewPass
import tvm
# Quick test
"
```

### 4. Iterate Quickly
```bash
# Edit-test loop (no compilation!)
while true; do
    vim tilelang/tenstorrent/passes/my_new_pass.py
    python test_quick.py
    read -p "Continue? " -n 1
done
```

---

## 🎯 First Day Goals

By the end of your first day, you should:

1. ✅ Have development environment setup
2. ✅ Successfully run the quickstart example
3. ✅ Implement one simple pass using a template
4. ✅ Write and run a test for your pass
5. ✅ Understand the basic IR structure

---

## 📚 Resources

### Essential Files to Study
```
tilelang/tenstorrent/passes/_common.py         # Utility functions
tilelang/tenstorrent/passes/infer_tt_layout_v5.py  # Good v5 pass example
tilelang/tenstorrent/codegen/kernel_generators.py  # Python codegen visitors
tilelang/tenstorrent/codegen/intrinsics.py     # Intrinsic registry
```

### TVM Documentation
- [TIR Intro](https://tvm.apache.org/docs/tutorial/tir_intro.html)
- [Pass Infrastructure](https://tvm.apache.org/docs/arch/pass_infra.html)
- [IR Module](https://tvm.apache.org/docs/reference/api/python/ir.html)

### Debugging Commands
```python
# Print IR as text
print(func.script())

# Explore attributes
for key, value in func.attrs.items():
    print(f"{key}: {value}")

# Check buffer properties
for buf in func.buffer_map.values():
    print(f"{buf.name}: {buf.shape} {buf.dtype}")
```

---

## 🚀 You're Ready!

With this guide, you should be able to:
- Start implementing passes immediately
- Test your changes without compilation
- Debug issues interactively
- Contribute to the TT backend quickly

**Remember:** Focus on getting it working first, optimize later!

---

**Document Version:** 1.0
**Last Updated:** 2025-10-15
**Questions?** Check #tt-backend channel or docs/