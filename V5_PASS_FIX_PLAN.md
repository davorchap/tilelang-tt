# Plan to Fix TVM Block Handling in v5 Passes

## Executive Summary
The v5 passes have fundamental issues with TVM's Block/BlockRealize structure introduced in newer TVM versions. This plan outlines a systematic approach to fix these issues.

## Problem Analysis

### Current Issues
1. **Block Structure Misunderstanding**:
   - Shared memory allocations are in `Block.alloc_buffers`, not as `Allocate` nodes
   - The passes expect old-style `Allocate` nodes but TVM now uses `Block` structures

2. **Visitor Pattern Issues**:
   - Custom visitor implementations don't properly handle Block/BlockRealize nodes
   - Missing proper reconstruction of Block nodes after transformation

3. **Constructor Signature Mismatch**:
   - Block constructor requires specific parameters in correct order
   - Current code passes incorrect arguments causing AttributeError

4. **Pass Decorator Confusion**:
   - Passes use `@tvm.tir.transform.prim_func_pass` but need proper integration

## Phase 1: Understanding TVM Block Structure

### 1.1 Research Block API
```python
# Key structures to understand:
- tir.Block: Contains alloc_buffers, body, iter_vars, reads, writes
- tir.BlockRealize: Wrapper around Block with iter_values and predicate
- Block.alloc_buffers: List of Buffer objects for local allocations
```

### 1.2 Create Test Infrastructure
```python
# Create simple test cases showing:
- How T.alloc_buffer creates Block.alloc_buffers
- How T.copy operations work within blocks
- How to properly traverse and modify Block structures
```

## Phase 2: Create Block Handling Utilities

### 2.1 BlockTransformer Base Class
```python
class BlockTransformer:
    """Base class for transforming TVM Block structures"""

    def transform_block_realize(self, block_realize):
        """Transform a BlockRealize node"""
        new_block = self.transform_block(block_realize.block)
        if new_block != block_realize.block:
            return tir.BlockRealize(
                block_realize.iter_values,
                block_realize.predicate,
                new_block,
                block_realize.span
            )
        return block_realize

    def transform_block(self, block):
        """Transform a Block node"""
        # Process alloc_buffers
        new_alloc_buffers = self.process_alloc_buffers(block.alloc_buffers)

        # Process body
        new_body = self.visit(block.body)

        # Reconstruct block if changed
        if new_body != block.body or new_alloc_buffers != block.alloc_buffers:
            return block.replace(
                body=new_body,
                alloc_buffers=new_alloc_buffers
            )
        return block
```

### 2.2 Buffer Scope Utilities
```python
def is_shared_buffer(buffer):
    """Check if buffer is in shared scope"""
    return buffer.scope() == "shared"

def create_cb_intrinsic(cb_name, shape, dtype):
    """Create a CB allocation intrinsic"""
    return tir.call_extern(
        "handle",
        "tt.alloc_cb",
        tir.StringImm(cb_name),
        *[tir.IntImm("int32", dim) for dim in shape],
        tir.StringImm(str(dtype))
    )
```

## Phase 3: Fix LowerSharedToCB_v5

### 3.1 Refactor SharedToCBTransformer
```python
class SharedToCBTransformer(BlockTransformer):
    def process_alloc_buffers(self, alloc_buffers):
        """Process Block.alloc_buffers for shared->CB conversion"""
        cb_metadata = []

        for buffer in alloc_buffers:
            if is_shared_buffer(buffer):
                cb_name = self.generate_cb_name(buffer.name)
                self.shared_to_cb_map[buffer.name] = cb_name

                # Store metadata
                cb_metadata.append({
                    'name': cb_name,
                    'shape': buffer.shape,
                    'dtype': buffer.dtype,
                    'original': buffer.name
                })

        # Keep original buffers, add CB info to metadata
        return alloc_buffers

    def transform_block_body(self, body, cb_metadata):
        """Insert CB allocations at start of body"""
        cb_allocs = []
        for cb_info in cb_metadata:
            cb_allocs.append(tir.Evaluate(
                create_cb_intrinsic(
                    cb_info['name'],
                    cb_info['shape'],
                    cb_info['dtype']
                )
            ))

        if cb_allocs:
            return tir.SeqStmt(cb_allocs + [body])
        return body
```

### 3.2 Handle Copy Operations
```python
def transform_copy(self, src, dst):
    """Transform T.copy to CB operations"""
    src_cb = self.get_cb_for_buffer(src)
    dst_cb = self.get_cb_for_buffer(dst)

    if src_cb and not dst_cb:
        # CB -> DRAM
        return create_write_from_cb(src_cb, dst)
    elif not src_cb and dst_cb:
        # DRAM -> CB
        return create_read_to_cb(src, dst_cb)

    return None  # No transformation needed
```

## Phase 4: Fix LowerTTTileIntrinsics_v5

### 4.1 Pattern Matching in Blocks
```python
class TileIntrinsicLowerer(BlockTransformer):
    def detect_compute_pattern(self, block):
        """Detect compute patterns in Block body"""
        patterns = []

        # Walk block body looking for patterns
        def visit_patterns(stmt):
            if is_gemm_pattern(stmt):
                patterns.append(('gemm', stmt))
            elif is_elementwise_pattern(stmt):
                patterns.append(('elementwise', stmt))

        stmt_functor.post_order_visit(block.body, visit_patterns)
        return patterns

    def lower_patterns(self, patterns):
        """Lower detected patterns to TT intrinsics"""
        replacements = {}

        for pattern_type, stmt in patterns:
            if pattern_type == 'gemm':
                replacements[stmt] = create_tt_matmul_intrinsic(stmt)
            elif pattern_type == 'elementwise':
                replacements[stmt] = create_tt_fpu_intrinsic(stmt)

        return replacements
```

## Phase 5: Fix GridToCoreGrid_v5

### 5.1 Grid Loop Detection
```python
class CoreGridTransformer(BlockTransformer):
    def is_grid_loop(self, for_node):
        """Detect GPU grid loops in Block context"""
        # Check for thread binding annotations
        if hasattr(for_node, 'thread_binding'):
            binding = for_node.thread_binding
            return 'blockIdx' in str(binding)
        return False

    def transform_to_core_launch(self, for_node):
        """Transform GPU grid loop to TT core launch"""
        # Extract grid dimensions
        grid_x = self.extract_grid_dim(for_node, 'x')
        grid_y = self.extract_grid_dim(for_node, 'y')

        # Create core launch
        return create_core_launch(
            grid_x, grid_y,
            for_node.body,
            self.metadata
        )
```

## Phase 6: Testing Strategy

### 6.1 Unit Tests
1. Test Block structure traversal
2. Test buffer scope detection
3. Test CB allocation generation
4. Test pattern matching in blocks
5. Test grid loop transformation

### 6.2 Integration Tests
```python
def test_full_pipeline():
    """Test all three passes in sequence"""

    # Create test module with Blocks
    @tvm.script.ir_module
    class TestModule:
        @T.prim_func
        def func(A: T.Buffer, B: T.Buffer, C: T.Buffer):
            A_shared = T.alloc_buffer(..., scope="shared")
            # ... computation ...

    # Apply passes
    mod = LowerSharedToCB_v5(TestModule)
    mod = LowerTTTileIntrinsics_v5(mod)
    mod = GridToCoreGrid_v5(mod)

    # Validate output
    validate_output(mod)
```

## Phase 7: Implementation Order

### Week 1: Foundation
1. Create BlockTransformer base class
2. Implement buffer utilities
3. Create test infrastructure

### Week 2: LowerSharedToCB_v5
1. Refactor to use BlockTransformer
2. Fix alloc_buffer handling
3. Fix copy operations
4. Add tests

### Week 3: LowerTTTileIntrinsics_v5
1. Implement pattern detection in Blocks
2. Fix intrinsic lowering
3. Add tests

### Week 4: GridToCoreGrid_v5
1. Fix grid loop detection
2. Implement core launch transformation
3. Add tests

### Week 5: Integration
1. Full pipeline testing
2. Performance validation
3. Documentation

## Key Implementation Files

### New Files to Create:
- `tilelang/tenstorrent/passes/block_transformer.py` - Base utilities
- `testing/python/tenstorrent/test_block_handling.py` - Block tests
- `testing/python/tenstorrent/test_v5_integration.py` - Integration tests

### Files to Modify:
- `tilelang/tenstorrent/passes/lower_shared_to_cb_v5.py`
- `tilelang/tenstorrent/passes/lower_tt_tile_intrinsics_v5.py`
- `tilelang/tenstorrent/passes/grid_to_core_grid_v5.py`

## Success Criteria

1. All v5 pass tests pass (test_v5_passes.py)
2. Integration tests pass
3. No TVM API errors
4. Proper Block structure handling
5. CB allocations correctly generated
6. Compute patterns properly lowered
7. Grid loops transformed to core launches

## Risk Mitigation

### Risk 1: TVM API Changes
- **Mitigation**: Pin TVM version, add version checks

### Risk 2: Block Structure Complexity
- **Mitigation**: Start with simple cases, incrementally add complexity

### Risk 3: Performance Regression
- **Mitigation**: Add performance benchmarks before/after

## Next Steps

1. Review and approve this plan
2. Create BlockTransformer base class
3. Begin implementation with LowerSharedToCB_v5
4. Iterate based on test results

## Notes

- Consider using TVM's built-in IRMutator if available in the version
- May need to handle nested Blocks
- Consider backward compatibility with non-Block IR