import tvm
from tvm.script import tir as T
import tvm.script

@tvm.script.ir_module
class TestModule:
    @T.prim_func
    def func(A: T.Buffer((256, 256), "float16")):
        A_shared = T.alloc_buffer((32, 32), "float16", scope="shared")
        T.evaluate(0)

func = TestModule["func"]

# Use post_order_visit to traverse the AST
found_shared = []

def visitor(node):
    # Check for DeclBuffer nodes with shared scope
    if isinstance(node, tvm.tir.DeclBuffer):
        if node.buffer.scope() == "shared":
            print(f"Found DeclBuffer with shared scope: {node.buffer.name}")
            found_shared.append(node.buffer.name)
    # Check for Allocate nodes
    elif isinstance(node, tvm.tir.Allocate):
        print(f"Found Allocate: {node.buffer_var.name}")
        # Check storage scope
        if hasattr(node.buffer_var, 'type_annotation'):
            print(f"  Type annotation: {node.buffer_var.type_annotation}")

tvm.tir.stmt_functor.post_order_visit(func.body, visitor)

print(f"\nFound shared buffers: {found_shared}")

# Also check function's buffer_map for any alloc_buffers
print(f"\nFunction buffer_map: {func.buffer_map}")

# Check for buffers in block's alloc_buffers
def check_blocks(node):
    if isinstance(node, tvm.tir.Block):
        print(f"Block alloc_buffers: {node.alloc_buffers}")
        for buf in node.alloc_buffers:
            print(f"  Buffer: {buf.name}, scope: {buf.scope()}")

tvm.tir.stmt_functor.post_order_visit(func.body, check_blocks)