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
print("Function script:")
print(func.script())
print("\nFunction body type:", type(func.body))

# Walk the body to understand structure
def walk_stmt(stmt, depth=0):
    indent = "  " * depth
    print(f"{indent}{type(stmt).__name__}")

    # Handle BlockRealize
    if isinstance(stmt, tvm.tir.BlockRealize):
        print(f"{indent}  block: {stmt.block}")
        if hasattr(stmt.block, 'body'):
            walk_stmt(stmt.block.body, depth + 1)
    # Handle Block
    elif isinstance(stmt, tvm.tir.Block):
        print(f"{indent}  iter_vars: {stmt.iter_vars}")
        if hasattr(stmt, 'body'):
            walk_stmt(stmt.body, depth + 1)
    # Handle DeclBuffer
    elif isinstance(stmt, tvm.tir.DeclBuffer):
        print(f"{indent}  buffer: {stmt.buffer.name}")
        print(f"{indent}  scope: {stmt.buffer.scope()}")
        if hasattr(stmt, 'body'):
            walk_stmt(stmt.body, depth + 1)
    # Handle Allocate
    elif isinstance(stmt, tvm.tir.Allocate):
        print(f"{indent}  buffer_var: {stmt.buffer_var.name}")
        print(f"{indent}  scope: {stmt.buffer_var.type_annotation}")
        if hasattr(stmt, 'body'):
            walk_stmt(stmt.body, depth + 1)
    # Handle other statements
    elif hasattr(stmt, 'body'):
        walk_stmt(stmt.body, depth + 1)
    elif hasattr(stmt, 'seq'):
        for s in stmt.seq:
            walk_stmt(s, depth + 1)

print("\nBody structure:")
walk_stmt(func.body)