"""
Pass C3: BuildTileDFGTT (v5 Specification)
Version: 5.0
Date: 2025-10-15

Purpose: Build tile-based dataflow graph from protocol-less IR.
         Tracks CB allocations and data movements to enable kernel splitting.

Input: Protocol-less IR with tt.alloc_cb, tt.read_to_cb, tt.write_from_cb, compute ops
Output: tt.tile_dfg containing nodes, edges, and role assignments
"""

from __future__ import annotations
from typing import Dict, Any, List, Set, Optional
import logging
from dataclasses import dataclass, field
from enum import Enum

try:
    import tvm
    from tvm import tir, IRModule
except ImportError:
    tvm = None
    tir = None
    IRModule = object

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the dataflow graph"""
    BUFFER = "buffer"  # External buffer (DRAM/L1)
    CB = "cb"  # Circular buffer
    COMPUTE = "compute"  # Compute operation
    READ = "read"  # Read operation (buffer -> CB)
    WRITE = "write"  # Write operation (CB -> buffer)


class EdgeType(Enum):
    """Types of edges in the dataflow graph"""
    DATA_READ = "data_read"  # Buffer -> CB via read
    DATA_WRITE = "data_write"  # CB -> Buffer via write
    COMPUTE_INPUT = "compute_in"  # CB -> Compute
    COMPUTE_OUTPUT = "compute_out"  # Compute -> CB


@dataclass
class DFGNode:
    """Node in the dataflow graph"""
    id: str
    type: NodeType
    attrs: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    kernel_role: Optional[str] = None  # reader, compute, writer


@dataclass
class DFGEdge:
    """Edge in the dataflow graph"""
    src: str
    dst: str
    type: EdgeType
    attrs: Dict[str, Any] = field(default_factory=dict)


class TileDFGBuilder:
    """Visitor to build tile dataflow graph from TIR"""

    def __init__(self):
        if tir is not None:
            # Use the correct TVM visitor class
            from tvm.tir.stmt_functor import post_order_visit
            self.post_order_visit = post_order_visit
        self.nodes: Dict[str, DFGNode] = {}
        self.edges: List[DFGEdge] = []
        self.cb_allocations: Dict[str, Dict[str, Any]] = {}
        self.current_compute_id = 0
        self.buffer_accesses: Dict[str, Set[str]] = {}  # buffer -> {read, write}

    def visit(self, stmt):
        """Custom visitor to traverse TIR statements"""

        def visit_node(op):
            if isinstance(op, tir.Evaluate):
                self.visit_evaluate(op)

        if tir is not None:
            from tvm.tir.stmt_functor import post_order_visit
            post_order_visit(stmt, visit_node)

    def visit_evaluate(self, op):
        """Visit T.evaluate nodes to find dataflow operations"""

        if hasattr(op, 'value') and isinstance(op.value, tir.Call):
            call = op.value

            # Check if this is a call_extern
            if hasattr(call, 'op') and 'call_extern' in str(call.op):
                # The C1/C2 passes use a different convention:
                # For tt operations, args[0] is the function name!
                if len(call.args) >= 1 and isinstance(call.args[0], tir.StringImm):
                    func_name = call.args[0].value

                    # Now check the function name for our operations
                    if "tt.alloc_cb" in func_name:
                        self._handle_cb_allocation(call)
                    elif "tt.read_to_cb" in func_name:
                        self._handle_read_to_cb(call)
                    elif "tt.write_from_cb" in func_name:
                        self._handle_write_from_cb(call)
                    elif any(x in func_name for x in ["tt.mm.mma", "tt.fpu.", "tt.sfpu."]):
                        self._handle_compute_op(call, func_name)
            else:
                # Handle other types of calls (non-call_extern)
                op_name = str(call.op) if hasattr(call.op, 'name') else str(call.op)

                if "alloc_cb" in op_name:
                    self._handle_cb_allocation(call)
                elif "read_to_cb" in op_name:
                    self._handle_read_to_cb(call)
                elif "write_from_cb" in op_name:
                    self._handle_write_from_cb(call)
                elif any(x in op_name for x in ["mm.mma", "fpu.", "sfpu."]):
                    self._handle_compute_op(call, op_name)

    def _handle_cb_allocation(self, call):
        """Handle tt.alloc_cb intrinsic"""

        # For call_extern, skip the first two args (return type and function name)
        args = self._get_actual_args(call)

        if len(args) >= 3:
            cb_name = self._extract_string_arg(args[0])

            # Shape can be passed as individual int args or as a single arg
            # Check if args[1] and args[2] are both integers (width, height)
            if len(args) >= 4 and isinstance(args[1], tir.IntImm) and isinstance(
                    args[2], tir.IntImm):
                # Shape is passed as two separate args
                shape = [self._extract_int(args[1]), self._extract_int(args[2])]
                dtype = self._extract_string_arg(args[3])
            else:
                # Shape is a single arg (list/tuple)
                shape = self._extract_shape(args[1])
                dtype = self._extract_string_arg(args[2])

            if cb_name:
                # Create CB node
                node = DFGNode(
                    id=cb_name,
                    type=NodeType.CB,
                    attrs={
                        "shape": shape,
                        "dtype": dtype,
                        "index": self._assign_cb_index(cb_name)
                    })
                self.nodes[cb_name] = node
                self.cb_allocations[cb_name] = node.attrs

                logger.debug(f"Added CB node: {cb_name} with shape {shape}")

    def _handle_read_to_cb(self, call):
        """Handle tt.read_to_cb intrinsic (buffer -> CB)"""

        args = self._get_actual_args(call)

        if len(args) >= 2:
            buffer_slice = args[0]
            cb_name = self._extract_string_or_cb_arg(args[1])

            if cb_name:
                buffer_name = self._extract_buffer_name(buffer_slice)

                if buffer_name:
                    # Create buffer node if not exists
                    if buffer_name not in self.nodes:
                        self.nodes[buffer_name] = DFGNode(
                            id=buffer_name,
                            type=NodeType.BUFFER,
                            attrs={"memory": "DRAM"},  # Default, updated from metadata
                            kernel_role="reader")

                    # Create read node
                    read_id = f"read_{buffer_name}_to_{cb_name}"
                    read_node = DFGNode(
                        id=read_id,
                        type=NodeType.READ,
                        inputs=[buffer_name],
                        outputs=[cb_name],
                        kernel_role="reader")
                    self.nodes[read_id] = read_node

                    # Create edges
                    self.edges.append(
                        DFGEdge(src=buffer_name, dst=read_id, type=EdgeType.DATA_READ))
                    self.edges.append(DFGEdge(src=read_id, dst=cb_name, type=EdgeType.DATA_READ))

                    # Mark CB as having reader role
                    if cb_name in self.nodes:
                        self.nodes[cb_name].kernel_role = "reader/compute"

                    logger.debug(f"Added read: {buffer_name} -> {cb_name}")

    def _handle_write_from_cb(self, call):
        """Handle tt.write_from_cb intrinsic (CB -> buffer)"""

        args = self._get_actual_args(call)

        if len(args) >= 2:
            cb_name = self._extract_string_or_cb_arg(args[0])
            buffer_slice = args[1]

            if cb_name:
                buffer_name = self._extract_buffer_name(buffer_slice)

                if buffer_name:
                    # Create buffer node if not exists
                    if buffer_name not in self.nodes:
                        self.nodes[buffer_name] = DFGNode(
                            id=buffer_name,
                            type=NodeType.BUFFER,
                            attrs={"memory": "DRAM"},
                            kernel_role="writer")

                    # Create write node
                    write_id = f"write_{cb_name}_to_{buffer_name}"
                    write_node = DFGNode(
                        id=write_id,
                        type=NodeType.WRITE,
                        inputs=[cb_name],
                        outputs=[buffer_name],
                        kernel_role="writer")
                    self.nodes[write_id] = write_node

                    # Create edges
                    self.edges.append(DFGEdge(src=cb_name, dst=write_id, type=EdgeType.DATA_WRITE))
                    self.edges.append(
                        DFGEdge(src=write_id, dst=buffer_name, type=EdgeType.DATA_WRITE))

                    # Mark CB as having writer role
                    if cb_name in self.nodes:
                        if self.nodes[cb_name].kernel_role == "reader/compute":
                            self.nodes[cb_name].kernel_role = "compute/writer"
                        else:
                            self.nodes[cb_name].kernel_role = "writer"

                    logger.debug(f"Added write: {cb_name} -> {buffer_name}")

    def _handle_compute_op(self, call, op_name):
        """Handle compute operations (mm.mma, fpu.add, etc.)"""

        compute_type = self._classify_compute_op(op_name)
        compute_id = f"compute_{compute_type}_{self.current_compute_id}"
        self.current_compute_id += 1

        # Extract input/output CBs
        input_cbs = []
        output_cb = None

        # Get actual args (skip return type and function name for call_extern)
        args = self._get_actual_args(call)

        # Most compute ops have CB inputs as first args
        for i, arg in enumerate(args):
            cb_name = self._extract_string_or_cb_arg(arg)
            if cb_name:
                if i < 2:  # First two args are usually inputs
                    input_cbs.append(cb_name)
                elif "cb_out" in str(cb_name):  # Output CB
                    output_cb = cb_name

        # For ops that write to DST, infer output CB from metadata
        if not output_cb and "dst" in op_name.lower():
            output_cb = "cb_out"  # Default output CB name

        # Create compute node
        compute_node = DFGNode(
            id=compute_id,
            type=NodeType.COMPUTE,
            attrs={
                "op_type": compute_type,
                "op_name": op_name
            },
            inputs=input_cbs,
            outputs=[output_cb] if output_cb else [],
            kernel_role="compute")
        self.nodes[compute_id] = compute_node

        # Create edges from input CBs to compute
        for cb in input_cbs:
            self.edges.append(
                DFGEdge(
                    src=cb,
                    dst=compute_id,
                    type=EdgeType.COMPUTE_INPUT,
                    attrs={"compute_op": compute_type}))

        # Create edge from compute to output CB
        if output_cb:
            self.edges.append(
                DFGEdge(
                    src=compute_id,
                    dst=output_cb,
                    type=EdgeType.COMPUTE_OUTPUT,
                    attrs={"compute_op": compute_type}))

        logger.debug(f"Added compute: {compute_id} ({compute_type}) with inputs {input_cbs}")

    def _classify_compute_op(self, op_name: str) -> str:
        """Classify compute operation type"""

        if "mm" in op_name or "mma" in op_name or "matmul" in op_name:
            return "matmul"
        elif "add" in op_name:
            return "add"
        elif "mul" in op_name:
            return "mul"
        elif "sfpu" in op_name:
            if "unary" in op_name:
                return "sfpu_unary"
            else:
                return "sfpu"
        elif "fpu" in op_name:
            return "fpu_binary"
        else:
            return "unknown"

    def _get_actual_args(self, call):
        """Get actual arguments from a call, handling call_extern"""
        if hasattr(call, 'op') and 'call_extern' in str(call.op):
            # For tt operations from C1/C2: args[0] is the function name
            # So actual args start at args[1]
            return call.args[1:] if len(call.args) > 1 else []
        else:
            return call.args

    def _assign_cb_index(self, cb_name: str) -> int:
        """Assign CB index (0-31)"""

        # Simple heuristic assignment
        if "in0" in cb_name:
            return 0
        elif "in1" in cb_name:
            return 1
        elif "in2" in cb_name:
            return 2
        elif "out" in cb_name:
            return 16
        else:
            # Assign sequentially from 3
            return len(self.cb_allocations) + 2

    def _extract_string_arg(self, arg) -> Optional[str]:
        """Extract string from TIR argument"""

        if isinstance(arg, str):
            return arg
        elif hasattr(arg, 'value'):
            if isinstance(arg.value, str):
                return arg.value
            else:
                return str(arg.value)
        else:
            return str(arg) if arg else None

    def _extract_string_or_cb_arg(self, arg) -> Optional[str]:
        """Extract CB name from argument (could be string or CB reference)"""

        result = self._extract_string_arg(arg)
        if result and ("cb" in result.lower() or result in self.cb_allocations):
            return result
        return None

    def _extract_shape(self, arg) -> List[int]:
        """Extract shape from TIR argument"""

        if isinstance(arg, (list, tuple)):
            return list(arg)
        elif hasattr(arg, 'args'):
            # Handle tir.Call representing shape
            return [self._extract_int(x) for x in arg.args]
        elif isinstance(arg, tir.IntImm):
            # Single dimension
            return [int(arg.value)]
        else:
            # For call_extern, shape might be passed as separate int args
            # Will be handled by the caller
            return []

    def _extract_int(self, arg) -> int:
        """Extract integer value from TIR argument"""

        if isinstance(arg, int):
            return arg
        elif hasattr(arg, 'value'):
            return int(arg.value)
        else:
            return 0

    def _extract_buffer_name(self, buffer_slice) -> Optional[str]:
        """Extract buffer name from buffer slice expression"""

        if hasattr(buffer_slice, 'buffer'):
            # Direct buffer access
            if hasattr(buffer_slice.buffer, 'name'):
                return buffer_slice.buffer.name
            elif hasattr(buffer_slice.buffer, 'data') and hasattr(buffer_slice.buffer.data, 'name'):
                return buffer_slice.buffer.data.name
        elif hasattr(buffer_slice, 'name'):
            return buffer_slice.name

        # Try to extract from string representation
        str_repr = str(buffer_slice)
        if '[' in str_repr:
            # Extract name before brackets
            return str_repr.split('[')[0].strip()

        return None


class BuildTileDFGTT:
    """
    Pass to build tile-based dataflow graph from protocol-less IR.

    This pass:
    1. Analyzes CB allocations and data movements
    2. Tracks compute operations and their CB dependencies
    3. Builds a dataflow graph showing producer-consumer relationships
    4. Assigns preliminary kernel roles to operations
    5. Prepares metadata for D1: SplitDeviceKernel
    """

    def __init__(self) -> None:
        """Initialize the pass."""
        pass

    def __call__(self, mod: IRModule) -> IRModule:
        """Apply the pass to an IRModule."""
        if tvm is None:
            return mod

        new_funcs = {}
        for gvar, func in mod.functions_items():
            if not isinstance(func, tir.PrimFunc):
                new_funcs[gvar] = func
                continue

            # Process this function
            func = self._process_function(func)
            new_funcs[gvar] = func

        return tvm.IRModule(new_funcs)

    def _process_function(self, func: "tir.PrimFunc") -> "tir.PrimFunc":
        """Process a single function to build dataflow graph."""

        # Build the dataflow graph
        builder = TileDFGBuilder()
        builder.visit(func.body)

        # Analyze the graph for splitting hints
        dfg_metadata = self._analyze_dfg(builder)

        # Attach the dataflow graph as metadata
        from tilelang.tenstorrent.attrs import TT_TILE_DFG, TT_CB_ASSIGNMENT
        func = func.with_attr(TT_TILE_DFG, tvm.runtime.convert(dfg_metadata))

        # Also attach CB assignment hints
        cb_assignment = self._generate_cb_assignment(builder)
        func = func.with_attr(TT_CB_ASSIGNMENT, tvm.runtime.convert(cb_assignment))

        logger.info(
            f"Built dataflow graph with {len(builder.nodes)} nodes and {len(builder.edges)} edges")

        return func

    def _analyze_dfg(self, builder: TileDFGBuilder) -> Dict[str, Any]:
        """Analyze the dataflow graph and prepare metadata."""

        # Serialize nodes
        nodes_dict = {}
        for node_id, node in builder.nodes.items():
            nodes_dict[node_id] = {
                "type": node.type.value,
                "attrs": node.attrs,
                "inputs": node.inputs,
                "outputs": node.outputs,
                "kernel_role": node.kernel_role or "unknown"
            }

        # Serialize edges
        edges_list = []
        for edge in builder.edges:
            edges_list.append({
                "src": edge.src,
                "dst": edge.dst,
                "type": edge.type.value,
                "attrs": edge.attrs
            })

        # Identify kernel boundaries
        kernel_roles = self._identify_kernel_roles(builder)

        # Identify CB reuse patterns
        cb_reuse = self._analyze_cb_reuse(builder)

        # Create the final DFG metadata
        dfg_metadata = {
            "nodes": nodes_dict,
            "edges": edges_list,
            "kernel_roles": kernel_roles,
            "cb_allocations": builder.cb_allocations,
            "cb_reuse": cb_reuse,
            "stats": {
                "num_nodes":
                    len(builder.nodes),
                "num_edges":
                    len(builder.edges),
                "num_cbs":
                    len(builder.cb_allocations),
                "num_compute_ops":
                    sum(1 for n in builder.nodes.values() if n.type == NodeType.COMPUTE)
            }
        }

        return dfg_metadata

    def _identify_kernel_roles(self, builder: TileDFGBuilder) -> Dict[str, List[str]]:
        """Identify which nodes belong to which kernel role."""

        roles = {"reader": [], "compute": [], "writer": []}

        for node_id, node in builder.nodes.items():
            if node.kernel_role == "reader":
                roles["reader"].append(node_id)
            elif node.kernel_role == "compute":
                roles["compute"].append(node_id)
            elif node.kernel_role == "writer":
                roles["writer"].append(node_id)
            elif node.kernel_role == "reader/compute":
                # CB used by both reader and compute
                roles["reader"].append(node_id)
                roles["compute"].append(node_id)
            elif node.kernel_role == "compute/writer":
                # CB used by both compute and writer
                roles["compute"].append(node_id)
                roles["writer"].append(node_id)

        return roles

    def _analyze_cb_reuse(self, builder: TileDFGBuilder) -> Dict[str, Any]:
        """Analyze CB reuse patterns for optimization."""

        cb_reuse = {}

        for cb_name, _cb_info in builder.cb_allocations.items():
            # Count how many times each CB is read/written
            read_count = sum(
                1 for e in builder.edges if e.dst == cb_name and e.type == EdgeType.DATA_READ)
            write_count = sum(
                1 for e in builder.edges if e.src == cb_name and e.type == EdgeType.DATA_WRITE)
            compute_in = sum(
                1 for e in builder.edges if e.src == cb_name and e.type == EdgeType.COMPUTE_INPUT)
            compute_out = sum(
                1 for e in builder.edges if e.dst == cb_name and e.type == EdgeType.COMPUTE_OUTPUT)

            cb_reuse[cb_name] = {
                "read_count":
                    read_count,
                "write_count":
                    write_count,
                "compute_input_count":
                    compute_in,
                "compute_output_count":
                    compute_out,
                "total_uses":
                    read_count + write_count + compute_in + compute_out,
                "is_intermediate":
                    read_count == 0 and write_count == 0 and compute_in + compute_out > 0
            }

        return cb_reuse

    def _generate_cb_assignment(self, builder: TileDFGBuilder) -> Dict[str, int]:
        """
        Generate CB index assignment (0-31) following TT SDK conventions:
        - Input CBs (cb_in*): indices 0-15
        - Output CBs (cb_out*): indices 16-31
        """

        assignment = {}

        # Separate CBs into input and output categories
        input_cbs = []
        output_cbs = []
        other_cbs = []

        for cb_name, cb_info in builder.cb_allocations.items():
            # Check if CB has explicit index hint
            if "index" in cb_info:
                assignment[cb_name] = cb_info["index"]
            # Categorize by name pattern
            elif "in" in cb_name.lower():
                input_cbs.append(cb_name)
            elif "out" in cb_name.lower():
                output_cbs.append(cb_name)
            else:
                # Check role from CB info if available
                role = cb_info.get("role", "").lower()
                if role in ["input", "read"]:
                    input_cbs.append(cb_name)
                elif role in ["output", "write"]:
                    output_cbs.append(cb_name)
                else:
                    other_cbs.append(cb_name)

        # Assign input CBs to indices 0-15
        next_input_index = 0
        for cb_name in input_cbs:
            if cb_name not in assignment:
                # Find next available input index
                while next_input_index < 16 and next_input_index in assignment.values():
                    next_input_index += 1
                if next_input_index < 16:
                    assignment[cb_name] = next_input_index
                    next_input_index += 1
                else:
                    logger.warning(
                        f"Input CB {cb_name} could not be assigned (input indices exhausted)")

        # Assign output CBs to indices 16-31
        next_output_index = 16
        for cb_name in output_cbs:
            if cb_name not in assignment:
                # Find next available output index
                while next_output_index < 32 and next_output_index in assignment.values():
                    next_output_index += 1
                if next_output_index < 32:
                    assignment[cb_name] = next_output_index
                    next_output_index += 1
                else:
                    logger.warning(
                        f"Output CB {cb_name} could not be assigned (output indices exhausted)")

        # Assign other CBs to any remaining indices (prefer input range)
        next_index = 0
        for cb_name in other_cbs:
            if cb_name not in assignment:
                # Find next available index (any range)
                while next_index < 32 and next_index in assignment.values():
                    next_index += 1
                if next_index < 32:
                    assignment[cb_name] = next_index
                    logger.info(f"CB {cb_name} assigned to index {next_index} (unclassified)")
                    next_index += 1
                else:
                    logger.error(f"CB {cb_name} could not be assigned (all indices exhausted)")

        return assignment


# Module-level pass function for compatibility
def build_tile_dfg_tt(mod: IRModule) -> IRModule:
    """Apply BuildTileDFGTT pass to a module."""
    pass_instance = BuildTileDFGTT()
    return pass_instance(mod)


# Example usage and testing
if __name__ == "__main__":
    import tvm.script
    from tvm.script import tir as T

    # Create test module with protocol-less IR
    @tvm.script.ir_module
    class TestModule:

        @T.prim_func
        def gemm_protocol_less(A: T.Buffer((256, 256), "float16"), B: T.Buffer(
            (256, 256), "float16"), C: T.Buffer((256, 256), "float16")):
            # Simulate output from C1/C2
            T.evaluate(T.call_extern("tt.alloc_cb", "cb_in0", [128, 32], "bf16"))
            T.evaluate(T.call_extern("tt.alloc_cb", "cb_in1", [32, 128], "bf16"))
            T.evaluate(T.call_extern("tt.alloc_cb", "cb_out", [128, 128], "bf16"))

            # Abstract data movements
            T.evaluate(T.call_extern("tt.read_to_cb", A[0:128, 0:32], "cb_in0"))
            T.evaluate(T.call_extern("tt.read_to_cb", B[0:32, 0:128], "cb_in1"))

            # Protocol-less compute
            T.evaluate(T.call_extern("tt.mm.mma", "cb_in0", "cb_in1", 0, True))

            # Abstract write back
            T.evaluate(T.call_extern("tt.write_from_cb", "cb_out", C[0:128, 0:128]))

    # Apply C3 pass
    pass_c3 = BuildTileDFGTT()
    result = pass_c3(TestModule)

    # Check results
    func = result["gemm_protocol_less"]
    if "tt.tile_dfg" in func.attrs:
        dfg = func.attrs["tt.tile_dfg"]
        print("=== Tile Dataflow Graph ===")
        print(f"Nodes: {dfg['stats']['num_nodes']}")
        print(f"Edges: {dfg['stats']['num_edges']}")
        print(f"CBs: {dfg['stats']['num_cbs']}")
        print(f"Compute ops: {dfg['stats']['num_compute_ops']}")

        print("\n=== Kernel Roles ===")
        for role, nodes in dfg["kernel_roles"].items():
            print(f"{role}: {nodes}")

        print("\n=== CB Assignment ===")
        if "tt.cb_assignment" in func.attrs:
            cb_assignment = func.attrs["tt.cb_assignment"]
            for cb_name, index in cb_assignment.items():
                print(f"  {cb_name}: CB{index}")
