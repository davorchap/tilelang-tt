"""
Pass: ValidateLoweredIR (Pre-Codegen Validation)
Version: 1.0
Date: 2025-10-18

Purpose: Validate that IR is fully lowered before codegen.
         Catches unlowered constructs that would cause C++ compilation errors.

         This pass is designed to be inserted after Stage D (Protocol Insertion)
         and before codegen to prevent cryptic C++ errors.

Input: IR after D5 (insert_dst_management_tt)
Output: Validated IR or raises ValueError with clear error messages
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Set
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


class ValidationLevel(Enum):
    """Validation severity levels"""

    ERROR = "error"  # Must fix - codegen will fail
    WARNING = "warning"  # Should fix - may cause issues
    INFO = "info"  # Informational


@dataclass
class ValidationIssue:
    """A single validation issue"""

    level: ValidationLevel
    category: str
    message: str
    location: Optional[str] = None
    ir_snippet: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationReport:
    """Validation report"""

    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def add_issue(
        self,
        level: ValidationLevel,
        category: str,
        message: str,
        location: Optional[str] = None,
        ir_snippet: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        """Add a validation issue"""
        self.issues.append(ValidationIssue(level, category, message, location, ir_snippet, suggestion))
        if level == ValidationLevel.ERROR:
            self.passed = False

    def get_summary(self) -> str:
        """Get summary string"""
        error_count = sum(1 for i in self.issues if i.level == ValidationLevel.ERROR)
        warning_count = sum(1 for i in self.issues if i.level == ValidationLevel.WARNING)
        info_count = sum(1 for i in self.issues if i.level == ValidationLevel.INFO)
        return f"Errors: {error_count}, Warnings: {warning_count}, Info: {info_count}"


class UnloweredConstructDetector:
    """Detector for unlowered IR constructs that should not reach codegen"""

    def __init__(self):
        self.unlowered_ops: List[Dict[str, Any]] = []
        self.current_function_name: Optional[str] = None
        self.current_kernel_role: Optional[str] = None

    def visit_stmt(self, stmt):
        """Visit a statement node"""
        if isinstance(stmt, tir.Evaluate):
            self._check_evaluate(stmt)

    def _check_evaluate(self, stmt):
        """Check an Evaluate statement for unlowered operations"""
        if not hasattr(stmt, "value"):
            return

        value = stmt.value
        if not hasattr(value, "op"):
            return

        # Extract operation name
        op_name = self._get_op_name(value)

        # Check for unlowered protocol-less operations
        if self._is_unlowered_operation(op_name, value):
            self.unlowered_ops.append(
                {
                    "op_name": op_name,
                    "function": self.current_function_name,
                    "kernel_role": self.current_kernel_role,
                    "stmt": stmt,
                }
            )

    def _get_op_name(self, call) -> str:
        """Extract operation name from a call node"""
        # Handle call_extern
        if str(call.op) == "Op(tir.call_extern)":
            # Function name is in the args
            for arg in call.args:
                if isinstance(arg, tir.StringImm):
                    func_name = arg.value
                    # Skip type strings
                    if func_name not in ["void", "int32", "uint32", "float32"]:
                        return func_name
            return str(call.op)
        elif hasattr(call.op, "name"):
            return call.op.name
        else:
            return str(call.op)

    def _is_unlowered_operation(self, op_name: str, call) -> bool:
        """
        Check if this is an unlowered operation that should have been transformed.

        Unlowered operations include:
        - tt.copy.protocol_less (should be lowered to cb_reserve_back + noc_async_read + etc.)
        - tt.read_to_cb (should be lowered to cb_reserve_back + noc_async_read + etc.)
        - tt.write_from_cb (should be lowered to cb_wait_front + noc_async_write + etc.)
        - Region expressions in device kernels (should be resolved to buffer accesses)

        Valid operations at this stage:
        - cb_reserve_back, cb_push_back, cb_wait_front, cb_pop_front
        - noc_async_read_tile, noc_async_write_tile, noc_async_read_barrier
        - get_write_ptr, get_read_ptr
        - mm_init, matmul_tiles, pack_tile, tile_regs_commit
        - acquire_dst, release_dst
        - Any standard TIR operations (Load, Store, For, etc.)
        """
        # Protocol-less operations that should have been lowered
        unlowered_patterns = [
            "tt.copy.protocol_less",
            "tt.read_to_cb",
            "tt.write_from_cb",
        ]

        for pattern in unlowered_patterns:
            if pattern in op_name:
                return True

        return False


class BufferRegionDetector:
    """Detector for BufferRegion nodes that should not reach codegen"""

    def __init__(self):
        self.buffer_regions: List[Dict[str, Any]] = []
        self.current_function_name: Optional[str] = None
        self.current_kernel_role: Optional[str] = None

    def visit_expr(self, expr):
        """Visit an expression node to check for BufferRegion"""
        # Check if this is a BufferRegion or BufferLoad with region
        if tir and isinstance(expr, (tir.BufferRegion,)):
            self.buffer_regions.append(
                {
                    "type": type(expr).__name__,
                    "function": self.current_function_name,
                    "kernel_role": self.current_kernel_role,
                    "expr": expr,
                }
            )

        # Recursively check sub-expressions
        if hasattr(expr, "a"):
            self.visit_expr(expr.a)
        if hasattr(expr, "b"):
            self.visit_expr(expr.b)
        if hasattr(expr, "args"):
            for arg in expr.args:
                if not isinstance(arg, (int, float, str)):
                    self.visit_expr(arg)


class TypeAnnotationDetector:
    """Detector for type annotations that should have been removed"""

    def __init__(self):
        self.type_annotations: List[Dict[str, Any]] = []
        self.current_function_name: Optional[str] = None

    def visit_expr(self, expr):
        """Visit an expression to check for type annotations"""
        # Check for Cast nodes that might indicate type annotations
        if tir and isinstance(expr, tir.Cast):
            # Some casts are valid, but excessive casting might indicate problems
            pass  # For now, we don't flag casts as errors

        # Recursively check sub-expressions
        if hasattr(expr, "value"):
            self.visit_expr(expr.value)


class IRTraverser:
    """Traverser to walk the IR tree and apply detectors"""

    def __init__(self, detectors: List):
        self.detectors = detectors

    def traverse_function(self, name: str, func: "tir.PrimFunc"):
        """Traverse a function and apply all detectors"""
        kernel_role = func.attrs.get("tt.kernel_role") if func.attrs else None

        # Set context for all detectors
        for detector in self.detectors:
            if hasattr(detector, "current_function_name"):
                detector.current_function_name = name
            if hasattr(detector, "current_kernel_role"):
                detector.current_kernel_role = kernel_role

        # Traverse the function body
        if func.body:
            self._traverse_stmt(func.body)

    def _traverse_stmt(self, stmt):
        """Recursively traverse a statement"""
        # Apply statement visitors
        for detector in self.detectors:
            if hasattr(detector, "visit_stmt"):
                detector.visit_stmt(stmt)

        # Check expressions in the statement
        if hasattr(stmt, "value"):
            self._traverse_expr(stmt.value)

        # Recursively traverse child statements
        if isinstance(stmt, tir.SeqStmt):
            for s in stmt.seq:
                self._traverse_stmt(s)
        elif isinstance(stmt, tir.For):
            self._traverse_stmt(stmt.body)
        elif isinstance(stmt, tir.IfThenElse):
            self._traverse_stmt(stmt.then_case)
            if stmt.else_case:
                self._traverse_stmt(stmt.else_case)
        elif isinstance(stmt, tir.LetStmt):
            self._traverse_expr(stmt.value)
            self._traverse_stmt(stmt.body)
        elif isinstance(stmt, tir.AttrStmt):
            self._traverse_stmt(stmt.body)
        elif isinstance(stmt, tir.BlockRealize):
            if hasattr(stmt, "block") and hasattr(stmt.block, "body"):
                self._traverse_stmt(stmt.block.body)

    def _traverse_expr(self, expr):
        """Recursively traverse an expression"""
        if expr is None:
            return

        # Apply expression visitors
        for detector in self.detectors:
            if hasattr(detector, "visit_expr"):
                detector.visit_expr(expr)

        # Recursively traverse based on expression type
        if hasattr(expr, "args"):
            for arg in expr.args:
                self._traverse_expr(arg)


class LoweredIRValidator:
    """Main validator that orchestrates detection and reporting"""

    def __init__(self):
        self.construct_detector = UnloweredConstructDetector()
        self.region_detector = BufferRegionDetector()
        self.type_detector = TypeAnnotationDetector()

    def validate(self, mod: IRModule, report: ValidationReport):
        """Validate that all IR is fully lowered"""

        # Create traverser with all detectors
        traverser = IRTraverser([self.construct_detector, self.region_detector, self.type_detector])

        # Traverse all functions
        for name, func in mod.functions_items():
            if isinstance(func, tir.PrimFunc):
                traverser.traverse_function(str(name), func)

        # Report unlowered constructs
        for unlowered in self.construct_detector.unlowered_ops:
            report.add_issue(
                ValidationLevel.ERROR,
                "Unlowered Operation",
                f"Found unlowered operation '{unlowered['op_name']}' in {unlowered['kernel_role']} kernel",
                location=unlowered["function"],
                ir_snippet=str(unlowered["stmt"])[:200],
                suggestion="This operation should have been lowered by Pass D3 (lower_cb_intrinsics). "
                "Check that the pass is correctly transforming protocol-less operations.",
            )

        # Report buffer regions
        for region in self.region_detector.buffer_regions:
            report.add_issue(
                ValidationLevel.ERROR,
                "Buffer Region",
                f"Found BufferRegion in {region['kernel_role']} kernel - should be resolved to buffer accesses",
                location=region["function"],
                ir_snippet=str(region["expr"])[:200],
                suggestion="BufferRegion nodes should be resolved during lowering. "
                "Check that all region-based operations are properly lowered.",
            )

        # Statistics
        report.stats["unlowered_op_count"] = len(self.construct_detector.unlowered_ops)
        report.stats["buffer_region_count"] = len(self.region_detector.buffer_regions)


class ValidateLoweredIR:
    """
    Pass to validate that IR is fully lowered before codegen.

    This pass:
    1. Detects unlowered protocol-less operations (tt.copy.protocol_less, etc.)
    2. Checks for BufferRegion nodes that should be resolved
    3. Validates that all operations are in lowered form

    This prevents cryptic C++ compilation errors by catching issues
    early in the pipeline where they're easier to debug.
    """

    def __init__(self, strict: bool = True) -> None:
        """
        Initialize validator.

        Args:
            strict: If True, warnings are treated as errors
        """
        self.strict = strict
        self.validator = LoweredIRValidator()

    def __call__(self, mod: IRModule) -> IRModule:
        """Apply validation to an IRModule."""
        if tvm is None:
            return mod

        report = ValidationReport(passed=True)

        # Validate lowering
        self.validator.validate(mod, report)

        # Apply strict mode
        if self.strict:
            for issue in report.issues:
                if issue.level == ValidationLevel.WARNING:
                    issue.level = ValidationLevel.ERROR
                    report.passed = False

        # Log results
        self._log_report(report)

        # Fail if validation failed
        if not report.passed:
            error_messages = []
            for issue in report.issues:
                if issue.level == ValidationLevel.ERROR:
                    msg = f"[{issue.category}] {issue.message}"
                    if issue.location:
                        msg += f"\n  Location: {issue.location}"
                    if issue.suggestion:
                        msg += f"\n  Suggestion: {issue.suggestion}"
                    if issue.ir_snippet:
                        msg += f"\n  IR Snippet: {issue.ir_snippet}"
                    error_messages.append(msg)

            raise ValueError(
                "IR validation failed - unlowered constructs detected:\n\n"
                + "\n\n".join(error_messages)
                + "\n\nThese constructs should have been lowered by Stage D passes "
                "but were found before codegen. This would cause C++ compilation errors."
            )

        return mod

    def _log_report(self, report: ValidationReport):
        """Log the validation report"""

        if report.passed:
            logger.info("✅ Lowered IR validation passed")
        else:
            logger.error("❌ Lowered IR validation failed")

        logger.info(f"Summary: {report.get_summary()}")

        # Log errors
        for issue in report.issues:
            if issue.level == ValidationLevel.ERROR:
                msg = f"ERROR [{issue.category}]: {issue.message}"
                if issue.location:
                    msg += f" at {issue.location}"
                if issue.suggestion:
                    msg += f"\n  Suggestion: {issue.suggestion}"
                logger.error(msg)

        # Log warnings
        for issue in report.issues:
            if issue.level == ValidationLevel.WARNING:
                msg = f"WARNING [{issue.category}]: {issue.message}"
                if issue.location:
                    msg += f" at {issue.location}"
                if issue.suggestion:
                    msg += f"\n  Suggestion: {issue.suggestion}"
                logger.warning(msg)

        # Log stats
        if report.stats:
            logger.info("Statistics:")
            for key, value in report.stats.items():
                logger.info(f"  {key}: {value}")


# Module-level pass function for compatibility
def validate_lowered_ir(mod: IRModule, strict: bool = True) -> IRModule:
    """Apply ValidateLoweredIR pass to a module."""
    pass_instance = ValidateLoweredIR(strict=strict)
    return pass_instance(mod)


# Example usage and testing
if __name__ == "__main__":
    import tvm.script
    from tvm.script import tir as T

    # Create test module with unlowered operations
    @tvm.script.ir_module
    class TestModuleUnlowered:

        @T.prim_func
        def bad_reader(A: T.Buffer((256, 256), "float16")):
            # This should have been lowered to cb_reserve_back + noc_async_read + etc.
            T.evaluate(T.call_extern("void", "tt.copy.protocol_less", A, "cb_in0"))

    # Add kernel role
    bad_reader = TestModuleUnlowered["bad_reader"]
    bad_reader = bad_reader.with_attr("tt.kernel_role", "reader")
    TestModuleUnlowered["bad_reader"] = bad_reader

    # Test unlowered module
    print("=== Testing Unlowered Module ===")
    try:
        validator = ValidateLoweredIR(strict=True)
        result = validator(TestModuleUnlowered)
        print("❌ Validation passed (unexpected!)\n")
    except ValueError as e:
        print(f"✅ Validation correctly failed:\n{e}\n")

    # Create test module with properly lowered operations
    @tvm.script.ir_module
    class TestModuleLowered:

        @T.prim_func
        def good_reader(A: T.Buffer((256, 256), "float16")):
            # Properly lowered NOC operations
            T.evaluate(T.call_extern("void", "cb_reserve_back", 0, 1))
            l1_addr = T.call_extern("uint32", "get_write_ptr", 0)
            T.evaluate(T.call_extern("void", "noc_async_read_tile", 0, 0, l1_addr))
            T.evaluate(T.call_extern("void", "noc_async_read_barrier"))
            T.evaluate(T.call_extern("void", "cb_push_back", 0, 1))

    # Add kernel role
    good_reader = TestModuleLowered["good_reader"]
    good_reader = good_reader.with_attr("tt.kernel_role", "reader")
    TestModuleLowered["good_reader"] = good_reader

    # Test lowered module
    print("=== Testing Lowered Module ===")
    try:
        validator = ValidateLoweredIR(strict=True)
        result = validator(TestModuleLowered)
        print("✅ Validation passed\n")
    except ValueError as e:
        print(f"❌ Validation failed (unexpected):\n{e}\n")
