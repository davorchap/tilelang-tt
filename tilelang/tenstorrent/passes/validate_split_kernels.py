"""
Pass: ValidateSplitKernels (v5 CodeGen Validation)
Version: 1.0
Date: 2025-10-17

Purpose: Validate that split kernels have complete IR before codegen.
         This pass identifies incomplete IR and fails loudly instead of
         allowing silent fallback to template-based code generation.

Input: IR after D1 (split_device_kernel)
Output: Validated IR or raises ValueError on incomplete IR
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
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
    ERROR = "error"  # Must fix - codegen will fail or use templates
    WARNING = "warning"  # Should fix - may cause issues
    INFO = "info"  # Informational


@dataclass
class ValidationIssue:
    """A single validation issue"""
    level: ValidationLevel
    kernel_role: str
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationReport:
    """Validation report for split kernels"""
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def add_issue(self,
                  level: ValidationLevel,
                  kernel_role: str,
                  message: str,
                  location: Optional[str] = None,
                  suggestion: Optional[str] = None):
        """Add a validation issue"""
        self.issues.append(ValidationIssue(level, kernel_role, message, location, suggestion))
        if level == ValidationLevel.ERROR:
            self.passed = False

    def get_summary(self) -> str:
        """Get summary string"""
        error_count = sum(1 for i in self.issues if i.level == ValidationLevel.ERROR)
        warning_count = sum(1 for i in self.issues if i.level == ValidationLevel.WARNING)
        info_count = sum(1 for i in self.issues if i.level == ValidationLevel.INFO)
        return f"Errors: {error_count}, Warnings: {warning_count}, Info: {info_count}"


class IRInspector:
    """Inspector to analyze IR structure"""

    def __init__(self):
        self.has_noc_read = False
        self.has_noc_write = False
        self.has_compute_op = False
        self.has_cb_ops = False
        self.statement_count = 0
        self.intrinsic_calls = []

    def visit_evaluate(self, op):
        """Visit evaluate nodes to check for intrinsic calls"""
        self.statement_count += 1

        if hasattr(op, 'value') and hasattr(op.value, 'op'):
            call = op.value

            # For call_extern, get the function name from arguments
            if str(call.op) == "Op(tir.call_extern)":
                if len(call.args) > 0 and hasattr(call.args[0], 'value'):
                    op_name = call.args[0].value
                else:
                    op_name = str(call.op)
            else:
                op_name = str(call.op) if hasattr(call.op, 'name') else str(call.op)

            self.intrinsic_calls.append(op_name)

            # Check for specific operation types
            # Accept both protocol-less (tt.*) and lowered (tir.*) forms
            if any(x in op_name for x in ["noc_async_read", "read_to_cb", "tt.read_to_cb"]):
                self.has_noc_read = True
                self.has_cb_ops = True

            if any(x in op_name for x in ["noc_async_write", "write_from_cb", "tt.write_from_cb"]):
                self.has_noc_write = True
                self.has_cb_ops = True

            if any(x in op_name
                   for x in ["cb_reserve_back", "cb_push_back", "cb_wait_front", "cb_pop_front"]):
                self.has_cb_ops = True

            if any(x in op_name for x in [
                    "mm_init", "matmul_tiles", "pack_tile", "tile_regs_commit",
                    "mm.mma", "tt.mm.mma",  # Accept protocol-less form
                    "fpu.", "tt.fpu.",      # Accept protocol-less form
                    "sfpu.", "tt.sfpu.",    # Accept protocol-less form
                    "gemm", "add", "mul"
            ]):
                self.has_compute_op = True


class SplitKernelValidator:
    """Validator for split kernel completeness"""

    def __init__(self):
        pass

    def validate(self, mod: IRModule, report: ValidationReport):
        """Validate all split kernels in the module"""

        # Group functions by kernel role
        kernels_by_role = {"reader": [], "compute": [], "writer": []}

        for name, func in mod.functions_items():
            if not isinstance(func, tir.PrimFunc):
                continue

            kernel_role = func.attrs.get("tt.kernel_role") if func.attrs else None
            if kernel_role in kernels_by_role:
                kernels_by_role[kernel_role].append((str(name), func))

        # Validate kernel count
        report.stats["reader_count"] = len(kernels_by_role["reader"])
        report.stats["compute_count"] = len(kernels_by_role["compute"])
        report.stats["writer_count"] = len(kernels_by_role["writer"])

        # Check that we have at least one of each
        for role in ["reader", "compute", "writer"]:
            if not kernels_by_role[role]:
                report.add_issue(
                    ValidationLevel.ERROR,
                    role,
                    f"No {role} kernel found after split_device_kernel",
                    suggestion="Check if D1 (split_device_kernel) was applied correctly")

        # Validate each kernel
        for role, kernels in kernels_by_role.items():
            for name, func in kernels:
                self._validate_kernel(name, func, role, report)

    def _validate_kernel(self, name: str, func: "tir.PrimFunc", role: str,
                         report: ValidationReport):
        """Validate a single kernel"""

        # Check that body exists
        if not func.body:
            report.add_issue(
                ValidationLevel.ERROR,
                role,
                f"Kernel {name} has no body",
                location=name,
                suggestion=f"Check Stage D passes - {role} kernel body is missing")
            return

        # Check that body is not just Evaluate(0)
        if self._is_empty_body(func.body):
            report.add_issue(
                ValidationLevel.ERROR,
                role,
                f"Kernel {name} has empty body (only Evaluate(0))",
                location=name,
                suggestion=f"Check Stage D passes - {role} kernel has no actual operations")
            return

        # Inspect the IR structure
        inspector = IRInspector()
        if tir and hasattr(tir, 'stmt_functor'):
            tir.stmt_functor.post_order_visit(func.body, inspector.visit_evaluate)
        else:
            # Fallback: manually traverse the IR tree if stmt_functor not available
            self._manual_inspect(func.body, inspector)

        # Validate based on role
        if role == "reader":
            self._validate_reader(name, inspector, report)
        elif role == "compute":
            self._validate_compute(name, inspector, report)
        elif role == "writer":
            self._validate_writer(name, inspector, report)

        # Log intrinsics found (for debugging)
        if inspector.intrinsic_calls:
            logger.debug(
                f"{name} ({role}) intrinsics: {inspector.intrinsic_calls[:10]}")  # First 10

    def _validate_reader(self, name: str, inspector: IRInspector, report: ValidationReport):
        """Validate reader kernel has NOC read operations"""

        if not inspector.has_noc_read:
            report.add_issue(
                ValidationLevel.ERROR,
                "reader",
                f"Reader kernel {name} has no NOC read operations",
                location=name,
                suggestion="Check that lower_cb_intrinsics (D3) properly lowered read_to_cb to noc_async_read_tile"
            )

        if not inspector.has_cb_ops:
            report.add_issue(
                ValidationLevel.WARNING,
                "reader",
                f"Reader kernel {name} has no CB operations",
                location=name,
                suggestion="Reader should have cb_reserve_back/cb_push_back operations")

        if inspector.statement_count < 2:
            report.add_issue(
                ValidationLevel.WARNING,
                "reader",
                f"Reader kernel {name} has very few statements ({inspector.statement_count})",
                location=name,
                suggestion="Verify that all read operations were included in the split")

    def _validate_compute(self, name: str, inspector: IRInspector, report: ValidationReport):
        """Validate compute kernel has compute operations"""

        if not inspector.has_compute_op:
            report.add_issue(
                ValidationLevel.ERROR,
                "compute",
                f"Compute kernel {name} has no compute operations",
                location=name,
                suggestion="Check that lower_tt_tile_intrinsics_v5 (C2) properly generated compute ops"
            )

        if inspector.statement_count < 1:
            report.add_issue(
                ValidationLevel.ERROR,
                "compute",
                f"Compute kernel {name} has no statements",
                location=name,
                suggestion="Check Stage D split - compute kernel body is empty")

    def _validate_writer(self, name: str, inspector: IRInspector, report: ValidationReport):
        """Validate writer kernel has NOC write operations"""

        if not inspector.has_noc_write:
            report.add_issue(
                ValidationLevel.ERROR,
                "writer",
                f"Writer kernel {name} has no NOC write operations",
                location=name,
                suggestion="Check that lower_cb_intrinsics (D3) properly lowered write_from_cb to noc_async_write_tile"
            )

        if not inspector.has_cb_ops:
            report.add_issue(
                ValidationLevel.WARNING,
                "writer",
                f"Writer kernel {name} has no CB operations",
                location=name,
                suggestion="Writer should have cb_wait_front/cb_pop_front operations")

        if inspector.statement_count < 2:
            report.add_issue(
                ValidationLevel.WARNING,
                "writer",
                f"Writer kernel {name} has very few statements ({inspector.statement_count})",
                location=name,
                suggestion="Verify that all write operations were included in the split")

    def _manual_inspect(self, stmt, inspector: IRInspector):
        """Manually traverse IR tree when stmt_functor is not available"""
        # This is a simple traversal - may not cover all cases
        if hasattr(stmt, '__iter__'):
            for s in stmt:
                self._manual_inspect(s, inspector)
        elif hasattr(stmt, 'body'):
            self._manual_inspect(stmt.body, inspector)

        # Check if this is an Evaluate node
        if tir and isinstance(stmt, tir.Evaluate):
            inspector.visit_evaluate(stmt)

    def _is_empty_body(self, body) -> bool:
        """Check if body is effectively empty (only Evaluate(0))"""
        if isinstance(body, tir.Evaluate) and hasattr(body, 'value'):
            # Check if it's just a constant 0
            if isinstance(body.value, (tir.IntImm, tir.FloatImm)):
                return body.value.value == 0
            # Check if it's a string literal (also considered empty)
            if isinstance(body.value, tir.StringImm):
                return True
        return False


class ValidateSplitKernels:
    """
    Pass to validate split kernel completeness.

    This pass:
    1. Checks that reader/compute/writer kernels exist
    2. Validates that each kernel has a non-empty body
    3. Ensures reader has NOC read operations
    4. Ensures compute has compute operations
    5. Ensures writer has NOC write operations

    This prevents codegen from silently falling back to templates.
    """

    def __init__(self, strict: bool = True) -> None:
        """
        Initialize validator.

        Args:
            strict: If True, warnings are treated as errors
        """
        self.strict = strict
        self.validator = SplitKernelValidator()

    def __call__(self, mod: IRModule) -> IRModule:
        """Apply validation to an IRModule."""
        if tvm is None:
            return mod

        report = ValidationReport(passed=True)

        # Validate split kernels
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
            error_messages = [
                f"[{i.kernel_role}] {i.message}" for i in report.issues
                if i.level == ValidationLevel.ERROR
            ]
            raise ValueError("Split kernel validation failed - IR is incomplete:\n" +
                             "\n".join(error_messages))

        return mod

    def _log_report(self, report: ValidationReport):
        """Log the validation report"""

        if report.passed:
            logger.info(" Split kernel validation passed")
        else:
            logger.error("L Split kernel validation failed")

        logger.info(f"Summary: {report.get_summary()}")

        # Log errors
        for issue in report.issues:
            if issue.level == ValidationLevel.ERROR:
                msg = f"ERROR [{issue.kernel_role}]: {issue.message}"
                if issue.location:
                    msg += f" at {issue.location}"
                if issue.suggestion:
                    msg += f"\n  -> Suggestion: {issue.suggestion}"
                logger.error(msg)

        # Log warnings
        for issue in report.issues:
            if issue.level == ValidationLevel.WARNING:
                msg = f"WARNING [{issue.kernel_role}]: {issue.message}"
                if issue.location:
                    msg += f" at {issue.location}"
                if issue.suggestion:
                    msg += f"\n  -> Suggestion: {issue.suggestion}"
                logger.warning(msg)

        # Log stats
        if report.stats:
            logger.info("Statistics:")
            for key, value in report.stats.items():
                logger.info(f"  {key}: {value}")


# Module-level pass function for compatibility
def validate_split_kernels(mod: IRModule, strict: bool = True) -> IRModule:
    """Apply ValidateSplitKernels pass to a module."""
    pass_instance = ValidateSplitKernels(strict=strict)
    return pass_instance(mod)


# Example usage and testing
if __name__ == "__main__":
    import tvm.script
    from tvm.script import tir as T

    # Create test module with good kernels
    @tvm.script.ir_module
    class GoodModule:

        @T.prim_func
        def matmul_reader(A: T.Buffer((256, 256), "float16")):
            T.evaluate(T.call_extern("tir.noc_async_read_tile", A[0, 0], "cb_in0", 0))
            T.evaluate(T.call_extern("tir.cb_reserve_back", "cb_in0", 1))
            T.evaluate(T.call_extern("tir.cb_push_back", "cb_in0", 1))

        @T.prim_func
        def matmul_compute():
            T.evaluate(T.call_extern("tir.mm_init"))
            T.evaluate(T.call_extern("tir.matmul_tiles", "cb_in0", "cb_in1", 0, True))
            T.evaluate(T.call_extern("tir.pack_tile", 0, "cb_out"))

        @T.prim_func
        def matmul_writer(C: T.Buffer((256, 256), "float16")):
            T.evaluate(T.call_extern("tir.cb_wait_front", "cb_out", 1))
            T.evaluate(T.call_extern("tir.noc_async_write_tile", "cb_out", C[0, 0]))
            T.evaluate(T.call_extern("tir.cb_pop_front", "cb_out", 1))

    # Add kernel role metadata
    reader = GoodModule["matmul_reader"]
    reader = reader.with_attr("tt.kernel_role", "reader")
    GoodModule["matmul_reader"] = reader

    compute = GoodModule["matmul_compute"]
    compute = compute.with_attr("tt.kernel_role", "compute")
    GoodModule["matmul_compute"] = compute

    writer = GoodModule["matmul_writer"]
    writer = writer.with_attr("tt.kernel_role", "writer")
    GoodModule["matmul_writer"] = writer

    # Test good module
    print("=== Testing Good Module ===")
    try:
        validator = ValidateSplitKernels(strict=False)
        result = validator(GoodModule)
        print(" Validation passed\n")
    except ValueError as e:
        print(f"L Validation failed: {e}\n")

    # Create test module with bad kernels (empty bodies)
    @tvm.script.ir_module
    class BadModule:

        @T.prim_func
        def bad_reader(A: T.Buffer((256, 256), "float16")):
            T.evaluate(0)  # Empty body!

        @T.prim_func
        def bad_compute():
            T.evaluate(0)  # Empty body!

        @T.prim_func
        def bad_writer(C: T.Buffer((256, 256), "float16")):
            T.evaluate(0)  # Empty body!

    # Add kernel role metadata
    bad_reader = BadModule["bad_reader"]
    bad_reader = bad_reader.with_attr("tt.kernel_role", "reader")
    BadModule["bad_reader"] = bad_reader

    bad_compute = BadModule["bad_compute"]
    bad_compute = bad_compute.with_attr("tt.kernel_role", "compute")
    BadModule["bad_compute"] = bad_compute

    bad_writer = BadModule["bad_writer"]
    bad_writer = bad_writer.with_attr("tt.kernel_role", "writer")
    BadModule["bad_writer"] = bad_writer

    # Test bad module
    print("=== Testing Bad Module (Empty Bodies) ===")
    try:
        validator = ValidateSplitKernels(strict=True)
        result = validator(BadModule)
        print(" Validation passed (unexpected!)\n")
    except ValueError as e:
        print(f" Validation correctly failed:\n{e}\n")
