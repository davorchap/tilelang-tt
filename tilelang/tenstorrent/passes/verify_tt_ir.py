"""
Pass F: VerifyTTIR (v5 Specification)
Version: 5.0
Date: 2025-10-15

Purpose: Verify TIR conformance for Tenstorrent backend.
         Checks capacity constraints, protocol correctness, and metadata completeness.

Input: Complete pipeline output
Output: Validation report (pass/fail with diagnostics)
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
    ERROR = "error"  # Must fix - will fail on hardware
    WARNING = "warning"  # Should fix - may cause issues
    INFO = "info"  # Informational - optimization opportunity


@dataclass
class ValidationIssue:
    """A single validation issue"""
    level: ValidationLevel
    category: str
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report"""
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def add_issue(self,
                  level: ValidationLevel,
                  category: str,
                  message: str,
                  location: Optional[str] = None,
                  suggestion: Optional[str] = None):
        """Add a validation issue to the report"""
        self.issues.append(ValidationIssue(level, category, message, location, suggestion))
        if level == ValidationLevel.ERROR:
            self.passed = False

    def get_summary(self) -> str:
        """Get a summary of the validation report"""
        error_count = sum(1 for i in self.issues if i.level == ValidationLevel.ERROR)
        warning_count = sum(1 for i in self.issues if i.level == ValidationLevel.WARNING)
        info_count = sum(1 for i in self.issues if i.level == ValidationLevel.INFO)

        return f"Errors: {error_count}, Warnings: {warning_count}, Info: {info_count}"


class CBValidator:
    """Validator for circular buffer constraints"""

    MAX_CBS = 32
    MAX_L1_BYTES = 1024 * 1024  # 1MB typical L1 size
    TILE_SIZE_BF16 = 32 * 32 * 2  # 2KB per BF16 tile

    def validate(self, func: "tir.PrimFunc", report: ValidationReport):
        """Validate CB constraints"""

        # Check CB count
        cb_indices = self._get_cb_indices(func)
        if cb_indices:
            max_index = max(cb_indices.values())
            if max_index >= self.MAX_CBS:
                report.add_issue(
                    ValidationLevel.ERROR,
                    "CB",
                    f"CB index {max_index} exceeds maximum of {self.MAX_CBS}",
                    location=func.name if hasattr(func, 'name') else None,
                    suggestion="Reduce CB usage or reuse indices")

            report.stats["cb_count"] = len(cb_indices)
            report.stats["max_cb_index"] = max_index

        # Check L1 capacity
        cb_descriptors = self._get_cb_descriptors(func)
        if cb_descriptors:
            total_l1_usage = 0
            for _cb_name, desc in cb_descriptors.items():
                page_size = desc.get("page_size", 0)
                depth = desc.get("depth", 1)
                cb_size = page_size * depth
                total_l1_usage += cb_size

            report.stats["l1_usage_bytes"] = total_l1_usage
            report.stats["l1_usage_percent"] = (total_l1_usage / self.MAX_L1_BYTES) * 100

            if total_l1_usage > self.MAX_L1_BYTES:
                report.add_issue(
                    ValidationLevel.ERROR,
                    "L1",
                    f"L1 usage {total_l1_usage} bytes exceeds capacity {self.MAX_L1_BYTES}",
                    location=func.name if hasattr(func, 'name') else None,
                    suggestion="Reduce CB depth or tile sizes")
            elif total_l1_usage > self.MAX_L1_BYTES * 0.8:
                report.add_issue(
                    ValidationLevel.WARNING,
                    "L1",
                    f"L1 usage {total_l1_usage} bytes is >80% of capacity",
                    location=func.name if hasattr(func, 'name') else None,
                    suggestion="Consider optimizing CB allocation")

    def _get_cb_indices(self, func: "tir.PrimFunc") -> Dict[str, int]:
        """Extract CB indices from function attributes"""
        if func.attrs and "tt.cb_indices" in func.attrs:
            return self._convert_to_dict(func.attrs["tt.cb_indices"])
        return {}

    def _get_cb_descriptors(self, func: "tir.PrimFunc") -> Dict[str, Dict[str, Any]]:
        """Extract CB descriptors from function attributes"""
        if func.attrs and "tt.cb_descriptors" in func.attrs:
            return self._convert_to_dict(func.attrs["tt.cb_descriptors"])
        return {}

    def _convert_to_dict(self, attr_value: Any) -> Dict[str, Any]:
        """Convert TVM attribute to dict"""
        if isinstance(attr_value, dict):
            return attr_value
        if hasattr(attr_value, "items"):
            result = {}
            for k, v in attr_value.items():
                if hasattr(v, "items"):
                    result[str(k)] = self._convert_to_dict(v)
                else:
                    result[str(k)] = v.value if hasattr(v, "value") else v
            return result
        return {}


class MetadataValidator:
    """Validator for required metadata"""

    REQUIRED_ATTRS = {
        "reader": ["tt.kernel_role", "tt.runtime_args", "tt.runtime_args_finalized"],
        "compute": [
            "tt.kernel_role", "tt.runtime_args", "tt.compute_init_inserted",
            "tt.dst_management_inserted"
        ],
        "writer": ["tt.kernel_role", "tt.runtime_args", "tt.runtime_args_finalized"],
        "monolithic": ["tt.core_grid", "tt.partition_mode"]
    }

    def validate(self, func: "tir.PrimFunc", report: ValidationReport):
        """Validate metadata completeness"""

        kernel_role = None
        if func.attrs and "tt.kernel_role" in func.attrs:
            kernel_role = func.attrs["tt.kernel_role"]
        else:
            report.add_issue(
                ValidationLevel.WARNING,
                "Metadata",
                "No kernel role specified",
                location=func.name if hasattr(func, 'name') else None,
                suggestion="Check if kernel splitting was applied")
            return

        # Check required attributes for role
        required = self.REQUIRED_ATTRS.get(kernel_role, [])
        missing = []

        for attr in required:
            if not func.attrs or attr not in func.attrs:
                missing.append(attr)

        if missing:
            report.add_issue(
                ValidationLevel.ERROR,
                "Metadata",
                f"Missing required attributes for {kernel_role}: {missing}",
                location=func.name if hasattr(func, 'name') else None,
                suggestion="Ensure all passes have been applied")

        # Check runtime args completeness
        if func.attrs and "tt.runtime_args" in func.attrs:
            runtime_args = func.attrs["tt.runtime_args"]
            if isinstance(runtime_args, (list, tuple)):
                report.stats[f"{kernel_role}_arg_count"] = len(runtime_args)

                # Verify essential args
                if kernel_role in ["reader", "writer"] and not any("addr" in str(arg) for arg in runtime_args):
                    report.add_issue(
                        ValidationLevel.WARNING,
                        "Runtime Args",
                        f"No address arguments found for {kernel_role}",
                        location=func.name if hasattr(func, 'name') else None)


class ProtocolValidator:
    """Validator for protocol correctness"""

    def validate(self, func: "tir.PrimFunc", report: ValidationReport):
        """Validate protocol sequences"""

        kernel_role = func.attrs.get("tt.kernel_role") if func.attrs else None
        if not kernel_role:
            return

        # Check protocol insertion markers
        if (kernel_role == "reader" or kernel_role == "writer") and (not func.attrs or "tt.cb_protocol_inserted" not in func.attrs):
            report.add_issue(
                ValidationLevel.WARNING,
                "Protocol",
                f"CB protocol may not be inserted for {kernel_role}",
                location=func.name if hasattr(func, 'name') else None,
                suggestion="Check if D3 (LowerCBIntrinsics) was applied")

        if kernel_role == "compute":
            if not func.attrs or "tt.compute_init_inserted" not in func.attrs:
                report.add_issue(
                    ValidationLevel.ERROR,
                    "Protocol",
                    "Compute engine init not inserted",
                    location=func.name if hasattr(func, 'name') else None,
                    suggestion="Apply D4 (InsertComputeInitTT)")

            if not func.attrs or "tt.dst_management_inserted" not in func.attrs:
                report.add_issue(
                    ValidationLevel.ERROR,
                    "Protocol",
                    "DST management not inserted",
                    location=func.name if hasattr(func, 'name') else None,
                    suggestion="Apply D5 (InsertDSTManagementTT)")


class DTypeValidator:
    """Validator for data type support"""

    SUPPORTED_DTYPES = ["bf16", "float16", "fp16", "float32", "fp32", "int8", "uint8"]
    OPTIMAL_DTYPES = ["bf16", "float16", "fp16"]

    def validate(self, func: "tir.PrimFunc", report: ValidationReport):
        """Validate data type support"""

        # Check buffer dtypes
        for param in func.params:
            if param in func.buffer_map:
                buffer = func.buffer_map[param]
                dtype_str = str(buffer.dtype)

                if not any(supported in dtype_str.lower() for supported in self.SUPPORTED_DTYPES):
                    report.add_issue(
                        ValidationLevel.ERROR,
                        "DType",
                        f"Unsupported dtype {dtype_str} for buffer {buffer.name}",
                        location=func.name if hasattr(func, 'name') else None,
                        suggestion=f"Use one of: {self.SUPPORTED_DTYPES}")
                elif not any(optimal in dtype_str.lower() for optimal in self.OPTIMAL_DTYPES):
                    report.add_issue(
                        ValidationLevel.INFO,
                        "DType",
                        f"Non-optimal dtype {dtype_str} for buffer {buffer.name}",
                        location=func.name if hasattr(func, 'name') else None,
                        suggestion="Consider using bf16 or fp16 for better performance")


class StructureValidator:
    """Validator for kernel structure"""

    def validate(self, mod: IRModule, report: ValidationReport):
        """Validate module structure and kernel organization"""

        # Count kernels by role
        kernel_roles = {"reader": [], "compute": [], "writer": [], "monolithic": [], "unknown": []}

        for name, func in mod.functions_items():
            if isinstance(func, tir.PrimFunc):
                role = func.attrs.get("tt.kernel_role", "unknown") if func.attrs else "unknown"
                kernel_roles[role].append(str(name))

        report.stats["kernel_counts"] = {k: len(v) for k, v in kernel_roles.items()}

        # Check for proper 3-kernel split
        if kernel_roles["monolithic"]:
            report.add_issue(
                ValidationLevel.WARNING,
                "Structure",
                f"Found monolithic kernels: {kernel_roles['monolithic']}",
                suggestion="Apply D1 (SplitDeviceKernel) for 3-kernel architecture")

        # Check for balanced split
        if kernel_roles["reader"] and kernel_roles["compute"] and kernel_roles["writer"] and (len(kernel_roles["reader"]) != len(kernel_roles["compute"]) or len(kernel_roles["compute"]) != len(kernel_roles["writer"])):
            report.add_issue(
                ValidationLevel.WARNING,
                "Structure",
                "Unbalanced kernel split detected",
                suggestion="Each kernel group should have equal count")


class VerifyTTIR:
    """
    Pass to verify TIR for Tenstorrent backend.

    This pass:
    1. Validates CB constraints (count, L1 capacity)
    2. Checks metadata completeness
    3. Verifies protocol insertion
    4. Validates data type support
    5. Checks kernel structure
    """

    def __init__(self, strict: bool = True) -> None:
        """
        Initialize verifier.

        Args:
            strict: If True, warnings are treated as errors
        """
        self.strict = strict
        self.validators = [
            CBValidator(),
            MetadataValidator(),
            ProtocolValidator(),
            DTypeValidator()
        ]
        self.structure_validator = StructureValidator()

    def __call__(self, mod: IRModule) -> IRModule:
        """Apply verification to an IRModule."""
        if tvm is None:
            return mod

        report = ValidationReport(passed=True)

        # Validate each function
        for name, func in mod.functions_items():
            if isinstance(func, tir.PrimFunc):
                logger.debug(f"Validating function: {name}")
                for validator in self.validators:
                    validator.validate(func, report)

        # Validate module structure
        self.structure_validator.validate(mod, report)

        # Apply strict mode
        if self.strict:
            for issue in report.issues:
                if issue.level == ValidationLevel.WARNING:
                    issue.level = ValidationLevel.ERROR
                    report.passed = False

        # Log results
        self._log_report(report)

        # Attach report to module
        if report.issues:
            # Convert report to serializable format
            report_dict = {
                "passed": report.passed,
                "summary": report.get_summary(),
                "issues": [{
                    "level": issue.level.value,
                    "category": issue.category,
                    "message": issue.message,
                    "location": issue.location,
                    "suggestion": issue.suggestion
                } for issue in report.issues],
                "stats": report.stats
            }

            # Note: In real implementation, we'd attach this to module metadata
            # For now, we'll just log it
            logger.info(f"Validation report: {report_dict}")

        if not report.passed:
            error_messages = [
                f"{i.category}: {i.message}" for i in report.issues
                if i.level == ValidationLevel.ERROR
            ]
            raise ValueError("TT IR validation failed:\n" + "\n".join(error_messages))

        return mod

    def _log_report(self, report: ValidationReport):
        """Log the validation report"""

        if report.passed:
            logger.info("✅ TT IR validation passed")
        else:
            logger.error("❌ TT IR validation failed")

        logger.info(f"Summary: {report.get_summary()}")

        # Log errors first
        for issue in report.issues:
            if issue.level == ValidationLevel.ERROR:
                msg = f"ERROR [{issue.category}]: {issue.message}"
                if issue.location:
                    msg += f" at {issue.location}"
                if issue.suggestion:
                    msg += f" | Suggestion: {issue.suggestion}"
                logger.error(msg)

        # Then warnings
        for issue in report.issues:
            if issue.level == ValidationLevel.WARNING:
                msg = f"WARNING [{issue.category}]: {issue.message}"
                if issue.location:
                    msg += f" at {issue.location}"
                if issue.suggestion:
                    msg += f" | Suggestion: {issue.suggestion}"
                logger.warning(msg)

        # Finally info
        for issue in report.issues:
            if issue.level == ValidationLevel.INFO:
                msg = f"INFO [{issue.category}]: {issue.message}"
                if issue.suggestion:
                    msg += f" | Suggestion: {issue.suggestion}"
                logger.info(msg)

        # Log statistics
        if report.stats:
            logger.info("Statistics:")
            for key, value in report.stats.items():
                logger.info(f"  {key}: {value}")


# Module-level pass function for compatibility
def verify_tt_ir(mod: IRModule, strict: bool = True) -> IRModule:
    """Apply VerifyTTIR pass to a module."""
    pass_instance = VerifyTTIR(strict=strict)
    return pass_instance(mod)


# Example usage and testing
if __name__ == "__main__":
    import tvm.script
    from tvm.script import tir as T

    # Create test module with various issues
    @tvm.script.ir_module
    class TestModule:

        @T.prim_func
        def good_reader(A: T.Buffer((256, 256), "bfloat16")):
            T.evaluate(0)

        @T.prim_func
        def bad_compute():
            T.evaluate(0)  # Missing protocol insertion

        @T.prim_func
        def bad_writer(C: T.Buffer((256, 256), "int64")  # Unsupported dtype
                      ):
            T.evaluate(0)

    # Add metadata
    good_reader = TestModule["good_reader"]
    good_reader = good_reader.with_attr("tt.kernel_role", "reader")
    good_reader = good_reader.with_attr("tt.runtime_args", ["A_addr", "start_id", "count"])
    good_reader = good_reader.with_attr("tt.runtime_args_finalized", True)
    good_reader = good_reader.with_attr("tt.cb_protocol_inserted", True)
    good_reader = good_reader.with_attr("tt.cb_indices", {"cb_in0": 0})
    TestModule["good_reader"] = good_reader

    bad_compute = TestModule["bad_compute"]
    bad_compute = bad_compute.with_attr("tt.kernel_role", "compute")
    # Missing required attributes!
    TestModule["bad_compute"] = bad_compute

    bad_writer = TestModule["bad_writer"]
    bad_writer = bad_writer.with_attr("tt.kernel_role", "writer")
    bad_writer = bad_writer.with_attr("tt.runtime_args", ["C_addr"])
    # Has unsupported dtype
    TestModule["bad_writer"] = bad_writer

    # Apply verification
    print("=== Running TT IR Verification ===\n")
    verifier = VerifyTTIR(strict=False)  # Non-strict for demonstration

    try:
        result = verifier(TestModule)
        print("Validation completed")
    except ValueError as e:
        print(f"Validation failed: {e}")
