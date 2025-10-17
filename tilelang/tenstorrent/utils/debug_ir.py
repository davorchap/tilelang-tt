"""
IR Debug Utilities for Tenstorrent Pipeline

This module provides utilities for debugging IR transformations,
including dumping IR after each pass and showing differences between passes.
"""

from __future__ import annotations
import os
import difflib
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
from datetime import datetime

try:
    import tvm
    from tvm import IRModule, tir
except ImportError:
    tvm = None
    IRModule = object
    tir = None

logger = logging.getLogger(__name__)


class IRDebugger:
    """Utility class for debugging IR transformations"""

    def __init__(self, dump_dir: str = "ir_dumps", enable: bool = True):
        """
        Initialize the IR debugger.

        Args:
            dump_dir: Directory to save IR dumps
            enable: Whether debugging is enabled
        """
        self.dump_dir = Path(dump_dir)
        self.enable = enable
        self.pass_count = 0
        self.ir_history: List[tuple[str, str]] = []  # (pass_name, ir_text)

        if self.enable:
            # Create dump directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.dump_dir = self.dump_dir / timestamp
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"IR debugger enabled. Dumps will be saved to: {self.dump_dir}")

    def dump_ir(self, mod: IRModule, pass_name: str, stage_name: Optional[str] = None) -> None:
        """
        Dump IR module to file after a pass.

        Args:
            mod: The IR module to dump
            pass_name: Name of the pass that just ran
            stage_name: Optional stage name (e.g., "A", "B", "C", "D", "E")
        """
        if not self.enable:
            return

        self.pass_count += 1

        # Generate filename
        stage_prefix = f"{stage_name}_" if stage_name else ""
        filename = f"{self.pass_count:03d}_{stage_prefix}{pass_name}.tir"
        filepath = self.dump_dir / filename

        # Convert module to string
        ir_text = str(mod)

        # Save to file
        with open(filepath, "w") as f:
            f.write(f"// Pass #{self.pass_count}: {pass_name}\n")
            if stage_name:
                f.write(f"// Stage: {stage_name}\n")
            f.write(f"// {'=' * 60}\n\n")
            f.write(ir_text)

        # Store in history
        self.ir_history.append((pass_name, ir_text))

        logger.debug(f"Dumped IR after pass {pass_name} to {filepath}")

    def show_diff(self, pass1_idx: int = -2, pass2_idx: int = -1, context_lines: int = 3) -> str:
        """
        Show diff between two passes in the history.

        Args:
            pass1_idx: Index of first pass (default: second to last)
            pass2_idx: Index of second pass (default: last)
            context_lines: Number of context lines in diff

        Returns:
            String containing the diff
        """
        if not self.enable or len(self.ir_history) < 2:
            return "Not enough IR history to show diff"

        try:
            pass1_name, ir1 = self.ir_history[pass1_idx]
            pass2_name, ir2 = self.ir_history[pass2_idx]
        except IndexError:
            return "Invalid pass indices"

        # Generate diff
        diff_lines = list(
            difflib.unified_diff(
                ir1.splitlines(keepends=True),
                ir2.splitlines(keepends=True),
                fromfile=f"After {pass1_name}",
                tofile=f"After {pass2_name}",
                n=context_lines))

        return "".join(diff_lines)

    def analyze_kernels(self, mod: IRModule) -> Dict[str, Any]:
        """
        Analyze kernels in the module and return statistics.

        Returns:
            Dictionary with kernel analysis results
        """
        if tvm is None:
            return {"error": "TVM not available"}

        stats = {
            "total_functions": 0,
            "kernels_by_role": {
                "reader": [],
                "compute": [],
                "writer": [],
                "host": [],
                "unknown": []
            },
            "empty_bodies": [],
            "statement_counts": {},
            "intrinsics_used": set(),
        }

        for name, func in mod.functions_items():
            if not isinstance(func, tir.PrimFunc):
                continue

            stats["total_functions"] += 1
            func_name = str(name)

            # Get kernel role
            kernel_role = func.attrs.get("tt.kernel_role", "unknown") if func.attrs else "unknown"
            stats["kernels_by_role"][kernel_role].append(func_name)

            # Check for empty body
            if not func.body or self._is_empty_body(func.body):
                stats["empty_bodies"].append(func_name)

            # Count statements
            stmt_count = self._count_statements(func.body) if func.body else 0
            stats["statement_counts"][func_name] = stmt_count

            # Collect intrinsics
            if func.body:
                intrinsics = self._collect_intrinsics(func.body)
                stats["intrinsics_used"].update(intrinsics)

        # Convert set to list for JSON serialization
        stats["intrinsics_used"] = list(stats["intrinsics_used"])

        return stats

    def _is_empty_body(self, body) -> bool:
        """Check if body is effectively empty"""
        if isinstance(body, tir.Evaluate) and hasattr(body, 'value'):
            if isinstance(body.value, (tir.IntImm, tir.FloatImm)):
                return body.value.value == 0
            if isinstance(body.value, tir.StringImm):
                return True
        return False

    def _count_statements(self, stmt) -> int:
        """Count statements in IR tree"""
        # Simple recursive counting
        count = 0
        if hasattr(stmt, '__iter__'):
            for s in stmt:
                count += self._count_statements(s)
        elif hasattr(stmt, 'body'):
            count += 1  # Count this statement
            count += self._count_statements(stmt.body)
        elif isinstance(stmt, tir.Evaluate):
            count += 1
        return count

    def _collect_intrinsics(self, stmt) -> set:
        """Collect all intrinsic calls in IR tree"""
        intrinsics = set()

        # Recursive collection
        if hasattr(stmt, '__iter__'):
            for s in stmt:
                intrinsics.update(self._collect_intrinsics(s))
        elif hasattr(stmt, 'body'):
            intrinsics.update(self._collect_intrinsics(stmt.body))
        elif isinstance(stmt, tir.Evaluate) and hasattr(stmt, 'value') and hasattr(
                stmt.value, 'op'):
            call = stmt.value
            op_name = str(call.op) if hasattr(call.op, 'name') else str(call.op)
            if "tir." in op_name or "T." in op_name:
                intrinsics.add(op_name)

        return intrinsics

    def save_summary(self, summary: Dict[str, Any], filename: str = "summary.txt") -> None:
        """
        Save analysis summary to file.

        Args:
            summary: Dictionary with summary data
            filename: Name of summary file
        """
        if not self.enable:
            return

        filepath = self.dump_dir / filename

        with open(filepath, "w") as f:
            f.write("IR Analysis Summary\n")
            f.write("=" * 60 + "\n\n")

            # Write general stats
            f.write(f"Total functions: {summary.get('total_functions', 0)}\n")
            f.write(f"Empty bodies: {len(summary.get('empty_bodies', []))}\n\n")

            # Write kernels by role
            f.write("Kernels by role:\n")
            for role, kernels in summary.get('kernels_by_role', {}).items():
                f.write(f"  {role}: {len(kernels)}\n")
                for kernel in kernels:
                    stmt_count = summary.get('statement_counts', {}).get(kernel, 0)
                    f.write(f"    - {kernel} ({stmt_count} statements)\n")
            f.write("\n")

            # Write empty bodies
            if summary.get('empty_bodies'):
                f.write("Empty body kernels:\n")
                for kernel in summary['empty_bodies']:
                    f.write(f"  - {kernel}\n")
                f.write("\n")

            # Write intrinsics used
            if summary.get('intrinsics_used'):
                f.write("Intrinsics used:\n")
                for intrinsic in sorted(summary['intrinsics_used']):
                    f.write(f"  - {intrinsic}\n")

        logger.info(f"Saved analysis summary to {filepath}")


def create_pipeline_wrapper(pipeline: List,
                            dump_ir: bool = False,
                            dump_dir: str = "ir_dumps") -> List:
    """
    Wrap a pipeline with IR dumping capability.

    Args:
        pipeline: List of pass functions/instances
        dump_ir: Whether to enable IR dumping
        dump_dir: Directory for IR dumps

    Returns:
        Wrapped pipeline with debugging
    """
    if not dump_ir:
        return pipeline

    # Check if IR dumping is enabled via environment variable
    if os.environ.get("TT_DUMP_IR", "").lower() in ["1", "true", "yes"]:
        dump_ir = True
        dump_dir = os.environ.get("TT_DUMP_IR_DIR", dump_dir)

    debugger = IRDebugger(dump_dir=dump_dir, enable=dump_ir)

    wrapped_pipeline = []
    stage_map = {
        "infer_tt_layout": "A",
        "propagate_tt_layout": "A",
        "attach_tensor_accessor": "A",
        "layout_aware_work_partition": "B",
        "grid_to_core_grid": "B",
        "lower_shared_to_cb": "C",
        "lower_tt_tile_intrinsics": "C",
        "build_tile_dfg": "C",
        "split_device_kernel": "D",
        "validate_split_kernels": "D",
        "configure_tensor_accessor": "D",
        "lower_cb_intrinsics": "D",
        "insert_compute_init": "D",
        "insert_dst_management": "D",
        "finalize_persistent_signature": "E",
    }

    for pass_instance in pipeline:
        # Get pass name
        if callable(pass_instance) and hasattr(pass_instance, '__name__'):
            pass_name = pass_instance.__name__
        elif hasattr(pass_instance, '__class__'):
            pass_name = pass_instance.__class__.__name__
        else:
            pass_name = "unknown_pass"

        # Determine stage
        stage = None
        for key, stage_letter in stage_map.items():
            if key in pass_name.lower():
                stage = stage_letter
                break

        # Create wrapper
        def make_wrapper(original_pass, name, stage_letter):

            def wrapper(mod):
                result = original_pass(mod)
                debugger.dump_ir(result, name, stage_letter)

                # Special handling after split_device_kernel
                if "split_device_kernel" in name:
                    logger.info("Analyzing kernels after split_device_kernel...")
                    analysis = debugger.analyze_kernels(result)
                    debugger.save_summary(analysis, "split_kernel_analysis.txt")

                    # Log warnings if we find issues
                    if analysis.get("empty_bodies"):
                        logger.warning(
                            f"Found {len(analysis['empty_bodies'])} kernels with empty bodies: "
                            f"{analysis['empty_bodies']}")

                return result

            return wrapper

        wrapped_pipeline.append(make_wrapper(pass_instance, pass_name, stage))

    # Save final summary after pipeline completes
    def final_summary(mod):
        if dump_ir:
            analysis = debugger.analyze_kernels(mod)
            debugger.save_summary(analysis, "final_analysis.txt")
            logger.info(f"Pipeline debugging complete. Results saved to: {debugger.dump_dir}")
        return mod

    wrapped_pipeline.append(final_summary)

    return wrapped_pipeline


# Convenience function to enable debugging via environment variable
def enable_ir_debugging():
    """Enable IR debugging by setting environment variable"""
    os.environ["TT_DUMP_IR"] = "1"
    logger.info("IR debugging enabled via TT_DUMP_IR environment variable")


def disable_ir_debugging():
    """Disable IR debugging by unsetting environment variable"""
    if "TT_DUMP_IR" in os.environ:
        del os.environ["TT_DUMP_IR"]
    logger.info("IR debugging disabled")


# Example usage
if __name__ == "__main__":
    # This would be used in the pipeline
    import logging
    logging.basicConfig(level=logging.INFO)

    # Enable debugging
    enable_ir_debugging()

    print("IR Debug utilities loaded. Set TT_DUMP_IR=1 to enable IR dumping.")
    print("Example usage:")
    print("  export TT_DUMP_IR=1")
    print("  export TT_DUMP_IR_DIR=my_ir_dumps")
    print("  python your_test.py")
