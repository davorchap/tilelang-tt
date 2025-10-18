import ast
import pathlib


def _is_within_main_block(node: ast.AST) -> bool:
    """Return True if the node is within an if __name__ == "__main__" block."""
    # Walk up parents using custom attribute set during traversal
    cur = getattr(node, "_parent", None)
    while cur is not None:
        if isinstance(cur, ast.If):
            # Check condition is __name__ == "__main__"
            cond = cur.test
            if isinstance(cond, ast.Compare) and isinstance(
                    cond.left, ast.Name) and cond.left.id == "__name__":
                # Compare to string "__main__"
                rights = cond.comparators
                if rights and isinstance(rights[0], ast.Constant) and rights[0].value == "__main__":
                    return True
        cur = getattr(cur, "_parent", None)
    return False


def _attach_parents(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._parent = parent  # type: ignore[attr-defined]


def _collect_offenses_in_file(py_path: pathlib.Path) -> list[tuple[int, str]]:
    text = py_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    _attach_parents(tree)

    offenses: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Detect obj.with_attr("tt.something", ...)
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "with_attr" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(
                        first.value, str) and first.value.startswith("tt."):
                    if _is_within_main_block(node):
                        continue
                    offenses.append((node.lineno, first.value))
    return offenses


def test_tt_attribute_keys_use_constants():
    # Scan production Tenstorrent backend Python files
    root = pathlib.Path(__file__).resolve().parents[3]  # repo root
    base = root / "tilelang" / "tenstorrent"
    assert base.exists()

    offenses_total: list[tuple[str, int, str]] = []
    for py_path in base.rglob("*.py"):
        # Skip the registry itself
        if py_path.name == "attrs.py":
            continue
        offenses = _collect_offenses_in_file(py_path)
        offenses_total.extend([(str(py_path.relative_to(root)), ln, key) for ln, key in offenses])

    # If there are offenses, produce a helpful message
    if offenses_total:
        lines = [f"{path}:{ln} -> {key}" for path, ln, key in offenses_total]
        joined = "\n".join(lines)
        raise AssertionError(
            "Found direct string uses of 'tt.*' in with_attr; use keys from tilelang.tenstorrent.attrs instead:\n"
            + joined)
