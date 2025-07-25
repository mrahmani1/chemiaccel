#!/usr/bin/env python3
"""
ChemiAccel Import Structure Validator
Ensures proper import patterns and prevents circular imports.
"""

import ast
import sys
from pathlib import Path
from typing import List


def check_import_structure(src_dir: Path) -> List[str]:
    """Check import structure for ChemiAccel modules."""
    errors = []
    modules = {}

    # Collect all Python files
    for py_file in src_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue

        relative_path = py_file.relative_to(src_dir)
        module_name = (
            str(relative_path.with_suffix("")).replace("/", ".").replace("\\", ".")
        )

        try:
            with open(py_file, encoding="utf-8") as f:
                tree = ast.parse(f.read())

            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            modules[module_name] = imports

        except Exception as e:
            errors.append(f"Error parsing {py_file}: {e}")

    # Check for potential circular imports
    for module, imports in modules.items():
        for imported in imports:
            if imported.startswith("chemi_accel"):
                # Check if imported module also imports back to this module
                imported_clean = imported.replace("chemi_accel.", "")
                if imported_clean in modules:
                    if any(
                        imp.startswith(f"chemi_accel.{module}")
                        for imp in modules[imported_clean]
                    ):
                        errors.append(
                            f"Potential circular import: {module} <-> {imported_clean}"
                        )

    # Check for forbidden imports
    forbidden_patterns = [
        ("test", "Production code should not import test modules"),
        ("examples", "Production code should not import examples"),
        ("benchmarks", "Production code should not import benchmarks"),
    ]

    for module, imports in modules.items():
        for pattern, message in forbidden_patterns:
            if any(pattern in imp for imp in imports):
                errors.append(f"{module}: {message}")

    return errors


def main():
    """Main validation function."""
    src_dir = Path("src/chemi_accel")
    if not src_dir.exists():
        print("ChemiAccel source directory not found")
        return 0

    errors = check_import_structure(src_dir)

    if errors:
        print("Import structure issues found:")
        for error in errors:
            print(f"  ERROR: {error}")
        return 1
    else:
        print("PASS: Import structure validation passed")
        return 0


if __name__ == "__main__":
    sys.exit(main())
