#!/usr/bin/env python3
"""
ChemiAccel Test Coverage Validator
Ensures that new modules have corresponding test files.
"""

import sys
from pathlib import Path
from typing import List


def check_test_coverage(files: List[str]) -> List[str]:
    """Check that modules have corresponding test files."""
    errors = []

    for file_path in files:
        path = Path(file_path)

        # Skip __init__.py and test files themselves
        if path.name == "__init__.py" or "test_" in path.name:
            continue

        # Skip certain directories that don't need tests
        skip_dirs = {"__pycache__", "build", "dist", ".git"}
        if any(skip_dir in path.parts for skip_dir in skip_dirs):
            continue

        # Convert src path to test path
        if "src/chemi_accel" in str(path):
            # src/chemi_accel/module.py -> tests/unit/test_module.py
            relative_path = path.relative_to(Path("src/chemi_accel"))
            test_path = Path("tests/unit") / f"test_{relative_path}"

            if not test_path.exists():
                # Also check for module-specific test directory
                module_test_dir = Path("tests/unit") / relative_path.with_suffix("")
                if not module_test_dir.exists():
                    errors.append(
                        f"Missing test file: {test_path} or {module_test_dir}/"
                    )

    return errors


def main():
    """Main validation function."""
    if len(sys.argv) < 2:
        print("Usage: check_test_coverage.py <file1> [file2] ...")
        return 0

    files = sys.argv[1:]
    errors = check_test_coverage(files)

    if errors:
        print("Test coverage issues found:")
        for error in errors:
            print(f"  WARNING: {error}")
        print("\nNote: This is a warning, not a blocking error.")
        print("Consider adding tests for new modules.")
        return 0  # Don't block commits, just warn
    else:
        print("PASS: Test coverage check passed")
        return 0


if __name__ == "__main__":
    sys.exit(main())
