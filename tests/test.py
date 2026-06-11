"""
Run the full test suite in this directory.

Usage:
    python tests/test.py
"""
from pathlib import Path

import pytest


def main() -> int:
    tests_dir = Path(__file__).resolve().parent
    return pytest.main([str(tests_dir), f"--ignore={Path(__file__).resolve()}"])


if __name__ == "__main__":
    raise SystemExit(main())
