#!/usr/bin/env python3
"""Development scripts for sllmp."""

import subprocess
import sys
from typing import List


def run_command(cmd: List[str]) -> int:
    """Run a command and return its exit code."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def test() -> int:
    """Run tests."""
    return run_command(["pytest"])


def test_verbose() -> int:
    """Run tests with verbose output."""
    return run_command(["pytest", "-v"])


def test_coverage() -> int:
    """Run tests with coverage."""
    return run_command(
        ["pytest", "--cov=sllmp", "--cov-report=html", "--cov-report=term"]
    )


def lint() -> int:
    """Run linting."""
    return run_command(["ruff", "check", "src", "tests"])


def format_code() -> int:
    """Format code."""
    return run_command(["ruff", "format", "src", "tests"])


def type_check() -> int:
    """Run type checking."""
    return run_command(["mypy", "src"])


def dev() -> int:
    """Start development server."""
    return run_command(
        ["uvicorn", "sllmp.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
    )


def serve() -> int:
    """Start production server."""
    return run_command(
        ["uvicorn", "sllmp.main:app", "--host", "0.0.0.0", "--port", "8000"]
    )


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Available commands:")
        print("  test          - Run tests")
        print("  test-verbose  - Run tests with verbose output")
        print("  test-coverage - Run tests with coverage")
        print("  lint          - Run linting")
        print("  format        - Format code")
        print("  type-check    - Run type checking")
        print("  dev           - Start development server")
        print("  serve         - Start production server")
        sys.exit(1)

    command = sys.argv[1].replace("-", "_")

    if hasattr(sys.modules[__name__], command):
        func = getattr(sys.modules[__name__], command)
        sys.exit(func())
    else:
        print(f"Unknown command: {sys.argv[1]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
