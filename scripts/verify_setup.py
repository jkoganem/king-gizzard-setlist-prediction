#!/usr/bin/env python3
"""Verify project setup and dependencies."""

import sys
from pathlib import Path


def check_dependencies():
    """Check Python dependencies."""
    print("Checking Python dependencies...")
    required = [
        "pandas",
        "numpy",
        "sklearn",
        "xgboost",
        "fastapi",
        "sqlalchemy",
        "pydantic",
    ]

    missing = []
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"  [OK] {pkg}")
        except ImportError:
            print(f"  [MISSING] {pkg}")
            missing.append(pkg)

    return missing


def check_structure():
    """Check project structure."""
    print("\nChecking project structure...")
    required_dirs = [
        "src",
        "src/ingest",
        "src/features",
        "src/models",
        "src/utils",
        "configs",
        "scripts",
        "experiments",
        "data",
        "tests",
    ]

    missing = []
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  [OK] {dir_path}/")
        else:
            print(f"  [MISSING] {dir_path}/")
            missing.append(dir_path)

    return missing


def check_config_files():
    """Check configuration files."""
    print("\nChecking configuration files...")
    required_files = [
        "pyproject.toml",
        "Makefile",
        "configs/models.yaml",
    ]

    missing = []
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  [OK] {file_path}")
        else:
            print(f"  [MISSING] {file_path}")
            missing.append(file_path)

    return missing


def check_env():
    """Check environment setup."""
    print("\nChecking environment...")
    env_file = Path(".env")

    if env_file.exists():
        print("  [OK] .env file exists")
        with open(env_file) as f:
            content = f.read()
            if "SETLISTFM_API_KEY" in content:
                print("  [OK] SETLISTFM_API_KEY configured")
            else:
                print("  [WARNING] SETLISTFM_API_KEY not set (will use synthetic data)")
    else:
        print("  [WARNING] .env file missing (copy from .env.example)")


def main():
    print("=" * 60)
    print("King Gizzard Setlist Predictor - Setup Verification")
    print("=" * 60)

    missing_deps = check_dependencies()
    missing_dirs = check_structure()
    missing_files = check_config_files()
    check_env()

    print("\n" + "=" * 60)
    if not any([missing_deps, missing_dirs, missing_files]):
        print("[SUCCESS] All checks passed! Setup looks good.")
        print("\nNext steps:")
        print("  1. Add SETLISTFM_API_KEY to .env")
        print("  2. make train-all  # Run Stage 1 factorial experiments")
        print("  3. make tune       # Run Stage 2 XGBoost tuning")
        print("  4. See TODO.md for full experimental pipeline")
    else:
        print("[FAILED] Some issues found:")
        if missing_deps:
            print(f"  - Missing dependencies: {', '.join(missing_deps)}")
            print("    Run: poetry install")
        if missing_dirs:
            print(f"  - Missing directories: {', '.join(missing_dirs)}")
        if missing_files:
            print(f"  - Missing files: {', '.join(missing_files)}")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
