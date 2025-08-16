import os
import sys

# Ensure local src path is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


def test_package_import():
    try:
        import gepa  # noqa: F401
    except ImportError as e:  # pragma: no cover
        assert False, f"Failed to import the 'gepa' package: {e}"


def test_gepa_optimize_import():
    try:
        from gepa import optimize  # noqa: F401
    except ImportError as e:  # pragma: no cover
        assert False, f"Failed to import the 'gepa.optimize' function: {e}"
