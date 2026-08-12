"""py_test entry point that actually runs pytest.

The test suite is pytest-style (bare test functions, no __main__ block), so a
plain py_test would import the file, collect nothing, and exit 0 — a false
green. Every py_test in //lattice_frx/testing.points its `main` here and passes the test
file as argv.
"""
import sys

import pytest

if __name__ == "__main__":
    sys.exit(pytest.main(["-p", "no:cacheprovider", *sys.argv[1:]]))
