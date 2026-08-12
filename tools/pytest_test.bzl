"""py_test wrapper for the pytest-style suite in //lattice_frx/testing.

A plain py_test on a pytest file collects nothing and exits 0 (false green);
this routes execution through tools/pytest_main.py instead.
"""

load("@lattice_frx_pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_test")

def pytest_test(name, srcs, deps = [], **kwargs):
    py_test(
        name = name,
        srcs = srcs + ["//tools:pytest_main.py"],
        main = "//tools:pytest_main.py",
        args = ["$(location %s)" % srcs[0]],
        deps = deps + [requirement("pytest")],
        **kwargs
    )
