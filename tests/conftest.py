"""Keep the default suite usable without a simulator or CUDA installation."""

import importlib.metadata

import pytest

BASE_TESTS = {"test_warp_device.py", "test_package_resources.py"}
INTEGRATION_TESTS = {"test_flex_bridge.py", "test_dense_effective.py", "test_newton_factor.py",
                     "test_newton_coupled.py"}
HOOK_TESTS = {"test_mujoco_simplex.py"}


def pytest_addoption(parser):
    parser.addoption("--integration", action="store_true", help="test the MuJoCo/MJWarp geometry bridge")
    parser.addoption("--experimental-gpu", action="store_true", help="test the pinned 4.0.1 GPU policies")
    parser.addoption("--source-hook", action="store_true", help="test a separately built MuJoCo hook")


def pytest_configure(config):
    if config.getoption("--experimental-gpu"):
        for name, expected in (("mujoco", "4.0.1"), ("mujoco-warp", "4.0.1"), ("warp-lang", "1.19.0")):
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                actual = "not installed"
            if actual.split("+")[0] != expected:
                raise pytest.UsageError(f"{name}: expected {expected}, found {actual}; install .[physics,test]")
        import warp as wp
        if not wp.is_cuda_available():
            raise pytest.UsageError("--experimental-gpu requires an available NVIDIA CUDA device")


def pytest_ignore_collect(collection_path, config):
    name = collection_path.name
    if not name.startswith("test_") or not name.endswith(".py"):
        return False
    if name in BASE_TESTS:
        return False
    if name in HOOK_TESTS:
        return not config.getoption("--source-hook")
    if name in INTEGRATION_TESTS:
        return not config.getoption("--integration")
    # Surface binding tests use a separately built C++ exporter, not pytest.
    if name == "test_surface_bindings.py":
        return True
    return not config.getoption("--experimental-gpu")
