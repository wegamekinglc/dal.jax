"""Test session setup."""

import jax
import pytest

import dal_jax

N_TEST_DEVICES = 4

dal_jax.config.configure(num_cpu_devices=N_TEST_DEVICES)


def pytest_addoption(parser):
    parser.addoption(
        "--run-gpu", action="store_true", help="run optional tests requiring an actual GPU backend"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-gpu"):
        return
    skip = pytest.mark.skip(
        reason="GPU validation is opt-in; pass --run-gpu with a CUDA/ROCm-enabled JAX"
    )
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def gpu_devices():
    devices = tuple(jax.local_devices(backend="gpu"))
    assert devices, "--run-gpu requires an actual GPU backend"
    return devices


@pytest.fixture(scope="session")
def cpu_devices() -> tuple[jax.Device, ...]:
    devices = tuple(jax.local_devices(backend="cpu"))
    assert len(devices) == N_TEST_DEVICES
    return devices


@pytest.fixture(scope="session")
def dal():
    """dal-python, the C++ oracle; tests using it are skipped when it is not installed."""
    module = pytest.importorskip("dal")
    module.EvaluationDate_Set(module.Date_(2022, 9, 15))
    return module
