"""Test session setup.

Four virtual CPU devices are configured before any JAX operation so the
parallel-consistency tests can compare one device against several.  Every
other test pins its own device list where it matters.
"""

import jax
import pytest

import dal_jax

N_TEST_DEVICES = 4

dal_jax.config.configure(num_cpu_devices=N_TEST_DEVICES)


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
