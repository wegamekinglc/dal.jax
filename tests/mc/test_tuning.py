"""Memory-based GPU sizing and explicit overrides without requiring CUDA in CI."""

from dataclasses import dataclass

import jax.numpy as jnp
import pytest
from support import bs_model, european_call, monthly_barrier_timeline, up_and_out_call

from dal_jax import MonteCarloEngine, MonteCarloSettings
from dal_jax.errors import InvalidSetting
from dal_jax.mc.tuning import estimated_path_bytes, resolve_block_size, resolve_dtype


@dataclass(frozen=True)
class Device:
    platform: str
    memory: int | None = None

    def memory_stats(self):
        return None if self.memory is None else {"bytes_limit": self.memory}


def resolve(product, *, memory=8*2**30, dtype="float64", enable_aad=True, **overrides):
    settings = MonteCarloSettings(enable_aad=enable_aad, **overrides)
    return resolve_block_size(settings, product, len(product.timeline), (Device("gpu", memory),), dtype)


def test_cpu_auto_preserves_existing_layout_and_precision(cpu_devices):
    engine = MonteCarloEngine(european_call(), bs_model(), MonteCarloSettings(devices=cpu_devices, dtype="auto"))
    assert engine.block_size == 8192 and engine.dtype == jnp.float64
    assert engine.layout(9000).block_size == 8192
    assert engine.layout(17).block_size == 17


def test_gpu_auto_precision_requires_an_explicit_request():
    devices = (Device("gpu"),)
    assert resolve_dtype("auto", devices) == "float32"
    assert resolve_dtype("float64", devices) == "float64"
    assert resolve_dtype("float32", (Device("cpu"),)) == "float32"


def test_gpu_blocks_shrink_with_horizon_memory_and_adjoint_storage():
    short = european_call()
    long = up_and_out_call(monthly_barrier_timeline(), tuple(range(1, 37)))
    assert resolve(short, memory=2**30) > resolve(long, memory=2**30)
    assert resolve(long, memory=2**30) < resolve(long)
    assert resolve(long, memory=2**30, enable_aad=True) < resolve(long, memory=2**30, enable_aad=False)
    assert resolve(long, dtype="float32") >= resolve(long, dtype="float64")


def test_explicit_block_and_unknown_memory_fallback():
    product = european_call()
    assert resolve(product, block_size=123) == 123
    assert resolve(product, memory=None) == 8192
    assert resolve(product, memory=0) == 8192
    assert resolve(product, memory=2**20) == 256


def test_smallest_device_memory_limits_each_devices_block():
    settings = MonteCarloSettings(enable_aad=True)
    product = up_and_out_call(monthly_barrier_timeline(), tuple(range(1, 37)))
    devices = (Device("gpu", 8*2**30), Device("gpu", 2**30))
    block = resolve_block_size(settings, product, 36, devices, "float64")
    assert block == resolve(product, memory=2**30)
    assert block & (block-1) == 0
    assert estimated_path_bytes(product, 36, "float64", True) > estimated_path_bytes(product, 36, "float32", False)


@pytest.mark.parametrize("block", ["AUTO", None, 0, -1, "8192"])
def test_invalid_auto_block_settings(block):
    with pytest.raises(InvalidSetting):
        MonteCarloSettings(block_size=block)
