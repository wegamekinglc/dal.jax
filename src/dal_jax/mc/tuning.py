"""Host-side precision and block-size choices for CPU and GPU execution."""

import numpy as np

from dal_jax.mc.settings import DEFAULT_BLOCK_SIZE

_MIN_GPU_BLOCK = 256
_MAX_GPU_BLOCK = 32768
_MEMORY_FRACTION = 0.20


def resolve_dtype(choice, devices):
    """Explicit auto opts into float32 GPU paths; compatibility defaults stay float64."""
    if choice == "auto":
        return "float32" if devices[0].platform == "gpu" else "float64"
    return choice


def _memory_limit(devices):
    limits = [(device.memory_stats() or {}).get("bytes_limit", 0) for device in devices]
    return min(limits) if limits and all(limit > 0 for limit in limits) else None


def estimated_path_bytes(product, sim_dim, dtype, enable_aad):
    """Conservative array estimate, including float64 RNG and temporary/adjoint headroom.

    This is a sizing heuristic, not an XLA memory bound. Scenario slots include
    spot, numeraire, requested observations and discounts on every event date.
    """
    samples = sum(
        2 + len(sample.index_names) + len(sample.discount_mats) for sample in product.sample_defs
    )
    slots = (
        samples + len(product.payoff_names) + len(product.error_messages) + product.path_state_size
    )
    arrays = max(1, sim_dim) * 8 + slots * np.dtype(dtype).itemsize
    return arrays * (16 if enable_aad else 4)


def resolve_block_size(settings, product, sim_dim, devices, dtype):
    """CPU auto keeps 8192; GPU auto budgets 20% of the smallest allocator limit.

    Use power-of-two blocks in [256, 32768]. Unknown GPU memory falls back to
    8192. An explicit block size always wins; path-count masking is unchanged.
    """
    if settings.block_size != "auto":
        return settings.block_size
    if devices[0].platform != "gpu" or (memory := _memory_limit(devices)) is None:
        return DEFAULT_BLOCK_SIZE
    budget = int(memory * _MEMORY_FRACTION)
    paths = max(1, budget // estimated_path_bytes(product, sim_dim, dtype, settings.enable_aad))
    power = 1 << (paths.bit_length() - 1)
    return min(_MAX_GPU_BLOCK, max(_MIN_GPU_BLOCK, power))
