"""Model protocol, aligned with DAL's ``Model_<T>``.

DAL's ``Allocate`` / ``Init`` / ``GeneratePath`` split into three stages:

* ``allocate(timeline, sample_defs)`` runs once on the host and returns a
  hashable static plan (time grid, slot layout, capability checks).
* ``init(params, plan)`` runs once per pricing on device, outside the path
  ``vmap``; it holds everything that depends only on parameters, so gradients
  flow through it.
* ``generate(state, plan, normals)`` describes a single path; the engine
  ``vmap``-s it over a block.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple, Protocol, runtime_checkable

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from dal_jax.errors import InvalidModelTimeline, UnsupportedModelObservation

type ModelParams = Mapping[str, ArrayLike]


@dataclass(frozen=True, slots=True, kw_only=True)
class SampleDef:
    """DAL's ``SampleDef_``: what a product needs from the model on one date."""

    numeraire: bool = True
    index_names: tuple[str, ...] = ()
    discount_mats: tuple[float, ...] = ()


class Sample(NamedTuple):
    """DAL's ``Sample_``: the model outputs on one product date."""

    spot: Array
    numeraire: Array
    observations: Array  # [max_observations]
    discounts: Array  # [max_discounts]


class Scenario(NamedTuple):
    """One simulated path.  Ragged per-sample vectors are padded to a fixed width.

    ``observations[i, j]`` is the j-th index requested on sample ``i``;
    ``discounts[i, k]`` is ``P(t_i, discount_mats[k])`` (padding is 1.0, as DAL
    initialises unused discount slots).
    """

    spot: Array  # [n_samples]
    numeraire: Array  # [n_samples]
    observations: Array  # [n_samples, max_observations]
    discounts: Array  # [n_samples, max_discounts]

    def samples(self) -> tuple[Sample, ...]:
        """Per-date views, like DAL's ``Scenario_ = Vector_<Sample_>``.

        Prefer this over repeated ``scenario.spot[i]`` when a payoff reads many
        dates: each static slice transposes to a full-size ``pad`` in reverse
        mode, while one ``unstack`` per field transposes to a single ``stack``
        (an order of magnitude faster on CPU for a 36-date barrier).
        """
        fields = [jnp.unstack(x, axis=0) for x in (self.spot, self.numeraire, self.observations, self.discounts)]
        return tuple(Sample(*row) for row in zip(*fields))


@runtime_checkable
class Model(Protocol):
    param_labels: tuple[str, ...]
    n_factors: int
    numeraire_is_deterministic: bool
    supports_discount_factors: bool
    max_observed_indices: int
    max_output_slots_per_sample: int

    @property
    def supports_bb(self) -> bool: ...

    def default_params(self) -> dict[str, Array]: ...

    def validate_params(self, params: ModelParams) -> None: ...

    def allocate(self, timeline: Sequence[float], sample_defs: Sequence[SampleDef]): ...

    def sim_dim(self, plan) -> int: ...

    def init(self, params: ModelParams, plan): ...

    def generate(self, state, plan, normals: Array) -> Scenario: ...


def validate_timeline(model: Model, timeline: Sequence[float], sample_defs: Sequence[SampleDef]) -> None:
    """``Model_::ValidateTimeline``: increasing non-negative times, sane maturities, index budget."""
    if not timeline or len(timeline) != len(sample_defs):
        raise InvalidModelTimeline("sample definitions must match dates")
    observed: set[str] = set()
    previous = -math.inf
    for time, definition in zip(timeline, sample_defs):
        if not (math.isfinite(time) and time >= 0.0 and time > previous):
            raise InvalidModelTimeline("times must be nonnegative, finite and strictly increasing")
        _validate_sample(model, time, definition, observed)
        previous = time


def _validate_sample(model: Model, time: float, definition: SampleDef, observed: set[str]) -> None:
    """One date's requests; ``observed`` accumulates the distinct index names seen so far."""
    if not all(math.isfinite(maturity) and maturity >= time for maturity in definition.discount_mats):
        raise InvalidModelTimeline("discount maturities must be finite and not precede their sample")
    if definition.discount_mats and not model.supports_discount_factors:
        raise UnsupportedModelObservation("model does not provide discount factors")
    if len(definition.index_names) > model.max_output_slots_per_sample:
        raise UnsupportedModelObservation("too many outputs per sample")
    observed.update(definition.index_names)
    if len(observed) > model.max_observed_indices:
        raise UnsupportedModelObservation("multiple future indices")
