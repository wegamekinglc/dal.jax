"""Model protocol, aligned with DAL's ``Model_<T>``."""

import math
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple, Protocol, runtime_checkable

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from dal_jax.errors import InvalidModelTimeline, UnsupportedModelObservation
from dal_jax.index import Index

type ModelParams = Mapping[str, ArrayLike]


@dataclass(frozen=True, slots=True, kw_only=True)
class SampleDef:
    """DAL's ``SampleDef_``: what a product needs from the model on one date."""

    numeraire: bool = True
    index_names: tuple[str, ...] = ()
    discount_mats: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "index_names", tuple(self.index_names))
        object.__setattr__(
            self, "discount_mats", tuple(float(value) for value in self.discount_mats)
        )


class Sample(NamedTuple):
    """DAL's ``Sample_``: the model outputs on one product date."""

    spot: Array
    numeraire: Array
    observations: Array  # [max_observations]
    discounts: Array  # [max_discounts]


class Scenario(NamedTuple):
    """Single-path arrays with padded observation and discount slots."""

    spot: Array  # [n_samples]
    numeraire: Array  # [n_samples]
    observations: Array  # [n_samples, max_observations]
    discounts: Array  # [n_samples, max_discounts]

    def samples(self) -> tuple[Sample, ...]:
        """Per-date views using one unstack per field to avoid repeated reverse-mode padding."""
        fields = [
            jnp.unstack(x, axis=0)
            for x in (self.spot, self.numeraire, self.observations, self.discounts)
        ]
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

    def supports_index(self, index: Index) -> bool: ...

    def allocate(self, timeline: Sequence[float], sample_defs: Sequence[SampleDef]): ...

    def sim_dim(self, plan) -> int: ...

    def init(self, params: ModelParams, plan): ...

    @abstractmethod
    def generate(self, state, plan, normals: Array) -> Scenario: ...


def validate_timeline(
    model: Model, timeline: Sequence[float], sample_defs: Sequence[SampleDef]
) -> None:
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
    if not all(
        math.isfinite(maturity) and maturity >= time for maturity in definition.discount_mats
    ):
        raise InvalidModelTimeline(
            "discount maturities must be finite and not precede their sample"
        )
    if definition.discount_mats and not model.supports_discount_factors:
        raise UnsupportedModelObservation("model does not provide discount factors")
    if len(definition.index_names) > model.max_output_slots_per_sample:
        raise UnsupportedModelObservation("too many outputs per sample")
    observed.update(definition.index_names)
    if len(observed) > model.max_observed_indices:
        raise UnsupportedModelObservation("multiple future indices")
