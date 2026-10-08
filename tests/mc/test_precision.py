"""Precision-specific fuzzy endpoint safety and unchanged compatibility defaults."""

import pytest
from script_cases import EVALUATION

from dal_jax.api import BSModelData_New, MonteCarlo_Value, Product_New


@pytest.mark.parametrize("kernel", ["dal", "smoothstep"])
@pytest.mark.parametrize(
    "spot,script",
    [
        (101.0, "IF SPOT()>100:2 THEN pay PAYS SPOT() ELSE pay PAYS LOG(-1) END"),
        (99.0, "IF SPOT()>100:2 THEN pay PAYS LOG(-1) ELSE pay PAYS SPOT() END"),
    ],
)
def test_float32_fuzzy_endpoints_mask_invalid_branches(spot, script, kernel):
    result = MonteCarlo_Value(
        Product_New([EVALUATION], [script]),
        BSModelData_New(spot, 0.0),
        1,
        enable_aad=True,
        dtype="float32",
        smoothing_kernel=kernel,
        parallel="none",
        evaluation_date=EVALUATION,
    )
    assert result["PV"] == spot and result["d_spot"] == 1.0
