from dal_jax import SampleDef
from dal_jax.models.gsrslv import GSRSLVSettings
from dal_jax.script.product import ScriptProductSettings


def test_public_settings_snapshot_input_sequences():
    indices, maturities, features, correlations = ["EQ[A]"], [1.], ["SPOT"], [.2]
    sample = SampleDef(index_names=indices, discount_mats=maturities)
    product = ScriptProductSettings(regression_features=features)
    rates = GSRSLVSettings(variance_correlations=correlations)
    for values in (indices, maturities, features, correlations):
        values.clear()
    assert sample.index_names == ("EQ[A]",) and sample.discount_mats == (1.,)
    assert product.regression_features == ("SPOT",)
    assert rates.variance_correlations == (.2,)
    assert all(isinstance(hash(value), int) for value in (sample, product, rates))
