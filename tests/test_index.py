"""Ports of DAL's ``tests/indice/parser/*.cpp`` and the naming parts of ``tests/indice/index/*.cpp``."""

import pytest

from dal_jax.errors import DalError, InvalidIndex, UnknownIndex
from dal_jax.index import parse_index


@pytest.mark.parametrize(
    "name",
    ["EQ[IBM]", "EQ[IBM]@2022-01-24", "EQ[IBM]>3M", "EQ[IBM]>1W", "EQ[IBM]>6M", "EQ[IBM]>2Y", "EQ[IBM]>3M&IMM", "FX[USD/JPY]", "FX[EUR/GBP]"],
)
def test_names_round_trip(name):
    index = parse_index(name)
    assert index.name == name and index.kind == name[:2]


def test_canonical_spellings():
    assert parse_index("eq[Aapl]").name == "EQ[Aapl]"
    assert parse_index("fx[eur/usd]").name == "FX[EUR/USD]"
    assert parse_index("EQ[x]@2026-1-5").name == "EQ[x]@2026-01-05"


@pytest.mark.parametrize(
    "name,canonical",
    [("IR:USD,LIBOR3MLCH", "IR:USD,LIBOR_3M_LCH"), ("IR:USD,LIBOR3MLCH,2022-02-03", "IR:USD,LIBOR_3M_LCH,2022-02-03"),
     ("IR:USD,10Y", "IR:USD,10Y"), ("IR:USD,10Y,2022-02-03", "IR:USD,10Y,2022-02-03"), ("IR[DF]:USD,2023-06-15", "IR[DF]:USD,2023-06-15"),
     ("IR[DF]:USD,3M", "IR[DF]:USD,3M"), ("IR[DF]:USD,2022-01-03,2023-06-15", "IR[DF]:USD,2022-01-03,2023-06-15"),
     ("IR[DF]:USD,2027-09-28,2028-09-28", "IR[DF]:USD,2027-09-28,2028-09-28"),
     ("IR[USD,DF,2028-09-28,2027-09-28]", "IR[DF]:USD,2027-09-28,2028-09-28"), ("IR[USD,SWAP,18M]", "IR[USD,SWAP,18M]"),
     ("IR[USD,SWAP,10Y]", "IR:USD,10Y")],
)
def test_interest_rate_names(name, canonical):
    index = parse_index(name)
    assert (index.kind, index.name) == ("IR", canonical)
    assert parse_index(canonical).name == canonical


@pytest.mark.parametrize(
    "name",
    ["EQ[IBM]@not-a-date", "EQ[IBM]@2026-02-30", "EQ[IBM]>invalid", "EQ[IBM]>999999999999999999999M", "EQ[IBM]>3M&", "EQ[IBM]>&3M",
     "EQ[IBM]>3M&&6M"],
)
def test_equity_delivery_errors_keep_the_index(name):
    with pytest.raises(DalError, match="InvalidIndex") as error:
        parse_index(name)
    assert name in str(error.value)


@pytest.mark.parametrize(
    "name",
    ["FXUSD/JPY]", "FX[USD/JPY", "FX[USDJPY]", "EQ[]", "EQ[AAPL", "EQ[[AAPL]]", "EQ[AAPL]]", "EQ[AAPL]junk", "EQ[AAPL]junk>3M",
     "EQ[AAPL]@2026-12-31junk", "EQ[AAPL]>", "EQ[AAPL]>3Mjunk", "EQ[AAPL]@", 'EQ["AAPL"]', "FX[EUR/USD]junk", "FX[[EUR/USD]]",
     "FX[EUR/USD]]", "FX[/USD]", "FX[EUR/]", "FX[EUR/USD/JPY]", "FX[EUR/USD]>3M", "FX[EUR/XYZ]", "IR:USD,3M", "IR:USD,,10Y"],
)
def test_malformed_names_are_rejected(name):
    with pytest.raises(DalError):
        parse_index(name)


def test_unknown_prefix_and_bare_names():
    with pytest.raises(UnknownIndex, match="no parser for 'XX\\[ABC\\]'"):
        parse_index("XX[ABC]")
    for name in ("IBM", ""):
        with pytest.raises(InvalidIndex, match="no index parsed"):
            parse_index(name)
