import math
import os
import subprocess
import sys

import pytest

from fin_pit import conformance
from fin_pit.conformance import CATALOG, ConformanceError, reference

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_catalogue_size_and_vectors_present():
    assert len(CATALOG) == 43
    doc = conformance.load()
    assert len(doc["fixtures"]) == 8
    combos = sum(len(conformance.variant_combinations(n)) for n in CATALOG)
    assert len(doc["cases"]) == combos * 8
    assert set(conformance.indicators()) == set(CATALOG)


def test_vectors_file_is_up_to_date():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "gen_vectors.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_vectors_equal_reference_today():
    # The stored vectors are what the reference functions compute now.
    cases = conformance.all_cases()
    assert len(cases) == len(conformance.load()["cases"])
    for case in cases:
        out = reference.FUNCTIONS[case.indicator](*case.args(), **case.params, **case.variant)
        case.check(out, rtol=0, atol=0)


def test_cases_selection():
    assert {c.fixture for c in conformance.cases("rsi")} == set(conformance.fixtures())
    assert all(c.variant == {"smoothing": "wilder"} for c in conformance.cases("rsi"))
    assert all(c.variant == {"smoothing": "ema"} for c in conformance.cases("rsi", smoothing="ema"))
    assert [c.fixture for c in conformance.cases("sma", fixtures=["flat"])] == ["flat"]
    with pytest.raises(ValueError):
        conformance.cases("rsi", smoothing="hull")
    with pytest.raises(ValueError):
        conformance.cases("rsi", seed="sma")
    with pytest.raises(ValueError):
        conformance.cases("nope")


def sma_case():
    return conformance.cases("sma", fixtures=["walk_a"])[0]


def test_check_accepts_forms_and_reports():
    case = sma_case()
    exp = case.expected["sma"]
    case.check(exp)
    case.check({"sma": exp})
    case.check((exp,))
    pd = pytest.importorskip("pandas")
    case.check(pd.DataFrame({"x": exp}))
    case.check(pd.DataFrame({"SMA_20": exp}), outputs={"sma": "SMA_20"})
    bad = list(exp)
    bad[100] += 1e-3
    with pytest.raises(ConformanceError, match=r"sma\[100\]"):
        case.check(bad)
    padded = list(exp)
    padded[0] = 1.0
    with pytest.raises(ConformanceError, match="warmup"):
        case.check(padded)
    case.check(padded, warmup="partial")
    with pytest.raises(ConformanceError, match="length"):
        case.check(exp[:-1])
    with pytest.raises(ConformanceError, match="NaN"):
        case.check(exp[:-1] + [math.nan])
    with pytest.raises(ValueError):
        case.check(exp, warmup="pad")
    with pytest.raises(ValueError):
        case.check(exp, zero_division="inf")


def test_zero_division_variants():
    case = conformance.cases("willr", fixtures=["flat"])[0]
    assert case.zero_div["willr"]
    exp = case.expected["willr"]
    with_nan = [math.nan if i in case.zero_div["willr"] else v for i, v in enumerate(exp)]
    with pytest.raises(ConformanceError):
        case.check(with_nan)
    case.check(with_nan, zero_division="nan")
    rsi = conformance.cases("rsi", fixtures=["flat"])[0]
    zeros = [0.0 if i in rsi.zero_div["rsi"] else v for i, v in enumerate(rsi.expected["rsi"])]
    with pytest.raises(ConformanceError):
        rsi.check(zeros)
    rsi.check(zeros, zero_division=0.0)


def test_multi_output_requires_all():
    case = conformance.cases("bbands", fixtures=["walk_a"])[0]
    with pytest.raises(ConformanceError, match="3 outputs"):
        case.check(case.expected["upper"])
    with pytest.raises(ConformanceError, match="missing"):
        case.check({"upper": case.expected["upper"]})


def test_results_reports_exceptions():
    def boom(case):
        raise RuntimeError("no")

    rep = conformance.results(boom, "sma")
    assert all(v and "RuntimeError" in v[0] for v in rep.values())
    ok = conformance.results(lambda c: c.expected["sma"], "sma")
    assert not any(ok.values())


@conformance.parametrize("ema", seed="adjusted")
def test_parametrize_helper(pit_case):
    assert pit_case.variant == {"seed": "adjusted"}
    out = reference.ema(pit_case.inputs["close"], 20, seed="adjusted")
    pit_case.check(out)


def test_short_fixture_is_all_warmup_for_long_lookbacks():
    for case in conformance.cases("adx", fixtures=["short"]):
        assert all(v != v for v in case.expected["adx"])
