import numpy as np
import pandas as pd
import pytest

import fin_pit as pit
from fin_pit import conformance

pytest.importorskip("fin_primitives")


def ohlcv(n, seed=1):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame(
        {
            "open": o,
            "high": np.maximum(o, c) + 0.5,
            "low": np.minimum(o, c) - 0.5,
            "close": c,
            "volume": rng.uniform(1e3, 1e4, n),
        }
    )


@pytest.mark.parametrize(
    "name,args,ref,ref_params",
    [
        ("sma", (20,), "sma", {"length": 20}),
        ("ema", (20,), "ema", {"length": 20}),
        ("rsi", (14,), "rsi", {"length": 14}),
        ("atr", (14,), "atr", {"length": 14}),
        ("wma", (10,), "wma", {"length": 10}),
        ("dema", (10,), "dema", {"length": 10}),
        ("tema", (10,), "tema", {"length": 10}),
        ("cci", (14,), "cci", {"length": 14}),
        ("willr", (14,), "willr", {"length": 14}),
        ("mfi", (14,), "mfi", {"length": 14}),
        ("adx", (14,), "adx", {"length": 14}),
    ],
)
def test_required_bars_matches_reference_lookback(name, args, ref, ref_params):
    # The measured warmup of fin-primitives equals the spec's warmup for the same indicator.
    from fin_pit.conformance import reference

    f = conformance.fixtures()["walk_a"]["data"]
    inputs = [f[c] for c in conformance.CATALOG[ref]["inputs"]]
    out = reference.FUNCTIONS[ref](*inputs, **ref_params)
    first = max(next(i for i, v in enumerate(vals) if v == v) for vals in out.values())
    assert pit.required_bars(name, *args) == first + 1


def test_tradingagents_1679_short_history():
    # ~45 sessions of history: a 200-bar SMA must not exist (stockstats would return the 45-bar mean).
    df = ohlcv(45)
    out = df.pit.indicator("sma", 200)
    assert out["SMA_200"].isna().all()
    assert out["bars_available"].iloc[-1] == 45
    assert (out["bars_required"] == 200).all()
    fifty = df.pit.indicator("sma", 50)
    assert fifty["SMA_50"].isna().all()
    ok = ohlcv(260).pit.indicator("sma", 200)
    assert ok["SMA_200"].iloc[199:].notna().all() and ok["SMA_200"].iloc[:199].isna().all()
    np.testing.assert_allclose(ok["SMA_200"].iloc[-1], ohlcv(260)["close"].iloc[-200:].mean(), rtol=1e-12)


def test_gap_resets_coverage():
    df = ohlcv(60)
    df.loc[30, "close"] = np.nan
    out = df.pit.indicator("sma", 10)
    assert out["bars_available"].iloc[30] == 0
    assert out["bars_available"].iloc[31] == 1
    assert out["SMA_10"].iloc[31:40].isna().all()
    assert out["SMA_10"].iloc[40:].notna().all()


def test_gap_in_unused_column_is_ignored():
    df = ohlcv(30)
    df.loc[5, "volume"] = np.nan
    out = df.pit.indicator("sma", 5)
    assert out["SMA_5"].iloc[4:].notna().all()


def test_settle_and_multi_output():
    df = ohlcv(80)
    out = df.pit.indicator("rsi", 14, settle=10)
    assert out["bars_required"].iloc[0] == 25
    assert out["RSI_14"].iloc[:24].isna().all() and out["RSI_14"].iloc[24:].notna().all()
    m = df.pit.indicator("macd", 12, 26, 9)
    assert list(m.columns[-2:]) == ["bars_available", "bars_required"]
    assert m.iloc[:33, :3].isna().all().all() and m.iloc[33:, :3].notna().all().all()
    with pytest.raises(ValueError):
        df.pit.indicator("rsi", 14, settle=-1)
    with pytest.raises(ValueError):
        df.pit.indicator("no_such_indicator")


def test_indicator_respects_asof():
    df = ohlcv(40)
    df["available_at"] = pd.date_range("2024-01-01 21:15", periods=40, freq="D", tz="UTC")
    with pit.asof("2024-01-30T00:00Z"):
        out = df.pit.indicator("sma", 5)
    assert len(out) == 29
    assert out.index[-1] == 28


def test_column_override():
    df = ohlcv(30).rename(columns={"close": "adj_close"})
    df.loc[3, "adj_close"] = np.nan
    out = df.pit.indicator("sma", 5, close="adj_close")
    assert out["bars_available"].iloc[3] == 0
    assert pit.required_bars("sma", 5) == 5


def test_series_input_rejected():
    with pytest.raises(TypeError):
        pit.indicator(pd.Series([1.0, 2.0]), "sma", 2)
