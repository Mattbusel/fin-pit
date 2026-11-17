"""The indicator catalogue behind SPEC.md: inputs, parameters, outputs and named variants.

Each entry's ``axes`` lists the variant axes and their options; the first option is the
canonical one (TA-Lib's behaviour where TA-Lib has the indicator). ``libraries`` records
which variant each library was measured to implement (see docs/conformance-results.md).
Two axes apply to every indicator and are not stored per entry:

``warmup``
    ``nan``: every warmup position must be NaN. ``partial``: warmup positions are not
    compared (a library that computes over the bars it has, like stockstats, declares this).
``zero_division``
    ``talib``: 0/0 and x/0 read TA-Lib's value (0, and 50 for RSI). ``nan``: NaN is
    accepted where the canonical value came from a zero denominator (positions listed in each
    vector's ``zero_div``). A number, for example ``0.0``: that value is accepted there.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

__all__ = ["CATALOG", "GLOBAL_AXES", "SPEC_VERSION"]

SPEC_VERSION = "1.0"

GLOBAL_AXES: Dict[str, Tuple[str, ...]] = {
    "warmup": ("nan", "partial"),
    # "talib": the canonical value; "nan": NaN accepted; a number (for example 0.0) is also
    # accepted and must then be the value at every 0/0 position.
    "zero_division": ("talib", "nan"),
}

C = ("close",)
HLC = ("high", "low", "close")
HL = ("high", "low")
HLCV = ("high", "low", "close", "volume")


def _e(
    inputs: Tuple[str, ...],
    params: Dict[str, Any],
    outputs: Tuple[str, ...],
    title: str,
    axes: Dict[str, Tuple[Any, ...]] | None = None,
    talib: str | None = None,
) -> Dict[str, Any]:
    return {
        "inputs": inputs,
        "params": params,
        "outputs": outputs,
        "title": title,
        "axes": axes or {},
        "talib": talib,
    }


CATALOG: Dict[str, Dict[str, Any]] = {
    # overlap studies
    "sma": _e(C, {"length": 20}, ("sma",), "Simple moving average", talib="SMA"),
    "ema": _e(C, {"length": 20}, ("ema",), "Exponential moving average, alpha = 2/(n+1)",
              {"seed": ("sma", "first", "adjusted")}, "EMA"),
    "wma": _e(C, {"length": 20}, ("wma",), "Linearly weighted moving average", talib="WMA"),
    "dema": _e(C, {"length": 10}, ("dema",), "Double EMA, 2*EMA - EMA(EMA)",
               {"seed": ("sma", "adjusted")}, "DEMA"),
    "tema": _e(C, {"length": 10}, ("tema",), "Triple EMA", {"seed": ("sma", "adjusted")}, "TEMA"),
    "trima": _e(C, {"length": 10}, ("trima",), "Triangular moving average (SMA of SMA)", talib="TRIMA"),
    "kama": _e(C, {"length": 10, "fast": 2, "slow": 30}, ("kama",), "Kaufman adaptive moving average",
               {"start": ("talib", "pandas_ta"), "seed": ("close", "sma")}, "KAMA"),
    "hma": _e(C, {"length": 10}, ("hma",), "Hull moving average"),
    "vwma": _e(("close", "volume"), {"length": 10}, ("vwma",), "Volume-weighted moving average"),
    "midpoint": _e(C, {"length": 14}, ("midpoint",), "(highest + lowest close) / 2", talib="MIDPOINT"),
    "midprice": _e(HL, {"length": 14}, ("midprice",), "(highest high + lowest low) / 2", talib="MIDPRICE"),
    "bbands": _e(C, {"length": 20, "std": 2.0}, ("lower", "middle", "upper"), "Bollinger bands",
                 {"ddof": (0, 1)}, "BBANDS"),
    # momentum
    "rsi": _e(C, {"length": 14}, ("rsi",), "Relative strength index",
              {"smoothing": ("wilder", "ema", "sma", "wilder_adjusted")}, "RSI"),
    "cmo": _e(C, {"length": 14}, ("cmo",), "Chande momentum oscillator",
              {"smoothing": ("wilder", "sum")}, "CMO"),
    "mom": _e(C, {"length": 10}, ("mom",), "Momentum, close - close[n]", talib="MOM"),
    "roc": _e(C, {"length": 10}, ("roc",), "Rate of change in percent", talib="ROC"),
    "macd": _e(C, {"fast": 12, "slow": 26, "signal": 9}, ("macd", "signal", "hist"), "MACD",
               {"seed": ("talib", "independent", "adjusted"), "align": ("all", "each")}, "MACD"),
    "apo": _e(C, {"fast": 12, "slow": 26}, ("apo",), "Absolute price oscillator",
              {"ma": ("ema", "sma")}, "APO"),
    "ppo": _e(C, {"fast": 12, "slow": 26}, ("ppo",), "Percentage price oscillator",
              {"ma": ("ema", "sma")}, "PPO"),
    "trix": _e(C, {"length": 10}, ("trix",), "1-bar rate of change of a triple EMA",
               {"seed": ("sma", "adjusted")}, "TRIX"),
    "willr": _e(HLC, {"length": 14}, ("willr",), "Williams %R", talib="WILLR"),
    "cci": _e(HLC, {"length": 14}, ("cci",), "Commodity channel index", talib="CCI"),
    "stoch": _e(HLC, {"fastk": 14, "slowk": 3, "slowd": 3}, ("slowk", "slowd"), "Slow stochastic",
                {"align": ("all", "each")}, "STOCH"),
    "stochf": _e(HLC, {"fastk": 14, "fastd": 3}, ("fastk", "fastd"), "Fast stochastic",
                 {"align": ("all", "each")}, "STOCHF"),
    "ultosc": _e(HLC, {"p1": 7, "p2": 14, "p3": 28}, ("ultosc",), "Ultimate oscillator", talib="ULTOSC"),
    "aroon": _e(HL, {"length": 14}, ("aroondown", "aroonup"), "Aroon down / up", talib="AROON"),
    "aroonosc": _e(HL, {"length": 14}, ("aroonosc",), "Aroon oscillator", talib="AROONOSC"),
    "bop": _e(("open", "high", "low", "close"), {}, ("bop",), "Balance of power", talib="BOP"),
    "mfi": _e(HLCV, {"length": 14}, ("mfi",), "Money flow index", talib="MFI"),
    "plus_di": _e(HLC, {"length": 14}, ("plus_di",), "Plus directional indicator", talib="PLUS_DI"),
    "minus_di": _e(HLC, {"length": 14}, ("minus_di",), "Minus directional indicator", talib="MINUS_DI"),
    "dx": _e(HLC, {"length": 14}, ("dx",), "Directional movement index", talib="DX"),
    "adx": _e(HLC, {"length": 14}, ("adx",), "Average directional index", talib="ADX"),
    # volatility
    "trange": _e(HLC, {}, ("trange",), "True range", talib="TRANGE"),
    "atr": _e(HLC, {"length": 14}, ("atr",), "Average true range",
              {"smoothing": ("wilder", "sma", "ema", "wilder_adjusted")}, "ATR"),
    "natr": _e(HLC, {"length": 14}, ("natr",), "Normalised ATR, 100 * ATR / close", talib="NATR"),
    "stddev": _e(C, {"length": 20}, ("stddev",), "Rolling standard deviation", {"ddof": (0, 1)}, "STDDEV"),
    "var": _e(C, {"length": 20}, ("var",), "Rolling variance", {"ddof": (0, 1)}, "VAR"),
    # volume
    "obv": _e(("close", "volume"), {}, ("obv",), "On-balance volume", {"start": ("volume", "zero")}, "OBV"),
    "ad": _e(HLCV, {}, ("ad",), "Chaikin accumulation/distribution line", talib="AD"),
    "adosc": _e(HLCV, {"fast": 3, "slow": 10}, ("adosc",), "Chaikin A/D oscillator", talib="ADOSC"),
    # statistics
    "linearreg": _e(C, {"length": 14}, ("linearreg",), "Linear regression end point", talib="LINEARREG"),
    "linearreg_slope": _e(C, {"length": 14}, ("linearreg_slope",), "Linear regression slope",
                          talib="LINEARREG_SLOPE"),
}


def variant_combinations(name: str) -> List[Dict[str, Any]]:
    """Every combination of ``name``'s variant axes (canonical first)."""
    axes = CATALOG[name]["axes"]
    combos: List[Dict[str, Any]] = [{}]
    for axis, options in axes.items():
        combos = [dict(c, **{axis: o}) for c in combos for o in options]
    return combos


def canonical(name: str) -> Dict[str, Any]:
    """The canonical variant of ``name`` (first option on every axis)."""
    return {axis: options[0] for axis, options in CATALOG[name]["axes"].items()}
