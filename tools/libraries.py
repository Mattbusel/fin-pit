"""Adapters that run TA-Lib, pandas-ta-classic, fin-primitives and stockstats on conformance
cases, with the variant each library declares. Used by tests/test_conformance_libraries.py
and tools/conformance_report.py.

Each entry: indicator -> (callable(case) -> outputs, declaration dict). The declaration holds
the variant axes plus ``warmup`` and ``zero_division``. A library missing from the
environment is skipped.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Tuple

Adapter = Tuple[Callable[[Any], Any], Dict[str, Any]]


def _np(case, *names):
    import numpy as np

    return [np.asarray(case.inputs[n], dtype=float) for n in names]


def _df(case):
    import pandas as pd

    return pd.DataFrame({k: list(v) for k, v in case.inputs.items()})


def talib_adapters() -> Dict[str, Adapter]:
    import talib

    P = lambda c, k: c.params[k]  # noqa: E731
    d = lambda **kw: dict(kw)  # noqa: E731
    return {
        "sma": (lambda c: talib.SMA(*_np(c, "close"), P(c, "length")), d()),
        "ema": (lambda c: talib.EMA(*_np(c, "close"), P(c, "length")), d(seed="sma")),
        "wma": (lambda c: talib.WMA(*_np(c, "close"), P(c, "length")), d()),
        "dema": (lambda c: talib.DEMA(*_np(c, "close"), P(c, "length")), d()),
        "tema": (lambda c: talib.TEMA(*_np(c, "close"), P(c, "length")), d()),
        "trima": (lambda c: talib.TRIMA(*_np(c, "close"), P(c, "length")), d()),
        "kama": (lambda c: talib.KAMA(*_np(c, "close"), P(c, "length")), d(start="talib", seed="close")),
        "midpoint": (lambda c: talib.MIDPOINT(*_np(c, "close"), P(c, "length")), d()),
        "midprice": (lambda c: talib.MIDPRICE(*_np(c, "high", "low"), P(c, "length")), d()),
        "bbands": (lambda c: tuple(reversed(talib.BBANDS(*_np(c, "close"), P(c, "length"), P(c, "std"), P(c, "std"), 0))), d(ddof=0)),
        "rsi": (lambda c: talib.RSI(*_np(c, "close"), P(c, "length")), d(smoothing="wilder")),
        "cmo": (lambda c: talib.CMO(*_np(c, "close"), P(c, "length")), d(smoothing="wilder")),
        "mom": (lambda c: talib.MOM(*_np(c, "close"), P(c, "length")), d()),
        "roc": (lambda c: talib.ROC(*_np(c, "close"), P(c, "length")), d()),
        "macd": (lambda c: talib.MACD(*_np(c, "close"), P(c, "fast"), P(c, "slow"), P(c, "signal")), d(seed="talib", align="all")),
        "apo": (lambda c: talib.APO(*_np(c, "close"), P(c, "fast"), P(c, "slow"), 1), d(ma="ema")),
        "ppo": (lambda c: talib.PPO(*_np(c, "close"), P(c, "fast"), P(c, "slow"), 1), d(ma="ema")),
        "trix": (lambda c: talib.TRIX(*_np(c, "close"), P(c, "length")), d()),
        "willr": (lambda c: talib.WILLR(*_np(c, "high", "low", "close"), P(c, "length")), d()),
        "cci": (lambda c: talib.CCI(*_np(c, "high", "low", "close"), P(c, "length")), d()),
        "stoch": (lambda c: talib.STOCH(*_np(c, "high", "low", "close"), P(c, "fastk"), P(c, "slowk"), 0, P(c, "slowd"), 0), d(align="all")),
        "stochf": (lambda c: talib.STOCHF(*_np(c, "high", "low", "close"), P(c, "fastk"), P(c, "fastd"), 0), d(align="all")),
        "ultosc": (lambda c: talib.ULTOSC(*_np(c, "high", "low", "close"), P(c, "p1"), P(c, "p2"), P(c, "p3")), d()),
        "aroon": (lambda c: talib.AROON(*_np(c, "high", "low"), P(c, "length")), d()),
        "aroonosc": (lambda c: talib.AROONOSC(*_np(c, "high", "low"), P(c, "length")), d()),
        "bop": (lambda c: talib.BOP(*_np(c, "open", "high", "low", "close")), d()),
        "mfi": (lambda c: talib.MFI(*_np(c, "high", "low", "close", "volume"), P(c, "length")), d()),
        "plus_di": (lambda c: talib.PLUS_DI(*_np(c, "high", "low", "close"), P(c, "length")), d()),
        "minus_di": (lambda c: talib.MINUS_DI(*_np(c, "high", "low", "close"), P(c, "length")), d()),
        "dx": (lambda c: talib.DX(*_np(c, "high", "low", "close"), P(c, "length")), d()),
        "adx": (lambda c: talib.ADX(*_np(c, "high", "low", "close"), P(c, "length")), d()),
        "trange": (lambda c: talib.TRANGE(*_np(c, "high", "low", "close")), d()),
        "atr": (lambda c: talib.ATR(*_np(c, "high", "low", "close"), P(c, "length")), d(smoothing="wilder")),
        "natr": (lambda c: talib.NATR(*_np(c, "high", "low", "close"), P(c, "length")), d()),
        "stddev": (lambda c: talib.STDDEV(*_np(c, "close"), P(c, "length"), 1.0), d(ddof=0)),
        "var": (lambda c: talib.VAR(*_np(c, "close"), P(c, "length"), 1.0), d(ddof=0)),
        "obv": (lambda c: talib.OBV(*_np(c, "close", "volume")), d(start="volume")),
        "ad": (lambda c: talib.AD(*_np(c, "high", "low", "close", "volume")), d()),
        "adosc": (lambda c: talib.ADOSC(*_np(c, "high", "low", "close", "volume"), P(c, "fast"), P(c, "slow")), d()),
        "linearreg": (lambda c: talib.LINEARREG(*_np(c, "close"), P(c, "length")), d()),
        "linearreg_slope": (lambda c: talib.LINEARREG_SLOPE(*_np(c, "close"), P(c, "length")), d()),
    }


def _pandas_style(accessor: Callable[[Any], Any], **extra: Any) -> Dict[str, Adapter]:
    """pandas-ta's method names and defaults; shared by pandas-ta-classic and df.fp."""
    P = lambda c, k: c.params[k]  # noqa: E731
    d = lambda **kw: dict(kw)  # noqa: E731
    A = lambda c: accessor(_df(c))  # noqa: E731

    def cols(frame, *idx):
        return tuple(frame.iloc[:, i] for i in idx)

    out = {
        "sma": (lambda c: A(c).sma(P(c, "length")), d()),
        "ema": (lambda c: A(c).ema(P(c, "length")), d(seed="sma")),
        "wma": (lambda c: A(c).wma(P(c, "length")), d()),
        "dema": (lambda c: A(c).dema(P(c, "length")), d()),
        "tema": (lambda c: A(c).tema(P(c, "length")), d()),
        "trima": (lambda c: A(c).trima(P(c, "length")), d()),
        "kama": (lambda c: A(c).kama(P(c, "length"), P(c, "fast"), P(c, "slow")), d(start="pandas_ta", seed="close")),
        "hma": (lambda c: A(c).hma(P(c, "length")), d()),
        "vwma": (lambda c: A(c).vwma(P(c, "length")), d()),
        "midpoint": (lambda c: A(c).midpoint(P(c, "length")), d()),
        "midprice": (lambda c: A(c).midprice(P(c, "length")), d()),
        "bbands": (lambda c: cols(A(c).bbands(P(c, "length"), P(c, "std")), 0, 1, 2), d(ddof=0)),
        "rsi": (lambda c: A(c).rsi(P(c, "length")), d(smoothing="wilder", zero_division=0.0)),
        "cmo": (lambda c: A(c).cmo(P(c, "length")), d(smoothing="sum")),
        "mom": (lambda c: A(c).mom(P(c, "length")), d()),
        "roc": (lambda c: A(c).roc(P(c, "length")), d()),
        "macd": (lambda c: cols(A(c).macd(P(c, "fast"), P(c, "slow"), P(c, "signal")), 0, 2, 1), d(seed="independent", align="each")),
        "apo": (lambda c: A(c).apo(P(c, "fast"), P(c, "slow")), d(ma="sma")),
        "ppo": (lambda c: cols(A(c).ppo(P(c, "fast"), P(c, "slow")), 0), d(ma="sma")),
        "trix": (lambda c: cols(A(c).trix(P(c, "length")), 0), d()),
        "willr": (lambda c: A(c).willr(P(c, "length")), d()),
        "cci": (lambda c: A(c).cci(P(c, "length")), d()),
        "stoch": (lambda c: cols(A(c).stoch(P(c, "fastk"), P(c, "slowd"), P(c, "slowk")), 0, 1), d(align="each")),
        "ultosc": (lambda c: A(c).uo(P(c, "p1"), P(c, "p2"), P(c, "p3")), d()),
        "aroon": (lambda c: cols(A(c).aroon(P(c, "length")), 0, 1), d()),
        "aroonosc": (lambda c: cols(A(c).aroon(P(c, "length")), 2), d()),
        "bop": (lambda c: A(c).bop(), d()),
        "mfi": (lambda c: A(c).mfi(P(c, "length")), d()),
        "plus_di": (lambda c: cols(A(c).adx(P(c, "length")), 1), d()),
        "minus_di": (lambda c: cols(A(c).adx(P(c, "length")), 2), d()),
        "adx": (lambda c: cols(A(c).adx(P(c, "length")), 0), d()),
        "trange": (lambda c: A(c).true_range(), d()),
        "atr": (lambda c: A(c).atr(P(c, "length")), d(smoothing="wilder")),
        "natr": (lambda c: A(c).natr(P(c, "length")), d()),
        "stddev": (lambda c: A(c).stdev(P(c, "length")), d(ddof=0)),
        "var": (lambda c: A(c).variance(P(c, "length")), d(ddof=0)),
        "obv": (lambda c: A(c).obv(), d(start="volume")),
        "ad": (lambda c: A(c).ad(), d()),
        "adosc": (lambda c: A(c).adosc(P(c, "fast"), P(c, "slow")), d()),
        "linearreg": (lambda c: A(c).linreg(P(c, "length")), d()),
    }
    for k, v in extra.items():
        out[k] = v
    return out


def pandas_ta_adapters() -> Dict[str, Adapter]:
    import functools

    import pandas_ta_classic  # noqa: F401  (registers df.ta)

    class NoTalib:
        def __init__(self, frame):
            self._acc = frame.ta

        def __getattr__(self, name):
            return functools.partial(getattr(self._acc, name), talib=False)

    out = _pandas_style(NoTalib)
    P = lambda c, k: c.params[k]  # noqa: E731
    out["linearreg_slope"] = (lambda c: NoTalib(_df(c)).linreg(P(c, "length"), slope=True), {})
    return out


def fin_primitives_adapters() -> Dict[str, Adapter]:
    import fin_primitives.pandas  # noqa: F401  (registers df.fp)

    return _pandas_style(lambda frame: frame.fp)


def stockstats_adapters() -> Dict[str, Adapter]:
    from stockstats import wrap

    P = lambda c, k: c.params[k]  # noqa: E731

    def S(c, key):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return wrap(_df(c))[key]

    def boll(c):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = wrap(_df(c))
            # stockstats' boll is fixed at 20 bars, 2 standard deviations (the catalogue's).
            return frame["boll_lb"], frame["boll"], frame["boll_ub"]

    def macd(c):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = wrap(_df(c))
            return frame["macd"], frame["macds"], frame["macdh"]

    partial = dict(warmup="partial")
    return {
        "sma": (lambda c: S(c, f"close_{P(c, 'length')}_sma"), dict(partial)),
        "ema": (lambda c: S(c, f"close_{P(c, 'length')}_ema"), dict(partial, seed="adjusted")),
        "rsi": (lambda c: S(c, f"rsi_{P(c, 'length')}"), dict(partial, smoothing="wilder_adjusted")),
        "bbands": (boll, dict(partial, ddof=0)),
        "macd": (macd, dict(partial, seed="adjusted", align="each")),
        "cci": (lambda c: S(c, f"cci_{P(c, 'length')}"), dict(partial)),
        "willr": (lambda c: S(c, f"wr_{P(c, 'length')}"), dict(partial)),
        "roc": (lambda c: S(c, f"close_{P(c, 'length')}_roc"), dict(partial)),
        "cmo": (lambda c: S(c, f"cmo_{P(c, 'length')}"), dict(partial, smoothing="sum")),
        "stddev": (lambda c: S(c, f"close_{P(c, 'length')}_mstd"), dict(partial, ddof=1)),
        "var": (lambda c: S(c, f"close_{P(c, 'length')}_mvar"), dict(partial, ddof=1)),
        "atr": (lambda c: S(c, f"atr_{P(c, 'length')}"), dict(partial, smoothing="wilder_adjusted")),
        "trange": (lambda c: S(c, "tr"), dict(partial)),
        "mfi": (lambda c: S(c, f"mfi_{P(c, 'length')}"), dict(partial)),
        "trix": (lambda c: S(c, f"close_{P(c, 'length')}_trix"), dict(partial, seed="adjusted")),
        "tema": (lambda c: S(c, f"close_{P(c, 'length')}_tema"), dict(partial, seed="adjusted")),
        "kama": (lambda c: S(c, f"close_{P(c, 'length')},{P(c, 'fast')},{P(c, 'slow')}_kama"), dict(partial, start="pandas_ta", seed="sma")),
        "linearreg": (lambda c: S(c, f"close_{P(c, 'length')}_lrma"), dict(partial)),
        "bop": (lambda c: S(c, "bop"), dict(partial)),
        "wma": (lambda c: S(c, f"close_{P(c, 'length')}_wma"), dict(partial)),
        "hma": (lambda c: S(c, f"close_{P(c, 'length')}_hma"), dict(partial)),
        "obv": (lambda c: S(c, "obv"), dict(partial, start="zero")),
        "aroonosc": (lambda c: S(c, f"aroon_{P(c, 'length')}"), dict(partial)),
    }


# Differences measured on the 2026-11-17 runs that no declared variant covers. Tests mark
# these as expected failures, and docs/conformance-results.md lists them.
KNOWN_DIFFERENCES: Dict[Tuple[str, str], str] = {
    ("pandas-ta-classic", "kama"): "returns no values when the series is shorter than max(length, fast, slow) bars",
    ("fin-primitives", "kama"): "follows pandas-ta: no values when the series is shorter than max(length, fast, slow) bars",
    ("stockstats", "mfi"): "returns MFI on a 0..1 scale instead of 0..100",
}

LIBRARIES: Dict[str, Callable[[], Dict[str, Adapter]]] = {
    "TA-Lib": talib_adapters,
    "pandas-ta-classic": pandas_ta_adapters,
    "fin-primitives": fin_primitives_adapters,
    "stockstats": stockstats_adapters,
}


def run(library: str, indicator: str, adapter: Adapter, rtol=1e-9, atol=1e-9):
    """(declaration, {case id: problems}) for one library/indicator."""
    from fin_pit import conformance

    fn, decl = adapter
    variant = {k: v for k, v in decl.items() if k not in ("warmup", "zero_division")}
    rep = conformance.results(
        fn,
        indicator,
        warmup=decl.get("warmup", "nan"),
        zero_division=decl.get("zero_division", "talib"),
        rtol=rtol,
        atol=atol,
        **variant,
    )
    return decl, rep
