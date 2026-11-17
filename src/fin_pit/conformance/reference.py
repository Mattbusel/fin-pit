"""Reference implementations behind the conformance vectors.

Plain Python, one loop per definition, float64 throughout, no numpy. They are written for
reading, not speed: each function is the formula in SPEC.md, and the test suite checks them
against TA-Lib, pandas-ta-classic and fin-primitives for the variants those libraries
implement. Warmup positions hold NaN; ``LOOKBACK`` gives how many.

Every function takes the input series as lists of floats and returns a dict of output name
to list of floats.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence

NAN = float("nan")
Series = List[float]
Out = Dict[str, Series]

__all__ = ["FUNCTIONS", "NAN"]


def _nan(n: int) -> Series:
    return [NAN] * n


def _isnan(x: float) -> bool:
    return x != x


# --- building blocks ---------------------------------------------------------------------


def _sma(x: Sequence[float], n: int, start: int = 0) -> Series:
    """SMA over x[start:], aligned to x; first value at start + n - 1."""
    out = _nan(len(x))
    for i in range(start + n - 1, len(x)):
        out[i] = math.fsum(x[i - n + 1 : i + 1]) / n
    return out


def _ema_seeded(x: Sequence[float], n: int, alpha: float, start: int = 0) -> Series:
    """EMA with an SMA seed over x[start:start+n]; first value at start + n - 1."""
    out = _nan(len(x))
    if len(x) - start < n:
        return out
    prev = math.fsum(x[start : start + n]) / n
    out[start + n - 1] = prev
    for i in range(start + n, len(x)):
        prev = prev + alpha * (x[i] - prev)
        out[i] = prev
    return out


def _ema(x: Sequence[float], n: int, seed: str = "sma", start: int = 0) -> Series:
    alpha = 2.0 / (n + 1)
    if seed == "sma":
        return _ema_seeded(x, n, alpha, start)
    out = _nan(len(x))
    if len(x) - start <= 0:
        return out
    if seed == "first":
        prev = x[start]
        out[start] = prev
        for i in range(start + 1, len(x)):
            prev = prev + alpha * (x[i] - prev)
            out[i] = prev
        return out
    if seed == "adjusted":
        # pandas ewm(span=n, adjust=True): weighted mean with weights (1 - alpha)^k
        return _adjusted(x, alpha, start)
    raise ValueError(f"unknown EMA seed {seed!r}")


def _adjusted(x: Sequence[float], alpha: float, start: int = 0) -> Series:
    """pandas ``ewm(alpha=alpha, adjust=True).mean()`` over x[start:]."""
    out = _nan(len(x))
    num = 0.0
    den = 0.0
    for i in range(start, len(x)):
        num = x[i] + (1.0 - alpha) * num
        den = 1.0 + (1.0 - alpha) * den
        out[i] = num / den
    return out


def _wilder(x: Sequence[float], n: int, start: int = 0) -> Series:
    return _ema_seeded(x, n, 1.0 / n, start)


def _wma(x: Sequence[float], n: int, start: int = 0) -> Series:
    out = _nan(len(x))
    denom = n * (n + 1) / 2.0
    for i in range(start + n - 1, len(x)):
        out[i] = math.fsum(x[i - n + 1 + k] * (k + 1) for k in range(n)) / denom
    return out


def _first_valid(x: Sequence[float]) -> int:
    for i, v in enumerate(x):
        if not _isnan(v):
            return i
    return len(x)


def _blank(out: Series, upto: int) -> Series:
    for i in range(min(upto, len(out))):
        out[i] = NAN
    return out


def _stdev(x: Sequence[float], n: int, ddof: int) -> Series:
    out = _nan(len(x))
    for i in range(n - 1, len(x)):
        w = x[i - n + 1 : i + 1]
        m = math.fsum(w) / n
        out[i] = math.sqrt(max(math.fsum((v - m) ** 2 for v in w) / (n - ddof), 0.0))
    return out


def _var(x: Sequence[float], n: int, ddof: int) -> Series:
    out = _nan(len(x))
    for i in range(n - 1, len(x)):
        w = x[i - n + 1 : i + 1]
        m = math.fsum(w) / n
        out[i] = math.fsum((v - m) ** 2 for v in w) / (n - ddof)
    return out


def _tr(h: Sequence[float], l: Sequence[float], c: Sequence[float]) -> Series:
    out = _nan(len(c))
    for i in range(1, len(c)):
        out[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    return out


# What a 0/0 or x/0 becomes. The canonical vectors use TA-Lib's value (0, or 50 for RSI);
# the generator also runs with NaN to find the positions where the ``zero_division="nan"``
# variant may read NaN.
_ZERO_DIVISION = [0.0]


def _z(value: float = 0.0) -> float:
    return value if _ZERO_DIVISION[0] == 0.0 else NAN


# --- overlap studies -----------------------------------------------------------------------


def sma(close: Series, length: int = 30) -> Out:
    return {"sma": _sma(close, length)}


def ema(close: Series, length: int = 30, seed: str = "sma") -> Out:
    out = _ema(close, length, seed)
    return {"ema": _blank(out, length - 1)}


def wma(close: Series, length: int = 30) -> Out:
    return {"wma": _wma(close, length)}


def _ema_chain(close: Sequence[float], length: int, depth: int, seed: str) -> List[Series]:
    """EMA, EMA of EMA, ... ``depth`` deep. ``seed="sma"``: each stage SMA-seeded on its own
    first full window (TA-Lib); ``seed="adjusted"``: pandas ``ewm(span, adjust=True)`` of
    each stage from the first bar (stockstats)."""
    if seed not in ("sma", "adjusted"):
        raise ValueError(f"unknown seed {seed!r}")
    chain: List[Series] = []
    x: Sequence[float] = close
    for k in range(depth):
        if seed == "sma":
            x = _ema(x, length, start=k * (length - 1))
        else:
            x = _ema(x, length, "adjusted")
        chain.append(list(x))
    return chain


def dema(close: Series, length: int = 30, seed: str = "sma") -> Out:
    e1, e2 = _ema_chain(close, length, 2, seed)
    out = _nan(len(close))
    for i in range(len(close)):
        if not _isnan(e2[i]):
            out[i] = 2.0 * e1[i] - e2[i]
    return {"dema": _blank(out, 2 * (length - 1))}


def tema(close: Series, length: int = 30, seed: str = "sma") -> Out:
    e1, e2, e3 = _ema_chain(close, length, 3, seed)
    out = _nan(len(close))
    for i in range(len(close)):
        if not _isnan(e3[i]):
            out[i] = 3.0 * e1[i] - 3.0 * e2[i] + e3[i]
    return {"tema": _blank(out, 3 * (length - 1))}


def trima(close: Series, length: int = 30) -> Out:
    # SMA of an SMA: TA-Lib's weights for both odd and even lengths.
    if length % 2 == 1:
        a = b = (length + 1) // 2
    else:
        a = length // 2
        b = a + 1
    first = _sma(close, a)
    out = _sma(first, b, start=a - 1)
    return {"trima": out}


def kama(
    close: Series, length: int = 30, fast: int = 2, slow: int = 30, start: str = "talib", seed: str = "close"
) -> Out:
    """Kaufman adaptive MA. ``seed="close"``: seeded with ``close[length - 1]`` (TA-Lib,
    pandas-ta); ``seed="sma"``: seeded with the SMA of the first ``length`` closes
    (StockCharts, stockstats). ``start="talib"``: first value at index ``length``;
    ``start="pandas_ta"``: the seed itself is emitted at ``length - 1``."""
    if start not in ("talib", "pandas_ta"):
        raise ValueError(f"unknown KAMA start {start!r}")
    if seed not in ("close", "sma"):
        raise ValueError(f"unknown KAMA seed {seed!r}")
    n = length
    out = _nan(len(close))
    if len(close) < n:
        return {"kama": out}
    first = close[n - 1] if seed == "close" else math.fsum(close[:n]) / n
    if start == "pandas_ta":
        out[n - 1] = first
    if len(close) <= n:
        return {"kama": out}
    fsc = 2.0 / (fast + 1)
    ssc = 2.0 / (slow + 1)
    prev = first
    for i in range(n, len(close)):
        change = abs(close[i] - close[i - n])
        vol = math.fsum(abs(close[k] - close[k - 1]) for k in range(i - n + 1, i + 1))
        er = change / vol if vol != 0 else 0.0
        sc = (er * (fsc - ssc) + ssc) ** 2
        prev = prev + sc * (close[i] - prev)
        out[i] = prev
    return {"kama": out}


def hma(close: Series, length: int = 10) -> Out:
    half = int(length / 2)
    root = int(math.sqrt(length))
    w_half = _wma(close, half)
    w_full = _wma(close, length)
    diff = _nan(len(close))
    for i in range(length - 1, len(close)):
        diff[i] = 2.0 * w_half[i] - w_full[i]
    return {"hma": _wma(diff, root, start=length - 1)}


def vwma(close: Series, volume: Series, length: int = 10) -> Out:
    out = _nan(len(close))
    for i in range(length - 1, len(close)):
        pv = math.fsum(close[k] * volume[k] for k in range(i - length + 1, i + 1))
        v = math.fsum(volume[i - length + 1 : i + 1])
        out[i] = pv / v if v != 0 else _z()
    return {"vwma": out}


def midpoint(close: Series, length: int = 14) -> Out:
    out = _nan(len(close))
    for i in range(length - 1, len(close)):
        w = close[i - length + 1 : i + 1]
        out[i] = (max(w) + min(w)) / 2.0
    return {"midpoint": out}


def midprice(high: Series, low: Series, length: int = 14) -> Out:
    out = _nan(len(high))
    for i in range(length - 1, len(high)):
        out[i] = (max(high[i - length + 1 : i + 1]) + min(low[i - length + 1 : i + 1])) / 2.0
    return {"midprice": out}


def bbands(close: Series, length: int = 5, std: float = 2.0, ddof: int = 0) -> Out:
    mid = _sma(close, length)
    sd = _stdev(close, length, ddof)
    upper = _nan(len(close))
    lower = _nan(len(close))
    for i in range(length - 1, len(close)):
        upper[i] = mid[i] + std * sd[i]
        lower[i] = mid[i] - std * sd[i]
    return {"lower": lower, "middle": mid, "upper": upper}


# --- momentum --------------------------------------------------------------------------------


def _gains_losses(close: Sequence[float]):
    g = _nan(len(close))
    l = _nan(len(close))
    for i in range(1, len(close)):
        d = close[i] - close[i - 1]
        g[i] = d if d > 0 else 0.0
        l[i] = -d if d < 0 else 0.0
    return g, l


def rsi(close: Series, length: int = 14, smoothing: str = "wilder") -> Out:
    g, l = _gains_losses(close)
    if smoothing == "wilder":
        ag, al = _wilder(g, length, 1), _wilder(l, length, 1)
    elif smoothing == "ema":
        ag, al = _ema(g, length, "sma", 1), _ema(l, length, "sma", 1)
    elif smoothing == "sma":
        ag, al = _sma(g, length, 1), _sma(l, length, 1)
    elif smoothing == "wilder_adjusted":
        ag, al = _adjusted(g, 1.0 / length, 1), _adjusted(l, 1.0 / length, 1)
    else:
        raise ValueError(f"unknown RSI smoothing {smoothing!r}")
    out = _nan(len(close))
    for i in range(length, len(close)):
        s = ag[i] + al[i]
        out[i] = 100.0 * ag[i] / s if s != 0 else _z(50.0)
    return {"rsi": out}


def cmo(close: Series, length: int = 14, smoothing: str = "wilder") -> Out:
    g, l = _gains_losses(close)
    if smoothing == "wilder":
        ag, al = _wilder(g, length, 1), _wilder(l, length, 1)
    elif smoothing == "sum":
        ag, al = _sma(g, length, 1), _sma(l, length, 1)
    else:
        raise ValueError(f"unknown CMO smoothing {smoothing!r}")
    out = _nan(len(close))
    for i in range(length, len(close)):
        s = ag[i] + al[i]
        out[i] = 100.0 * (ag[i] - al[i]) / s if s != 0 else _z()
    return {"cmo": out}


def mom(close: Series, length: int = 10) -> Out:
    out = _nan(len(close))
    for i in range(length, len(close)):
        out[i] = close[i] - close[i - length]
    return {"mom": out}


def roc(close: Series, length: int = 10) -> Out:
    out = _nan(len(close))
    for i in range(length, len(close)):
        out[i] = (close[i] / close[i - length] - 1.0) * 100.0 if close[i - length] != 0 else _z()
    return {"roc": out}


def macd(
    close: Series, fast: int = 12, slow: int = 26, signal: int = 9, seed: str = "talib", align: str = "all"
) -> Out:
    """MACD line, signal and histogram.

    ``seed="talib"``: both EMAs start at index ``slow - 1``, the fast one seeded with the SMA
    of the ``fast`` closes ending there (TA-Lib). ``seed="independent"``: each EMA seeded
    with the SMA of its own first window (pandas-ta, fin-primitives). ``seed="adjusted"``:
    pandas ``ewm(span, adjust=True)`` from the first bar for all three EMAs (stockstats).
    ``align="all"`` starts every output at ``slow + signal - 2`` (TA-Lib); ``align="each"``
    starts the line at ``slow - 1`` (pandas-ta).
    """
    if fast > slow:
        fast, slow = slow, fast
    n = len(close)
    if seed == "talib":
        e_fast = _ema(close, fast, start=slow - fast)
        e_slow = _ema(close, slow)
    elif seed == "independent":
        e_fast = _ema(close, fast)
        e_slow = _ema(close, slow)
    elif seed == "adjusted":
        e_fast = _ema(close, fast, "adjusted")
        e_slow = _ema(close, slow, "adjusted")
    else:
        raise ValueError(f"unknown MACD seed {seed!r}")
    if align not in ("all", "each"):
        raise ValueError(f"unknown MACD align {align!r}")
    if seed == "adjusted":
        line = [e_fast[i] - e_slow[i] for i in range(n)]
        sig = _ema(line, signal, "adjusted")
    else:
        line = _nan(n)
        for i in range(slow - 1, n):
            line[i] = e_fast[i] - e_slow[i]
        sig = _ema(line, signal, start=slow - 1)
    first = slow + signal - 2
    hist = _nan(n)
    for i in range(n):
        if not _isnan(line[i]) and not _isnan(sig[i]):
            hist[i] = line[i] - sig[i]
    line = _blank(line, first if align == "all" else slow - 1)
    return {"macd": line, "signal": _blank(sig, first), "hist": _blank(hist, first)}


def _ma(x: Sequence[float], n: int, kind: str, start: int = 0) -> Series:
    if kind == "sma":
        return _sma(x, n, start)
    if kind == "ema":
        return _ema(x, n, "sma", start)
    raise ValueError(f"unknown moving average {kind!r}")


def apo(close: Series, fast: int = 12, slow: int = 26, ma: str = "ema") -> Out:
    if fast > slow:
        fast, slow = slow, fast
    f = _ma(close, fast, ma)
    s = _ma(close, slow, ma)
    out = _nan(len(close))
    for i in range(slow - 1, len(close)):
        out[i] = f[i] - s[i]
    return {"apo": out}


def ppo(close: Series, fast: int = 12, slow: int = 26, ma: str = "ema") -> Out:
    if fast > slow:
        fast, slow = slow, fast
    f = _ma(close, fast, ma)
    s = _ma(close, slow, ma)
    out = _nan(len(close))
    for i in range(slow - 1, len(close)):
        out[i] = 100.0 * (f[i] - s[i]) / s[i] if s[i] != 0 else _z()
    return {"ppo": out}


def trix(close: Series, length: int = 30, seed: str = "sma") -> Out:
    e3 = _ema_chain(close, length, 3, seed)[2]
    out = _nan(len(close))
    for i in range(3 * (length - 1) + 1, len(close)):
        out[i] = (e3[i] / e3[i - 1] - 1.0) * 100.0 if e3[i - 1] != 0 else _z()
    return {"trix": out}


def willr(high: Series, low: Series, close: Series, length: int = 14) -> Out:
    out = _nan(len(close))
    for i in range(length - 1, len(close)):
        hh = max(high[i - length + 1 : i + 1])
        ll = min(low[i - length + 1 : i + 1])
        out[i] = -100.0 * (hh - close[i]) / (hh - ll) if hh != ll else _z()
    return {"willr": out}


def cci(high: Series, low: Series, close: Series, length: int = 14) -> Out:
    tp = [(h + l + c) / 3.0 for h, l, c in zip(high, low, close)]
    out = _nan(len(close))
    for i in range(length - 1, len(close)):
        w = tp[i - length + 1 : i + 1]
        m = math.fsum(w) / length
        md = math.fsum(abs(v - m) for v in w) / length
        out[i] = (tp[i] - m) / (0.015 * md) if md != 0 else _z()
    return {"cci": out}


def _fastk(high, low, close, k):
    out = _nan(len(close))
    for i in range(k - 1, len(close)):
        hh = max(high[i - k + 1 : i + 1])
        ll = min(low[i - k + 1 : i + 1])
        out[i] = 100.0 * (close[i] - ll) / (hh - ll) if hh != ll else _z()
    return out


def stoch(
    high: Series, low: Series, close: Series, fastk: int = 5, slowk: int = 3, slowd: int = 3, align: str = "all"
) -> Out:
    """Slow stochastic with SMA smoothing. ``align="all"`` starts %K with %D (TA-Lib);
    ``align="each"`` starts %K as soon as it exists (pandas-ta)."""
    if align not in ("all", "each"):
        raise ValueError(f"unknown STOCH align {align!r}")
    fk = _fastk(high, low, close, fastk)
    sk = _sma(fk, slowk, fastk - 1)
    sd = _sma(sk, slowd, fastk + slowk - 2)
    first = fastk + slowk + slowd - 3
    return {"slowk": _blank(sk, first) if align == "all" else sk, "slowd": sd}


def stochf(high: Series, low: Series, close: Series, fastk: int = 5, fastd: int = 3, align: str = "all") -> Out:
    """Fast stochastic; ``align`` as in ``stoch``."""
    if align not in ("all", "each"):
        raise ValueError(f"unknown STOCHF align {align!r}")
    fk = _fastk(high, low, close, fastk)
    fd = _sma(fk, fastd, fastk - 1)
    first = fastk + fastd - 2
    return {"fastk": _blank(fk, first) if align == "all" else fk, "fastd": fd}


def ultosc(high: Series, low: Series, close: Series, p1: int = 7, p2: int = 14, p3: int = 28) -> Out:
    bp = _nan(len(close))
    tr = _nan(len(close))
    for i in range(1, len(close)):
        tl = min(low[i], close[i - 1])
        bp[i] = close[i] - tl
        tr[i] = max(high[i], close[i - 1]) - tl
    longest = max(p1, p2, p3)
    out = _nan(len(close))
    for i in range(longest, len(close)):
        parts = []
        for p in (p1, p2, p3):
            b = math.fsum(bp[i - p + 1 : i + 1])
            t = math.fsum(tr[i - p + 1 : i + 1])
            parts.append(b / t if t != 0 else _z())
        out[i] = 100.0 * (4.0 * parts[0] + 2.0 * parts[1] + parts[2]) / 7.0
    return {"ultosc": out}


def aroon(high: Series, low: Series, length: int = 14) -> Out:
    """Window of ``length + 1`` bars; ties go to the most recent bar (TA-Lib)."""
    up = _nan(len(high))
    down = _nan(len(high))
    for i in range(length, len(high)):
        hi_k = i - length
        lo_k = i - length
        for k in range(i - length, i + 1):
            if high[k] >= high[hi_k]:
                hi_k = k
            if low[k] <= low[lo_k]:
                lo_k = k
        up[i] = 100.0 * (length - (i - hi_k)) / length
        down[i] = 100.0 * (length - (i - lo_k)) / length
    return {"aroondown": down, "aroonup": up}


def aroonosc(high: Series, low: Series, length: int = 14) -> Out:
    a = aroon(high, low, length)
    out = _nan(len(high))
    for i in range(length, len(high)):
        out[i] = a["aroonup"][i] - a["aroondown"][i]
    return {"aroonosc": out}


def bop(open: Series, high: Series, low: Series, close: Series) -> Out:
    return {"bop": [(c - o) / (h - l) if h != l else _z() for o, h, l, c in zip(open, high, low, close)]}


def mfi(high: Series, low: Series, close: Series, volume: Series, length: int = 14) -> Out:
    tp = [(h + l + c) / 3.0 for h, l, c in zip(high, low, close)]
    pos = _nan(len(close))
    neg = _nan(len(close))
    for i in range(1, len(close)):
        flow = tp[i] * volume[i]
        pos[i] = flow if tp[i] > tp[i - 1] else 0.0
        neg[i] = flow if tp[i] < tp[i - 1] else 0.0
    out = _nan(len(close))
    for i in range(length, len(close)):
        p = math.fsum(pos[i - length + 1 : i + 1])
        n = math.fsum(neg[i - length + 1 : i + 1])
        out[i] = 100.0 * p / (p + n) if (p + n) != 0 else _z()
    return {"mfi": out}


# --- directional movement (Wilder) ------------------------------------------------------------


def _dm(high, low, close, length):
    """Wilder-smoothed +DM, -DM and TR as TA-Lib runs them: sums seeded over the first
    ``length - 1`` changes, then ``prev - prev / length + today``."""
    n = len(close)
    pdm = [0.0] * n
    mdm = [0.0] * n
    tr = [0.0] * n
    for i in range(1, n):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        mdm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    s_p = _nan(n)
    s_m = _nan(n)
    s_t = _nan(n)
    if n <= length:
        return s_p, s_m, s_t
    p = math.fsum(pdm[1:length])
    m = math.fsum(mdm[1:length])
    t = math.fsum(tr[1:length])
    for i in range(length, n):
        p = p - p / length + pdm[i]
        m = m - m / length + mdm[i]
        t = t - t / length + tr[i]
        s_p[i], s_m[i], s_t[i] = p, m, t
    return s_p, s_m, s_t


def plus_di(high: Series, low: Series, close: Series, length: int = 14) -> Out:
    p, _, t = _dm(high, low, close, length)
    return {"plus_di": [NAN if _isnan(a) else (100.0 * a / b if b != 0 else _z()) for a, b in zip(p, t)]}


def minus_di(high: Series, low: Series, close: Series, length: int = 14) -> Out:
    _, m, t = _dm(high, low, close, length)
    return {"minus_di": [NAN if _isnan(a) else (100.0 * a / b if b != 0 else _z()) for a, b in zip(m, t)]}


def dx(high: Series, low: Series, close: Series, length: int = 14) -> Out:
    p, m, t = _dm(high, low, close, length)
    out = _nan(len(close))
    for i in range(length, len(close)):
        if t[i] == 0:
            out[i] = _z()
            continue
        pdi = 100.0 * p[i] / t[i]
        mdi = 100.0 * m[i] / t[i]
        s = pdi + mdi
        out[i] = 100.0 * abs(pdi - mdi) / s if s != 0 else _z()
    return {"dx": out}


def adx(high: Series, low: Series, close: Series, length: int = 14) -> Out:
    d = dx(high, low, close, length)["dx"]
    out = _nan(len(close))
    first = 2 * length - 1
    if len(close) <= first:
        return {"adx": out}
    prev = math.fsum(d[length : 2 * length]) / length
    out[first] = prev
    for i in range(first + 1, len(close)):
        prev = (prev * (length - 1) + d[i]) / length
        out[i] = prev
    return {"adx": out}


# --- volatility ------------------------------------------------------------------------------


def trange(high: Series, low: Series, close: Series) -> Out:
    return {"trange": _tr(high, low, close)}


def atr(high: Series, low: Series, close: Series, length: int = 14, smoothing: str = "wilder") -> Out:
    tr = _tr(high, low, close)
    if smoothing == "wilder":
        out = _wilder(tr, length, 1)
    elif smoothing == "sma":
        out = _sma(tr, length, 1)
    elif smoothing == "ema":
        out = _ema(tr, length, "sma", 1)
    elif smoothing == "wilder_adjusted":
        # stockstats: the first bar's true range is its high - low, then pandas
        # ewm(alpha=1/n, adjust=True) from the first bar.
        tr0 = list(tr)
        if tr0:
            tr0[0] = high[0] - low[0]
        out = _blank(_adjusted(tr0, 1.0 / length, 0), length)
    else:
        raise ValueError(f"unknown ATR smoothing {smoothing!r}")
    return {"atr": out}


def natr(high: Series, low: Series, close: Series, length: int = 14) -> Out:
    a = atr(high, low, close, length)["atr"]
    return {"natr": [NAN if _isnan(v) else (100.0 * v / c if c != 0 else _z()) for v, c in zip(a, close)]}


def stddev(close: Series, length: int = 5, ddof: int = 0) -> Out:
    return {"stddev": _stdev(close, length, ddof)}


def var(close: Series, length: int = 5, ddof: int = 0) -> Out:
    return {"var": _var(close, length, ddof)}


# --- volume ----------------------------------------------------------------------------------


def obv(close: Series, volume: Series, start: str = "volume") -> Out:
    """On-balance volume. ``start="volume"``: the first value is the first bar's volume
    (TA-Lib, pandas-ta); ``start="zero"``: it is 0 (stockstats)."""
    if start not in ("volume", "zero"):
        raise ValueError(f"unknown OBV start {start!r}")
    out = _nan(len(close))
    if not close:
        return {"obv": out}
    acc = volume[0] if start == "volume" else 0.0
    out[0] = acc
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            acc += volume[i]
        elif close[i] < close[i - 1]:
            acc -= volume[i]
        out[i] = acc
    return {"obv": out}


def _ad_line(high, low, close, volume):
    out = []
    acc = 0.0
    for h, l, c, v in zip(high, low, close, volume):
        if h != l:
            acc += ((c - l) - (h - c)) / (h - l) * v
        out.append(acc)
    return out


def ad(high: Series, low: Series, close: Series, volume: Series) -> Out:
    return {"ad": _ad_line(high, low, close, volume)}


def adosc(high: Series, low: Series, close: Series, volume: Series, fast: int = 3, slow: int = 10) -> Out:
    """Chaikin oscillator, TA-Lib form: both EMAs of the A/D line start from its first value
    (no SMA seed); output from index ``slow - 1``."""
    line = _ad_line(high, low, close, volume)
    f = _ema(line, fast, "first")
    s = _ema(line, slow, "first")
    out = _nan(len(close))
    longest = max(fast, slow)
    for i in range(longest - 1, len(close)):
        out[i] = f[i] - s[i]
    return {"adosc": out}


# --- statistics ------------------------------------------------------------------------------


def _linreg_fit(w: Sequence[float]):
    n = len(w)
    sx = n * (n - 1) / 2.0
    sxx = (n - 1) * n * (2 * n - 1) / 6.0
    sy = math.fsum(w)
    sxy = math.fsum(k * v for k, v in enumerate(w))
    d = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / d
    intercept = (sy - slope * sx) / n
    return slope, intercept


def linearreg(close: Series, length: int = 14) -> Out:
    out = _nan(len(close))
    for i in range(length - 1, len(close)):
        slope, b = _linreg_fit(close[i - length + 1 : i + 1])
        out[i] = b + slope * (length - 1)
    return {"linearreg": out}


def linearreg_slope(close: Series, length: int = 14) -> Out:
    out = _nan(len(close))
    for i in range(length - 1, len(close)):
        out[i] = _linreg_fit(close[i - length + 1 : i + 1])[0]
    return {"linearreg_slope": out}


FUNCTIONS: Dict[str, Callable[..., Out]] = {
    f.__name__: f
    for f in (
        sma, ema, wma, dema, tema, trima, kama, hma, vwma, midpoint, midprice, bbands,
        rsi, cmo, mom, roc, macd, apo, ppo, trix, willr, cci, stoch, stochf, ultosc,
        aroon, aroonosc, bop, mfi, plus_di, minus_di, dx, adx,
        trange, atr, natr, stddev, var, obv, ad, adosc, linearreg, linearreg_slope,
    )
}
