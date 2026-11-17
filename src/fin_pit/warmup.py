"""Strict-warmup indicators: ``df.pit.indicator("sma", 200)``.

An indicator value needs a window of history. Libraries disagree on what to emit before the
window is full: TA-Lib and pandas-ta emit NaN; stockstats computes over whatever bars exist
(its rolling windows use ``min_periods=1``), so a 200-bar SMA on 45 bars is a 45-bar mean
(TradingAgents #1679). A point-in-time backtest also cuts history at the decision time, so
the first months of a run are exactly where short history bites.

``indicator`` computes with fin-primitives (``df.fp``, pandas-ta's formulas in Rust) and then
enforces coverage itself:

``bars_available``
    The number of consecutive bars, ending at this row, whose inputs are all present
    (a missing input resets the count, because the window behind it has a hole).
``bars_required``
    How many bars the indicator needs before its first value, measured from fin-primitives
    on a clean series (see ``required_bars``), plus ``settle`` extra bars if asked.

Every row with ``bars_available < bars_required`` reads NaN, whatever the library emitted.
Inside ``fin_pit.asof`` the frame is first cut to rows available at the decision time.
"""
from __future__ import annotations

import functools
from typing import Any, Dict, Optional, Sequence, Tuple

__all__ = ["indicator", "required_bars", "coverage"]


def _fp() -> Any:
    try:
        import fin_primitives.pandas  # noqa: F401  (registers df.fp)
    except ImportError as exc:  # pragma: no cover
        raise ImportError("fin-pit indicators need fin-primitives: pip install fin-primitives") from exc
    import fin_primitives.pandas as fpp

    return fpp


def _inputs(name: str) -> Tuple[str, ...]:
    fpp = _fp()
    fn = getattr(fpp, "_inputs", None)
    if fn is not None:
        return tuple(fn(name))
    return ("close",)  # pragma: no cover


def _probe_frame(n: int) -> Any:
    import numpy as np
    import pandas as pd

    # A deterministic random walk (64-bit LCG), so the probe never depends on numpy's RNG.
    state = 0x2545F4914F6CDD1D
    steps = np.empty(n)
    for i in range(n):
        state = (6364136223846793005 * state + 1442695040888963407) % (1 << 64)
        steps[i] = ((state >> 11) / float(1 << 53)) - 0.5
    close = 100.0 * np.exp(np.cumsum(steps * 0.02))
    open_ = np.concatenate(([close[0]], close[:-1]))
    spread = np.abs(np.roll(steps, 1)) * 0.02 * close + 0.01
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = 1000.0 + np.abs(steps) * 10000.0
    index = pd.date_range("2000-01-03", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
    )


def _first_valid(result: Any) -> Optional[int]:
    import numpy as np
    import pandas as pd

    frame = result.to_frame() if isinstance(result, pd.Series) else result
    firsts = []
    for c in frame.columns:
        arr = frame[c].to_numpy(dtype=float, na_value=np.nan)
        valid = np.flatnonzero(~np.isnan(arr))
        if len(valid):
            firsts.append(int(valid[0]))
    return max(firsts) if firsts else None


def _key(args: Sequence[Any], params: Dict[str, Any]) -> Tuple[Any, ...]:
    return tuple(args), tuple(sorted(params.items()))


@functools.lru_cache(maxsize=512)
def _required_cached(name: str, key: Tuple[Any, ...]) -> int:
    args, items = key
    params = dict(items)
    _fp()
    n = 512
    while n <= 16384:
        frame = _probe_frame(n)
        method = getattr(frame.fp, name, None)
        if method is None:
            raise ValueError(f"fin-primitives has no indicator {name!r} (see df.fp.indicators())")
        first = _first_valid(method(*args, **params))
        if first is not None and first < n // 2:
            return first + 1
        n *= 4
    raise ValueError(f"{name}{args or ''} produced no values on a {n // 4}-bar probe")


def required_bars(name: str, *args: Any, **params: Any) -> int:
    """Bars ``df.fp.<name>(*args, **params)`` needs before its first value.

    Measured: the indicator is run on a clean, deterministic OHLCV random walk and the
    position of the first row where every output column has a value is read off. For
    multi-column indicators the latest column decides. Cached per (name, parameters).
    """
    return _required_cached(name, _key(args, params))


def coverage(frame: Any, inputs: Sequence[str]) -> Any:
    """Consecutive bars, ending at each row, with every input column present."""
    import numpy as np
    import pandas as pd

    ok = np.ones(len(frame), dtype=bool)
    for c in inputs:
        col = _column(frame, c)
        ok &= ~np.isnan(col.to_numpy(dtype=float, na_value=np.nan))
    run = np.zeros(len(frame), dtype=np.int64)
    count = 0
    for i, good in enumerate(ok):
        count = count + 1 if good else 0
        run[i] = count
    return pd.Series(run, index=frame.index, name="bars_available")


def _column(frame: Any, name: str) -> Any:
    if name in frame.columns:
        return frame[name]
    for c in frame.columns:
        if isinstance(c, str) and c.lower() == name:
            return frame[c]
    raise KeyError(f"no {name!r} column")


def indicator(frame: Any, name: str, *args: Any, settle: int = 0, **params: Any) -> Any:
    """Compute ``frame.fp.<name>(*args, **params)`` with strict warmup and coverage columns.

    Returns a DataFrame: the indicator's column(s) (fin-primitives / pandas-ta names), then
    ``bars_available`` and ``bars_required``. ``settle`` adds bars to the requirement, for
    recursive indicators (EMA, RSI, ATR, ADX) whose first values still depend on the seed.
    Inside an ``asof`` block, rows not available at the decision time are dropped first.
    """
    import numpy as np
    import pandas as pd

    from .guard import current, filter as pit_filter

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("indicator() needs a pandas DataFrame")
    if not isinstance(settle, int) or isinstance(settle, bool) or settle < 0:
        raise ValueError("settle must be a non-negative int")
    state = current()
    if state is not None and (
        state.column in frame.columns or state.column in (frame.index.names or [])
    ):
        frame = pit_filter(frame)
    _fp()
    method = getattr(frame.fp, name, None)
    if method is None:
        raise ValueError(f"fin-primitives has no indicator {name!r} (see df.fp.indicators())")
    cols = ("open", "high", "low", "close", "volume")
    probe_params = {k: v for k, v in params.items() if k not in cols}
    required = required_bars(name, *args, **probe_params) + settle
    result = method(*args, **params)
    out = result.to_frame() if isinstance(result, pd.Series) else result.copy()
    used = []
    for c in _inputs(name):
        given = params.get(c)
        if isinstance(given, str):
            used.append(given)
        elif given is None and c in cols:
            used.append(c)
    have = coverage(frame, used)
    short = (have < required).to_numpy()
    for c in out.columns:
        values = out[c].to_numpy(dtype=float, na_value=np.nan, copy=True)
        values[short] = np.nan
        out[c] = values
    out["bars_available"] = have.to_numpy()
    out["bars_required"] = np.int64(required)
    return out
