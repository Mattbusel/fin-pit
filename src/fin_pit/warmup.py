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

Only causal outputs are returned. Two pandas-ta conventions read later bars: the centred DPO
(``dpo`` is computed with ``centered=False`` here) and Ichimoku's chikou span (``ICS_*``,
the close 26 bars ahead, which is dropped). Every indicator is also checked at run time: it is
computed on a probe series and on the same series cut short, and any output column whose
earlier values change is dropped (``noncausal_columns``).

Inside ``fin_pit.asof``: a frame with an ``available_at`` column (or index level) is cut to
the rows available at the decision time first. A frame without one (plain OHLCV) cannot be
cut exactly, because a bar's label is not when it closed: rows labelled at or after the
decision time are dropped, ``unstamped="raise"`` refuses the frame, and otherwise an
``UnstampedFrameWarning`` says the last bar may still be open. Stamp bars with
``fin_pit.adapters.yfinance.stamp_bars`` (or your own ``available_at``) to cut exactly.
"""
from __future__ import annotations

import functools
import math
import warnings
from typing import Any, Dict, Optional, Sequence, Tuple

__all__ = [
    "UnstampedFrameWarning",
    "coverage",
    "indicator",
    "indicators",
    "noncausal_columns",
    "required_bars",
]

_OHLCV = ("open", "high", "low", "close", "volume")

# Parameters forced so the indicator only reads past bars.
_CAUSAL_PARAMS: Dict[str, Dict[str, Any]] = {"dpo": {"centered": False}}


class UnstampedFrameWarning(UserWarning):
    """``indicator`` ran inside ``asof`` on a frame with no ``available_at``."""


def _fp() -> Any:
    try:
        import fin_primitives.pandas  # noqa: F401  (registers df.fp)
    except ImportError as exc:  # pragma: no cover
        raise ImportError("fin-pit indicators need fin-primitives: pip install fin-primitives") from exc
    import fin_primitives.pandas as fpp

    return fpp


def indicators() -> Tuple[str, ...]:
    """Names ``indicator`` accepts (fin-primitives' ``df.fp`` indicators)."""
    import pandas as pd

    _fp()
    return tuple(pd.DataFrame({c: [1.0] for c in _OHLCV}).fp.indicators())


def _inputs(name: str) -> Tuple[str, ...]:
    fpp = _fp()
    fn = getattr(fpp, "_inputs", None)
    if fn is not None:
        return tuple(fn(name))
    # fin-primitives no longer exposes its input map: require every OHLCV column instead,
    # which can only mask more rows, never fewer.
    warnings.warn(
        "fin_primitives.pandas._inputs is missing; coverage counts every OHLCV column",
        RuntimeWarning,
        stacklevel=3,
    )
    return _OHLCV


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


def _frame_of(result: Any) -> Any:
    import pandas as pd

    return result.to_frame() if isinstance(result, pd.Series) else result


def _first_valid(result: Any) -> Optional[int]:
    import numpy as np

    frame = _frame_of(result)
    firsts = []
    for c in frame.columns:
        arr = frame[c].to_numpy(dtype=float, na_value=np.nan)
        valid = np.flatnonzero(~np.isnan(arr))
        if len(valid):
            firsts.append(int(valid[0]))
    return max(firsts) if firsts else None


def _key(args: Sequence[Any], params: Dict[str, Any]) -> Tuple[Any, ...]:
    return tuple(args), tuple(sorted(params.items()))


def _largest_number(values: Sequence[Any]) -> float:
    nums = [abs(float(v)) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return max(nums) if nums else 0.0


def _differs(a: Any, b: Any) -> bool:
    import numpy as np

    na, nb = np.isnan(a), np.isnan(b)
    if (na != nb).any():
        return True
    both = ~na
    return bool((np.abs(a[both] - b[both]) > 1e-9 * (1.0 + np.abs(a[both]))).any())


@functools.lru_cache(maxsize=512)
def _probe(name: str, key: Tuple[Any, ...]) -> Tuple[int, Tuple[str, ...]]:
    """(bars required, non-causal output columns) for ``df.fp.<name>(*args, **params)``."""
    import numpy as np

    args, items = key
    params = dict(items)
    _fp()
    big = _largest_number(list(args) + list(params.values()))
    n = 512
    if big * 4 > n:
        n = 1 << int(math.ceil(math.log2(big * 4)))
    limit = max(16384, n * 16)
    while True:
        frame = _probe_frame(n)
        method = getattr(frame.fp, name, None)
        if method is None:
            raise ValueError(f"fin-primitives has no indicator {name!r} (see fin_pit.warmup.indicators())")
        full = _frame_of(method(*args, **params))
        first = _first_valid(full)
        last_try = n * 4 > limit
        if first is not None and (first < n // 2 or (last_try and first < n - 1)):
            break
        if last_try:
            if first is None:
                raise ValueError(f"{name}{args or ''} produced no values on a {n}-bar probe")
            raise ValueError(
                f"{name}{args or ''} needs {first + 1} bars, too close to the {n}-bar probe "
                f"to check it"
            )
        n *= 4
    # Causality: an output that only reads past bars gives the same value at row i whether or
    # not later rows exist. Cut the probe short and compare.
    noncausal = set()
    for m in sorted({n - 1, n - n // 4}):
        cut = _frame_of(getattr(frame.iloc[:m].fp, name)(*args, **params))
        for c in full.columns:
            if c in noncausal:
                continue
            a = full[c].to_numpy(dtype=float, na_value=np.nan)[:m]
            b = cut[c].to_numpy(dtype=float, na_value=np.nan) if c in cut.columns else np.full(m, np.nan)
            if _differs(a, b):
                noncausal.add(c)
    return first + 1, tuple(c for c in full.columns if c in noncausal)


def _probe_params(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in params.items() if k not in _OHLCV}
    for k, v in _CAUSAL_PARAMS.get(name, {}).items():
        out.setdefault(k, v)
    return out


def required_bars(name: str, *args: Any, **params: Any) -> int:
    """Bars ``df.fp.<name>(*args, **params)`` needs before its first value.

    Measured: the indicator is run on a clean, deterministic OHLCV random walk (at least four
    times the largest numeric parameter long) and the position of the first row where every
    output column has a value is read off. For multi-column indicators the latest column
    decides. Cached per (name, parameters).
    """
    return _probe(name, _key(args, _probe_params(name, params)))[0]


def noncausal_columns(name: str, *args: Any, **params: Any) -> Tuple[str, ...]:
    """Output columns of ``df.fp.<name>(*args, **params)`` whose values depend on later
    bars (measured by truncating a probe series); ``indicator`` drops them."""
    return _probe(name, _key(args, _probe_params(name, params)))[1]


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


def _cut_unstamped(frame: Any, state: Any) -> Any:
    """Inside ``asof`` with a frame that has no ``available_at``."""
    import numpy as np
    import pandas as pd

    from .guard import UnstampedError

    if state.unstamped == "raise":
        raise UnstampedError(
            f"indicator input has no {state.column!r} column or index level; cannot prove its "
            f"bars were available at {state.time.isoformat()} (unstamped='raise')",
            decision_time=state.time,
        )
    idx = frame.index
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        # A bar labelled at or after the decision time had not started (bars are labelled
        # by their start), so it cannot have been available then.
        frame = frame[np.asarray(idx.tz_convert("UTC") < pd.Timestamp(state.time))]
    warnings.warn(
        f"indicator input has no {state.column!r}: "
        + (
            "rows labelled at or after the decision time were dropped, but the last bar may still "
            "have been open. "
            if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None
            else "it was not cut to the decision time (no tz-aware DatetimeIndex). "
        )
        + "Stamp the bars (fin_pit.adapters.yfinance.stamp_bars) to cut exactly.",
        UnstampedFrameWarning,
        stacklevel=4,
    )
    return frame


def indicator(frame: Any, name: str, *args: Any, settle: int = 0, **params: Any) -> Any:
    """Compute ``frame.fp.<name>(*args, **params)`` with strict warmup and coverage columns.

    Returns a DataFrame: the indicator's causal column(s) (fin-primitives / pandas-ta names),
    then ``bars_available`` and ``bars_required``. ``settle`` adds bars to the requirement,
    for recursive indicators (EMA, RSI, ATR, ADX) whose first values still depend on the
    seed. Inside an ``asof`` block the frame is cut first (see the module notes for frames
    with no ``available_at``).
    """
    import numpy as np
    import pandas as pd

    from .guard import LookaheadError, current, filter as pit_filter

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("indicator() needs a pandas DataFrame")
    if not isinstance(settle, int) or isinstance(settle, bool) or settle < 0:
        raise ValueError("settle must be a non-negative int")
    for k, v in _CAUSAL_PARAMS.get(name, {}).items():
        if k in params and params[k] != v:
            raise LookaheadError(
                f"{name}({k}={params[k]!r}) reads later bars; fin-pit computes it with {k}={v!r}"
            )
        params.setdefault(k, v)
    state = current()
    if state is not None:
        if state.column in frame.columns or state.column in (frame.index.names or []):
            frame = pit_filter(frame)
        else:
            frame = _cut_unstamped(frame, state)
    _fp()
    method = getattr(frame.fp, name, None)
    if method is None:
        raise ValueError(f"fin-primitives has no indicator {name!r} (see fin_pit.warmup.indicators())")
    probe_params = {k: v for k, v in params.items() if k not in _OHLCV}
    required, noncausal = _probe(name, _key(args, probe_params))
    required += settle
    result = method(*args, **params)
    out = result.to_frame() if isinstance(result, pd.Series) else result.copy()
    if noncausal:
        keep = [c for c in out.columns if c not in noncausal]
        if not keep:
            raise LookaheadError(
                f"{name} reads later bars in every output column ({', '.join(noncausal)}); "
                f"it cannot be used point-in-time"
            )
        out = out[keep].copy()
    used = []
    for c in _inputs(name):
        given = params.get(c)
        if isinstance(given, str):
            used.append(given)
        elif given is None and c in _OHLCV:
            if c in frame.columns or any(isinstance(x, str) and x.lower() == c for x in frame.columns):
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
