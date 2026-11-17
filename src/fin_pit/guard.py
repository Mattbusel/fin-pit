"""The point-in-time firewall: ``asof``, ``filter``, ``check``, ``guard`` and ``latest``.

A decision time is set with ``asof`` (a context manager and a decorator). It lives in a
``contextvars.ContextVar``, so it follows the code that set it through ``await`` and into
``contextvars.copy_context().run`` without leaking into other tasks or threads.

Inside an ``asof`` block every guarded read is checked against that time:

* a row or record whose ``available_at`` is after the decision time is a look-ahead. In
  ``mode="raise"`` (the default) it raises ``LookaheadError``; in ``mode="filter"`` it is
  dropped and the rest is returned;
* a row whose ``available_at`` is missing (``None``/``NaT``) counts as not available, the
  conservative reading;
* values that carry no availability time at all are "unstamped". ``unstamped="allow"`` (the
  default) passes them through, ``unstamped="raise"`` refuses them.
"""
from __future__ import annotations

import contextvars
import datetime as _dt
import functools
import inspect
import re
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

from ._time import UTC, to_utc
from .records import PITRecord

__all__ = [
    "AVAILABLE_AT",
    "LookaheadError",
    "UnstampedError",
    "asof",
    "current",
    "decision_time",
    "filter",
    "check",
    "guard",
    "latest",
    "scan_text",
]

AVAILABLE_AT = "available_at"
MODES = ("raise", "filter")
UNSTAMPED = ("allow", "raise")

F = TypeVar("F", bound=Callable[..., Any])


class LookaheadError(LookupError):
    """A read inside ``asof(t)`` returned something that was not knowable at ``t``."""

    def __init__(
        self,
        message: str,
        *,
        decision_time: Optional[_dt.datetime] = None,
        available_at: Optional[_dt.datetime] = None,
        count: int = 0,
    ) -> None:
        super().__init__(message)
        self.decision_time = decision_time
        self.available_at = available_at
        self.count = count


class UnstampedError(LookaheadError):
    """``unstamped="raise"`` was set and a value carried no ``available_at``."""


class _State:
    __slots__ = ("time", "mode", "unstamped", "column")

    def __init__(self, time: _dt.datetime, mode: str, unstamped: str, column: str) -> None:
        self.time = time
        self.mode = mode
        self.unstamped = unstamped
        self.column = column

    def __repr__(self) -> str:
        return (
            f"asof({self.time.isoformat()}, mode={self.mode!r}, "
            f"unstamped={self.unstamped!r}, column={self.column!r})"
        )


_STATE: "contextvars.ContextVar[Optional[_State]]" = contextvars.ContextVar(
    "fin_pit_asof", default=None
)


def current() -> Optional[_State]:
    """The active ``asof`` state, or ``None`` outside any ``asof`` block."""
    return _STATE.get()


def decision_time() -> Optional[_dt.datetime]:
    """The active decision time (aware, UTC), or ``None`` outside any ``asof`` block."""
    state = _STATE.get()
    return None if state is None else state.time


class asof:
    """Run code as of ``when``: ``with asof(t): ...`` or ``@asof(t)``.

    ``when`` is an aware datetime, or a naive one / ISO string / date with ``tz=``. A bare
    date means 00:00 of that day in ``tz``.

    Nesting may only move the clock backwards: an inner ``asof`` later than the enclosing one
    raises ``LookaheadError``, because it would let the inner code see what the outer
    decision could not.
    """

    def __init__(
        self,
        when: Any,
        *,
        tz: Any = None,
        mode: str = "raise",
        unstamped: str = "allow",
        column: str = AVAILABLE_AT,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if unstamped not in UNSTAMPED:
            raise ValueError(f"unstamped must be one of {UNSTAMPED}, got {unstamped!r}")
        if not isinstance(column, str) or not column:
            raise ValueError("column must be a non-empty string")
        self.time = to_utc(when, tz, what="decision time")
        self.mode = mode
        self.unstamped = unstamped
        self.column = column
        self._tokens: List[contextvars.Token] = []

    def __repr__(self) -> str:
        return f"asof({self.time.isoformat()}, mode={self.mode!r})"

    def _enter(self) -> contextvars.Token:
        outer = _STATE.get()
        if outer is not None and self.time > outer.time:
            raise LookaheadError(
                f"nested asof({self.time.isoformat()}) is later than the enclosing "
                f"asof({outer.time.isoformat()})",
                decision_time=outer.time,
                available_at=self.time,
                count=1,
            )
        return _STATE.set(_State(self.time, self.mode, self.unstamped, self.column))

    def __enter__(self) -> "asof":
        self._tokens.append(self._enter())
        return self

    def __exit__(self, *exc: Any) -> None:
        _STATE.reset(self._tokens.pop())

    async def __aenter__(self) -> "asof":
        return self.__enter__()

    async def __aexit__(self, *exc: Any) -> None:
        self.__exit__(*exc)

    def __call__(self, fn: F) -> F:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                token = self._enter()
                try:
                    return await fn(*args, **kwargs)
                finally:
                    _STATE.reset(token)

            return awrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            token = self._enter()
            try:
                return fn(*args, **kwargs)
            finally:
                _STATE.reset(token)

        return wrapper  # type: ignore[return-value]


# ---------------------------------------------------------------------------------------
# filtering
# ---------------------------------------------------------------------------------------


def _resolve(
    when: Any, mode: Optional[str], unstamped: Optional[str], column: Optional[str]
) -> Optional[_State]:
    state = _STATE.get()
    if when is not None:
        t = to_utc(when, what="decision time")
        if state is not None and t > state.time:
            raise LookaheadError(
                f"explicit decision time {t.isoformat()} is later than the active "
                f"asof({state.time.isoformat()})",
                decision_time=state.time,
                available_at=t,
                count=1,
            )
        base = state or _State(t, "raise", "allow", AVAILABLE_AT)
        state = _State(t, base.mode, base.unstamped, base.column)
    if state is None:
        return None
    return _State(
        state.time,
        mode if mode is not None else state.mode,
        unstamped if unstamped is not None else state.unstamped,
        column if column is not None else state.column,
    )


def _late_message(state: _State, n: int, first: Any, what: str) -> str:
    first_s = first.isoformat() if hasattr(first, "isoformat") else str(first)
    return (
        f"{n} {what} not available at the decision time {state.time.isoformat()} "
        f"(first available_at: {first_s})"
    )


def _unstamped(obj: Any, state: _State) -> Any:
    if state.unstamped == "raise":
        raise UnstampedError(
            f"{type(obj).__name__} has no {state.column!r}; cannot prove it was available at "
            f"{state.time.isoformat()} (unstamped='raise')",
            decision_time=state.time,
        )
    return obj


def _mod(obj: Any) -> str:
    return type(obj).__module__.split(".", 1)[0]


def _record_time(item: Any, column: str) -> Tuple[bool, Optional[_dt.datetime]]:
    """(stamped?, available_at or None when stamped but missing)."""
    if isinstance(item, PITRecord):
        return True, item.available_at
    if isinstance(item, dict) and column in item:
        raw = item[column]
        if raw is None or (isinstance(raw, float) and raw != raw):
            return True, None
        if type(raw).__name__ == "NaTType":
            return True, None
        return True, to_utc(raw, what=column)
    t = getattr(item, column, None) if not isinstance(item, (str, bytes, dict)) else None
    if t is not None:
        return True, to_utc(t, what=column)
    return False, None


def _apply(obj: Any, state: _State) -> Any:
    col = state.column
    if obj is None:
        return None
    mod = _mod(obj)
    if mod == "pandas":
        return _apply_pandas(obj, state)
    if mod == "polars":
        return _apply_polars(obj, state)
    if isinstance(obj, (PITRecord, dict)) and _record_time(obj, col)[0]:
        _, t = _record_time(obj, col)
        if t is None or t > state.time:
            raise LookaheadError(
                _late_message(state, 1, t if t is not None else "missing", "record is"),
                decision_time=state.time,
                available_at=t,
                count=1,
            )
        return obj
    if isinstance(obj, (list, tuple)):
        stamped = [_record_time(x, col) for x in obj]
        if not obj or not any(s for s, _ in stamped):
            return _unstamped(obj, state) if obj else obj
        if not all(s for s, _ in stamped):
            # Partly stamped: the unstamped items cannot be cleared.
            if state.unstamped == "raise":
                return _unstamped(obj, state)
        late = [t for s, t in stamped if s and (t is None or t > state.time)]
        if not late:
            return obj
        if state.mode == "raise":
            known = [t for t in late if t is not None]
            raise LookaheadError(
                _late_message(state, len(late), min(known) if known else "missing", "records are"),
                decision_time=state.time,
                available_at=min(known) if known else None,
                count=len(late),
            )
        kept = [x for x, (s, t) in zip(obj, stamped) if not s or (t is not None and t <= state.time)]
        return type(obj)(kept) if isinstance(obj, tuple) else kept
    if isinstance(obj, PITRecord):  # pragma: no cover - handled above
        return obj
    stamped, t = _record_time(obj, col)
    if stamped:
        if t is None or t > state.time:
            raise LookaheadError(
                _late_message(state, 1, t if t is not None else "missing", "record is"),
                decision_time=state.time,
                available_at=t,
                count=1,
            )
        return obj
    return _unstamped(obj, state)


def _pandas_times(values: Any, column: str) -> Any:
    import pandas as pd

    s = values
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    dtype = s.dtype
    if isinstance(dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert("UTC")
    if str(dtype).startswith("datetime64"):
        raise ValueError(
            f"column {column!r} holds naive datetimes; localise it first, e.g. "
            f"df[{column!r}] = df[{column!r}].dt.tz_localize('America/New_York')"
        )
    # object / string column: parse each value strictly (naive strings raise)
    parsed = [
        pd.NaT if (v is None or (isinstance(v, float) and v != v) or v is pd.NaT) else to_utc(v, what=column)
        for v in s.tolist()
    ]
    return pd.Series(pd.to_datetime(parsed, utc=True), index=s.index)


def _apply_pandas(obj: Any, state: _State) -> Any:
    import pandas as pd

    col = state.column
    if isinstance(obj, pd.DataFrame):
        if col in obj.columns:
            times = _pandas_times(obj[col], col)
        elif col in (obj.index.names or []):
            times = _pandas_times(pd.Series(obj.index.get_level_values(col), index=obj.index), col)
        else:
            return _unstamped(obj, state)
    elif isinstance(obj, pd.Series):
        if col in (obj.index.names or []):
            times = _pandas_times(pd.Series(obj.index.get_level_values(col), index=obj.index), col)
        else:
            return _unstamped(obj, state)
    else:
        return _unstamped(obj, state)
    t = pd.Timestamp(state.time)
    ok = (times <= t).fillna(False).to_numpy(dtype=bool)
    if ok.all():
        return obj
    n_late = int((~ok).sum())
    if state.mode == "raise":
        late_times = times[~ok].dropna()
        first = late_times.min().to_pydatetime() if len(late_times) else None
        raise LookaheadError(
            _late_message(state, n_late, first if first is not None else "missing", "rows are"),
            decision_time=state.time,
            available_at=first,
            count=n_late,
        )
    return obj[ok]


def _apply_polars(obj: Any, state: _State) -> Any:
    import polars as pl

    col = state.column
    if isinstance(obj, pl.LazyFrame):
        schema = obj.collect_schema()
        if col not in schema.names():
            return _unstamped(obj, state)
        dtype = schema[col]
        _polars_check_dtype(dtype, col)
        if state.mode == "raise":
            raise ValueError(
                "a LazyFrame cannot be checked without collecting it; use mode='filter' "
                "or collect first"
            )
        return obj.filter(pl.col(col).dt.convert_time_zone("UTC") <= pl.lit(state.time))
    if not isinstance(obj, pl.DataFrame):
        return _unstamped(obj, state)
    if col not in obj.columns:
        return _unstamped(obj, state)
    _polars_check_dtype(obj.schema[col], col)
    times = obj.get_column(col).dt.convert_time_zone("UTC")
    ok = (times <= state.time).fill_null(False)
    if bool(ok.all()):
        return obj
    n_late = int((~ok).sum())
    if state.mode == "raise":
        late = times.filter(~ok).drop_nulls()
        first = late.min() if late.len() else None
        raise LookaheadError(
            _late_message(state, n_late, first if first is not None else "missing", "rows are"),
            decision_time=state.time,
            available_at=first,
            count=n_late,
        )
    return obj.filter(ok)


def _polars_check_dtype(dtype: Any, col: str) -> None:
    import polars as pl

    if not isinstance(dtype, pl.Datetime):
        raise ValueError(f"column {col!r} must be a polars Datetime, got {dtype}")
    if dtype.time_zone is None:
        raise ValueError(
            f"column {col!r} holds naive datetimes; use "
            f"pl.col({col!r}).dt.replace_time_zone('America/New_York') first"
        )


def filter(
    obj: Any,
    when: Any = None,
    *,
    column: Optional[str] = None,
    unstamped: Optional[str] = None,
) -> Any:
    """Drop everything in ``obj`` not available at the decision time.

    The decision time is ``when`` or the active ``asof``. With neither, ``obj`` comes back
    unchanged. Works on pandas DataFrames (an ``available_at`` column or index level), Polars
    DataFrames and LazyFrames, lists/tuples of ``PITRecord`` or dicts, and single records
    (a single late record cannot be dropped, so it raises).
    """
    state = _resolve(when, "filter", unstamped, column)
    if state is None:
        return obj
    return _apply(obj, state)


def check(
    obj: Any,
    when: Any = None,
    *,
    column: Optional[str] = None,
    unstamped: Optional[str] = None,
) -> Any:
    """Raise ``LookaheadError`` if anything in ``obj`` was not available at the decision
    time; otherwise return ``obj`` unchanged."""
    state = _resolve(when, "raise", unstamped, column)
    if state is None:
        return obj
    return _apply(obj, state)


# ---------------------------------------------------------------------------------------
# text scanning (opt-in): dates written into tool output
# ---------------------------------------------------------------------------------------

_DATE_RE = re.compile(
    r"(?<![0-9])((?:19|20)[0-9]{2})-([01][0-9])-([0-3][0-9])"
    r"(?:[T ]([0-2][0-9]):([0-5][0-9])(?::([0-5][0-9])(?:\.[0-9]+)?)?"
    r"(Z|[+-][0-2][0-9]:?[0-5][0-9])?)?(?![0-9])"
)


def scan_text(text: str, when: Any = None) -> List[str]:
    """Find dates in ``text`` that are later than the decision time.

    A bare ``YYYY-MM-DD`` counts as later only when the whole day is after the decision's
    UTC calendar day; a timestamp with an offset is compared exactly; a timestamp without an
    offset is read as UTC. Returns the offending substrings (empty when clean). This is a
    heuristic for free-text tool output, used by ``guard(..., scan_text=True)``.
    """
    state = _resolve(when, None, None, None)
    if state is None or not isinstance(text, str):
        return []
    t = state.time
    found: List[str] = []
    for m in _DATE_RE.finditer(text):
        y, mo, d, hh, mi, ss, off = m.groups()
        try:
            day = _dt.date(int(y), int(mo), int(d))
        except ValueError:
            continue
        if hh is None:
            if day > t.date():
                found.append(m.group(0))
            continue
        try:
            stamp = _dt.datetime(int(y), int(mo), int(d), int(hh), int(mi), int(ss or 0))
        except ValueError:
            continue
        if off is None or off in ("Z", "z"):
            stamp = stamp.replace(tzinfo=UTC)
        else:
            sign = 1 if off[0] == "+" else -1
            digits = off[1:].replace(":", "")
            stamp = stamp.replace(
                tzinfo=_dt.timezone(sign * _dt.timedelta(hours=int(digits[:2]), minutes=int(digits[2:])))
            )
        if stamp > t:
            found.append(m.group(0))
    return found


# ---------------------------------------------------------------------------------------
# guard: wrap any callable
# ---------------------------------------------------------------------------------------

DEFAULT_DATE_ARGS: Tuple[str, ...] = (
    "as_of",
    "asof",
    "as_of_date",
    "curr_date",
    "current_date",
    "trade_date",
    "end",
    "end_date",
    "end_time",
    "date",
    "to",
    "to_date",
    "until",
)


def _check_args(
    sig: Optional[inspect.Signature],
    args: Sequence[Any],
    kwargs: dict,
    names: Iterable[str],
    state: _State,
    tz: Any,
) -> None:
    bound: dict = dict(kwargs)
    if sig is not None:
        try:
            b = sig.bind_partial(*args, **kwargs)
            bound = dict(b.arguments)
        except TypeError:
            bound = dict(kwargs)
        else:
            for p in sig.parameters.values():
                if p.kind is inspect.Parameter.VAR_KEYWORD and isinstance(bound.get(p.name), dict):
                    bound.update(bound.pop(p.name))
    for name in names:
        if name not in bound:
            continue
        raw = bound[name]
        if raw is None:
            continue
        try:
            t = to_utc(raw, tz, what=f"argument {name}")
        except (TypeError, ValueError):
            continue
        # A bare date argument means "up to and including that day": the day must not
        # start after the decision time. Datetimes are compared exactly.
        if t > state.time:
            raise LookaheadError(
                f"argument {name}={raw!r} is later than the decision time "
                f"{state.time.isoformat()}",
                decision_time=state.time,
                available_at=t,
                count=1,
            )


def guard(
    fn: Optional[F] = None,
    *,
    mode: Optional[str] = None,
    unstamped: Optional[str] = None,
    column: Optional[str] = None,
    date_args: Optional[Iterable[str]] = DEFAULT_DATE_ARGS,
    date_tz: Any = "UTC",
    scan_text: bool = False,
    require_asof: bool = False,
) -> Any:
    """Wrap a callable so every call is checked against the active ``asof``.

    Before the call, arguments named in ``date_args`` (``curr_date``, ``end_date``, ...) that
    parse as a time later than the decision time raise ``LookaheadError``; naive values are
    read in ``date_tz``. After the call, the result goes through ``filter``/``check``
    (``mode``), and with ``scan_text=True`` string results are scanned for later dates.

    Outside any ``asof`` the call passes through untouched, unless ``require_asof=True``.
    Works as ``guard(fn)`` and as ``@guard(mode="filter")``; async functions stay async.
    """
    if mode is not None and mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if unstamped is not None and unstamped not in UNSTAMPED:
        raise ValueError(f"unstamped must be one of {UNSTAMPED}, got {unstamped!r}")
    names = tuple(date_args or ())

    def deco(f: F) -> F:
        try:
            sig: Optional[inspect.Signature] = inspect.signature(f)
        except (TypeError, ValueError):
            sig = None

        def before(args: Sequence[Any], kwargs: dict) -> Optional[_State]:
            state = _resolve(None, mode, unstamped, column)
            if state is None:
                if require_asof:
                    raise LookaheadError(
                        f"{getattr(f, '__name__', 'call')} needs an active asof (require_asof=True)"
                    )
                return None
            _check_args(sig, args, kwargs, names, state, date_tz)
            return state

        def after(result: Any, state: Optional[_State]) -> Any:
            if state is None:
                return result
            if scan_text and isinstance(result, str):
                hits = _scan_with(result, state)
                if hits:
                    raise LookaheadError(
                        f"{len(hits)} date(s) in the output are later than the decision time "
                        f"{state.time.isoformat()}: {', '.join(hits[:5])}",
                        decision_time=state.time,
                        count=len(hits),
                    )
                return result
            if isinstance(result, str):
                return _unstamped(result, state) if state.unstamped == "raise" else result
            return _apply(result, state)

        if inspect.iscoroutinefunction(f):

            @functools.wraps(f)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                state = before(args, kwargs)
                return after(await f(*args, **kwargs), state)

            awrapper.__fin_pit_guarded__ = True  # type: ignore[attr-defined]
            return awrapper  # type: ignore[return-value]

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            state = before(args, kwargs)
            return after(f(*args, **kwargs), state)

        wrapper.__fin_pit_guarded__ = True  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    if fn is not None:
        return deco(fn)
    return deco


def _scan_with(text: str, state: _State) -> List[str]:
    token = _STATE.set(state)
    try:
        return scan_text(text)
    finally:
        _STATE.reset(token)


# ---------------------------------------------------------------------------------------
# vintages
# ---------------------------------------------------------------------------------------


def latest(
    obj: Any,
    on: Union[str, Sequence[str]] = "event_time",
    when: Any = None,
    *,
    column: Optional[str] = None,
) -> Any:
    """The newest vintage of each ``on`` key that was available at the decision time.

    ``obj`` is a pandas or Polars DataFrame (or a list of ``PITRecord``) with several rows per
    key, one per vintage (ALFRED revisions, restated XBRL facts). Rows not yet available are
    dropped first, then the row with the greatest ``available_at`` per key is kept. Ties keep
    the last row in input order. Without a decision time, the newest vintage overall is kept.
    """
    keys = [on] if isinstance(on, str) else list(on)
    col = column or (current().column if current() is not None else AVAILABLE_AT)
    state = _resolve(when, "filter", "allow", col)
    if state is not None:
        obj = _apply(obj, state)
    mod = _mod(obj)
    if mod == "pandas":
        import pandas as pd

        if not isinstance(obj, pd.DataFrame):
            raise TypeError("latest() needs a DataFrame")
        times = _pandas_times(obj[col], col)
        order = pd.Series(range(len(obj)), index=obj.index)
        tmp = obj.assign(__t=times.to_numpy(), __o=order.to_numpy())
        tmp = tmp.sort_values(["__t", "__o"], kind="mergesort")
        out = tmp.groupby(keys, sort=False, dropna=False).tail(1)
        out = out.sort_values("__o", kind="mergesort").drop(columns=["__t", "__o"])
        return out
    if mod == "polars":
        import polars as pl

        if not isinstance(obj, pl.DataFrame):
            raise TypeError("latest() needs a polars DataFrame")
        return (
            obj.with_row_index("__o")
            .sort([col, "__o"])
            .group_by(keys, maintain_order=True)
            .last()
            .sort("__o")
            .drop("__o")
            .select(obj.columns)
        )
    if isinstance(obj, (list, tuple)):
        best: dict = {}
        for i, rec in enumerate(obj):
            if not isinstance(rec, PITRecord):
                raise TypeError("latest() on a list needs PITRecord items")
            k = tuple(getattr(rec, name) for name in keys)
            if k not in best or rec.available_at >= best[k][1].available_at:
                best[k] = (i, rec)
        return [rec for _, rec in sorted(best.values(), key=lambda p: p[0])]
    raise TypeError(f"latest() does not support {type(obj).__name__}")
