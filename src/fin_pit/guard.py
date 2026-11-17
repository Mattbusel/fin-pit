"""The point-in-time firewall: ``asof``, ``filter``, ``check``, ``guard`` and ``latest``.

A decision time is set with ``asof`` (a context manager and a decorator). It lives in a
``contextvars.ContextVar``, so it follows the code that set it through ``await`` and into
``contextvars.copy_context().run`` without leaking into other tasks or threads. A plain
``ThreadPoolExecutor.submit`` does not copy the context; wrap the callable with ``bind`` first.

Inside an ``asof`` block every guarded read is checked against that time:

* a row or record whose ``available_at`` is after the decision time is a look-ahead. In
  ``mode="raise"`` (the default) it raises ``LookaheadError``; in ``mode="filter"`` it is
  dropped and the rest is returned;
* a row whose ``available_at`` is missing (``None``/``NaT``/``pd.NA``) counts as not
  available, the conservative reading;
* records nested inside dicts, lists and tuples (``{"rows": [...]}``, ``(content, artifact)``)
  are found and checked too;
* values that carry no availability time at all are "unstamped". ``unstamped="allow"`` (the
  default) passes them through, ``unstamped="raise"`` refuses them.
"""
from __future__ import annotations

import contextvars
import datetime as _dt
import functools
import inspect
import numbers
import re
import warnings
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

from ._time import UTC, source_zone, to_utc
from .records import PITRecord

__all__ = [
    "AVAILABLE_AT",
    "LookaheadError",
    "UnstampedError",
    "UnguardedCallWarning",
    "UnparsedDateWarning",
    "asof",
    "bind",
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
UNPARSED = ("warn", "raise", "ignore")

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


class UnguardedCallWarning(UserWarning):
    """A guarded callable ran with no active ``asof``, so nothing was checked."""


class UnparsedDateWarning(UserWarning):
    """A date argument of a guarded call could not be read as a time, so it was not checked."""


class _State:
    __slots__ = ("time", "mode", "unstamped", "column", "zone")

    def __init__(
        self, time: _dt.datetime, mode: str, unstamped: str, column: str, zone: Any = UTC
    ) -> None:
        self.time = time
        self.mode = mode
        self.unstamped = unstamped
        self.column = column
        self.zone = zone

    def replace(self, **kw: Any) -> "_State":
        vals = {k: getattr(self, k) for k in self.__slots__}
        vals.update({k: v for k, v in kw.items() if v is not None})
        return _State(**vals)

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


# ---------------------------------------------------------------------------------------
# generator drivers: keep a context variable set while a generator body runs, step by step
# ---------------------------------------------------------------------------------------

_SKIP = object()


def _drive(gen: Any, enter: Callable[[], Any], leave: Callable[[Any], None],
           transform: Callable[[Any], Any]) -> Any:
    """Drive ``gen`` with ``enter()``/``leave(token)`` around every step, forwarding
    ``send``, ``throw`` and ``close``; ``transform`` maps each yielded item (``_SKIP`` drops
    it)."""
    value: Any = None
    pending: Optional[BaseException] = None
    while True:
        token = enter()
        try:
            if pending is not None:
                exc, pending = pending, None
                item = gen.throw(exc)
            else:
                item = gen.send(value)
        except StopIteration as stop:
            return stop.value
        finally:
            leave(token)
        out = transform(item)
        if out is _SKIP:
            value = None
            continue
        try:
            value = yield out
        except GeneratorExit:
            token = enter()
            try:
                gen.close()
            finally:
                leave(token)
            raise
        except BaseException as exc:  # forwarded into the wrapped generator
            pending = exc


async def _adrive(agen: Any, enter: Callable[[], Any], leave: Callable[[Any], None],
                  transform: Callable[[Any], Any]) -> Any:
    """Async-generator twin of ``_drive``."""
    value: Any = None
    pending: Optional[BaseException] = None
    while True:
        token = enter()
        try:
            if pending is not None:
                exc, pending = pending, None
                item = await agen.athrow(exc)
            else:
                item = await agen.asend(value)
        except StopAsyncIteration:
            return
        finally:
            leave(token)
        out = transform(item)
        if out is _SKIP:
            value = None
            continue
        try:
            value = yield out
        except GeneratorExit:
            token = enter()
            try:
                await agen.aclose()
            finally:
                leave(token)
            raise
        except BaseException as exc:
            pending = exc


def _identity(x: Any) -> Any:
    return x


class asof:
    """Run code as of ``when``: ``with asof(t): ...`` or ``@asof(t)``.

    ``when`` is an aware datetime, or a naive one / ISO string / date with ``tz=``. A bare
    date means 00:00 of that day in ``tz``. The zone the decision was given in (``tz=``, or
    the offset of an aware ``when``) is kept: guarded calls read naive date arguments and
    naive timestamps in text in that zone.

    Nesting may only move the clock backwards: an inner ``asof`` later than the enclosing one
    raises ``LookaheadError``, because it would let the inner code see what the outer
    decision could not. As a decorator it works on plain, async, generator and async
    generator functions; a generator body sees the decision time on every step.
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
        self.zone = source_zone(when, tz)
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
        return _STATE.set(_State(self.time, self.mode, self.unstamped, self.column, self.zone))

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
        if inspect.isasyncgenfunction(fn):

            @functools.wraps(fn)
            def agwrapper(*args: Any, **kwargs: Any) -> Any:
                return _adrive(fn(*args, **kwargs), self._enter, _STATE.reset, _identity)

            return agwrapper  # type: ignore[return-value]

        if inspect.isgeneratorfunction(fn):

            @functools.wraps(fn)
            def gwrapper(*args: Any, **kwargs: Any) -> Any:
                return _drive(fn(*args, **kwargs), self._enter, _STATE.reset, _identity)

            return gwrapper  # type: ignore[return-value]

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


def bind(fn: F) -> F:
    """Capture the active ``asof`` now and run ``fn`` under it later, on any thread.

    ``ThreadPoolExecutor.submit`` and ``loop.run_in_executor`` do not copy context
    variables, so a guarded call on a worker thread would see no decision time. Use
    ``executor.submit(pit.bind(fetch), ticker)``. Outside any ``asof``, ``fn`` is returned
    unchanged.
    """
    state = _STATE.get()
    if state is None:
        return fn

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def abound(*args: Any, **kwargs: Any) -> Any:
            token = _STATE.set(state)
            try:
                return await fn(*args, **kwargs)
            finally:
                _STATE.reset(token)

        return abound  # type: ignore[return-value]

    @functools.wraps(fn)
    def bound(*args: Any, **kwargs: Any) -> Any:
        token = _STATE.set(state)
        try:
            return fn(*args, **kwargs)
        finally:
            _STATE.reset(token)

    return bound  # type: ignore[return-value]


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
        state = _State(t, base.mode, base.unstamped, base.column, source_zone(when))
    if state is None:
        return None
    return state.replace(mode=mode, unstamped=unstamped, column=column)


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


def _is_frame(obj: Any) -> bool:
    return _mod(obj) in ("pandas", "polars")


_MISSING_TEXT = ("", "nat", "nan", "none", "null", "<na>")


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and v != v:
        return True
    if type(v).__name__ in ("NaTType", "NAType"):
        return True
    if isinstance(v, str) and v.strip().lower() in _MISSING_TEXT:
        return True
    return False


def _record_time(item: Any, column: str) -> Tuple[bool, Optional[_dt.datetime]]:
    """(stamped?, available_at or None when stamped but missing)."""
    if isinstance(item, PITRecord):
        return True, item.available_at
    if isinstance(item, (str, bytes, numbers.Number)) or _is_frame(item):
        return False, None
    if isinstance(item, dict):
        if column not in item:
            return False, None
        raw = item[column]
    else:
        raw = getattr(item, column, None)
        if raw is None:
            return False, None
    if _is_missing(raw):
        return True, None
    return True, to_utc(raw, what=column)


_MAX_DEPTH = 8


def _late_record(state: _State, t: Optional[_dt.datetime]) -> LookaheadError:
    return LookaheadError(
        _late_message(state, 1, t if t is not None else "missing", "record is"),
        decision_time=state.time,
        available_at=t,
        count=1,
    )


def _walk(obj: Any, state: _State, depth: int) -> Tuple[Any, bool]:
    """Check or filter ``obj`` and whatever it nests. Returns (result, found_stamped)."""
    mod = _mod(obj)
    if mod == "pandas":
        return _apply_pandas(obj, state)
    if mod == "polars":
        return _apply_polars(obj, state)
    stamped, t = _record_time(obj, state.column)
    if stamped:
        if t is None or t > state.time:
            raise _late_record(state, t)
        return obj, True
    if depth >= _MAX_DEPTH or hasattr(obj, "_fields"):  # namedtuples: never re-shaped
        return obj, False
    if isinstance(obj, dict):
        out = {}
        found = changed = False
        for k, v in obj.items():
            nv, f = _walk(v, state, depth + 1)
            found = found or f
            changed = changed or nv is not v
            out[k] = nv
        if not changed:
            return obj, found
        if type(obj) is not dict:
            try:
                return type(obj)(out), found
            except Exception:  # pragma: no cover - exotic dict subclasses
                pass
        return out, found
    if isinstance(obj, (list, tuple)):
        items = list(obj)
        direct = [_record_time(x, state.column) for x in items]
        any_direct = any(s for s, _ in direct)
        if any_direct and not all(s for s, _ in direct) and state.unstamped == "raise":
            # Partly stamped: the unstamped items cannot be cleared.
            _unstamped(obj, state)
        late = [i for i, (s, t) in enumerate(direct) if s and (t is None or t > state.time)]
        if late and state.mode == "raise":
            known = [direct[i][1] for i in late if direct[i][1] is not None]
            first = min(known) if known else None
            raise LookaheadError(
                _late_message(state, len(late), first if first is not None else "missing", "records are"),
                decision_time=state.time,
                available_at=first,
                count=len(late),
            )
        drop = set(late)
        kept = []
        found = any_direct
        changed = bool(drop)
        for i, x in enumerate(items):
            if i in drop:
                continue
            if direct[i][0]:
                kept.append(x)
                continue
            nx, f = _walk(x, state, depth + 1)
            found = found or f
            changed = changed or nx is not x
            kept.append(nx)
        if not changed:
            return obj, found
        return (tuple(kept) if isinstance(obj, tuple) else kept), found
    return obj, False


def _apply(obj: Any, state: _State) -> Any:
    if obj is None:
        return None
    out, found = _walk(obj, state, 0)
    if found:
        return out
    if isinstance(obj, (list, tuple, dict)) and len(obj) == 0:
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
    # object / string column: parse each value strictly (naive strings raise); missing
    # values (None, NaN, NaT, pd.NA, "NaT") count as not available
    parsed = [pd.NaT if _is_missing(v) else to_utc(v, what=column) for v in s.tolist()]
    return pd.Series(pd.to_datetime(parsed, utc=True), index=s.index)


def _apply_pandas(obj: Any, state: _State) -> Tuple[Any, bool]:
    import pandas as pd

    col = state.column
    if isinstance(obj, pd.DataFrame):
        if col in obj.columns:
            times = _pandas_times(obj[col], col)
        elif col in (obj.index.names or []):
            times = _pandas_times(pd.Series(obj.index.get_level_values(col), index=obj.index), col)
        else:
            return obj, False
    elif isinstance(obj, pd.Series):
        if col in (obj.index.names or []):
            times = _pandas_times(pd.Series(obj.index.get_level_values(col), index=obj.index), col)
        else:
            return obj, False
    else:
        return obj, False
    t = pd.Timestamp(state.time)
    ok = (times <= t).fillna(False).to_numpy(dtype=bool)
    if ok.all():
        return obj, True
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
    return obj[ok], True


def _apply_polars(obj: Any, state: _State) -> Tuple[Any, bool]:
    import polars as pl

    col = state.column
    if isinstance(obj, pl.LazyFrame):
        schema = obj.collect_schema()
        if col not in schema.names():
            return obj, False
        dtype = schema[col]
        _polars_check_dtype(dtype, col)
        if state.mode == "raise":
            raise ValueError(
                "a LazyFrame cannot be checked without collecting it; use mode='filter' "
                "or collect first"
            )
        return obj.filter(pl.col(col).dt.convert_time_zone("UTC") <= pl.lit(state.time)), True
    if not isinstance(obj, pl.DataFrame):
        return obj, False
    if col not in obj.columns:
        return obj, False
    _polars_check_dtype(obj.schema[col], col)
    times = obj.get_column(col).dt.convert_time_zone("UTC")
    ok = (times <= state.time).fill_null(False)
    if bool(ok.all()):
        return obj, True
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
    return obj.filter(ok), True


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
    DataFrames and LazyFrames, lists/tuples of ``PITRecord`` or dicts, single records (a
    single late record cannot be dropped, so it raises), and dicts, lists and tuples that
    nest any of these.
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
    r"(?<![0-9])((?:19|20)[0-9]{2})([-/.])([01]?[0-9])\2([0-3]?[0-9])"
    r"(?:[T ]([0-2][0-9]):([0-5][0-9])(?::([0-5][0-9])(?:\.[0-9]+)?)?"
    r"(Z|[+-][0-2][0-9]:?[0-5][0-9])?)?(?![0-9])"
)
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH = (
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
    r"|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
)
_MDY_RE = re.compile(
    r"(?<![A-Za-z])" + _MONTH + r"\s+([0-3]?[0-9])(?:st|nd|rd|th)?,?\s+((?:19|20)[0-9]{2})(?![0-9])",
    re.IGNORECASE,
)
_DMY_RE = re.compile(
    r"(?<![0-9])([0-3]?[0-9])(?:st|nd|rd|th)?\s+" + _MONTH + r",?\s+((?:19|20)[0-9]{2})(?![0-9])",
    re.IGNORECASE,
)


def scan_text(text: str, when: Any = None) -> List[str]:
    """Find dates in ``text`` that are later than the decision time.

    Matched forms: ``YYYY-MM-DD``, ``YYYY/MM/DD`` and ``YYYY.MM.DD`` (optionally with a
    time and an offset), ``May 11, 2024`` and ``11 May 2024``. A bare date counts as later
    only when the whole day is after the decision's calendar day in the ``asof`` zone; a
    timestamp with an offset is compared exactly; a timestamp without an offset is read in
    the ``asof`` zone (the ``tz=`` the decision was given in, else UTC). Numeric
    day-first/month-first dates (``05/11/2024``) are ambiguous and not matched. Returns the
    offending substrings (empty when clean). This is a heuristic for free-text tool output,
    used by ``guard(..., scan_text=True)``.
    """
    state = _resolve(when, None, None, None)
    if state is None or not isinstance(text, str):
        return []
    t = state.time
    local_day = t.astimezone(state.zone).date()
    found: List[Tuple[int, str]] = []
    for m in _DATE_RE.finditer(text):
        y, _sep, mo, d, hh, mi, ss, off = m.groups()
        try:
            day = _dt.date(int(y), int(mo), int(d))
        except ValueError:
            continue
        if hh is None:
            if day > local_day:
                found.append((m.start(), m.group(0)))
            continue
        try:
            stamp = _dt.datetime(int(y), int(mo), int(d), int(hh), int(mi), int(ss or 0))
        except ValueError:
            continue
        if off is None:
            stamp = stamp.replace(tzinfo=state.zone)
        elif off in ("Z", "z"):
            stamp = stamp.replace(tzinfo=UTC)
        else:
            sign = 1 if off[0] == "+" else -1
            digits = off[1:].replace(":", "")
            stamp = stamp.replace(
                tzinfo=_dt.timezone(sign * _dt.timedelta(hours=int(digits[:2]), minutes=int(digits[2:])))
            )
        if stamp > t:
            found.append((m.start(), m.group(0)))
    for rx, order in ((_MDY_RE, "mdy"), (_DMY_RE, "dmy")):
        for m in rx.finditer(text):
            if order == "mdy":
                mon, d, y = m.groups()
            else:
                d, mon, y = m.groups()
            try:
                day = _dt.date(int(y), _MONTHS[mon[:3].lower()], int(d))
            except (ValueError, KeyError):
                continue
            if day > local_day:
                found.append((m.start(), m.group(0)))
    return [s for _, s in sorted(found)]


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

_NOW_WORDS = ("now", "today", "latest", "current", "live")
_YMD8 = re.compile(r"((?:19|20)[0-9]{2})([01][0-9])([0-3][0-9])")
_NUMERIC = re.compile(r"[+-]?[0-9]+(?:\.[0-9]+)?")
_SLASHED = re.compile(r"((?:19|20)[0-9]{2})[/.]([01]?[0-9])[/.]([0-3]?[0-9])((?:[T ].*)?)")


def _ymd(y: int, m: int, d: int) -> Optional[_dt.date]:
    try:
        return _dt.date(y, m, d)
    except ValueError:
        return None


def _epoch(value: float) -> Optional[_dt.datetime]:
    """Seconds, milliseconds, microseconds or nanoseconds since 1970, told apart by size."""
    a = abs(value)
    if a >= 1e17:
        value /= 1e9
    elif a >= 1e14:
        value /= 1e6
    elif a >= 1e11:
        value /= 1e3
    try:
        return _dt.datetime.fromtimestamp(value, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _arg_time(raw: Any, tz: Any) -> Optional[_dt.datetime]:
    """Read a date-like argument value as an aware UTC time, or ``None`` if it is not one.

    Besides what ``to_utc`` reads: ``YYYYMMDD`` ints and strings (tushare, akshare),
    ``YYYY/MM/DD`` and ``YYYY.MM.DD``, epoch seconds / milliseconds / microseconds /
    nanoseconds (ccxt ``until``), and the words now / today / latest (the wall clock, so
    always later than a backtest's decision time).
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, numbers.Real) and not isinstance(raw, bool):
        try:
            v = float(raw)
        except (OverflowError, ValueError):
            return None
        if v != v or v in (float("inf"), float("-inf")):
            return None
        if v == int(v) and 19000101 <= v <= 21001231:
            iv = int(v)
            day = _ymd(iv // 10000, iv // 100 % 100, iv % 100)
            if day is not None:
                return to_utc(day, tz)
        return _epoch(v)
    if isinstance(raw, str):
        s = raw.strip()
        if s.lower() in _NOW_WORDS:
            return _dt.datetime.now(UTC)
        m = _YMD8.fullmatch(s)
        if m:
            day = _ymd(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return to_utc(day, tz) if day is not None else None
        if _NUMERIC.fullmatch(s):
            return _epoch(float(s))
        m = _SLASHED.fullmatch(s)
        if m:
            s = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}{m.group(4)}"
        try:
            return to_utc(s, tz, what="argument")
        except (TypeError, ValueError, OverflowError, OSError):
            return None
    try:
        return to_utc(raw, tz, what="argument")
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _check_args(
    sig: Optional[inspect.Signature],
    args: Sequence[Any],
    kwargs: dict,
    names: Iterable[str],
    state: _State,
    tz: Any,
    unparsed: str = "warn",
    where: str = "call",
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
    zone = tz if tz is not None else state.zone
    for name in names:
        if name not in bound:
            continue
        raw = bound[name]
        if raw is None:
            continue
        t = _arg_time(raw, zone)
        if t is None:
            msg = (
                f"{where}: argument {name}={raw!r} is not a time fin-pit can read, so it was "
                f"not checked against the decision time {state.time.isoformat()}"
            )
            if unparsed == "raise":
                raise LookaheadError(msg, decision_time=state.time, count=1)
            if unparsed == "warn":
                warnings.warn(msg, UnparsedDateWarning, stacklevel=4)
            continue
        # A bare date argument means "up to and including that day": the day must not
        # start after the decision time. Datetimes are compared exactly; naive ones are read
        # in the asof zone (or date_tz).
        if t > state.time:
            raise LookaheadError(
                f"argument {name}={raw!r} is later than the decision time "
                f"{state.time.isoformat()}",
                decision_time=state.time,
                available_at=t,
                count=1,
            )


def _item_filter(state: _State) -> Callable[[Any], Any]:
    """Per-item check for guarded generators: a late single record is dropped in
    ``mode="filter"`` (raised in ``"raise"``); anything else goes through ``_apply``."""

    def transform(item: Any) -> Any:
        stamped, t = _record_time(item, state.column)
        if stamped:
            if t is None or t > state.time:
                if state.mode == "filter":
                    return _SKIP
                raise _late_record(state, t)
            return item
        if isinstance(item, str):
            return _unstamped(item, state) if state.unstamped == "raise" else item
        return _apply(item, state)

    return transform


def guard(
    fn: Optional[F] = None,
    *,
    mode: Optional[str] = None,
    unstamped: Optional[str] = None,
    column: Optional[str] = None,
    date_args: Optional[Iterable[str]] = DEFAULT_DATE_ARGS,
    date_tz: Any = None,
    scan_text: bool = False,
    require_asof: Optional[bool] = None,
    unparsed: str = "warn",
) -> Any:
    """Wrap a callable so every call is checked against the active ``asof``.

    Before the call, arguments named in ``date_args`` (``curr_date``, ``end_date``, ...) that
    read as a time later than the decision time raise ``LookaheadError``. Naive values are
    read in ``date_tz``, by default the zone the ``asof`` was given in. ``YYYYMMDD``,
    ``YYYY/MM/DD``, epoch seconds or milliseconds and "today"/"now" are understood. A value
    that cannot be read warns (``unparsed="warn"``), raises (``"raise"``) or passes
    (``"ignore"``).

    After the call, the result goes through ``filter``/``check`` (``mode``), including
    records nested in dicts, lists and tuples; with ``scan_text=True`` string results are
    scanned for later dates. Generator and async generator functions are checked item by
    item.

    Outside any ``asof`` nothing can be checked: ``require_asof=True`` raises, ``False``
    passes silently, and the default (``None``) passes with a one-time
    ``UnguardedCallWarning`` per wrapped function, because the usual cause is a worker
    thread that did not inherit the context (see ``bind``).
    Works as ``guard(fn)`` and as ``@guard(mode="filter")``; async functions stay async.
    """
    opts = _GuardOptions(mode, unstamped, column, date_args, date_tz, scan_text, require_asof, unparsed)

    def deco(f: F) -> F:
        g = _Guard(f, opts)

        if inspect.iscoroutinefunction(f):

            @functools.wraps(f)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                state = g.before(args, kwargs)
                return g.after(await f(*args, **kwargs), state)

            awrapper.__fin_pit_guarded__ = True  # type: ignore[attr-defined]
            return awrapper  # type: ignore[return-value]

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            state = g.before(args, kwargs)
            return g.after(f(*args, **kwargs), state)

        wrapper.__fin_pit_guarded__ = True  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    if fn is not None:
        return deco(fn)
    return deco


class _GuardOptions:
    __slots__ = ("mode", "unstamped", "column", "names", "date_tz", "scan_text", "require_asof", "unparsed")

    def __init__(self, mode: Optional[str], unstamped: Optional[str], column: Optional[str],
                 date_args: Optional[Iterable[str]], date_tz: Any, scan_text: bool,
                 require_asof: Optional[bool], unparsed: str) -> None:
        if mode is not None and mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if unstamped is not None and unstamped not in UNSTAMPED:
            raise ValueError(f"unstamped must be one of {UNSTAMPED}, got {unstamped!r}")
        if unparsed not in UNPARSED:
            raise ValueError(f"unparsed must be one of {UNPARSED}, got {unparsed!r}")
        self.mode = mode
        self.unstamped = unstamped
        self.column = column
        self.names = tuple(date_args or ())
        self.date_tz = date_tz
        self.scan_text = scan_text
        self.require_asof = require_asof
        self.unparsed = unparsed


class _Guard:
    """The checks ``guard`` runs before and after one wrapped callable."""

    def __init__(self, f: Any, opts: _GuardOptions, name: Optional[str] = None) -> None:
        try:
            self.sig: Optional[inspect.Signature] = inspect.signature(f)
        except (TypeError, ValueError):
            self.sig = None
        self.opts = opts
        self.name = name or getattr(f, "__qualname__", getattr(f, "__name__", "call"))
        self._warned = False

    def before(self, args: Sequence[Any], kwargs: dict) -> Optional[_State]:
        o = self.opts
        state = _resolve(None, o.mode, o.unstamped, o.column)
        if state is None:
            if o.require_asof:
                raise LookaheadError(f"{self.name} needs an active asof (require_asof=True)")
            if o.require_asof is None and not self._warned:
                self._warned = True
                warnings.warn(
                    f"{self.name} is guarded by fin-pit but ran with no active asof, so nothing "
                    f"was checked. On a worker thread, submit pit.bind(fn) so the decision "
                    f"time follows; pass require_asof=False to silence this.",
                    UnguardedCallWarning,
                    stacklevel=4,
                )
            return None
        _check_args(self.sig, args, kwargs, o.names, state, o.date_tz, o.unparsed, self.name)
        return state

    def scan(self, text: str, state: _State) -> None:
        hits = _scan_with(text, state)
        if hits:
            raise LookaheadError(
                f"{len(hits)} date(s) in the output are later than the decision time "
                f"{state.time.isoformat()}: {', '.join(hits[:5])}",
                decision_time=state.time,
                count=len(hits),
            )

    def after(self, result: Any, state: Optional[_State]) -> Any:
        if state is None:
            return result
        if inspect.isgenerator(result) or inspect.isasyncgen(result):
            # Checked item by item while it is iterated, under the decision time captured
            # at call time.
            drive = _drive if inspect.isgenerator(result) else _adrive
            return drive(result, lambda: _STATE.set(state), _STATE.reset, _item_filter(state))
        if isinstance(result, str):
            if self.opts.scan_text:
                self.scan(result, state)
                return result
            return _unstamped(result, state) if state.unstamped == "raise" else result
        return _apply(result, state)


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
        # NaT sorts first, so a key's newest real vintage wins (as in Polars).
        tmp = tmp.sort_values(["__t", "__o"], kind="mergesort", na_position="first")
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
