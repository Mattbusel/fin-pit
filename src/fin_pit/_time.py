"""Timestamp parsing. Every time inside fin-pit is a timezone-aware ``datetime`` in UTC.

Naive timestamps are refused: a naive "2024-05-10 16:00" is New York to one data vendor and
UTC to the next, and that ambiguity is how a decision dated d ends up reading news from the
evening of d (TradingAgents #1518). Pass an aware datetime, or a naive one together with
``tz=``.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Optional

try:  # Python 3.9+
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment,misc]

UTC = _dt.timezone.utc

__all__ = ["UTC", "source_zone", "to_utc", "zone"]


def zone(tz: Any) -> _dt.tzinfo:
    """Turn ``"America/New_York"``, ``"UTC"`` or a tzinfo into a tzinfo."""
    if isinstance(tz, _dt.tzinfo):
        return tz
    if isinstance(tz, str):
        if tz.upper() in ("UTC", "Z"):
            return UTC
        if ZoneInfo is None:  # pragma: no cover
            raise ValueError("zoneinfo is unavailable; pass a tzinfo object")
        try:
            return ZoneInfo(tz)
        except Exception as exc:  # ZoneInfoNotFoundError subclasses KeyError
            raise ValueError(
                f"unknown time zone {tz!r} (on Windows, `pip install tzdata`)"
            ) from exc
    raise TypeError(f"tz must be a str or tzinfo, got {type(tz).__name__}")


def to_utc(value: Any, tz: Optional[Any] = None, *, what: str = "time") -> _dt.datetime:
    """Parse ``value`` into an aware UTC datetime.

    Accepts an aware ``datetime``, a pandas ``Timestamp``, a ``numpy.datetime64`` (read as
    UTC, the way numpy stores it), an ISO 8601 string, a ``date`` or a POSIX timestamp in
    seconds. A naive datetime or string, or a bare ``date``, needs ``tz``; a bare date means
    00:00 of that day in ``tz``, the earliest moment of the day, so it never admits anything
    published later that day.
    """
    if value is None:
        raise ValueError(f"{what} is None")
    if isinstance(value, bool):
        raise TypeError(f"{what} must be a time, got bool")
    # pandas.Timestamp is a datetime subclass; numpy.datetime64 has no tzinfo at all.
    tname = type(value).__name__
    if tname == "datetime64":
        import numpy as np

        if np.isnat(value):
            raise ValueError(f"{what} is NaT")
        ns = int(value.astype("datetime64[ns]").astype("int64"))
        return _dt.datetime(1970, 1, 1, tzinfo=UTC) + _dt.timedelta(microseconds=ns // 1000)
    if tname == "NaTType":
        raise ValueError(f"{what} is NaT")
    if isinstance(value, (int, float)):
        if value != value:  # NaN
            raise ValueError(f"{what} is NaN")
        try:
            return _dt.datetime.fromtimestamp(float(value), tz=UTC)
        except (OverflowError, OSError) as exc:  # out of range (OSError on Windows)
            raise ValueError(f"{what} {value!r} is out of range as POSIX seconds") from exc
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{what} is an empty string")
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            parsed: Any = _dt.datetime.fromisoformat(text)
        except ValueError:
            try:
                parsed = _dt.date.fromisoformat(text)
            except ValueError as exc:
                raise ValueError(f"{what}: cannot parse {value!r} as ISO 8601") from exc
        value = parsed
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            if tz is None:
                raise ValueError(
                    f"{what} {value.isoformat()} has no time zone; pass an aware datetime "
                    f"or tz= (for example tz='America/New_York')"
                )
            value = value.replace(tzinfo=zone(tz))
        if hasattr(value, "to_pydatetime"):
            value = value.to_pydatetime()
        return value.astimezone(UTC)
    if isinstance(value, _dt.date):
        if tz is None:
            raise ValueError(
                f"{what} {value.isoformat()} is a date with no time; pass a datetime, or "
                f"tz= to read it as 00:00 of that day in tz"
            )
        return _dt.datetime(value.year, value.month, value.day, tzinfo=zone(tz)).astimezone(UTC)
    raise TypeError(f"{what} must be a datetime, date, str or number, got {type(value).__name__}")


def source_zone(value: Any, tz: Optional[Any] = None) -> _dt.tzinfo:
    """The zone a time was given in: ``tz`` when passed, else the offset of an aware
    datetime or ISO string, else UTC. Used to read later naive values the same way."""
    if tz is not None:
        return zone(tz)
    if isinstance(value, _dt.datetime):
        if value.tzinfo is not None and value.tzinfo.utcoffset(value) is not None:
            return value.tzinfo
        return UTC
    if isinstance(value, str):
        text = value.strip()
        if text.endswith(("Z", "z")):
            return UTC
        try:
            parsed = _dt.datetime.fromisoformat(text)
        except ValueError:
            return UTC
        if parsed.tzinfo is not None:
            return parsed.tzinfo
    return UTC
