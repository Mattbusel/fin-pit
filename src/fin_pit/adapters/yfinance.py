"""yfinance bars and snapshots, stamped with when they became knowable.

Rules:

* **Daily (and longer) bars** (``confidence="rule"``): the bar for session D is available at
  D's close in the exchange's time zone plus ``lag``. D is the bar label's date in the
  index's own zone (yfinance labels a session 00:00 exchange time). The default close is
  16:00 ``America/New_York`` and the default lag 15 minutes (a conservative allowance for
  the closing auction print and vendor delay; it is a parameter, not a measured figure).
  Any other exchange needs ``calendar=`` (an ``exchange_calendars`` name such as
  ``"XSHG"``, ``"XHKG"``, ``"XTKS"``, ``"XLON"``) or ``close_time=``; without one,
  ``stamp_bars`` raises rather than assume a New York close. With ``calendar=`` each
  session's real close is used, so early-close days (13:00 New York) are right; otherwise
  early closes are stamped late, which is safe. Weekly and monthly bars are stamped at the
  close of the last day they cover (``bar_end``).
* **Intraday bars**: a bar labelled t with interval i is available at t + i + ``lag``.
* **24/7 markets** (``session="24/7"``, crypto): a daily bar dated D (UTC) is available at
  D+1 00:00 UTC plus ``lag``. A daily index in UTC without ``session="24/7"``, ``tz=`` or
  ``calendar=`` raises, because a crypto bar stamped at a New York close would be about a
  day early.
* **Snapshots** (``Ticker.info``, ``fast_info``, ``get_*`` fundamentals tables with no
  publication date): ``confidence="live"``, stamped with the retrieval time, so any
  ``asof`` dated earlier refuses them (TradingAgents #1300). Quarterly statement tables
  only carry the period end, which is not when they were published; use
  ``fin_pit.adapters.edgar`` for filing-time fundamentals.

``stamp_bars`` is pure (no network). ``history`` and ``snapshot`` call yfinance if it is
installed.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, Optional

from .._time import UTC, to_utc, zone

__all__ = ["stamp_bars", "history", "snapshot", "DEFAULT_LAG"]

DEFAULT_LAG = _dt.timedelta(minutes=15)
NEW_YORK = "America/New_York"

_INTRADAY = {
    "1m": _dt.timedelta(minutes=1),
    "2m": _dt.timedelta(minutes=2),
    "5m": _dt.timedelta(minutes=5),
    "15m": _dt.timedelta(minutes=15),
    "30m": _dt.timedelta(minutes=30),
    "60m": _dt.timedelta(minutes=60),
    "90m": _dt.timedelta(minutes=90),
    "1h": _dt.timedelta(hours=1),
}
_DAILY = {"1d", "5d", "1wk", "1mo", "3mo"}


def _session_closes(days: Any, calendar: Any) -> Dict[Any, Any]:
    try:
        import exchange_calendars as xcals
    except ImportError as exc:  # pragma: no cover
        raise ImportError("calendar= needs exchange_calendars: pip install exchange_calendars") from exc
    import pandas as pd

    cal = xcals.get_calendar(calendar) if isinstance(calendar, str) else calendar
    out = {}
    for d in days:
        ts = pd.Timestamp(d)
        try:
            out[d] = cal.session_close(ts).tz_convert("UTC").to_pydatetime()
        except Exception:
            out[d] = None
    return out


def _clean_zone(tz: Any) -> Any:
    """A zoneinfo zone for an index tz (pytz zones misbehave with ``replace(tzinfo=)``)."""
    try:
        return zone(_zone_name(tz))
    except (TypeError, ValueError):
        return tz


def _zone_name(tz: Any) -> str:
    return str(getattr(tz, "key", None) or getattr(tz, "zone", None) or tz)


def _is_utc(tz: Any) -> bool:
    name = _zone_name(tz)
    if name.upper() in ("UTC", "Z", "ETC/UTC", "GMT", "ETC/GMT", "UTC+00:00"):
        return True
    try:
        off = _dt.datetime(2024, 1, 1, tzinfo=tz).utcoffset()
        off_july = _dt.datetime(2024, 7, 1, tzinfo=tz).utcoffset()
    except Exception:  # pragma: no cover - exotic tzinfo
        return False
    return off == _dt.timedelta(0) and off_july == _dt.timedelta(0)


def stamp_bars(
    bars: Any,
    interval: str = "1d",
    *,
    tz: Optional[str] = None,
    close_time: Optional[_dt.time] = None,
    lag: _dt.timedelta = DEFAULT_LAG,
    calendar: Any = None,
    session: str = "exchange",
) -> Any:
    """Return a copy of ``bars`` (a pandas DataFrame indexed by bar time, as
    ``yfinance.download``/``Ticker.history`` return it) with ``event_time``,
    ``available_at``, ``source`` and ``confidence`` columns.

    Daily bars: the session day is the bar label's date in the index's own time zone
    (``Ticker.history`` labels a session 00:00 in the exchange's zone), or in ``tz``
    (default ``America/New_York``) for a naive index. The close is ``calendar``'s session
    close when given, else ``close_time`` in ``tz`` (or the index's zone). Only New York has a
    default close (16:00): for any other zone pass ``calendar=`` (``"XSHG"``, ``"XHKG"``,
    ``"XTKS"``, ``"XLON"``, ...) or ``close_time=``, otherwise ``ValueError``, so a non-US bar
    is never stamped with a New York close. A UTC index needs ``session="24/7"`` (crypto) or
    an exchange via ``tz``/``calendar``. ``session="24/7"`` stamps daily bars at the next UTC
    midnight. A calendar session missing from the calendar falls back to ``close_time``,
    else to the end of that day.
    """
    import pandas as pd

    if not isinstance(bars, pd.DataFrame):
        raise TypeError("stamp_bars needs a pandas DataFrame")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise TypeError("stamp_bars needs a DatetimeIndex (bar times)")
    if not isinstance(lag, _dt.timedelta) or lag < _dt.timedelta(0):
        raise ValueError("lag must be a non-negative timedelta")
    if session not in ("exchange", "24/7"):
        raise ValueError("session must be 'exchange' or '24/7'")
    if close_time is not None and not isinstance(close_time, _dt.time):
        raise TypeError("close_time must be a datetime.time")
    idx = bars.index
    naive = idx.tz is None
    if naive:
        idx = idx.tz_localize(zone(tz or NEW_YORK) if session == "exchange" else UTC)
    out = bars.copy()
    if interval in _INTRADAY:
        step = _INTRADAY[interval]
        avail = [t.to_pydatetime().astimezone(UTC) + step + lag for t in idx]
    elif interval in _DAILY:
        if session == "24/7":
            day_zone: Any = UTC
            close_zone: Any = UTC
        else:
            day_zone = _clean_zone(idx.tz)
            close_zone = zone(tz) if tz is not None else day_zone
            if not naive and tz is None and calendar is None and _is_utc(idx.tz):
                raise ValueError(
                    "daily bars with a UTC index: pass session='24/7' for a 24/7 market "
                    "(crypto), or tz= / calendar= for an exchange"
                )
            if close_time is None and calendar is None and _zone_name(close_zone) != NEW_YORK:
                raise ValueError(
                    f"no default close for {_zone_name(close_zone)}: pass calendar= (an "
                    f"exchange_calendars name such as 'XSHG', 'XHKG', 'XTKS' or 'XLON') or "
                    f"close_time= (the session close in that zone)"
                )
        local_days = [t.tz_convert(day_zone).date() for t in idx]
        ends = _bar_end_days(local_days, interval)
        closes: Dict[Any, Any] = {}
        if calendar is not None and session == "exchange":
            closes = _session_closes(ends, calendar)
        when_closed = close_time or _dt.time(16, 0)
        avail = []
        for end in ends:
            if session == "24/7":
                base = _dt.datetime.combine(end + _dt.timedelta(days=1), _dt.time(0)).replace(tzinfo=UTC)
            elif closes.get(end) is not None:
                base = closes[end]
            elif close_time is None and calendar is not None:
                # Not a session in the calendar and no close_time: the end of the day.
                base = _dt.datetime.combine(end + _dt.timedelta(days=1), _dt.time(0)).replace(
                    tzinfo=zone(close_zone)
                ).astimezone(UTC)
            else:
                base = _dt.datetime.combine(end, when_closed).replace(tzinfo=zone(close_zone)).astimezone(UTC)
            avail.append(base + lag)
    else:
        raise ValueError(f"unknown interval {interval!r}")
    out["event_time"] = pd.DatetimeIndex(idx).tz_convert("UTC")
    out["available_at"] = pd.to_datetime(avail, utc=True)
    out["source"] = "yfinance"
    out["confidence"] = "rule"
    return out


def _bar_end_days(days: Any, interval: str) -> Any:
    """Last calendar day each bar covers (the same day for daily bars)."""
    if interval in ("1d",):
        return list(days)
    ends = []
    for i, d in enumerate(days):
        nxt = days[i + 1] if i + 1 < len(days) else None
        if interval in ("5d", "1wk"):
            natural = d + _dt.timedelta(days=(4 - d.weekday()) % 7)  # the Friday on or after d
        elif interval == "1mo":
            first_next = (d.replace(day=1) + _dt.timedelta(days=32)).replace(day=1)
            natural = first_next - _dt.timedelta(days=1)
        else:  # 3mo
            m = d.month - 1 + 3
            first_next = _dt.date(d.year + m // 12, m % 12 + 1, 1)
            natural = first_next - _dt.timedelta(days=1)
        if nxt is not None and nxt <= natural:
            natural = nxt - _dt.timedelta(days=1)
        ends.append(max(natural, d))
    return ends


def history(ticker: str, *, interval: str = "1d", **kwargs: Any) -> Any:
    """``yfinance.Ticker(ticker).history(interval=..., **kwargs)`` stamped with
    ``stamp_bars``. Stamping options (``tz``, ``close_time``, ``lag``, ``calendar``,
    ``session``) are taken out of ``kwargs`` first. Inside ``asof`` the bars are cut to
    those available at the decision time."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("history() needs yfinance: pip install yfinance") from exc
    from ..guard import filter as pit_filter

    stamp = {k: kwargs.pop(k) for k in ("tz", "close_time", "lag", "calendar", "session") if k in kwargs}
    bars = yf.Ticker(ticker).history(interval=interval, **kwargs)
    return pit_filter(stamp_bars(bars, interval, **stamp))


def snapshot(data: Any, retrieved_at: Any = None) -> Dict[str, Any]:
    """Stamp a live snapshot (``Ticker.info``, a quote dict) with its retrieval time.

    Returns a dict with the snapshot's fields plus ``available_at`` (``retrieved_at`` or
    now, UTC) and ``confidence="live"``. ``fin_pit.check`` on it inside any earlier
    ``asof`` raises ``LookaheadError``; a ``guard``-ed tool returning it does the same.
    If ``data`` is a ticker symbol, ``yfinance.Ticker(data).info`` is fetched.
    """
    if isinstance(data, str):
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            raise ImportError("snapshot('SYMBOL') needs yfinance") from exc
        data = yf.Ticker(data).info
    if not isinstance(data, dict):
        raise TypeError("snapshot needs a dict (Ticker.info) or a ticker symbol")
    when = to_utc(retrieved_at, what="retrieved_at") if retrieved_at is not None else _dt.datetime.now(UTC)
    out = dict(data)
    out["available_at"] = when
    out["source"] = "yfinance"
    out["confidence"] = "live"
    return out
