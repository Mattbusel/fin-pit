"""yfinance bars and snapshots, stamped with when they became knowable.

Rules:

* **Daily (and longer) bars** (``confidence="rule"``): the bar for session D is available at
  D's close in the exchange's time zone plus ``lag``. The defaults are 16:00
  ``America/New_York`` and a 15-minute lag (a conservative allowance for the closing
  auction print and vendor delay; it is a parameter, not a measured figure). With
  ``calendar=`` an ``exchange_calendars`` calendar (for example ``"XNYS"``) gives each
  session's real close, so early-close days (13:00 New York) are right; otherwise early
  closes are stamped late, which is safe. Weekly and monthly bars are stamped at the close
  of the last day they cover (``bar_end``).
* **Intraday bars**: a bar labelled t with interval i is available at t + i + ``lag``.
* **24/7 markets** (``session="24/7"``, crypto): a daily bar dated D (UTC) is available at
  D+1 00:00 UTC plus ``lag``.
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


def stamp_bars(
    bars: Any,
    interval: str = "1d",
    *,
    tz: str = NEW_YORK,
    close_time: _dt.time = _dt.time(16, 0),
    lag: _dt.timedelta = DEFAULT_LAG,
    calendar: Any = None,
    session: str = "exchange",
) -> Any:
    """Return a copy of ``bars`` (a pandas DataFrame indexed by bar time, as
    ``yfinance.download``/``Ticker.history`` return it) with ``event_time``,
    ``available_at``, ``source`` and ``confidence`` columns.

    A naive index is read in ``tz``. ``session="24/7"`` stamps daily bars at the next UTC
    midnight. ``calendar`` takes an ``exchange_calendars`` name or object for real session
    closes (daily bars only; a session missing from the calendar falls back to
    ``close_time``).
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
    idx = bars.index
    if idx.tz is None:
        idx = idx.tz_localize(zone(tz) if session == "exchange" else UTC)
    out = bars.copy()
    if interval in _INTRADAY:
        step = _INTRADAY[interval]
        avail = [t.to_pydatetime().astimezone(UTC) + step + lag for t in idx]
    elif interval in _DAILY:
        local_days = []
        for t in idx:
            local = t.tz_convert(UTC if session == "24/7" else zone(tz))
            local_days.append(local.date())
        ends = _bar_end_days(local_days, interval)
        closes: Dict[Any, Any] = {}
        if calendar is not None and session == "exchange":
            closes = _session_closes(ends, calendar)
        avail = []
        for end in ends:
            if session == "24/7":
                base = _dt.datetime.combine(end + _dt.timedelta(days=1), _dt.time(0)).replace(tzinfo=UTC)
            else:
                base = closes.get(end) or _dt.datetime.combine(end, close_time).replace(tzinfo=zone(tz)).astimezone(UTC)
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
