"""FRED / ALFRED vintages as point-in-time records.

FRED serves each series twice over: ``observation_start``/``observation_end`` bound the
period the data describes, and ``realtime_start``/``realtime_end`` bound the vintage (which
values were current on which days). The vintage window defaults to today, so a request
bounded only by observation date returns today's revised numbers, including observations
published after the decision date (TradingAgents #1275).

Rules used here (``confidence="rule"``):

* a vintage row with ``realtime_start = D`` became available on day D at the series'
  release time in New York. FRED dates are days, not times, so the release time is a
  parameter: ``release_time=None`` (the default) means the end of day D (23:59:59.999999
  New York), which is never early; pass ``datetime.time(8, 30)`` for 08:30 releases such as
  CPI and payrolls if you need same-day decisions to see them;
* ``event_time`` is the observation ``date`` (the start of the period it describes), read
  as 00:00 New York.

Two ways in:

* ``from_fredapi(df, series_id)`` turns ``fredapi.Fred.get_series_all_releases(...)`` (a
  DataFrame with ``date``, ``realtime_start``, ``value``) into a stamped DataFrame;
* ``observation_params(decision_time, release_time)`` gives the ``realtime_start`` /
  ``realtime_end`` query parameters for a raw ``series/observations`` call, so the vendor
  returns the vintage that was current at the decision time.

``FredSource`` wraps a ``fredapi.Fred`` client so ``get(series_id)`` inside ``asof`` returns
the as-of series directly.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional

from .._time import UTC, to_utc, zone
from ..records import PITRecord

__all__ = [
    "NEW_YORK",
    "available_at",
    "from_fredapi",
    "records_from_fredapi",
    "observation_params",
    "asof_series",
    "FredSource",
]

NEW_YORK = "America/New_York"
_END_OF_DAY = _dt.time(23, 59, 59, 999999)


def _day(value: Any) -> _dt.date:
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().date()
    if isinstance(value, str):
        return _dt.date.fromisoformat(value.strip()[:10])
    raise TypeError(f"expected a date, got {type(value).__name__}")


def available_at(realtime_start: Any, release_time: Optional[_dt.time] = None, tz: str = NEW_YORK) -> _dt.datetime:
    """When a vintage dated ``realtime_start`` became available (aware UTC)."""
    day = _day(realtime_start)
    t = release_time if release_time is not None else _END_OF_DAY
    if not isinstance(t, _dt.time):
        raise TypeError("release_time must be a datetime.time")
    local = _dt.datetime.combine(day, t.replace(tzinfo=None)).replace(tzinfo=zone(tz))
    return local.astimezone(UTC)


def observation_params(
    decision_time: Any, release_time: Optional[_dt.time] = None, tz: str = NEW_YORK
) -> Dict[str, str]:
    """``realtime_start``/``realtime_end`` for FRED's ``series/observations`` as of a time.

    The vintage date is the decision's New York date, or the day before when the decision
    comes before that day's release time (the end of day by default, so by default the
    vintage date is always the previous day unless the decision is at 23:59:59.999999).
    Pass both values with ``observation_end`` set as you already do.
    """
    t = to_utc(decision_time, what="decision_time")
    local = t.astimezone(zone(tz))
    day = local.date()
    if available_at(day, release_time, tz) > t:
        day = day - _dt.timedelta(days=1)
    iso = day.isoformat()
    return {"realtime_start": iso, "realtime_end": iso}


def from_fredapi(
    frame: Any,
    series_id: str,
    release_time: Optional[_dt.time] = None,
    tz: str = NEW_YORK,
) -> Any:
    """Stamp ``Fred.get_series_all_releases`` output.

    Returns a pandas DataFrame with ``series_id``, ``event_time``, ``available_at``,
    ``vintage`` (the ``realtime_start`` date, ISO), ``value``, ``source`` and
    ``confidence``, one row per (observation, vintage). Use ``fin_pit.latest(df,
    "event_time")`` inside ``asof`` to get the series as it stood at the decision time.
    """
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("from_fredapi needs the DataFrame from Fred.get_series_all_releases")
    missing = [c for c in ("date", "realtime_start", "value") if c not in frame.columns]
    if missing:
        raise ValueError(f"missing column(s) {missing}; expected fredapi's all-releases frame")
    nyc = zone(tz)
    rows: List[Dict[str, Any]] = []
    for date, rs, value in zip(frame["date"], frame["realtime_start"], frame["value"]):
        if date is None or rs is None or date is pd.NaT or rs is pd.NaT:
            raise ValueError("date and realtime_start must not be missing")
        d = _day(date)
        rows.append(
            {
                "series_id": series_id,
                "event_time": _dt.datetime(d.year, d.month, d.day, tzinfo=nyc).astimezone(UTC),
                "available_at": available_at(rs, release_time, tz),
                "vintage": _day(rs).isoformat(),
                "value": float(value) if value is not None else float("nan"),
                "source": "fred",
                "confidence": "rule",
            }
        )
    out = pd.DataFrame(rows, columns=["series_id", "event_time", "available_at", "vintage", "value", "source", "confidence"])
    for c in ("event_time", "available_at"):
        out[c] = pd.to_datetime(out[c], utc=True)
    return out


def records_from_fredapi(
    frame: Any, series_id: str, release_time: Optional[_dt.time] = None, tz: str = NEW_YORK
) -> List[PITRecord]:
    """Same as ``from_fredapi`` but as a list of ``PITRecord``."""
    df = from_fredapi(frame, series_id, release_time, tz)
    return [
        PITRecord(
            event_time=r.event_time.to_pydatetime(),
            available_at=r.available_at.to_pydatetime(),
            vintage=r.vintage,
            source="fred",
            confidence="rule",
            value=r.value,
            key=series_id,
        )
        for r in df.itertuples(index=False)
    ]


def asof_series(stamped: Any, when: Any = None) -> Any:
    """The series as it stood at the decision time: a pandas Series indexed by observation
    date (New York), holding the newest vintage of each observation available then."""
    import pandas as pd

    from ..guard import latest

    cut = latest(stamped, "event_time", when)
    s = pd.Series(
        cut["value"].to_numpy(),
        index=pd.DatetimeIndex(cut["event_time"]).tz_convert(NEW_YORK).tz_localize(None).normalize(),
        name=cut["series_id"].iloc[0] if len(cut) else None,
    )
    return s.sort_index()


class FredSource:
    """A ``fredapi.Fred`` client behind the point-in-time guard.

    ``get(series_id)`` fetches every vintage once (cached), stamps it, and inside ``asof``
    returns the series as it stood at the decision time. Outside ``asof`` it returns the
    latest vintage, like ``Fred.get_series``.
    """

    def __init__(self, client: Any, release_times: Optional[Dict[str, _dt.time]] = None, tz: str = NEW_YORK) -> None:
        if not hasattr(client, "get_series_all_releases"):
            raise TypeError("client must be a fredapi.Fred (needs get_series_all_releases)")
        self.client = client
        self.release_times = dict(release_times or {})
        self.tz = tz
        self._cache: Dict[str, Any] = {}

    def vintages(self, series_id: str) -> Any:
        """Every vintage of ``series_id``, stamped (cached after the first call)."""
        if series_id not in self._cache:
            raw = self.client.get_series_all_releases(series_id)
            self._cache[series_id] = from_fredapi(raw, series_id, self.release_times.get(series_id), self.tz)
        return self._cache[series_id]

    def get(self, series_id: str, when: Any = None) -> Any:
        """The series as of ``when`` or the active ``asof``."""
        return asof_series(self.vintages(series_id), when)
