"""``PITRecord``: one value plus the two times that matter.

``event_time`` is the time the value is about (the quarter end, the observation month, the
bar's session). ``available_at`` is the first moment someone could have known it (the filing
was accepted, the vintage was released, the bar closed). Look-ahead bias is any read where
``available_at`` is after the decision time; lining data up on ``event_time`` instead is the
bug behind TradingAgents #1275 and FinRL #1463.

``confidence`` says how ``available_at`` was obtained, so a backtest can choose how strict to
be:

``exact``
    The source publishes the moment itself (EDGAR ``acceptanceDateTime``).
``rule``
    Derived by a documented rule from the source's own fields (a daily bar is available at
    the session close plus a lag; a FRED vintage at its release time).
``bound``
    A conservative upper bound: the true time is at or before it (EDGAR's ``filed`` date
    read as the end of that day in New York when the acceptance time is unknown).
``live``
    The source only serves the current value (``yfinance`` ``Ticker.info``). It is stamped
    with the retrieval time, so any backtest dated earlier refuses it.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
from typing import Any, Dict, Optional

from ._time import to_utc

__all__ = ["PITRecord", "CONFIDENCE_LEVELS"]

CONFIDENCE_LEVELS = ("exact", "rule", "bound", "live")


@dataclasses.dataclass(frozen=True)
class PITRecord:
    """A value stamped with when it describes and when it became knowable.

    Times are normalised to aware UTC datetimes on construction; naive input raises.
    ``available_at`` earlier than ``event_time`` is allowed (a forecast or a guidance figure
    describes the future), so it is not checked.
    """

    event_time: _dt.datetime
    available_at: _dt.datetime
    vintage: Optional[str] = None
    source: str = ""
    confidence: str = "rule"
    value: Any = None
    key: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_time", to_utc(self.event_time, what="event_time"))
        object.__setattr__(self, "available_at", to_utc(self.available_at, what="available_at"))
        if self.confidence not in CONFIDENCE_LEVELS:
            raise ValueError(
                f"confidence must be one of {CONFIDENCE_LEVELS}, got {self.confidence!r}"
            )
        if self.vintage is not None and not isinstance(self.vintage, str):
            object.__setattr__(self, "vintage", str(self.vintage))

    def known_at(self, decision_time: Any) -> bool:
        """True when this record was available at ``decision_time`` (an aware time)."""
        return self.available_at <= to_utc(decision_time, what="decision_time")

    def to_dict(self) -> Dict[str, Any]:
        """Plain dict with ISO 8601 times (for JSON or a DataFrame row)."""
        return {
            "key": self.key,
            "event_time": self.event_time.isoformat(),
            "available_at": self.available_at.isoformat(),
            "vintage": self.vintage,
            "source": self.source,
            "confidence": self.confidence,
            "value": self.value,
        }
