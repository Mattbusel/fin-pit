"""fin-pit: point-in-time guards for backtests and trading agents.

Every value carries ``available_at``, the first moment it could have been known. Code runs
under ``asof(decision_time)``, and every guarded read refuses (or drops) what was not yet
knowable::

    import fin_pit as pit

    with pit.asof("2023-03-31 16:00", tz="America/New_York"):
        cpi = pit.filter(vintages)          # only the CPI prints released by then
        tool = pit.guard_tool(get_news)     # LangChain / LangGraph tool behind the firewall

    df.pit.indicator("sma", 200)            # NaN until 200 bars exist, plus coverage columns

See https://github.com/Mattbusel/fin-pit/tree/main/docs for the guide and SPEC.md for the
indicator conformance vectors.
"""
from __future__ import annotations

from ._time import to_utc
from .guard import (
    AVAILABLE_AT,
    LookaheadError,
    UnstampedError,
    asof,
    check,
    current,
    decision_time,
    filter,
    guard,
    latest,
    scan_text,
)
from .join import asof_join
from .langchain import guard_tool, guard_tools
from .records import CONFIDENCE_LEVELS, PITRecord
from .warmup import coverage, indicator, required_bars

__version__ = "0.1.0"

__all__ = [
    "AVAILABLE_AT",
    "CONFIDENCE_LEVELS",
    "LookaheadError",
    "PITRecord",
    "UnstampedError",
    "__version__",
    "asof",
    "asof_join",
    "check",
    "coverage",
    "current",
    "decision_time",
    "filter",
    "guard",
    "guard_tool",
    "guard_tools",
    "indicator",
    "latest",
    "required_bars",
    "scan_text",
    "to_utc",
]

from .accessors import register as _register

_register()
