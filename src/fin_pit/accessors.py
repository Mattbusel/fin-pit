"""``df.pit`` on pandas DataFrames and Polars DataFrames.

Registered when ``fin_pit`` is imported and pandas / Polars are installed::

    import fin_pit as pit

    df.pit.asof(t)                   # rows available at t (pit.filter)
    df.pit.check(t)                  # raise LookaheadError if any row is not
    df.pit.latest("event_time", t)   # newest vintage per key known at t
    df.pit.indicator("sma", 200)     # strict warmup + bars_available / bars_required (pandas)
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Union

from .guard import check as _check
from .guard import filter as _filter
from .guard import latest as _latest

__all__ = ["register"]

_registered = {"pandas": False, "polars": False}


class _PitAccessor:
    def __init__(self, obj: Any) -> None:
        self._obj = obj

    def asof(self, when: Any = None, *, column: Optional[str] = None) -> Any:
        """Rows available at ``when`` (or the active ``asof``); see ``fin_pit.filter``."""
        return _filter(self._obj, when, column=column)

    def check(self, when: Any = None, *, column: Optional[str] = None) -> Any:
        """Raise ``LookaheadError`` if any row is not available; see ``fin_pit.check``."""
        return _check(self._obj, when, column=column)

    def latest(
        self, on: Union[str, Sequence[str]] = "event_time", when: Any = None, *, column: Optional[str] = None
    ) -> Any:
        """Newest vintage per key available at ``when``; see ``fin_pit.latest``."""
        return _latest(self._obj, on, when, column=column)


class _PandasPit(_PitAccessor):
    def indicator(self, name: str, *args: Any, settle: int = 0, **params: Any) -> Any:
        """Strict-warmup indicator; see ``fin_pit.warmup.indicator``."""
        from .warmup import indicator

        return indicator(self._obj, name, *args, settle=settle, **params)

    def required_bars(self, name: str, *args: Any, **params: Any) -> int:
        """Bars the indicator needs before its first value; see ``fin_pit.required_bars``."""
        from .warmup import required_bars

        return required_bars(name, *args, **params)


def register() -> None:
    """Register ``.pit`` on pandas and Polars DataFrames (safe to call more than once)."""
    import warnings

    if not _registered["pandas"]:
        try:
            import pandas as pd
        except ImportError:
            pass
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pd.api.extensions.register_dataframe_accessor("pit")(_PandasPit)
            _registered["pandas"] = True
    if not _registered["polars"]:
        try:
            import polars as pl
        except ImportError:
            pass
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pl.api.register_dataframe_namespace("pit")(_PitAccessor)
                pl.api.register_lazyframe_namespace("pit")(_PitAccessor)
            _registered["polars"] = True
