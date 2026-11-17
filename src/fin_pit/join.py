"""``asof_join``: attach to each decision row the newest record known at its decision time.

This is the fix for the most common look-ahead in factor backtests: fundamentals joined on
the fiscal period end and back-filled (FinRL #1463). ``asof_join`` joins on availability
instead: each left row gets, per ``by`` key, the right row with the greatest
``available_at`` that is at or before the left row's decision time. Nothing is back-filled,
and keys never bleed into each other.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Union

from .guard import AVAILABLE_AT, _pandas_times

__all__ = ["asof_join"]


def asof_join(
    left: Any,
    right: Any,
    left_time: str,
    *,
    by: Union[None, str, Sequence[str]] = None,
    right_time: str = AVAILABLE_AT,
    tolerance: Any = None,
    suffixes: Sequence[str] = ("", "_right"),
) -> Any:
    """Point-in-time join of two pandas DataFrames.

    ``left[left_time]`` is each row's decision time, ``right[right_time]`` each record's
    availability; both must be timezone-aware. ``by`` keys (for example ``"tic"``) must
    match exactly. ``tolerance`` (a ``Timedelta``) drops matches older than that. The
    result keeps ``left``'s row order and index; right columns are NaN where nothing was
    known yet.
    """
    import pandas as pd

    if not isinstance(left, pd.DataFrame) or not isinstance(right, pd.DataFrame):
        raise TypeError("asof_join needs two pandas DataFrames")
    keys = [] if by is None else ([by] if isinstance(by, str) else list(by))
    lt = _pandas_times(left[left_time], left_time)
    rt = _pandas_times(right[right_time], right_time)
    if rt.isna().any():
        raise ValueError(f"right[{right_time!r}] has missing times; drop or fill them first")
    if lt.isna().any():
        raise ValueError(f"left[{left_time!r}] has missing times")
    if len(left) == 0 or len(right) == 0:
        # Nothing to match: every right column is missing (merge_asof refuses empty keys of
        # mismatched dtypes, so this case is built directly).
        out = left.copy()
        for c in right.columns:
            if c in keys:
                continue
            name = f"{c}{suffixes[1]}" if c in left.columns else c
            out[name] = right[c].iloc[:0].reindex(range(len(left))).to_numpy()
        return out
    lcol, rcol = "__fin_pit_left_t", "__fin_pit_right_t"
    lo = left.assign(**{lcol: lt, "__fin_pit_order": range(len(left))})
    ro = right.assign(**{rcol: rt})
    lo = lo.sort_values(lcol, kind="mergesort")
    ro = ro.sort_values(rcol, kind="mergesort")
    merged = pd.merge_asof(
        lo,
        ro,
        left_on=lcol,
        right_on=rcol,
        by=keys or None,
        direction="backward",
        allow_exact_matches=True,
        tolerance=tolerance,
        suffixes=tuple(suffixes),
    )
    merged = merged.sort_values("__fin_pit_order", kind="mergesort")
    merged.index = left.index
    return merged.drop(columns=[lcol, rcol, "__fin_pit_order"])
