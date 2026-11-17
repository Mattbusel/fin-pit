"""SEC EDGAR filings and XBRL facts stamped with their acceptance time.

EDGAR's XBRL ``companyfacts`` API gives each fact a ``filed`` date but no time. Comparing
that date with a decision date admits a 10-Q or an earnings 8-K accepted at 16:05 New York to
a decision made at the 16:00 close of the same day (TradingAgents #1518). The submissions API
(``https://data.sec.gov/submissions/CIK##########.json``) carries ``acceptanceDateTime`` for
every accession number; it is UTC (Apple's 16:30 New York earnings 8-Ks read ``20:30:..Z`` in
summer and ``21:30:..Z`` in winter, checked against the 2025-2026 filings on 2026-11-17).

Rules:

* ``available_at`` = the filing's ``acceptanceDateTime`` (``confidence="exact"``);
* when the accession number is not in the submissions data given, ``available_at`` = the
  end of the ``filed`` day in New York (``confidence="bound"``). EDGAR gives a filing
  accepted after 17:30 the next business day's filing date, so acceptance is always on or
  before the filing date and this bound is never early;
* ``event_time`` = the fact's period ``end`` (00:00 New York);
* ``vintage`` = the accession number, so restatements in later filings are later vintages
  of the same (concept, unit, period) key.

These functions parse JSON you fetched; they do not fetch. SEC asks automated clients to
send a descriptive User-Agent and to stay under 10 requests per second.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from .._time import UTC, to_utc, zone
from ..records import PITRecord

__all__ = [
    "acceptance_times",
    "filings",
    "facts",
    "fact_records",
    "filing_available_at",
]

NEW_YORK = "America/New_York"
_END_OF_DAY = _dt.time(23, 59, 59, 999999)


def _recent_blocks(submissions: Union[Mapping[str, Any], Iterable[Mapping[str, Any]]]) -> List[Mapping[str, Any]]:
    """Accept the submissions JSON, its ``filings.recent`` block, or the extra
    ``CIK##########-submissions-###.json`` pages (or a list of any of these)."""
    if isinstance(submissions, Mapping):
        items: List[Any] = [submissions]
    else:
        items = list(submissions)
    blocks: List[Mapping[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise TypeError("submissions must be a dict (or a list of dicts) from the EDGAR submissions API")
        if "filings" in item:
            recent = item["filings"].get("recent") if isinstance(item["filings"], Mapping) else None
            if not isinstance(recent, Mapping):
                raise ValueError("submissions JSON has no filings.recent block")
            blocks.append(recent)
        elif "accessionNumber" in item:
            blocks.append(item)
        else:
            raise ValueError("not an EDGAR submissions document (no filings.recent or accessionNumber)")
    return blocks


def _columns(block: Mapping[str, Any], names: Iterable[str]) -> List[List[Any]]:
    cols = []
    length = None
    for n in names:
        col = block.get(n)
        if col is None:
            col = []
        if not isinstance(col, list):
            raise ValueError(f"filings.{n} must be a list")
        if length is None:
            length = len(col)
        elif col and len(col) != length:
            raise ValueError(f"filings.{n} has {len(col)} entries, expected {length}")
        cols.append(col)
    length = length or 0
    return [c if c else [None] * length for c in cols]


def _parse_acceptance(raw: Any) -> _dt.datetime:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"bad acceptanceDateTime {raw!r}")
    text = raw.strip()
    if text[-1:] not in ("Z", "z") and "+" not in text[10:] and "-" not in text[10:]:
        # The API writes "...Z"; anything without an offset is read as UTC as well.
        text = text + "Z"
    return to_utc(text, what="acceptanceDateTime")


def acceptance_times(submissions: Any) -> Dict[str, _dt.datetime]:
    """Accession number -> acceptance time (aware UTC) from submissions JSON."""
    out: Dict[str, _dt.datetime] = {}
    for block in _recent_blocks(submissions):
        accn, acc = _columns(block, ("accessionNumber", "acceptanceDateTime"))
        for a, t in zip(accn, acc):
            if not isinstance(a, str) or t is None:
                continue
            out[a] = _parse_acceptance(t)
    return out


def filing_available_at(filed: Any, acceptance: Optional[Any] = None) -> Any:
    """``(available_at, confidence)`` for one filing: the acceptance time when known
    (``exact``), else the end of the ``filed`` day in New York (``bound``)."""
    if acceptance is not None:
        return _parse_acceptance(acceptance) if isinstance(acceptance, str) else to_utc(acceptance), "exact"
    day = filed if isinstance(filed, _dt.date) else _dt.date.fromisoformat(str(filed)[:10])
    if isinstance(day, _dt.datetime):
        day = day.date()
    local = _dt.datetime.combine(day, _END_OF_DAY).replace(tzinfo=zone(NEW_YORK))
    return local.astimezone(UTC), "bound"


def filings(submissions: Any, forms: Optional[Iterable[str]] = None) -> Any:
    """One row per filing: ``accession``, ``form``, ``items``, ``report_date``,
    ``filing_date``, ``available_at`` (acceptance, UTC), ``primary_document``.

    ``forms`` keeps only those form types (for example ``{"8-K", "10-Q", "10-K"}``).
    """
    import pandas as pd

    keep = set(forms) if forms is not None else None
    rows = []
    for block in _recent_blocks(submissions):
        accn, form, items, report, filed, acc, doc = _columns(
            block,
            ("accessionNumber", "form", "items", "reportDate", "filingDate", "acceptanceDateTime", "primaryDocument"),
        )
        for i in range(len(accn)):
            if keep is not None and form[i] not in keep:
                continue
            when, conf = filing_available_at(filed[i], acc[i]) if (acc[i] or filed[i]) else (None, None)
            if when is None:
                raise ValueError(f"filing {accn[i]} has neither acceptanceDateTime nor filingDate")
            rows.append(
                {
                    "accession": accn[i],
                    "form": form[i],
                    "items": items[i] or "",
                    "report_date": report[i] or None,
                    "filing_date": filed[i],
                    "available_at": when,
                    "confidence": conf,
                    "primary_document": doc[i],
                }
            )
    out = pd.DataFrame(
        rows,
        columns=["accession", "form", "items", "report_date", "filing_date", "available_at", "confidence", "primary_document"],
    )
    out["available_at"] = pd.to_datetime(out["available_at"], utc=True)
    return out


def _iter_facts(companyfacts: Mapping[str, Any], concepts: Optional[Iterable[str]], taxonomy: Optional[str]):
    if not isinstance(companyfacts, Mapping) or "facts" not in companyfacts:
        raise ValueError("not an EDGAR companyfacts document (no 'facts')")
    want = set(concepts) if concepts is not None else None
    if not isinstance(companyfacts["facts"], Mapping):
        raise ValueError("companyfacts 'facts' must be an object")
    for tax, by_concept in companyfacts["facts"].items():
        if taxonomy is not None and tax != taxonomy:
            continue
        if not isinstance(by_concept, Mapping):
            raise ValueError(f"facts.{tax} must be an object")
        for concept, body in by_concept.items():
            if want is not None and concept not in want and f"{tax}:{concept}" not in want:
                continue
            units = body.get("units", {}) if isinstance(body, Mapping) else {}
            if not isinstance(units, Mapping):
                raise ValueError(f"facts.{tax}.{concept}.units must be an object")
            for unit, entries in units.items():
                if not isinstance(entries, list):
                    raise ValueError(f"facts.{tax}.{concept}.units.{unit} must be a list")
                for e in entries:
                    if not isinstance(e, Mapping):
                        raise ValueError("each fact must be an object")
                    yield tax, concept, unit, e


def facts(
    companyfacts: Mapping[str, Any],
    submissions: Any = None,
    concepts: Optional[Iterable[str]] = None,
    taxonomy: Optional[str] = None,
) -> Any:
    """XBRL facts as a stamped pandas DataFrame, one row per (fact, filing).

    Columns: ``taxonomy``, ``concept``, ``unit``, ``start``, ``end``, ``value``, ``fy``,
    ``fp``, ``form``, ``accession`` (= ``vintage``), ``filed``, ``frame``, ``event_time``,
    ``available_at``, ``confidence``. Pass ``submissions`` (one or more submissions JSON
    documents) to stamp acceptance times; facts whose accession is not found fall back to
    the end of the ``filed`` day (``bound``).

    ``fin_pit.latest(df, ["concept", "unit", "start", "end"])`` inside ``asof`` gives each
    figure as last reported before the decision time.
    """
    import pandas as pd

    accept = acceptance_times(submissions) if submissions is not None else {}
    nyc = zone(NEW_YORK)
    rows = []
    for tax, concept, unit, e in _iter_facts(companyfacts, concepts, taxonomy):
        accn = e.get("accn")
        filed = e.get("filed")
        end = e.get("end")
        if not isinstance(end, str) or not isinstance(filed, str):
            raise ValueError(f"fact {tax}:{concept} needs 'end' and 'filed' dates")
        when, conf = filing_available_at(filed, accept.get(accn) if isinstance(accn, str) else None)
        end_day = _dt.date.fromisoformat(end[:10])
        rows.append(
            {
                "taxonomy": tax,
                "concept": concept,
                "unit": unit,
                "start": e.get("start"),
                "end": end,
                "value": e.get("val"),
                "fy": e.get("fy"),
                "fp": e.get("fp"),
                "form": e.get("form"),
                "accession": accn,
                "vintage": accn,
                "filed": filed,
                "frame": e.get("frame"),
                "event_time": _dt.datetime(end_day.year, end_day.month, end_day.day, tzinfo=nyc).astimezone(UTC),
                "available_at": when,
                "confidence": conf,
            }
        )
    cols = ["taxonomy", "concept", "unit", "start", "end", "value", "fy", "fp", "form", "accession",
            "vintage", "filed", "frame", "event_time", "available_at", "confidence"]
    out = pd.DataFrame(rows, columns=cols)
    for c in ("event_time", "available_at"):
        out[c] = pd.to_datetime(out[c], utc=True)
    return out


def fact_records(
    companyfacts: Mapping[str, Any],
    submissions: Any = None,
    concepts: Optional[Iterable[str]] = None,
    taxonomy: Optional[str] = None,
) -> List[PITRecord]:
    """``facts`` as ``PITRecord`` objects; ``key`` is ``taxonomy:concept[unit]start..end``."""
    df = facts(companyfacts, submissions, concepts, taxonomy)
    out = []
    for r in df.itertuples(index=False):
        out.append(
            PITRecord(
                event_time=r.event_time.to_pydatetime(),
                available_at=r.available_at.to_pydatetime(),
                vintage=r.accession,
                source="sec-edgar",
                confidence=r.confidence,
                value=r.value,
                key=f"{r.taxonomy}:{r.concept}[{r.unit}]{r.start or ''}..{r.end}",
            )
        )
    return out
