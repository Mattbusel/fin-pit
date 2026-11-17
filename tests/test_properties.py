"""Property and fuzz tests. Inputs that come from outside (vendor JSON, timestamps, frames)
must either work or fail with ValueError/TypeError, and the firewall must never let a late
row through."""
import datetime as dt
import json

import pandas as pd
import pytest

hyp = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import fin_pit as pit  # noqa: E402
from fin_pit.adapters import edgar, fred  # noqa: E402

UTC = dt.timezone.utc
BASE = dt.datetime(2020, 1, 1, tzinfo=UTC)
minutes = st.integers(min_value=0, max_value=60 * 24 * 365 * 3)
SETTINGS = settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def t(m):
    return BASE + dt.timedelta(minutes=m)


@SETTINGS
@given(st.lists(st.one_of(minutes, st.none()), max_size=40), minutes)
def test_filter_never_admits_late_rows(avail, decision):
    df = pd.DataFrame(
        {"i": range(len(avail)), "available_at": pd.to_datetime([None if a is None else t(a) for a in avail], utc=True)}
    )
    out = pit.filter(df, t(decision))
    expected = [i for i, a in enumerate(avail) if a is not None and a <= decision]
    assert out["i"].tolist() == expected
    late = len(avail) - len(expected)
    if late:
        with pytest.raises(pit.LookaheadError) as e:
            pit.check(df, t(decision))
        assert e.value.count == late
    else:
        assert pit.check(df, t(decision)) is df


@SETTINGS
@given(st.lists(st.tuples(st.sampled_from("abc"), minutes, st.integers()), max_size=30), minutes)
def test_latest_is_max_available_per_key(rows, decision):
    df = pd.DataFrame(rows, columns=["k", "m", "v"])
    df["available_at"] = pd.to_datetime([t(m) for m in df["m"]], utc=True) if len(df) else pd.Series([], dtype="datetime64[ns, UTC]")
    out = pit.latest(df, "k", t(decision))
    for k in "abc":
        known = [(m, i) for i, (kk, m, v) in enumerate(rows) if kk == k and m <= decision]
        got = out[out.k == k]
        if not known:
            assert got.empty
        else:
            assert len(got) == 1
            assert got["m"].iloc[0] == max(m for m, _ in known)


@SETTINGS
@given(st.lists(minutes, max_size=20), st.lists(st.tuples(st.sampled_from("ab"), minutes, st.integers()), max_size=20))
def test_asof_join_never_uses_future_rows(left_times, right_rows):
    left = pd.DataFrame({"k": ["a", "b"] * (len(left_times) // 2) + ["a"] * (len(left_times) % 2),
                         "dt": pd.to_datetime([t(m) for m in left_times], utc=True)})
    right = pd.DataFrame(right_rows, columns=["k", "m", "v"])
    right["available_at"] = pd.to_datetime([t(m) for m in right["m"]], utc=True)
    out = pit.asof_join(left, right, "dt", by="k")
    assert len(out) == len(left) and out.index.equals(left.index)
    for _, row in out.iterrows():
        if pd.notna(row["m"]):
            assert t(int(row["m"])) <= row["dt"]
            best = max(m for k, m, _ in right_rows if k == row["k"] and t(m) <= row["dt"])
            assert row["m"] == best


@SETTINGS
@given(minutes, st.integers(min_value=0, max_value=3000), st.text(alphabet=" abc:", max_size=8))
def test_scan_text_date_ordering(decision, days, filler):
    d = t(decision)
    earlier = (d - dt.timedelta(days=days)).date().isoformat()
    later = (d + dt.timedelta(days=days + 1)).date().isoformat()
    assert pit.scan_text(filler + earlier + filler, d) == []
    assert pit.scan_text(filler + later + filler, d) == [later]


@SETTINGS
@given(st.one_of(st.text(max_size=40), st.integers(min_value=-(10**12), max_value=10**12), st.floats(allow_nan=True),
                 st.none(), st.booleans(), st.lists(st.integers(), max_size=3)))
def test_to_utc_fails_cleanly(value):
    try:
        out = pit.to_utc(value, "UTC")
    except (ValueError, TypeError, OverflowError, OSError):
        return
    assert out.tzinfo is not None and out.utcoffset() == dt.timedelta(0)


json_leaf = st.one_of(st.none(), st.booleans(), st.integers(), st.floats(allow_nan=False), st.text(max_size=20))
json_value = st.recursive(json_leaf, lambda c: st.one_of(st.lists(c, max_size=4), st.dictionaries(st.text(max_size=12), c, max_size=4)), max_leaves=20)


@SETTINGS
@given(json_value)
def test_edgar_submissions_fuzz(doc):
    try:
        edgar.acceptance_times(doc)
        edgar.filings(doc)
    except (ValueError, TypeError):
        pass


@SETTINGS
@given(st.dictionaries(st.sampled_from(["facts", "cik", "entityName"]), json_value, max_size=3))
def test_edgar_companyfacts_fuzz(doc):
    try:
        edgar.facts(doc)
    except (ValueError, TypeError):
        pass


fact = st.fixed_dictionaries(
    {"end": st.text(max_size=12), "filed": st.text(max_size=12)},
    optional={"accn": st.text(max_size=20), "val": json_leaf, "start": st.text(max_size=12)},
)


@SETTINGS
@given(st.lists(fact, max_size=5))
def test_edgar_fact_fields_fuzz(entries):
    doc = {"facts": {"us-gaap": {"X": {"units": {"USD": entries}}}}}
    try:
        edgar.facts(doc)
    except (ValueError, TypeError):
        pass


@SETTINGS
@given(st.lists(st.tuples(st.dates(min_value=dt.date(1950, 1, 1), max_value=dt.date(2030, 1, 1)),
                          st.dates(min_value=dt.date(1950, 1, 1), max_value=dt.date(2030, 1, 1)),
                          st.floats(allow_nan=False, allow_infinity=False)), max_size=20),
       st.dates(min_value=dt.date(1950, 1, 1), max_value=dt.date(2030, 1, 1)))
def test_fred_asof_never_returns_unreleased(rows, when):
    raw = pd.DataFrame(rows, columns=["date", "realtime_start", "value"])
    raw["date"] = pd.to_datetime(raw["date"])
    raw["realtime_start"] = pd.to_datetime(raw["realtime_start"])
    stamped = fred.from_fredapi(raw, "X")
    decision = pit.to_utc(when, "America/New_York")
    cut = pit.filter(stamped, decision)
    assert (cut["available_at"] <= pd.Timestamp(decision)).all()
    # a vintage dated D is never visible before the end of day D in New York
    assert all(pd.Timestamp(v).date() < when for v in cut["vintage"])


def test_vectors_are_valid_json_numbers():
    from fin_pit import conformance

    doc = conformance.load()
    json.dumps(doc, allow_nan=False)
