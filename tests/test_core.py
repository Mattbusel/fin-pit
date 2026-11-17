import asyncio
import contextvars
import datetime as dt
import threading

import pandas as pd
import pytest

import fin_pit as pit
from fin_pit import LookaheadError, PITRecord, UnstampedError

UTC = dt.timezone.utc


def T(s):
    return pit.to_utc(s)


# --- to_utc ---------------------------------------------------------------------------------


def test_to_utc_forms():
    assert T("2024-05-10T16:00:00Z") == dt.datetime(2024, 5, 10, 16, tzinfo=UTC)
    assert T("2024-05-10T12:00:00-04:00") == dt.datetime(2024, 5, 10, 16, tzinfo=UTC)
    assert pit.to_utc("2024-05-10 12:00", "America/New_York") == dt.datetime(2024, 5, 10, 16, tzinfo=UTC)
    assert pit.to_utc(dt.date(2024, 1, 2), "America/New_York") == dt.datetime(2024, 1, 2, 5, tzinfo=UTC)
    assert pit.to_utc(pd.Timestamp("2024-01-02 10:00", tz="UTC")) == dt.datetime(2024, 1, 2, 10, tzinfo=UTC)
    assert pit.to_utc(0) == dt.datetime(1970, 1, 1, tzinfo=UTC)
    import numpy as np

    assert pit.to_utc(np.datetime64("2024-01-02T10:00")) == dt.datetime(2024, 1, 2, 10, tzinfo=UTC)


@pytest.mark.parametrize("bad", ["2024-05-10 16:00", dt.datetime(2024, 5, 10), dt.date(2024, 5, 10), "", "nope"])
def test_to_utc_refuses_ambiguous(bad):
    with pytest.raises(ValueError):
        pit.to_utc(bad)


@pytest.mark.parametrize("bad", [None, float("nan"), pd.NaT])
def test_to_utc_refuses_missing(bad):
    with pytest.raises(ValueError):
        pit.to_utc(bad)


def test_to_utc_type_errors():
    with pytest.raises(TypeError):
        pit.to_utc(True)
    with pytest.raises(TypeError):
        pit.to_utc([2024])
    with pytest.raises(ValueError):
        pit.to_utc("2024-01-01 00:00", "Not/AZone")


# --- PITRecord ------------------------------------------------------------------------------


def test_record_normalises_and_validates():
    r = PITRecord("2024-03-31T00:00Z", "2024-04-30T20:30Z", vintage=1, source="x", confidence="exact", value=1.5)
    assert r.available_at.tzinfo is UTC and r.vintage == "1"
    assert r.known_at("2024-04-30T20:30Z") and not r.known_at("2024-04-30T20:29Z")
    assert r.to_dict()["available_at"] == "2024-04-30T20:30:00+00:00"
    with pytest.raises(ValueError):
        PITRecord("2024-03-31T00:00Z", "2024-04-30T20:30Z", confidence="sure")
    with pytest.raises(ValueError):
        PITRecord("2024-03-31T00:00Z", dt.datetime(2024, 4, 30))


# --- asof -----------------------------------------------------------------------------------


def test_asof_context_and_nesting():
    assert pit.decision_time() is None
    with pit.asof("2024-01-02T00:00Z") as a:
        assert pit.decision_time() == a.time
        with pit.asof("2024-01-01T00:00Z"):
            assert pit.decision_time() == T("2024-01-01T00:00Z")
        assert pit.decision_time() == a.time
        with pytest.raises(LookaheadError):
            with pit.asof("2024-01-03T00:00Z"):
                pass
        assert pit.decision_time() == a.time
    assert pit.decision_time() is None


def test_asof_reentrant_instance():
    a = pit.asof("2024-01-02T00:00Z")
    with a:
        with a:
            assert pit.decision_time() == a.time
        assert pit.decision_time() == a.time
    assert pit.decision_time() is None


def test_asof_options_validated():
    with pytest.raises(ValueError):
        pit.asof("2024-01-02T00:00Z", mode="drop")
    with pytest.raises(ValueError):
        pit.asof("2024-01-02T00:00Z", unstamped="ignore")
    with pytest.raises(ValueError):
        pit.asof("2024-01-02T00:00Z", column="")
    with pytest.raises(ValueError):
        pit.asof("2024-01-02")  # no tz


def test_asof_decorator_sync_async_and_isolation():
    seen = {}

    @pit.asof("2024-01-02T00:00Z")
    def f():
        return pit.decision_time()

    @pit.asof("2024-01-03T00:00Z")
    async def g(tag):
        await asyncio.sleep(0)
        seen[tag] = pit.decision_time()
        return seen[tag]

    assert f() == T("2024-01-02T00:00Z")
    assert pit.decision_time() is None

    async def main():
        return await asyncio.gather(g("a"), g("b"))

    assert asyncio.run(main()) == [T("2024-01-03T00:00Z")] * 2
    assert pit.decision_time() is None


def test_asof_does_not_leak_into_other_threads():
    out = []
    with pit.asof("2024-01-02T00:00Z"):
        t = threading.Thread(target=lambda: out.append(pit.decision_time()))
        t.start()
        t.join()
        ctx = contextvars.copy_context()
    assert out == [None]
    assert ctx.run(pit.decision_time) == T("2024-01-02T00:00Z")


# --- filter / check -------------------------------------------------------------------------


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "v": [1, 2, 3, 4],
            "available_at": pd.to_datetime(
                ["2024-01-01T10:00Z", "2024-01-02T10:00Z", None, "2024-01-03T10:00Z"], utc=True
            ),
        }
    )


def test_filter_pandas(frame):
    assert pit.filter(frame) is frame  # no decision time: unchanged
    assert pit.filter(frame, "2024-01-02T10:00Z").v.tolist() == [1, 2]  # inclusive, NaT dropped
    with pit.asof("2024-01-02T12:00Z"):
        assert pit.filter(frame).v.tolist() == [1, 2]
        assert frame.pit.asof().v.tolist() == [1, 2]
        with pytest.raises(LookaheadError) as e:
            pit.check(frame)
        assert e.value.count == 2
        assert e.value.available_at == T("2024-01-03T10:00Z")
    assert pit.check(frame.iloc[:2], "2024-01-02T10:00Z") is not None


def test_filter_index_level_and_series(frame):
    idx = frame.set_index("available_at")
    assert pit.filter(idx, "2024-01-02T12:00Z")["v"].tolist() == [1, 2]
    s = idx["v"]
    assert pit.filter(s, "2024-01-02T12:00Z").tolist() == [1, 2]


def test_filter_refuses_naive_and_parses_strings():
    naive = pd.DataFrame({"v": [1], "available_at": pd.to_datetime(["2024-01-01 10:00"])})
    with pytest.raises(ValueError, match="naive"):
        pit.filter(naive, "2024-01-02T00:00Z")
    strs = pd.DataFrame({"v": [1, 2], "available_at": ["2024-01-01T10:00Z", "2024-01-03T10:00+00:00"]})
    assert pit.filter(strs, "2024-01-02T00:00Z").v.tolist() == [1]
    bad = pd.DataFrame({"v": [1], "available_at": ["2024-01-01 10:00"]})
    with pytest.raises(ValueError):
        pit.filter(bad, "2024-01-02T00:00Z")


def test_explicit_time_cannot_exceed_active():
    with pit.asof("2024-01-02T00:00Z"):
        with pytest.raises(LookaheadError):
            pit.filter([], "2024-01-03T00:00Z")


def test_records_lists_dicts():
    recs = [PITRecord("2024-01-01T00:00Z", f"2024-01-0{d}T00:00Z", value=d) for d in (1, 2, 3)]
    assert [r.value for r in pit.filter(recs, "2024-01-02T00:00Z")] == [1, 2]
    assert isinstance(pit.filter(tuple(recs), "2024-01-02T00:00Z"), tuple)
    with pytest.raises(LookaheadError):
        pit.check(recs, "2024-01-02T00:00Z")
    dicts = [{"t": "a", "available_at": "2024-01-01T00:00Z"}, {"t": "b", "available_at": None}]
    assert pit.filter(dicts, "2024-01-02T00:00Z") == dicts[:1]
    with pytest.raises(LookaheadError):
        pit.filter(recs[2], "2024-01-02T00:00Z")  # a single late record cannot be dropped
    assert pit.filter(recs[0], "2024-01-02T00:00Z") is recs[0]


def test_unstamped_policy():
    with pit.asof("2024-01-02T00:00Z"):
        assert pit.filter({"x": 1}) == {"x": 1}
        assert pit.filter(pd.DataFrame({"x": [1]})).shape == (1, 1)
    with pit.asof("2024-01-02T00:00Z", unstamped="raise"):
        with pytest.raises(UnstampedError):
            pit.filter({"x": 1})
        with pytest.raises(UnstampedError):
            pit.filter([{"x": 1}, {"available_at": "2024-01-01T00:00Z"}])
        with pytest.raises(UnstampedError):
            pit.filter(pd.DataFrame({"x": [1]}))


def test_custom_column():
    df = pd.DataFrame({"v": [1, 2], "knowable": pd.to_datetime(["2024-01-01T00:00Z", "2024-01-03T00:00Z"], utc=True)})
    with pit.asof("2024-01-02T00:00Z", column="knowable", mode="filter"):
        assert pit.filter(df).v.tolist() == [1]
    assert pit.filter(df, "2024-01-02T00:00Z", column="knowable").v.tolist() == [1]


def test_polars():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame(
        {"v": [1, 2, 3], "available_at": [dt.datetime(2024, 1, d, tzinfo=UTC) for d in (1, 2, 3)]}
    )
    assert pit.filter(df, "2024-01-02T00:00Z")["v"].to_list() == [1, 2]
    with pytest.raises(LookaheadError):
        pit.check(df, "2024-01-02T00:00Z")
    assert df.pit.asof("2024-01-02T00:00Z")["v"].to_list() == [1, 2]
    lazy = df.lazy()
    assert pit.filter(lazy, "2024-01-02T00:00Z").collect()["v"].to_list() == [1, 2]
    with pytest.raises(ValueError):
        pit.check(lazy, "2024-01-02T00:00Z")
    naive = pl.DataFrame({"available_at": [dt.datetime(2024, 1, 1)]})
    with pytest.raises(ValueError, match="naive"):
        pit.filter(naive, "2024-01-02T00:00Z")


# --- latest ---------------------------------------------------------------------------------


def test_latest_vintages():
    df = pd.DataFrame(
        {
            "event_time": ["2023-02", "2023-02", "2023-03", "2023-03"],
            "value": [1.0, 1.1, 2.0, 2.1],
            "available_at": pd.to_datetime(
                ["2023-03-14T23:59Z", "2024-02-13T23:59Z", "2023-04-12T23:59Z", "2024-02-13T23:59Z"], utc=True
            ),
        }
    )
    assert pit.latest(df, "event_time").value.tolist() == [1.1, 2.1]
    assert pit.latest(df, "event_time", "2023-03-31T20:00Z").value.tolist() == [1.0]
    assert pit.latest(df, "event_time", "2023-05-01T00:00Z").value.tolist() == [1.0, 2.0]
    with pit.asof("2023-05-01T00:00Z"):
        assert df.pit.latest().value.tolist() == [1.0, 2.0]


def test_latest_polars_and_records():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame(
        {
            "k": ["a", "a", "b"],
            "value": [1, 2, 3],
            "available_at": [dt.datetime(2024, 1, d, tzinfo=UTC) for d in (1, 3, 2)],
        }
    )
    assert pit.latest(df, "k", "2024-01-02T00:00Z")["value"].to_list() == [1, 3]
    recs = [PITRecord("2024-01-01T00:00Z", f"2024-01-0{d}T00:00Z", value=d, key="k") for d in (1, 2)]
    assert [r.value for r in pit.latest(recs, "key")] == [2]
    with pytest.raises(TypeError):
        pit.latest([1, 2], "key")


# --- guard ----------------------------------------------------------------------------------


def test_guard_arguments_and_outputs():
    rows = pd.DataFrame(
        {"v": [1, 2], "available_at": pd.to_datetime(["2024-05-10T19:00Z", "2024-05-10T21:00Z"], utc=True)}
    )

    @pit.guard
    def fetch(ticker, curr_date):
        return rows

    @pit.guard(mode="filter")
    def fetch_filtered(ticker, end_date=None):
        return rows

    assert len(fetch("X", "2030-01-01")) == 2  # outside asof: untouched
    with pit.asof("2024-05-10 16:00", tz="America/New_York"):
        with pytest.raises(LookaheadError):
            fetch("X", "2024-05-11")
        with pytest.raises(LookaheadError):
            fetch("X", curr_date="2024-05-10")  # the 21:00Z row leaks
        assert fetch_filtered("X", end_date="2024-05-10").v.tolist() == [1]
        assert fetch_filtered("X").v.tolist() == [1]


def test_guard_require_asof_and_async():
    @pit.guard(require_asof=True)
    def f():
        return 1

    with pytest.raises(LookaheadError):
        f()
    with pit.asof("2024-01-01T00:00Z"):
        assert f() == 1

    @pit.guard(mode="filter")
    async def g(as_of=None):
        return [{"available_at": "2023-12-31T00:00Z"}, {"available_at": "2024-01-02T00:00Z"}]

    async def run():
        with pit.asof("2024-01-01T00:00Z"):
            return await g()

    assert len(asyncio.run(run())) == 1
    assert f.__fin_pit_guarded__ and f.__name__ == "f"


def test_guard_text_scan():
    @pit.guard(scan_text=True)
    def report():
        return "# Data retrieved on: 2026-09-04 12:00:00\nprice 63000"

    @pit.guard(scan_text=True)
    def clean():
        return "As of 2024-05-10, close 63000 (filed 2024-05-09T21:00:00Z)"

    with pit.asof("2024-05-10T20:00Z"):
        with pytest.raises(LookaheadError, match="2026-09-04"):
            report()
        assert clean().startswith("As of")


def test_scan_text_rules():
    t = "2024-05-10T20:00Z"
    assert pit.scan_text("2024-05-10", t) == []  # same day: allowed
    assert pit.scan_text("2024-05-11", t) == ["2024-05-11"]
    assert pit.scan_text("2024-05-10T20:00:01Z", t) == ["2024-05-10T20:00:01Z"]
    assert pit.scan_text("2024-05-10T16:30-04:00", t) == ["2024-05-10T16:30-04:00"]
    assert pit.scan_text("2024-05-10T16:00-04:00", t) == []
    assert pit.scan_text("2024-02-30 or 12024-05-11 or 2024-13-01", t) == []
    assert pit.scan_text("no dates", None) == []


def test_guard_validates_options():
    with pytest.raises(ValueError):
        pit.guard(mode="maybe")
    with pytest.raises(ValueError):
        pit.guard(unstamped="maybe")
