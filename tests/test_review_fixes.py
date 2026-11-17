"""Regression tests for the 0.1.1 fixes: non-causal indicator outputs, non-US daily bars,
date-argument parsing, generators, nested outputs, threads, text scanning, LangChain
ToolCall forwarding and untrusted-input edges."""
import asyncio
import concurrent.futures as cf
import datetime as dt
import warnings

import numpy as np
import pandas as pd
import pytest

import fin_pit as pit
from fin_pit import LookaheadError, UnstampedError
from fin_pit.adapters import edgar
from fin_pit.adapters import yfinance as yfa

UTC = dt.timezone.utc
LATE = {"available_at": "2030-01-01T00:00Z", "v": "late"}
EARLY = {"available_at": "2020-01-01T00:00Z", "v": "early"}
NY_CLOSE = ("2024-05-10 16:00", "America/New_York")


# --- strict-warmup indicators -------------------------------------------------------------

fp = pytest.importorskip("fin_primitives")


def probe(n):
    from fin_pit.warmup import _probe_frame

    return _probe_frame(n)


def test_fin_primitives_input_map_is_present():
    # warmup reads fin_primitives.pandas._inputs (private); fail loudly if it moves.
    import fin_primitives.pandas as fpp

    assert callable(getattr(fpp, "_inputs", None))
    assert fpp._inputs("sma") == ("close",)


def test_indicator_names():
    names = pit.warmup.indicators()
    assert "sma" in names and "ichimoku" in names and len(names) == len(set(names))


@pytest.mark.parametrize("name", sorted(__import__("fin_pit").warmup.indicators()))
def test_every_indicator_is_causal(name):
    # Values at rows < K must not change when rows >= K are added.
    df = probe(400)
    K = 300
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = df.pit.indicator(name)
        cut = df.iloc[:K].pit.indicator(name)
    assert list(full.columns) == list(cut.columns)
    for c in full.columns:
        a = full[c].iloc[:K].to_numpy(dtype=float)
        b = cut[c].to_numpy(dtype=float)
        assert np.array_equal(np.isnan(a), np.isnan(b)), (name, c)
        both = ~np.isnan(a)
        assert np.allclose(a[both], b[both], rtol=1e-9, atol=1e-12), (name, c)


def test_ichimoku_chikou_and_centred_dpo_are_refused():
    df = probe(300)
    ich = df.pit.indicator("ichimoku")
    assert not [c for c in ich.columns if c.startswith("ICS")]
    assert pit.warmup.noncausal_columns("ichimoku") == ("ICS_26",)
    d = df.pit.indicator("dpo")
    sma20 = df["close"].rolling(20).mean()
    assert d["DPO_20"].iloc[150] == pytest.approx(df["close"].iloc[150] - sma20.iloc[150 - 11])
    with pytest.raises(LookaheadError):
        df.pit.indicator("dpo", centered=True)


def test_required_bars_long_windows():
    assert pit.required_bars("sma", 200) == 200
    assert pit.required_bars("sma", 4097) == 4097
    assert pit.required_bars("sma", 5000) == 5000


def test_indicator_inside_asof_on_unstamped_frame():
    df = probe(300)  # daily index from 2000-01-03, UTC
    with pit.asof("2000-03-01T00:00Z"):
        with pytest.warns(pit.warmup.UnstampedFrameWarning):
            r = df.pit.indicator("sma", 10)
        assert r.index[-1] == pd.Timestamp("2000-02-29T00:00Z") and len(r) == 58  # Jan 3 .. Feb 29
    with pit.asof("2000-03-01T00:00Z", unstamped="raise"):
        with pytest.raises(UnstampedError):
            df.pit.indicator("sma", 10)
    stamped = df.assign(available_at=df.index + pd.Timedelta(hours=21))
    with pit.asof("2000-03-01T00:00Z"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r = stamped.pit.indicator("sma", 10)
    assert len(r) == 58  # the Mar 1 bar closes at 21:00Z


# --- yfinance daily bars outside New York ---------------------------------------------------


def bars(tz, n=3):
    return pd.DataFrame({"Close": range(n)}, index=pd.date_range("2024-05-06", periods=n, freq="D", tz=tz))


@pytest.mark.parametrize("tz", ["Asia/Shanghai", "Asia/Tokyo", "Europe/London", "Asia/Hong_Kong"])
def test_non_us_daily_bars_need_a_close(tz):
    with pytest.raises(ValueError, match="calendar="):
        yfa.stamp_bars(bars(tz))


@pytest.mark.parametrize(
    "tz,close,expected",
    [
        ("Asia/Shanghai", dt.time(15, 0), "2024-05-06T07:00Z"),
        ("Asia/Tokyo", dt.time(15, 30), "2024-05-06T06:30Z"),
        ("Europe/London", dt.time(16, 30), "2024-05-06T15:30Z"),
    ],
)
def test_non_us_daily_bars_use_the_index_day(tz, close, expected):
    out = yfa.stamp_bars(bars(tz), close_time=close, lag=dt.timedelta(0))
    assert out["available_at"].iloc[0] == pd.Timestamp(expected)


def test_non_us_daily_bars_with_a_calendar():
    pytest.importorskip("exchange_calendars")
    out = yfa.stamp_bars(bars("Asia/Shanghai"), calendar="XSHG", lag=dt.timedelta(0))
    assert out["available_at"].iloc[0] == pd.Timestamp("2024-05-06T07:00Z")


def test_utc_daily_index_needs_a_session():
    with pytest.raises(ValueError, match="24/7"):
        yfa.stamp_bars(bars("UTC"))
    out = yfa.stamp_bars(bars("UTC"), session="24/7", lag=dt.timedelta(0))
    assert out["available_at"].iloc[0] == pd.Timestamp("2024-05-07T00:00Z")
    ny = yfa.stamp_bars(bars("America/New_York"), lag=dt.timedelta(0))
    assert ny["available_at"].iloc[0] == pd.Timestamp("2024-05-06T20:00Z")


# --- guard: date arguments ----------------------------------------------------------------


@pit.guard(require_asof=False)
def news(ticker, end_date=None, end_time=None, until=None, date=None, to=None):
    return "ok"


@pytest.mark.parametrize(
    "kw",
    [
        {"end_date": "2024-05-11"},
        {"end_date": "20240511"},
        {"date": 20240511},
        {"end_date": "2024/05/11"},
        {"end_date": "2024.5.11"},
        {"end_time": 1715400000000},  # milliseconds (ccxt / Binance)
        {"end_time": "1715400000000"},
        {"end_time": 1715400000},  # seconds
        {"end_time": np.int64(1715400000000)},
        {"end_date": "2024-05-10 18:00"},  # naive: read in the asof zone (New York)
        {"end_date": dt.datetime(2024, 5, 10, 18, 0)},
        {"until": "today"},
        {"to": "latest"},
    ],
)
def test_late_date_arguments_are_refused(kw):
    with pit.asof(NY_CLOSE[0], tz=NY_CLOSE[1]):
        with pytest.raises(LookaheadError):
            news("A", **kw)


@pytest.mark.parametrize(
    "kw",
    [
        {"end_date": "2024-05-10"},
        {"end_date": "20240510"},
        {"date": 20240509},
        {"end_date": "2024/05/10"},
        {"end_time": 1715300000000},
        {"end_date": "2024-05-10 15:59"},
    ],
)
def test_earlier_date_arguments_pass(kw):
    with pit.asof(NY_CLOSE[0], tz=NY_CLOSE[1]):
        assert news("A", **kw) == "ok"


def test_unreadable_date_arguments_warn_or_raise():
    with pit.asof(NY_CLOSE[0], tz=NY_CLOSE[1]):
        with pytest.warns(pit.UnparsedDateWarning):
            assert news("A", to="EUR") == "ok"
        with pytest.warns(pit.UnparsedDateWarning):
            assert news("A", end_time=10**40) == "ok"  # out of range: no crash

    @pit.guard(unparsed="raise")
    def strict(end_date=None):
        return 1

    with pit.asof("2024-01-01T00:00Z"):
        with pytest.raises(LookaheadError, match="not a time"):
            strict(end_date="soon")
    with pytest.raises(ValueError):
        pit.guard(unparsed="sometimes")


def test_date_tz_overrides_the_asof_zone():
    @pit.guard(date_tz="UTC")
    def f(end_date=None):
        return 1

    with pit.asof(NY_CLOSE[0], tz=NY_CLOSE[1]):  # 20:00Z
        assert f(end_date="2024-05-10 19:00") == 1
        with pytest.raises(LookaheadError):
            f(end_date="2024-05-10 21:00")


# --- guard: generators, nested outputs, threads ------------------------------------------


def test_asof_decorated_generators_see_the_decision_time():
    @pit.asof("2024-01-01T00:00Z")
    def gen():
        yield pit.decision_time()
        received = yield "ready"
        yield received

    g = gen()
    assert next(g) == dt.datetime(2024, 1, 1, tzinfo=UTC)
    assert next(g) == "ready"
    assert g.send(42) == 42
    assert pit.decision_time() is None

    @pit.asof("2024-01-01T00:00Z")
    async def agen():
        yield pit.decision_time()

    async def collect():
        return [x async for x in agen()]

    assert asyncio.run(collect()) == [dt.datetime(2024, 1, 1, tzinfo=UTC)]


def test_guarded_generators_are_checked_per_item():
    @pit.guard
    def rows():
        yield EARLY
        yield LATE

    @pit.guard(mode="filter")
    def rows_filtered():
        yield EARLY
        yield LATE
        yield pd.DataFrame([EARLY, LATE]).assign(available_at=lambda d: pd.to_datetime(d.available_at, utc=True))

    with pit.asof("2024-01-01T00:00Z"):
        g = rows()
        gf = rows_filtered()
    with pytest.raises(LookaheadError):
        list(g)  # checked against the time captured at call
    out = list(gf)
    assert out[0] == EARLY and len(out) == 2 and out[1]["v"].tolist() == ["early"]

    @pit.guard(mode="filter")
    async def arows():
        yield EARLY
        yield LATE

    async def collect():
        with pit.asof("2024-01-01T00:00Z"):
            return [x async for x in arows()]

    assert asyncio.run(collect()) == [EARLY]


def test_nested_outputs_are_checked():
    @pit.guard
    def nested():
        return {"rows": [EARLY, LATE], "meta": "x"}

    @pit.guard(mode="filter")
    def nested_filtered():
        return {"rows": [EARLY, LATE], "meta": "x", "pair": ("text", [LATE])}

    with pit.asof("2024-01-01T00:00Z"):
        with pytest.raises(LookaheadError):
            nested()
        out = nested_filtered()
    assert out == {"rows": [EARLY], "meta": "x", "pair": ("text", [])}


def test_unguarded_thread_warns_and_bind_carries_the_decision_time():
    @pit.guard
    def late():
        return [LATE]

    with pit.asof("2024-01-01T00:00Z"):
        with cf.ThreadPoolExecutor(1) as ex:
            with pytest.warns(pit.UnguardedCallWarning):
                assert ex.submit(late).result() == [LATE]
            with pytest.raises(LookaheadError):
                ex.submit(pit.bind(late)).result()
    assert pit.bind(late) is late  # outside asof


# --- text scanning ---------------------------------------------------------------------------


def test_scan_text_reads_naive_times_in_the_asof_zone_and_more_formats():
    with pit.asof(NY_CLOSE[0], tz=NY_CLOSE[1]):
        assert pit.scan_text("Published 2024-05-10 17:30") == ["2024-05-10 17:30"]
        assert pit.scan_text("Published 2024-05-10 15:30") == []
        assert pit.scan_text("Published May 11, 2024") == ["May 11, 2024"]
        assert pit.scan_text("Published 11 May 2024") == ["11 May 2024"]
        assert pit.scan_text("Published Sept. 3rd, 2024") == ["Sept. 3rd, 2024"]
        assert pit.scan_text("Published 2024/05/11") == ["2024/05/11"]
        assert pit.scan_text("Published May 10, 2024 and 2024/05/10") == []
    # A bare date is compared with the decision's day in the asof zone: 23:00 New York is
    # already the 11th in UTC, but a story dated the 11th is still in the future.
    with pit.asof("2024-05-10 23:00", tz="America/New_York"):
        assert pit.scan_text("2024-05-11") == ["2024-05-11"]


# --- missing values and vintages ----------------------------------------------------------


def test_missing_markers_in_string_columns():
    df = pd.DataFrame(
        {"available_at": pd.array(["2023-06-01T00:00Z", None, "NaT"], dtype="string"), "v": [1, 2, 3]}
    )
    with pit.asof("2024-01-01T00:00Z"):
        assert pit.filter(df)["v"].tolist() == [1]


def test_latest_ignores_missing_times_like_polars():
    df = pd.DataFrame(
        {
            "k": ["a", "a", "a"],
            "available_at": pd.to_datetime(["2023-06-01", "2024-06-01", None], utc=True),
            "v": [1, 2, 3],
        }
    )
    assert pit.latest(df, "k")["v"].tolist() == [2]
    pl = pytest.importorskip("polars")
    p = pl.DataFrame(
        {
            "k": ["a", "a", "a"],
            "available_at": [dt.datetime(2023, 6, 1, tzinfo=UTC), dt.datetime(2024, 6, 1, tzinfo=UTC), None],
            "v": [1, 2, 3],
        }
    )
    assert pit.latest(p, "k")["v"].to_list() == [2]


# --- EDGAR untrusted input ----------------------------------------------------------------


def submissions(acc):
    return {
        "filings": {
            "recent": {
                "accessionNumber": ["a1"],
                "form": ["8-K"],
                "items": [""],
                "reportDate": [""],
                "filingDate": ["2024-01-02"],
                "acceptanceDateTime": [acc],
                "primaryDocument": ["x.htm"],
            }
        }
    }


@pytest.mark.parametrize("acc", [20240102, "", "  ", None, dt.datetime(2024, 1, 2, 21)])
def test_edgar_unusable_acceptance_falls_back_to_bound(acc):
    row = edgar.filings(submissions(acc)).iloc[0]
    assert row["confidence"] == "bound"
    assert row["available_at"] > pd.Timestamp("2024-01-03T04:59Z")
    assert edgar.acceptance_times(submissions(acc)) == {}


# --- LangChain ToolCall forwarding --------------------------------------------------------


def test_guard_tool_forwards_tool_call_id_config_and_artifact():
    lc = pytest.importorskip("langchain_core.tools")
    from typing import Annotated

    from langchain_core.messages import ToolMessage
    from langchain_core.runnables import RunnableConfig

    @lc.tool
    def tcid_tool(ticker: str, tool_call_id: Annotated[str, lc.InjectedToolCallId]) -> str:
        """Reads its tool call id."""
        return "tcid=" + tool_call_id

    @lc.tool
    def cfg_tool(ticker: str, config: RunnableConfig) -> str:
        """Reads the run config."""
        return "thread=" + str(config.get("configurable", {}).get("thread_id"))

    @lc.tool(response_format="content_and_artifact")
    def bars_tool(ticker: str) -> tuple:
        """Content plus stamped rows."""
        return "rows attached", [EARLY, LATE]

    call = {"type": "tool_call", "name": "tcid_tool", "args": {"ticker": "A"}, "id": "call-7"}
    with pit.asof("2024-01-01T00:00Z"):
        msg = pit.guard_tool(tcid_tool).invoke(call)
        assert isinstance(msg, ToolMessage) and msg.content == "tcid=call-7" and msg.tool_call_id == "call-7"
        out = pit.guard_tool(cfg_tool).invoke({"ticker": "A"}, config={"configurable": {"thread_id": "T1"}})
        assert out == "thread=T1"
        g = pit.guard_tool(bars_tool, mode="filter")
        assert g.response_format == "content_and_artifact"
        m = g.invoke({**call, "name": "bars_tool"})
        assert m.content == "rows attached" and m.artifact == [EARLY] and m.tool_call_id == "call-7"
        with pytest.raises(LookaheadError):
            pit.guard_tool(bars_tool).invoke({**call, "name": "bars_tool"})


# --- property: date-argument reading never crashes ---------------------------------------

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


@settings(max_examples=300, deadline=None)
@given(
    st.one_of(
        st.text(max_size=30),
        st.integers(min_value=-(10**30), max_value=10**30),
        st.floats(allow_nan=True, allow_infinity=True),
        st.none(),
        st.booleans(),
        st.dates(),
    )
)
def test_arg_time_never_raises(value):
    from fin_pit.guard import _arg_time

    t = _arg_time(value, "America/New_York")
    assert t is None or (t.tzinfo is not None and t.utcoffset() == dt.timedelta(0))
