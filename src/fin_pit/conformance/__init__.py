"""Indicator conformance vectors with named variants (SPEC.md).

43 indicators, 8 fixture series, every combination of each indicator's variant axes: 568
cases in ``vectors/vectors.json.gz`` (plain JSON inside; NaN is ``null``). Libraries
disagree on legitimate choices (RSI smoothing, EMA seeding, warmup), so a library declares
the variant it implements and is checked against that, instead of failing for being
different::

    from fin_pit import conformance

    for case in conformance.cases("rsi", smoothing="wilder"):
        case.check(my_rsi(case.inputs["close"], **case.params))

    # in a test suite
    @conformance.parametrize("ema", seed="adjusted")
    def test_ema(pit_case):
        pit_case.check(my_ema(pit_case.inputs["close"], 20), warmup="partial")

``check`` raises ``ConformanceError`` with the first mismatches. ``results`` runs a whole
indicator and returns pass/fail per case without raising.
"""
from __future__ import annotations

import functools
import gzip
import json
import math
import os
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from .catalog import CATALOG, GLOBAL_AXES, SPEC_VERSION, canonical, variant_combinations

__all__ = [
    "CATALOG",
    "SPEC_VERSION",
    "Case",
    "ConformanceError",
    "all_cases",
    "cases",
    "fixtures",
    "indicators",
    "load",
    "parametrize",
    "results",
    "canonical",
    "variant_combinations",
]

_PATH = os.path.join(os.path.dirname(__file__), "vectors", "vectors.json.gz")

DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = 1e-9


class ConformanceError(AssertionError):
    """An implementation's output differs from a conformance vector."""


@functools.lru_cache(maxsize=1)
def load() -> Dict[str, Any]:
    """The whole vectors document (cached)."""
    with gzip.open(_PATH, "rb") as fh:
        doc = json.loads(fh.read().decode("utf-8"))
    if doc.get("spec_version") != SPEC_VERSION:
        raise RuntimeError(f"vectors file is spec {doc.get('spec_version')}, code expects {SPEC_VERSION}")
    return doc


def indicators() -> List[str]:
    """Indicator names in the conformance set."""
    return list(load()["indicators"])


def fixtures() -> Dict[str, Dict[str, Any]]:
    """Fixture name -> ``{"description", "data": {"open", "high", "low", "close", "volume"}}``."""
    return load()["fixtures"]


def _f(x: Any) -> float:
    return float("nan") if x is None else float(x)


class Case:
    """One vector: an indicator, a parameter set, a variant and a fixture series."""

    __slots__ = ("id", "indicator", "params", "variant", "fixture", "inputs", "outputs",
                 "expected", "first_valid", "zero_div")

    def __init__(self, raw: Mapping[str, Any], fixture_data: Mapping[str, List[float]]) -> None:
        self.id: str = raw["id"]
        self.indicator: str = raw["indicator"]
        self.params: Dict[str, Any] = dict(raw["params"])
        self.variant: Dict[str, Any] = dict(raw["variant"])
        self.fixture: str = raw["fixture"]
        self.inputs: Dict[str, List[float]] = {k: list(v) for k, v in fixture_data.items()}
        self.outputs: List[str] = list(CATALOG[self.indicator]["outputs"])
        self.expected: Dict[str, List[float]] = {k: [_f(x) for x in v] for k, v in raw["expected"].items()}
        self.first_valid: Dict[str, int] = dict(raw["first_valid"])
        self.zero_div: Dict[str, frozenset] = {k: frozenset(v) for k, v in raw.get("zero_div", {}).items()}

    def __repr__(self) -> str:
        return f"Case({self.id!r})"

    def args(self) -> List[List[float]]:
        """The indicator's inputs in catalogue order (for ``fn(*case.args(), **case.params)``)."""
        return [self.inputs[c] for c in CATALOG[self.indicator]["inputs"]]

    def _actual(self, actual: Any, outputs: Optional[Mapping[str, Any]]) -> Dict[str, List[float]]:
        names = self.outputs
        if outputs is not None:
            if hasattr(actual, "columns") or isinstance(actual, Mapping):
                return {n: _floats(actual[outputs[n]]) for n in names if n in outputs}
            raise TypeError("outputs= maps output names to DataFrame columns or dict keys")
        if isinstance(actual, Mapping):
            missing = [n for n in names if n not in actual]
            if missing:
                raise ConformanceError(f"{self.id}: missing output(s) {missing}; got keys {list(actual)}")
            return {n: _floats(actual[n]) for n in names}
        if hasattr(actual, "columns"):
            cols = list(actual.columns)
            if len(cols) < len(names):
                raise ConformanceError(f"{self.id}: expected {len(names)} columns {names}, got {cols}")
            return {n: _floats(actual[c]) for n, c in zip(names, cols)}
        if isinstance(actual, (tuple, list)) and actual and _is_seq(actual[0]):
            if len(actual) != len(names):
                raise ConformanceError(f"{self.id}: expected {len(names)} outputs {names}, got {len(actual)}")
            return {n: _floats(a) for n, a in zip(names, actual)}
        if len(names) != 1:
            raise ConformanceError(f"{self.id}: {len(names)} outputs expected ({names}); pass a tuple or dict")
        return {names[0]: _floats(actual)}

    def check(
        self,
        actual: Any,
        *,
        warmup: str = "nan",
        zero_division: Any = "talib",
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        outputs: Optional[Mapping[str, Any]] = None,
        max_report: int = 5,
    ) -> None:
        """Compare ``actual`` with this vector; raise ``ConformanceError`` on a mismatch.

        ``actual`` is one array-like (single-output indicators), a tuple/list of arrays in
        catalogue output order, a dict keyed by output name, or a DataFrame (columns in
        catalogue order, or mapped with ``outputs={"upper": "BBU_20_2.0", ...}``).
        A value matches when ``|a - e| <= atol + rtol * |e|``.
        """
        problems = self.mismatches(actual, warmup=warmup, zero_division=zero_division, rtol=rtol,
                                   atol=atol, outputs=outputs)
        if problems:
            shown = "\n  ".join(problems[:max_report])
            more = f"\n  ... {len(problems) - max_report} more" if len(problems) > max_report else ""
            raise ConformanceError(f"{self.id} ({_variant_label(warmup, zero_division)}):\n  {shown}{more}")

    def mismatches(
        self,
        actual: Any,
        *,
        warmup: str = "nan",
        zero_division: Any = "talib",
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        outputs: Optional[Mapping[str, Any]] = None,
    ) -> List[str]:
        """The list of problems ``check`` would report (empty when it conforms)."""
        if warmup not in GLOBAL_AXES["warmup"]:
            raise ValueError(f"warmup must be one of {GLOBAL_AXES['warmup']}")
        zd_value: Optional[float] = None
        if isinstance(zero_division, (int, float)) and not isinstance(zero_division, bool):
            zd_value = float(zero_division)
        elif zero_division not in GLOBAL_AXES["zero_division"]:
            raise ValueError(f"zero_division must be one of {GLOBAL_AXES['zero_division']} or a number")
        got = self._actual(actual, outputs)
        problems: List[str] = []
        for name, values in got.items():
            exp = self.expected[name]
            if len(values) != len(exp):
                problems.append(f"{name}: length {len(values)}, expected {len(exp)}")
                continue
            start = self.first_valid[name]
            zd = self.zero_div.get(name, frozenset())
            for i, (a, e) in enumerate(zip(values, exp)):
                if e != e:  # expected NaN: only warmup positions are NaN in the vectors
                    if warmup == "partial" and i < start:
                        continue
                    if a == a:
                        problems.append(f"{name}[{i}]: expected NaN (warmup), got {a!r}")
                    continue
                if i in zd and zd_value is not None and a == zd_value:
                    continue
                if a != a:
                    if zero_division == "nan" and i in zd:
                        continue
                    problems.append(f"{name}[{i}]: expected {e!r}, got NaN")
                    continue
                if math.isinf(a) or abs(a - e) > atol + rtol * abs(e):
                    problems.append(f"{name}[{i}]: expected {e!r}, got {a!r} (diff {a - e:.3g})")
        return problems


def _is_seq(x: Any) -> bool:
    return hasattr(x, "__len__") and not isinstance(x, (str, bytes)) and not isinstance(x, (int, float))


def _floats(x: Any) -> List[float]:
    if hasattr(x, "to_numpy"):
        x = x.to_numpy(dtype=float, na_value=float("nan"))
    if hasattr(x, "tolist"):
        x = x.tolist()
    return [float("nan") if v is None else float(v) for v in x]


def _variant_label(warmup: str, zero_division: str) -> str:
    return f"warmup={warmup}, zero_division={zero_division}"


def cases(
    indicator: Optional[str] = None,
    *,
    fixtures: Optional[Iterable[str]] = None,
    **variant: Any,
) -> List[Case]:
    """Vectors for ``indicator`` (all indicators if None) and the given variant.

    Variant axes not named take their canonical (first) option, so ``cases("rsi")`` is
    Wilder's RSI and ``cases("rsi", smoothing="ema")`` is the EMA-smoothed one. Unknown
    axes or options raise ``ValueError``.
    """
    doc = load()
    names = [indicator] if indicator is not None else list(doc["indicators"])
    want_fx = set(fixtures) if fixtures is not None else None
    out: List[Case] = []
    for name in names:
        if name not in CATALOG:
            raise ValueError(f"unknown indicator {name!r}; see fin_pit.conformance.indicators()")
        axes = CATALOG[name]["axes"]
        for k, v in variant.items():
            if indicator is None:
                continue
            if k not in axes:
                raise ValueError(f"{name} has no variant axis {k!r} (axes: {list(axes) or 'none'})")
            if v not in axes[k]:
                raise ValueError(f"{name}.{k} must be one of {list(axes[k])}, got {v!r}")
        target = canonical(name)
        if indicator is not None:
            target.update(variant)
        for raw in doc["cases"]:
            if raw["indicator"] != name or raw["variant"] != target:
                continue
            if want_fx is not None and raw["fixture"] not in want_fx:
                continue
            out.append(Case(raw, doc["fixtures"][raw["fixture"]]["data"]))
    return out


def all_cases() -> List[Case]:
    """Every case in the file: all indicators, all variant combinations, all fixtures."""
    doc = load()
    return [Case(raw, doc["fixtures"][raw["fixture"]]["data"]) for raw in doc["cases"]]


def parametrize(indicator: Optional[str] = None, *, argname: str = "pit_case", **variant: Any) -> Any:
    """``pytest.mark.parametrize`` over ``cases(indicator, **variant)``, ids = case ids."""
    import pytest

    selected = cases(indicator, **variant)
    return pytest.mark.parametrize(argname, selected, ids=[c.id for c in selected])


def results(
    fn: Callable[..., Any],
    indicator: str,
    *,
    warmup: str = "nan",
    zero_division: Any = "talib",
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    outputs: Optional[Mapping[str, Any]] = None,
    **variant: Any,
) -> Dict[str, List[str]]:
    """Run ``fn(case)`` for every case of ``indicator``/``variant``; case id -> problems
    (empty list = pass). An exception from ``fn`` is reported as a problem."""
    report: Dict[str, List[str]] = {}
    for case in cases(indicator, **variant):
        try:
            actual = fn(case)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            report[case.id] = [f"raised {type(exc).__name__}: {exc}"]
            continue
        report[case.id] = case.mismatches(actual, warmup=warmup, zero_division=zero_division,
                                          rtol=rtol, atol=atol, outputs=outputs)
    return report
