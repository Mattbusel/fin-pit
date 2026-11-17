"""Generate the conformance vectors: src/fin_pit/conformance/vectors/vectors.json.gz.

    python tools/gen_vectors.py          # write
    python tools/gen_vectors.py --check  # exit 1 if the committed file is out of date

The eight fixture series are built here from a 64-bit LCG (no numpy, no RNG library), so the
file is reproducible byte for byte on any platform. Expected values come from
fin_pit.conformance.reference; tests/test_conformance_libraries.py checks those against
TA-Lib, pandas-ta-classic, fin-primitives and stockstats.
"""
from __future__ import annotations

import gzip
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from fin_pit.conformance import catalog, reference  # noqa: E402

OUT = os.path.join(ROOT, "src", "fin_pit", "conformance", "vectors", "vectors.json.gz")


class LCG:
    def __init__(self, seed: int) -> None:
        self.state = seed & ((1 << 64) - 1)

    def uniform(self) -> float:
        self.state = (6364136223846793005 * self.state + 1442695040888963407) % (1 << 64)
        return (self.state >> 11) / float(1 << 53)

    def normalish(self) -> float:
        # Irwin-Hall: sum of 12 uniforms minus 6, mean 0, variance 1.
        return sum(self.uniform() for _ in range(12)) - 6.0


def r4(x: float) -> float:
    return float(round(x, 4))


def ohlcv_from_close(close, rng, wick=0.6, base_volume=50000.0):
    o, h, l, v = [], [], [], []
    prev = close[0]
    for i, c in enumerate(close):
        op = prev if i else c
        hi = max(op, c) + abs(rng.normalish()) * wick
        lo = min(op, c) - abs(rng.normalish()) * wick
        o.append(r4(op))
        h.append(r4(hi))
        l.append(r4(max(lo, 0.01)))
        v.append(float(round(base_volume * (1.0 + abs(rng.normalish())))))
        prev = c
    return {"open": o, "high": h, "low": l, "close": [r4(c) for c in close], "volume": v}


def walk(seed, n, start, vol):
    # Only +, -, * and / (no exp/pow): IEEE 754 makes those identical on every platform, while
    # libm's exp differs in the last bit between glibc and the Windows CRT.
    rng = LCG(seed)
    c = [start]
    for _ in range(n - 1):
        c.append(c[-1] * (1.0 + vol * rng.normalish()))
    return c, rng


def _decline(start, factor, n):
    out = [start]
    for _ in range(n - 1):
        out.append(out[-1] * factor)
    return out


def fixtures():
    out = {}
    c, rng = walk(0xA11CE, 260, 100.0, 0.015)
    out["walk_a"] = ("Random walk, 260 bars, 1.5% daily volatility", ohlcv_from_close(c, rng))
    c, rng = walk(0xB0B, 260, 25.0, 0.035)
    out["walk_b"] = ("Random walk, 260 bars, 3.5% daily volatility", ohlcv_from_close(c, rng, wick=0.3))
    rng = LCG(7)
    out["trend_up"] = ("Straight line up, 120 bars, 0.5 per bar",
                       ohlcv_from_close([50.0 + 0.5 * i for i in range(120)], rng, wick=0.25))
    rng = LCG(11)
    out["trend_down"] = ("Geometric decline, 120 bars, -1% per bar",
                         ohlcv_from_close(_decline(200.0, 0.99, 120), rng, wick=0.5))
    flat = {k: [100.0] * 60 for k in ("open", "high", "low", "close")}
    flat["volume"] = [10000.0] * 60
    out["flat"] = ("Constant 100, zero range, constant volume, 60 bars (every 0/0 case)", flat)
    zz = [100.0 + (1.0 if i % 2 else -1.0) for i in range(100)]
    rng = LCG(13)
    out["zigzag"] = ("Alternating 99 / 101, 100 bars", ohlcv_from_close(zz, rng, wick=0.2))
    c, rng = walk(0x5EED, 150, 80.0, 0.01)
    c = c[:75] + [x * 0.7 for x in c[75:]]
    shock = ohlcv_from_close(c, rng)
    shock["volume"][75] *= 20
    out["shock"] = ("Random walk, 150 bars, 30% gap down at bar 75 with a 20x volume spike", shock)
    c, rng = walk(0x5407, 12, 40.0, 0.02)
    out["short"] = ("12 bars, shorter than most warmups", ohlcv_from_close(c, rng))
    return out


def enc(x):
    return None if x != x else x


def first_valid(values):
    for i, v in enumerate(values):
        if v == v:
            return i
    return len(values)


def build():
    fx = fixtures()
    cases = []
    for name, entry in catalog.CATALOG.items():
        fn = reference.FUNCTIONS[name]
        for variant in catalog.variant_combinations(name):
            for fname, (_, data) in fx.items():
                args = [data[c] for c in entry["inputs"]]
                reference._ZERO_DIVISION[0] = 0.0
                expected = fn(*args, **entry["params"], **variant)
                reference._ZERO_DIVISION[0] = 1.0
                with_nan = fn(*args, **entry["params"], **variant)
                reference._ZERO_DIVISION[0] = 0.0
                zero_div = {}
                for out_name in entry["outputs"]:
                    e, z = expected[out_name], with_nan[out_name]
                    idx = [i for i in range(len(e)) if e[i] == e[i] and z[i] != z[i]]
                    if idx:
                        zero_div[out_name] = idx
                vid = "/".join(
                    [name] + [f"{k}={v}" for k, v in variant.items()] + [fname]
                )
                cases.append(
                    {
                        "id": vid,
                        "indicator": name,
                        "params": entry["params"],
                        "variant": variant,
                        "fixture": fname,
                        "first_valid": {o: first_valid(expected[o]) for o in entry["outputs"]},
                        "expected": {o: [enc(x) for x in expected[o]] for o in entry["outputs"]},
                        "zero_div": zero_div,
                    }
                )
    doc = {
        "spec": "fin-pit indicator conformance",
        "spec_version": catalog.SPEC_VERSION,
        "global_axes": {k: list(v) for k, v in catalog.GLOBAL_AXES.items()},
        "indicators": {
            k: {
                "title": v["title"],
                "inputs": list(v["inputs"]),
                "params": v["params"],
                "outputs": list(v["outputs"]),
                "axes": {a: list(o) for a, o in v["axes"].items()},
                "talib": v["talib"],
            }
            for k, v in catalog.CATALOG.items()
        },
        "fixtures": {k: {"description": d, "data": data} for k, (d, data) in fx.items()},
        "cases": cases,
    }
    raw = json.dumps(doc, separators=(",", ":"), sort_keys=False, allow_nan=False).encode()
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, filename="") as gz:
        gz.write(raw)
    return buf.getvalue(), len(cases)


def main():
    data, n = build()
    if "--check" in sys.argv:
        try:
            with open(OUT, "rb") as fh:
                current = fh.read()
        except FileNotFoundError:
            current = b""
        old = gzip.decompress(current) if current else b""
        if old != gzip.decompress(data):
            print("vectors.json.gz is out of date: run python tools/gen_vectors.py")
            return 1
        print(f"vectors.json.gz is up to date ({n} cases)")
        return 0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as fh:
        fh.write(data)
    print(f"wrote {OUT}: {n} cases, {len(data)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
