"""Reference implementations vs. real libraries.

Each library runs with the variant it declares (tools/libraries.py). This is what makes the
golden vectors trustworthy: the canonical vectors are TA-Lib's numbers, and the other
variants are pandas-ta-classic's, fin-primitives' and stockstats' numbers.
"""
import os
import sys
import warnings

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import libraries  # noqa: E402

PARAMS = []
for lib, make in libraries.LIBRARIES.items():
    try:
        adapters = make()
    except ImportError:
        continue
    for ind, adapter in adapters.items():
        marks = []
        reason = libraries.KNOWN_DIFFERENCES.get((lib, ind))
        if reason:
            marks.append(pytest.mark.xfail(reason=reason, strict=True))
        PARAMS.append(pytest.param(lib, ind, adapter, id=f"{lib}-{ind}", marks=marks))


@pytest.mark.skipif(not PARAMS, reason="no reference library installed")
@pytest.mark.parametrize("lib,ind,adapter", PARAMS)
def test_library_conforms_to_declared_variant(lib, ind, adapter):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        decl, rep = libraries.run(lib, ind, adapter)
    failed = {k: v[:3] for k, v in rep.items() if v}
    assert not failed, f"{lib} {ind} {decl}: {failed}"


def test_talib_covers_every_canonical_indicator():
    pytest.importorskip("talib")
    names = set(libraries.talib_adapters())
    from fin_pit.conformance import CATALOG

    no_talib = {n for n, e in CATALOG.items() if e["talib"] is None}
    assert names | no_talib == set(CATALOG)
