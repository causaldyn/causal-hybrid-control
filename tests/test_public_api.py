"""Invariants of the top-level namespace itself, not of any one routine in it.

Both of these are cheap and both have already caught something. The shadowing check is the reason
the prescribe facade lives in ``chc.decision``: a module named after the function it exports is
overwritten by that function during ``chc/__init__``, so ``import chc.prescribe`` then
``chc.prescribe.Lever`` raises ``AttributeError`` -- a failure with no symptom until a user writes
the most natural line there is.
"""

from __future__ import annotations

import pkgutil

import chc


def test_no_export_shadows_a_submodule() -> None:
    modules = {found.name for found in pkgutil.iter_modules(chc.__path__)}
    assert sorted(modules & set(chc.__all__)) == []


def test_every_advertised_symbol_exists() -> None:
    missing = [name for name in chc.__all__ if not hasattr(chc, name)]
    assert missing == []


def test_the_namespace_advertises_no_duplicates() -> None:
    assert len(chc.__all__) == len(set(chc.__all__))
