"""Seed blocks and exact binomial floors for the misspecification tests that count logs.

A test that counts reads ``count`` logs, ``LOGS`` unless it says otherwise, from one block of seeds:
block ``b`` of a test whose seeds start at ``base`` is ``base + b * count`` to
``base + (b + 1) * count - 1``. ``CHC_MISSPEC_BLOCK`` names the block, 0 when unset, so CI reads
fixed seeds and a run on another block reads fresh ones: a floor that held only on the seeds it was
measured on fails there.

A floor is set from the exact binomial law of the count, so that a certificate that holds at the
level ``p0`` its test states fails it on at most ``ALPHA`` of the blocks: :func:`at_least` is the
largest floor a count of ``Binomial(count, p0)`` falls below with probability ``ALPHA`` or less, and
:func:`at_most` the least ceiling it rises above with that probability.
``validation/misspec_power.mac`` reproduces each floor in exact rationals and computes each test's
power against the defect its docstring names.
"""

from __future__ import annotations

import math
import os
from fractions import Fraction

LOGS = 200
ALPHA = Fraction(1, 1000)


def _block() -> int:
    text = os.environ.get("CHC_MISSPEC_BLOCK", "0")
    if not text.isdigit():
        raise ValueError(f"CHC_MISSPEC_BLOCK is a block's number, 0 or more, not {text!r}")
    return int(text)


BLOCK = _block()


def seeds(base: int, count: int = LOGS) -> range:
    """The seeds of this run's block for a test that reads ``count`` logs from ``base`` on."""
    return range(base + BLOCK * count, base + (BLOCK + 1) * count)


def _mass(k: int, p0: Fraction, count: int) -> Fraction:
    return math.comb(count, k) * p0**k * (1 - p0) ** (count - k)


def at_least(p0: str, count: int = LOGS) -> int:
    """The floor of a count of ``count`` logs whose rate is ``p0`` (a decimal, read exactly) at
    least: the largest ``c`` with ``P(count < c) <= ALPHA``."""
    rate, tail, floor = Fraction(p0), Fraction(0), 0
    while floor < count and tail + _mass(floor, rate, count) <= ALPHA:
        tail += _mass(floor, rate, count)
        floor += 1
    return floor


def at_most(p0: str, count: int = LOGS) -> int:
    """The ceiling of a count of ``count`` logs whose rate is ``p0`` at most: the least ``c`` with
    ``P(count > c) <= ALPHA``."""
    rate, tail, ceiling = Fraction(p0), Fraction(0), count
    while ceiling > 0 and tail + _mass(ceiling, rate, count) <= ALPHA:
        tail += _mass(ceiling, rate, count)
        ceiling -= 1
    return ceiling
