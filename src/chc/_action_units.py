"""The units a Riccati solve reads each action in. NumPy only, and it imports nothing from ``chc``,
so the LQR solvers share it without pulling in each other's modules."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def action_units(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """A power of two per action that brings ``r``'s diagonal to within a factor of 2 of 1.

    SciPy's Riccati solvers read ``b`` and ``r`` apart, so one problem written in other units of
    the actions read another gain: off by 1e-4 at a billionth of the units, and refused at 1e-12.
    Solved in these units, every such problem is one problem to the bit, since a power of two
    rescales exactly. An action whose diagonal entry is not positive keeps its units.
    """
    diagonal = np.diag(np.atleast_2d(r))
    return np.exp2(-np.round(np.log2(np.where(diagonal > 0.0, diagonal, 1.0)) / 2.0))
