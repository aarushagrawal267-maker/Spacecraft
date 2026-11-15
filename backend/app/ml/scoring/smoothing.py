"""Causal exponentially weighted moving average (EWMA).

Matches ``pandas.Series.ewm(span=s, adjust=True).mean()`` exactly (the smoothing
Telemanom uses), but supports incremental updates so the historical replay
produces the same values as the offline computation.

y_t = sum_i (1-a)^i x_{t-i} / sum_i (1-a)^i,   a = 2 / (span + 1)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


class EWMA:
    def __init__(self, span: int) -> None:
        if span < 1:
            raise ValueError("span must be >= 1")
        self.span = span
        self.decay = 1.0 - 2.0 / (span + 1.0)
        self._num = 0.0
        self._den = 0.0

    def update(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return x
        b, a = [1.0], [1.0, -self.decay]
        num, zf_n = lfilter(b, a, x, zi=[self.decay * self._num])
        den, zf_d = lfilter(b, a, np.ones_like(x), zi=[self.decay * self._den])
        self._num, self._den = float(num[-1]), float(den[-1])
        return num / den


def ewma(x: np.ndarray, span: int) -> np.ndarray:
    return EWMA(span).update(x)
