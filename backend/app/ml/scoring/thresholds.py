"""Anomaly thresholding of smoothed prediction errors.

``DynamicThreshold`` re-implements the nonparametric dynamic thresholding (NDT)
of Hundman et al. (KDD 2018) following the published Telemanom code
(``telemanom/errors.py``):

* errors are processed in overlapping windows of ``window_size`` batches of
  ``batch_size`` points; a window only *reports* anomalies in its newest batch;
* per window, epsilon = mean + z*sd is chosen (z in [2.5, 12) step 0.5) to
  maximise  (Δmean/mean + Δsd/sd) / (n_sequences² + n_flagged), subject to at
  most 5 sequences and fewer than 50 % of points flagged;
* the same search runs on errors reflected about the window mean ("inverse"
  pass) to catch unusually *low* errors;
* flags are buffered by ``error_buffer`` points, then pruned: sequences whose
  peak error is not at least ``p`` (relative) above the next-highest peak,
  ending with the highest non-anomalous error, are removed.

Deliberate, documented deviations from Telemanom (docs/methodology.md):
1. no initial errors are ignored by default (Telemanom ignores up to 2·l_s and
   special-cases channel C-2);
2. the EWMA is not overwritten during warm-up (the adjust=True EWMA has no
   start-up bias);
3. the final window reports every point after the previous window's zone
   (Telemanom's final window can leave a gap of up to batch_size-1 points that
   no window ever evaluates).

The class is streaming-capable: points can be pushed incrementally and each
window is processed as soon as it is complete, which is what the historical
replay uses. Offline detection is the same object fed all points at once.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

HIGH, LOW = 1, 2


@dataclass
class ThresholdConfig:
    mode: str = "ndt"  # "ndt" (dynamic) or "static" (mean + k·sd of training errors)
    batch_size: int = 70
    window_size: int = 30
    smoothing_perc: float = 0.05
    error_buffer: int = 100
    p: float = 0.13
    z_min: float = 2.5
    z_max: float = 12.0
    z_step: float = 0.5
    max_sequences: int = 5
    max_flag_fraction: float = 0.5
    min_error: float = 0.05
    guard_fraction: float = 0.05
    inverse: bool = True
    ignore_initial: int = 0
    static_k: float = 3.0

    @property
    def span(self) -> int:
        return max(1, int(self.batch_size * self.window_size * self.smoothing_perc))

    @classmethod
    def from_dict(cls, d: dict | None) -> "ThresholdConfig":
        """Build a config from user/stored settings; raises ``ValueError`` with a readable
        message for a wrong type or an out-of-range value (instead of failing deep in NDT)."""
        d = d or {}
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        cfg = cls(**known)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        def num(name: str, *, integer: bool = False, lo: float | None = None, hi: float | None = None,
                lo_open: bool = False, hi_open: bool = False) -> None:
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
                raise ValueError(f"threshold '{name}' must be a number, got {v!r}")
            if not np.isfinite(v) or (integer and float(v) != int(v)):
                raise ValueError(f"threshold '{name}' must be a finite {'integer' if integer else 'number'}, got {v!r}")
            if lo is not None and (v <= lo if lo_open else v < lo):
                raise ValueError(f"threshold '{name}' must be {'>' if lo_open else '>='} {lo}, got {v!r}")
            if hi is not None and (v >= hi if hi_open else v > hi):
                raise ValueError(f"threshold '{name}' must be {'<' if hi_open else '<='} {hi}, got {v!r}")
            setattr(self, name, int(v) if integer else float(v))

        if self.mode not in ("ndt", "static"):
            raise ValueError(f"threshold 'mode' must be 'ndt' or 'static', got {self.mode!r}")
        if not isinstance(self.inverse, bool):
            raise ValueError(f"threshold 'inverse' must be true or false, got {self.inverse!r}")
        num("batch_size", integer=True, lo=1)
        num("window_size", integer=True, lo=1)
        num("smoothing_perc", lo=0, hi=1, lo_open=True)
        num("error_buffer", integer=True, lo=0)
        num("p", lo=0, hi=1)
        num("z_min", lo=0)
        num("z_max", lo=0)
        num("z_step", lo=0, lo_open=True)
        if self.z_max <= self.z_min:
            raise ValueError("threshold 'z_max' must be greater than 'z_min'")
        num("max_sequences", integer=True, lo=1)
        num("max_flag_fraction", lo=0, hi=1, lo_open=True)
        num("min_error", lo=0)
        num("guard_fraction", lo=0, hi=1, hi_open=True)
        num("ignore_initial", integer=True, lo=0)
        num("static_k", lo=0)

    def to_dict(self) -> dict:
        return asdict(self)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) of consecutive True runs."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    d = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return mask.copy()
    k = np.ones(2 * radius + 1)
    return np.convolve(mask.astype(np.float64), k, mode="same") > 0.5


@dataclass
class WindowInfo:
    index: int
    start: int
    end: int
    zone: tuple[int, int]
    mean: float
    sd: float
    z_high: float
    epsilon_high: float
    z_low: float | None
    epsilon_low: float | None  # expressed in error space (lower threshold)
    guard_passed: bool
    n_flagged: int = 0


@dataclass
class Interval:
    start: int
    end: int  # inclusive
    score: float
    peak_error: float
    threshold: float
    direction: str
    window: dict = field(default_factory=dict)


class DynamicThreshold:
    def __init__(self, n_total: int, cfg: ThresholdConfig) -> None:
        if n_total < 2:
            raise ValueError("Need at least 2 points for thresholding")
        self.cfg = cfg
        bs = cfg.batch_size
        ws = cfg.window_size
        n_windows = int((n_total - bs * ws) / bs)
        while n_windows < 0:
            ws -= 1
            if ws < 1:
                # Sequence shorter than one batch: treat the whole series as a single window.
                ws, bs = 1, n_total
                n_windows = 0
                break
            n_windows = int((n_total - bs * ws) / bs)
        self.bs, self.ws, self.n_windows, self.n_total = bs, ws, n_windows, n_total

        self._e = np.full(n_total, np.nan)
        self._y = np.full(n_total, np.nan)
        self._n = 0
        self.next_window = 0
        self.flags = np.zeros(n_total, dtype=bool)
        self.direction = np.zeros(n_total, dtype=np.int8)
        self.margin = np.full(n_total, -np.inf)
        self.window_of = np.full(n_total, -1, dtype=np.int32)
        self.threshold_high = np.full(n_total, np.nan)
        self.threshold_low = np.full(n_total, np.nan)
        self.windows: list[WindowInfo] = []
        self._zone_end = 0

    # ------------------------------------------------------------------ geometry
    def bounds(self, i: int) -> tuple[int, int]:
        start = i * self.bs
        end = (self.ws + i) * self.bs if i < self.n_windows else self.n_total
        return start, end

    def zone(self, i: int) -> tuple[int, int]:
        _, end = self.bounds(i)
        if i == 0:
            return min(self.cfg.ignore_initial, end), end
        if i == self.n_windows:
            return self._zone_end, end
        return end - self.bs, end

    @property
    def processed_upto(self) -> int:
        """Points before this index have received their final verdict."""
        return self._zone_end

    # ------------------------------------------------------------------ streaming
    def push(self, e_s: np.ndarray, y: np.ndarray) -> list[int]:
        e_s = np.asarray(e_s, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        k = len(e_s)
        if self._n + k > self.n_total:
            raise ValueError("Pushed more points than n_total")
        self._e[self._n : self._n + k] = e_s
        self._y[self._n : self._n + k] = y
        self._n += k
        done = []
        while self.next_window <= self.n_windows and self.bounds(self.next_window)[1] <= self._n:
            self._process(self.next_window)
            done.append(self.next_window)
            self.next_window += 1
        return done

    def finalize(self) -> None:
        if self._n != self.n_total:
            raise ValueError(f"finalize() called after {self._n}/{self.n_total} points")
        while self.next_window <= self.n_windows:
            self._process(self.next_window)
            self.next_window += 1

    # ------------------------------------------------------------------ core
    def _find_epsilon(self, es: np.ndarray, mean: float, sd: float) -> tuple[float, float]:
        cfg = self.cfg
        best_eps, best_z, best_score = mean + 12.0 * sd, 12.0, -np.inf
        if sd <= 0:
            return np.inf, np.inf
        n = len(es)
        for z in np.arange(cfg.z_min, cfg.z_max, cfg.z_step):
            eps = mean + z * sd
            above = es >= eps
            if not above.any():
                continue
            below = es[~above]
            if below.size == 0:
                continue
            buffered = _dilate(above, cfg.error_buffer - 1)
            n_flag = int(buffered.sum())
            n_seq = sum(1 for s, e in runs(buffered) if e > s)
            mean_dec = (mean - below.mean()) / mean if mean > 0 else 0.0
            sd_dec = (sd - below.std()) / sd
            score = (mean_dec + sd_dec) / (n_seq**2 + n_flag)
            if score >= best_score and n_seq <= cfg.max_sequences and n_flag < n * cfg.max_flag_fraction:
                best_score, best_eps, best_z = score, eps, float(z)
        return best_eps, best_z

    def _compare(
        self, es: np.ndarray, eps: float, inter_range: float, zone_local: tuple[int, int], s: int
    ) -> np.ndarray:
        cfg = self.cfg
        n = len(es)
        hit = (es >= eps) & (es > cfg.guard_fraction * inter_range)
        if not hit.any():
            return np.zeros(n, dtype=bool)
        flagged = _dilate(hit, cfg.error_buffer)
        zone_mask = np.zeros(n, dtype=bool)
        zone_mask[zone_local[0] : zone_local[1]] = True
        flagged &= zone_mask
        if not flagged.any():
            return flagged
        # Highest smoothed error in the window that is not (already or newly) anomalous.
        prior = self.flags[s : s + n]
        candidates = ~(prior | flagged)
        non_anom_max = float(es[candidates].max()) if candidates.any() else 0.0
        return self._prune(flagged, es, non_anom_max)

    def _prune(self, flagged: np.ndarray, es: np.ndarray, non_anom_max: float) -> np.ndarray:
        seqs = [(a, b) for a, b in runs(flagged) if b > a]  # single points are not sequences
        if not seqs:
            return np.zeros_like(flagged)
        peaks = np.array([es[a : b + 1].max() for a, b in seqs])
        order = np.sort(peaks)[::-1]
        chain = np.append(order, non_anom_max)
        remove: set[int] = set()
        for i in range(len(chain) - 1):
            cur, nxt = chain[i], chain[i + 1]
            rel = (cur - nxt) / cur if cur > 0 else 0.0
            if rel < self.cfg.p:
                remove.update(np.flatnonzero(peaks == cur).tolist())
            else:
                remove = set()
        keep = np.zeros_like(flagged)
        for j, (a, b) in enumerate(seqs):
            if j not in remove:
                keep[a : b + 1] = True
        return flagged & keep

    def _process(self, i: int) -> None:
        cfg = self.cfg
        s, e = self.bounds(i)
        zs, ze = self.zone(i)
        es = self._e[s:e]
        yw = self._y[s:e]
        mean, sd = float(es.mean()), float(es.std())
        sd_values = float(yw.std())
        inter_range = float(np.percentile(yw, 95) - np.percentile(yw, 5))

        eps_hi, z_hi = self._find_epsilon(es, mean, sd)
        eps_lo_inv, z_lo = (np.inf, None)
        es_inv = None
        if cfg.inverse:
            es_inv = mean + (mean - es)
            eps_lo_inv, z_lo = self._find_epsilon(es_inv, mean, sd)
        eps_lo = (2 * mean - eps_lo_inv) if np.isfinite(eps_lo_inv) else None

        self.threshold_high[zs:ze] = eps_hi if np.isfinite(eps_hi) else np.nan
        if eps_lo is not None:
            self.threshold_low[zs:ze] = eps_lo

        es_max = float(es.max())
        guard = (sd > cfg.guard_fraction * sd_values or es_max > cfg.guard_fraction * inter_range) and (
            es_max > cfg.min_error
        )
        zl = (zs - s, ze - s)
        n_flag = 0
        if guard:
            denom = mean + sd if (mean + sd) > 0 else 1.0
            hi = self._compare(es, eps_hi, inter_range, zl, s) if np.isfinite(eps_hi) else np.zeros(len(es), bool)
            lo = (
                self._compare(es_inv, eps_lo_inv, inter_range, zl, s)
                if (es_inv is not None and np.isfinite(eps_lo_inv))
                else np.zeros(len(es), bool)
            )
            for mask, direction, margin in (
                (hi, HIGH, (es - eps_hi) / denom),
                (lo, LOW, (es_inv - eps_lo_inv) / denom if es_inv is not None else None),
            ):
                if margin is None or not mask.any():
                    continue
                gidx = np.flatnonzero(mask) + s
                self.flags[gidx] = True
                self.direction[gidx] |= direction
                self.margin[gidx] = np.maximum(self.margin[gidx], margin[mask])
                self.window_of[gidx] = i
                n_flag += int(mask.sum())

        self.windows.append(
            WindowInfo(
                index=i, start=s, end=e, zone=(zs, ze), mean=mean, sd=sd,
                z_high=float(z_hi), epsilon_high=float(eps_hi),
                z_low=None if z_lo is None else float(z_lo),
                epsilon_low=None if eps_lo is None else float(eps_lo),
                guard_passed=bool(guard), n_flagged=n_flag,
            )
        )
        self._zone_end = ze

    # ------------------------------------------------------------------ results
    def intervals(self, offset: int = 0, upto: int | None = None) -> list[Interval]:
        """Final anomaly sequences (length > 1), shifted by ``offset``."""
        limit = self.processed_upto if upto is None else min(upto, self.processed_upto)
        out: list[Interval] = []
        for a, b in runs(self.flags[:limit]):
            if b == a:
                continue
            seg = slice(a, b + 1)
            peak_local = a + int(np.nanargmax(self._e[seg]))
            dirs = int(np.bitwise_or.reduce(self.direction[seg]))
            w_idx = int(self.window_of[peak_local]) if self.window_of[peak_local] >= 0 else int(
                max(self.window_of[seg])
            )
            win = asdict(self.windows[w_idx]) if 0 <= w_idx < len(self.windows) else {}
            if win:
                win["start"] += offset
                win["end"] += offset
                win["zone"] = [win["zone"][0] + offset, win["zone"][1] + offset]
            out.append(
                Interval(
                    start=a + offset,
                    end=b + offset,
                    score=float(np.max(self.margin[seg])),
                    peak_error=float(self._e[peak_local]),
                    threshold=float(self.threshold_high[peak_local]),
                    direction={HIGH: "high", LOW: "low", HIGH | LOW: "both"}.get(dirs, "high"),
                    window=win,
                )
            )
        return out


@dataclass
class StaticThreshold:
    """Fixed threshold epsilon = mean + k·sd of *training-split* smoothed errors.

    Uses no test information, so it is a fair, label-free comparison point for
    the dynamic method.
    """

    epsilon: float
    n_total: int
    k: float

    def __post_init__(self) -> None:
        self.flags = np.zeros(self.n_total, dtype=bool)
        self._e = np.full(self.n_total, np.nan)
        self._n = 0
        self.threshold_high = np.full(self.n_total, self.epsilon)
        self.threshold_low = np.full(self.n_total, np.nan)
        self.windows: list[WindowInfo] = []

    @classmethod
    def from_training_errors(cls, train_e_s: np.ndarray, n_total: int, k: float) -> "StaticThreshold":
        t = train_e_s[np.isfinite(train_e_s)]
        eps = float(t.mean() + k * t.std()) if t.size else np.inf
        return cls(epsilon=eps, n_total=n_total, k=k)

    @property
    def processed_upto(self) -> int:
        return self._n

    def push(self, e_s: np.ndarray, y: np.ndarray | None = None) -> None:
        k = len(e_s)
        self._e[self._n : self._n + k] = e_s
        self.flags[self._n : self._n + k] = np.asarray(e_s) >= self.epsilon
        self._n += k

    def finalize(self) -> None:
        pass

    def intervals(self, offset: int = 0, upto: int | None = None) -> list[Interval]:
        limit = self._n if upto is None else min(upto, self._n)
        out = []
        for a, b in runs(self.flags[:limit]):
            if b == a:
                continue
            seg = self._e[a : b + 1]
            denom = abs(self.epsilon) if self.epsilon not in (0, np.inf) else 1.0
            out.append(
                Interval(
                    start=a + offset, end=b + offset,
                    score=float((seg.max() - self.epsilon) / denom),
                    peak_error=float(seg.max()), threshold=self.epsilon, direction="high",
                    window={"mode": "static", "k": self.k, "epsilon": self.epsilon},
                )
            )
        return out
