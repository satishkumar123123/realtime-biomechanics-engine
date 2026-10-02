"""Adaptive One-Euro smoothing and optional calibrated segment consistency checks.

Algorithm: Casiez, Roussel and Vogel, CHI 2012 (https://gery.casiez.net/1euro/).
No model inference, camera I/O, or additional dependencies are used here.
"""

from dataclasses import dataclass
import math
import time
from typing import Optional

import numpy as np


class OneEuroFilter:
    """Filter scalar or fixed-shape arrays independently along every component.

    Timestamps are monotonic seconds, e.g. capture.timestamp_ns / 1e9. Cutoffs
    are Hz; beta is speed sensitivity and depends on signal units. The defaults
    are starting values, not a zero-lag guarantee. Increase beta when fast motion
    still lags. Use one instance per stream; calls must be externally serialized.

    Invalid components (None, NaN, infinity, or valid_mask=False) return NaN and clear
    that component's history. Its next valid observation starts without a stale
    derivative. Repeated/backward timestamps hold valid components; invalid
    components still clear history and return NaN. Missing samples must reach
    this filter to clear history,
    or callers should reset it after a tracking gap.
    """

    MIN_DT = 1e-12

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        for name, value, positive in (("min_cutoff", min_cutoff, True),
                                      ("beta", beta, False),
                                      ("d_cutoff", d_cutoff, True)):
            if not math.isfinite(value) or (value <= 0 if positive else value < 0):
                raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self):
        """Clear all signal, derivative and timestamp history (shape may change)."""
        self._timestamp = None
        self._raw = None
        self._filtered = None
        self._derivative = None
        self._initialized = None
        self._cutoff = None

    @property
    def cutoff(self):
        """Copy of the last component-wise cutoff in Hz, or None before use."""
        return None if self._cutoff is None else self._cutoff.copy()

    @staticmethod
    def _alpha(cutoff, dt):
        # Reciprocal form avoids forming large products dt * cutoff.
        with np.errstate(over='ignore', divide='ignore', under='ignore'):
            return 1.0 / (1.0 + (1.0 / cutoff) / (2.0 * math.pi) / dt)

    def __call__(self, value, timestamp=None, *, valid_mask=None):
        return self.filter(value, timestamp, valid_mask=valid_mask)

    def filter(self, value, timestamp=None, *, valid_mask=None):
        """Return an owned float64 array (0-D for scalar input).

        valid_mask must broadcast to the input shape: for (33, 3) landmarks,
        supply a (33, 1) visibility mask to gate whole keypoints. Nonfinite input
        is always invalid. A whole-frame None invalidates all established components;
        before the first shaped sample it returns scalar NaN without state. Input
        shape is fixed until reset. Nonfinite timestamps
        and malformed input raise ValueError without changing state. Positive
        intervals <= 1e-12 seconds are treated like duplicates for numeric safety.
        """
        timestamp = time.perf_counter() if timestamp is None else float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite monotonic seconds")
        if value is None and self._raw is None:
            return np.asarray(np.nan)
        sample = (np.full_like(self._raw, np.nan) if value is None
                  else np.asarray(value, dtype=np.float64))
        if self._raw is not None and sample.shape != self._raw.shape:
            raise ValueError("Input shape changed; reset before changing shape")
        valid = np.isfinite(sample)
        if valid_mask is not None:
            valid = valid & np.broadcast_to(np.asarray(valid_mask, dtype=bool), sample.shape)

        if self._timestamp is not None:
            dt = timestamp - self._timestamp
            if not math.isfinite(dt):
                raise ValueError("Timestamp interval is not finite")
            if dt <= self.MIN_DT:
                # Time errors must not turn current occlusion/NaNs into stale poses.
                self._initialized &= valid
                self._raw = np.where(valid, self._raw, 0.0)
                self._filtered = np.where(valid, self._filtered, np.nan)
                self._derivative = np.where(valid, self._derivative, 0.0)
                self._cutoff = np.where(valid, self._cutoff, np.nan)
                return self._filtered.copy()
        else:
            dt = None

        if self._raw is None:
            raw = np.where(valid, sample, 0.0)
            filtered = np.where(valid, sample, np.nan)
            derivative = np.zeros_like(sample)
            cutoff = np.where(valid, self.min_cutoff, np.nan)
        else:
            continuing = valid & self._initialized
            # Mask before math so invalid/uninitialized data never enter derivatives.
            current = np.where(continuing, sample, 0.0)
            previous = np.where(continuing, self._raw, 0.0)
            previous_filtered = np.where(continuing, self._filtered, 0.0)
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                velocity = (current - previous) / dt
                alpha_d = self._alpha(self.d_cutoff, dt)
                derivative = (alpha_d * velocity
                              + (1.0 - alpha_d) * self._derivative)
                derivative = np.where(continuing, derivative, 0.0)
                cutoff = self.min_cutoff + self.beta * np.abs(derivative)
                alpha = self._alpha(cutoff, dt)
                smoothed = alpha * current + (1.0 - alpha) * previous_filtered
            if (not np.all(np.isfinite(derivative))
                    or not np.all(np.isfinite(cutoff))
                    or not np.all(np.isfinite(smoothed))):
                raise ValueError("Signal magnitude exceeds safe numeric range")
            raw = np.where(valid, sample, 0.0)
            filtered = np.where(valid, np.where(continuing, smoothed, sample), np.nan)
            cutoff = np.where(valid, cutoff, np.nan)

        # Commit state only after all validations/calculations succeed.
        self._timestamp = timestamp
        self._raw = raw.copy()
        self._filtered = filtered.copy()
        self._derivative = derivative.copy()
        self._initialized = valid.copy()
        self._cutoff = cutoff.copy()
        return filtered.copy()


@dataclass(frozen=True)
class SegmentCheck:
    """Consistency verdict; None verdict means unavailable, not a valid segment."""

    length: Optional[float]
    reference_length: float
    relative_deviation: Optional[float]
    consistent: Optional[bool]


class BoneLengthConstraintChecker:
    """Flag segment length drift against an explicit, frozen calibration.

    This is a relative consistency check, not a population anthropometric test.
    It neither warps landmarks nor updates its baseline from noisy live data.
    Calibrate using trustworthy neutral frames (median across multiple frames is
    recommended). Recalibrate for a new person, camera setup or tracking session.
    Gate downstream measurements using verdicts; missing data must not be passed.
    """

    DEFAULT_SEGMENTS = {
        "L_UpperArm": (11, 13), "R_UpperArm": (12, 14),
        "L_Shin": (25, 27), "R_Shin": (26, 28),
    }
    EPSILON = 1e-7

    def __init__(self, relative_tolerance=0.20, visibility_threshold=0.65,
                 segments=None):
        if not math.isfinite(relative_tolerance) or not 0 <= relative_tolerance < 1:
            raise ValueError("relative_tolerance must be finite and in [0, 1)")
        if not math.isfinite(visibility_threshold) or not 0 <= visibility_threshold <= 1:
            raise ValueError("visibility_threshold must be in [0, 1]")
        self.relative_tolerance = float(relative_tolerance)
        self.visibility_threshold = float(visibility_threshold)
        self.segments = dict(self.DEFAULT_SEGMENTS if segments is None else segments)
        if not self.segments:
            raise ValueError("At least one segment is required")
        for pair in self.segments.values():
            if (len(pair) != 2 or any(not isinstance(i, (int, np.integer)) or not 0 <= i < 33
                                      for i in pair) or pair[0] == pair[1]):
                raise ValueError("Segments must contain two distinct landmark indices in [0, 32]")
        self._reference = None

    def reset(self):
        """Discard calibration; check() requires another calibration."""
        self._reference = None

    @classmethod
    def _points(cls, landmarks):
        points = np.asarray(landmarks, dtype=np.float64)
        if points.shape != (33, 3):
            raise ValueError("Expected landmark shape (33, 3)")
        return points

    def _length(self, points, pair, visibilities=None):
        if visibilities is not None:
            for index in pair:
                try:
                    raw_confidence = visibilities[index]
                    if np.ndim(raw_confidence) != 0:
                        return None
                    confidence = float(raw_confidence)
                except (KeyError, IndexError, TypeError, ValueError, OverflowError):
                    return None
                if not np.isfinite(confidence) or not self.visibility_threshold < confidence <= 1:
                    return None
        if not np.all(np.isfinite(points[list(pair)])):
            return None
        with np.errstate(over='ignore', invalid='ignore'):
            length = float(np.hypot.reduce(points[pair[0]] - points[pair[1]]))
        return length if np.isfinite(length) and length > self.EPSILON else None

    def calibrate(self, landmarks, visibilities=None):
        """Freeze lengths from a reliable (33, 3) frame; reject invalid segments.

        For robust calibration pass a median landmark frame from stable,
        visibility-gated observations. Failure leaves any old calibration intact.
        """
        points = self._points(landmarks)
        reference = {name: self._length(points, pair, visibilities)
                     for name, pair in self.segments.items()}
        if any(length is None for length in reference.values()):
            raise ValueError("Calibration contains missing, occluded or degenerate segments")
        self._reference = reference
        return dict(reference)

    def check(self, landmarks, visibilities=None):
        """Return per-segment verdicts without modifying points or baseline."""
        if self._reference is None:
            raise RuntimeError("Calibrate segment lengths before checking")
        points = self._points(landmarks)
        verdicts = {}
        for name, pair in self.segments.items():
            reference = self._reference[name]
            length = self._length(points, pair, visibilities)
            deviation = None if length is None else abs(length / reference - 1.0)
            verdicts[name] = SegmentCheck(
                length, reference, deviation,
                None if deviation is None else deviation <= self.relative_tolerance + 1e-12)
        return verdicts
