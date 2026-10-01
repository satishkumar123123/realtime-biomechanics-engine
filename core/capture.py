"""Continuously drain a camera into a bounded, latest-frame mailbox."""

from dataclasses import dataclass
import math
import threading
import time
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameSnapshot:
    """One publication; timestamp is monotonic host time after camera read."""

    sequence: int
    timestamp_ns: int
    frame: np.ndarray


class VideoCaptureAsync:
    """Camera polling on a daemon thread, independent of inference speed.

    Resolution and FPS are requests: a driver may negotiate different values.
    Frames superseded before consumption are intentionally discarded. No camera
    or OS can guarantee zero dropped frames. ``read`` never waits for camera I/O;
    use ``wait_for_frame`` to wait for a distinct publication without busy polling.
    Configuration applies on each start; stop before changing camera settings.
    """

    def __init__(self, source=0, width=640, height=480, fps=60.0,
                 backend=cv2.CAP_ANY, *, capture_factory=None):
        if (not isinstance(width, int) or not isinstance(height, int)
                or width <= 0 or height <= 0
                or not math.isfinite(fps) or fps <= 0):
            raise ValueError("Resolution and FPS must be positive and finite")
        self.source, self.width, self.height = source, width, height
        self.fps, self.backend = fps, backend
        # Optional OpenCV-compatible source factory for paced replay/benchmarks.
        self._capture_factory = capture_factory
        self._lifecycle = threading.Lock()
        self._condition = threading.Condition(threading.Lock())
        self._stop = threading.Event()
        self._thread = None
        self._latest = None
        self._sequence = 0
        self._running = False
        self._error = None
        self.actual_settings = {}

    @property
    def running(self):
        """Whether the capture worker is currently active."""
        with self._condition:
            return self._running

    @property
    def error(self):
        """Last worker failure, or None; stale frames are cleared on failure."""
        with self._condition:
            return self._error

    def start(self):
        """Open/configure the camera and start polling; idempotent while running.

        Opening/configuration may block on the driver, but no consumer read does.
        Raises OSError if opening fails; the instance can be restarted after stop.
        """
        with self._lifecycle:
            if self._thread is not None and self._thread.is_alive():
                if self._stop.is_set():
                    raise RuntimeError("Previous camera worker has not stopped")
                return self
            factory = cv2.VideoCapture if self._capture_factory is None else self._capture_factory
            cap = factory(self.source, self.backend)
            try:
                if not cap.isOpened():
                    raise OSError(f"Cannot open camera {self.source!r}")
                for prop, value in ((cv2.CAP_PROP_FRAME_WIDTH, self.width),
                                    (cv2.CAP_PROP_FRAME_HEIGHT, self.height),
                                    (cv2.CAP_PROP_FPS, self.fps),
                                    (cv2.CAP_PROP_BUFFERSIZE, 1)):
                    cap.set(prop, value)
                self.actual_settings = {
                    "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                    "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                    "fps": cap.get(cv2.CAP_PROP_FPS),
                }
                self._stop.clear()
                with self._condition:
                    self._latest = None
                    self._error = None
                    self._running = True
                self._thread = threading.Thread(
                    target=self._capture, args=(cap,), name="camera-capture",
                    daemon=True)
                self._thread.start()
            except BaseException:
                cap.release()
                with self._condition:
                    self._running = False
                    self._condition.notify_all()
                raise
        return self

    def _capture(self, cap):
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()  # Only the worker touches camera I/O.
                timestamp = time.perf_counter_ns()
                if self._stop.is_set():
                    break
                if not ok or frame is None or frame.size == 0:
                    raise OSError("Camera read failed or stream ended")
                # Own the memory even if a backend reuses its input buffer.
                frame = frame.copy()
                frame.flags.writeable = False
                with self._condition:
                    if self._stop.is_set():
                        break
                    self._sequence += 1
                    self._latest = FrameSnapshot(self._sequence, timestamp, frame)
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._latest = None
        finally:
            try:
                cap.release()
            finally:
                with self._condition:
                    self._running = False
                    self._latest = None
                    self._condition.notify_all()

    @staticmethod
    def _snapshot_copy(snapshot, copy):
        if snapshot is None or not copy:
            return snapshot
        return FrameSnapshot(snapshot.sequence, snapshot.timestamp_ns,
                             snapshot.frame.copy())

    def snapshot(self, *, copy=True) -> Optional[FrameSnapshot]:
        """Return latest frame and metadata; copy happens outside the lock.

        ``copy=False`` borrows an immutable frame. Keep it read-only; use the
        default writable copy for overlays or in-place processing.
        """
        with self._condition:
            latest = self._latest
        return self._snapshot_copy(latest, copy)

    def read(self, *, copy=True):
        """Return OpenCV-style (available, frame) immediately, possibly repeated."""
        latest = self.snapshot(copy=copy)
        return (False, None) if latest is None else (True, latest.frame)

    def wait_for_frame(self, after_sequence=0, timeout=1.0, *, copy=True):
        """Wait for a newer frame; return None on timeout, failure, or stop."""
        with self._condition:
            ready = self._condition.wait_for(
                lambda: (self._latest is not None
                         and self._latest.sequence > after_sequence)
                or not self._running or self._stop.is_set(), timeout)
            latest = self._latest
            if not ready or latest is None or latest.sequence <= after_sequence:
                return None
        return self._snapshot_copy(latest, copy)

    def stop(self, timeout=2.0):
        """Request shutdown and join; release is owned exclusively by the worker.

        Raises TimeoutError if a native driver read remains blocked. Python cannot
        safely cancel that read; do not release the device from another thread.
        Once the read returns the worker releases it; a later stop can join it.
        """
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("Stop timeout must be finite and nonnegative")
        with self._lifecycle:
            self._stop.set()
            with self._condition:
                self._latest = None
                self._condition.notify_all()
            if self._thread is not None:
                self._thread.join(timeout)
                if self._thread.is_alive():
                    raise TimeoutError("Camera driver did not return before shutdown timeout")
                self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
