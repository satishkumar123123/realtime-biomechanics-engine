"""Hardware-independent lifecycle and concurrency tests (stdlib unittest)."""
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import unittest
from unittest.mock import patch

import numpy as np

from core.capture import VideoCaptureAsync


class FakeCamera:
    def __init__(self, opened=True):
        self.opened = opened
        self.released = 0
        self.settings = {}
        self.fail = False
        self.gate = threading.Event()
        self.gate.set()
        self.buffer = np.zeros((480, 640, 3), dtype=np.uint8)

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.settings[prop] = value
        return True

    def get(self, prop):
        return self.settings.get(prop, 0)

    def read(self):
        self.gate.wait()
        time.sleep(0.002)
        self.buffer.fill((int(self.buffer[0, 0, 0]) + 1) % 256)
        return (False, None) if self.fail else (True, self.buffer)

    def release(self):
        self.released += 1


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.camera = FakeCamera()
        self.mock = patch('core.capture.cv2.VideoCapture', return_value=self.camera)
        self.mock.start()
        self.capture = VideoCaptureAsync()

    def tearDown(self):
        self.camera.gate.set()
        self.capture.stop()
        self.mock.stop()

    def test_latest_frame_and_memory_ownership(self):
        self.assertEqual(self.capture.read(), (False, None))
        self.capture.start()
        first = self.capture.wait_for_frame()
        self.assertIsNotNone(first)
        borrowed = self.capture.snapshot(copy=False)
        original = borrowed.frame.copy()
        self.assertFalse(borrowed.frame.flags.writeable)
        newer = self.capture.wait_for_frame(first.sequence)
        self.assertGreater(newer.sequence, first.sequence)
        np.testing.assert_array_equal(borrowed.frame, original)
        first.frame.fill(0)
        self.assertTrue(self.capture.read()[0])

    def test_slow_consumer_does_not_stall_ingestion(self):
        self.capture.start()
        first = self.capture.wait_for_frame()
        time.sleep(0.05)
        later = self.capture.wait_for_frame(first.sequence + 1, timeout=2)
        self.assertIsNotNone(later)
        self.assertGreater(later.sequence, first.sequence + 1)

    def test_concurrent_lifecycle_and_restart(self):
        with ThreadPoolExecutor(max_workers=8) as workers:
            starts = [workers.submit(self.capture.start) for _ in range(8)]
            for start in starts:
                self.assertIs(start.result(timeout=2), self.capture)
        self.assertIsNotNone(self.capture.wait_for_frame())
        self.capture.stop()
        self.capture.stop()
        self.assertEqual(self.camera.released, 1)
        self.capture.start()
        self.assertIsNotNone(self.capture.wait_for_frame())

    def test_open_failure_releases_device(self):
        self.camera.opened = False
        with self.assertRaises(OSError):
            self.capture.start()
        self.assertEqual(self.camera.released, 1)

    def test_failure_clears_stale_frame(self):
        self.capture.start()
        first = self.capture.wait_for_frame()
        self.camera.fail = True
        deadline = time.perf_counter() + 1
        while self.capture.running and time.perf_counter() < deadline:
            time.sleep(0.001)
        self.assertFalse(self.capture.running)
        self.assertIsInstance(self.capture.error, OSError)
        self.assertEqual(self.capture.read(), (False, None))
        self.assertIsNone(self.capture.wait_for_frame(first.sequence))

    def test_blocked_driver_timeout_and_eventual_cleanup(self):
        self.camera.gate.clear()
        self.capture.start()
        with self.assertRaises(TimeoutError):
            self.capture.stop(timeout=0.01)
        with self.assertRaises(RuntimeError):
            self.capture.start()
        self.assertEqual(self.camera.released, 0)
        self.camera.gate.set()
        self.capture.stop()
        self.assertEqual(self.camera.released, 1)

    def test_wait_timeout_and_validation(self):
        with self.assertRaises(ValueError):
            VideoCaptureAsync(fps=0)
        self.camera.gate.clear()
        self.capture.start()
        self.assertIsNone(self.capture.wait_for_frame(timeout=0.01))

    def test_thread_start_failure_can_be_stopped_and_restarted(self):
        with patch('core.capture.threading.Thread.start', side_effect=RuntimeError('Cannot start')):
            with self.assertRaises(RuntimeError):
                self.capture.start()
        self.capture.stop()  # Formerly tried to join an unstarted Thread.
        self.assertEqual(self.camera.released, 1)
        self.capture.start()
        self.assertIsNotNone(self.capture.wait_for_frame())

    def test_backend_release_failure_is_reported_without_daemon_crash(self):
        camera = FakeCamera()
        def fail_release():
            raise OSError('Backend release failed')
        camera.release = fail_release
        capture = VideoCaptureAsync(capture_factory=lambda *args: camera)
        capture.start()
        self.assertIsNotNone(capture.wait_for_frame())
        with self.assertRaises(OSError):
            capture.stop()
        self.assertFalse(capture.running)
        self.assertIsInstance(capture.error, OSError)

    def test_wait_parameter_validation(self):
        for timeout in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.capture.wait_for_frame(timeout=timeout)
        with self.assertRaises(ValueError):
            self.capture.wait_for_frame(after_sequence=-1)

    def test_repeated_lifecycle_releases_every_device_and_thread(self):
        cameras = []
        def factory(*args):
            camera = FakeCamera()
            cameras.append(camera)
            return camera
        capture = VideoCaptureAsync(capture_factory=factory)
        try:
            for _ in range(20):
                capture.start()
                self.assertIsNotNone(capture.wait_for_frame(timeout=2))
                capture.stop()
                self.assertFalse(capture.running)
                self.assertIsNone(capture._thread)
        finally:
            capture.stop()
        self.assertTrue(all(camera.released == 1 for camera in cameras))


if __name__ == '__main__':
    unittest.main()
