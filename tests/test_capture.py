"""Hardware-independent lifecycle and concurrency tests (stdlib unittest)."""
import threading
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
        self.assertGreater(self.capture.snapshot().sequence, first.sequence + 1)

    def test_concurrent_lifecycle_and_restart(self):
        workers = [threading.Thread(target=self.capture.start) for _ in range(8)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
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


if __name__ == '__main__':
    unittest.main()
