"""Run: python -m tests.benchmark_capture [--synthetic] [--frames 300]."""
import argparse
from contextlib import nullcontext
import json
import time
from unittest.mock import patch

import numpy as np

from core.capture import VideoCaptureAsync


class SyntheticCamera:
    """Paced source for repeatable mailbox checks, not a hardware FPS result."""
    def __init__(self, width, height, fps):
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.period = 1 / fps
        self.deadline = time.perf_counter()
        self.settings = {}

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.settings[prop] = value
        return True

    def get(self, prop):
        return self.settings.get(prop, 0)

    def read(self):
        self.deadline += self.period
        time.sleep(max(0, self.deadline - time.perf_counter()))
        return True, self.frame

    def release(self):
        pass


def benchmark(args):
    if args.frames < 2:
        raise ValueError('At least two frames are required')
    source = (patch('core.capture.cv2.VideoCapture', return_value=SyntheticCamera(
        args.width, args.height, args.fps)) if args.synthetic else nullcontext())
    with source, VideoCaptureAsync(args.camera, args.width, args.height, args.fps) as cap:
        first = cap.wait_for_frame(timeout=5, copy=False)
        if first is None:
            raise RuntimeError(f'No initial frame: {cap.error}')
        last = first
        observed = 1
        overhead = []
        age = []
        deadline = time.perf_counter() + max(30, args.frames / args.fps * 4)
        while observed < args.frames:
            if time.perf_counter() > deadline:
                raise TimeoutError('Benchmark exceeded total deadline')
            current = cap.wait_for_frame(last.sequence, timeout=5, copy=False)
            if current is None:
                raise RuntimeError(f'Capture stalled: {cap.error}')
            age.append((time.perf_counter_ns() - current.timestamp_ns) / 1e6)
            # Measure nonblocking mailbox access, excluding sensor waiting and copy.
            start = time.perf_counter_ns()
            cap.snapshot(copy=False)
            overhead.append((time.perf_counter_ns() - start) / 1e6)
            last = current
            observed += 1
        elapsed = (last.timestamp_ns - first.timestamp_ns) / 1e9
        result = {
            'source': 'synthetic' if args.synthetic else 'webcam',
            'requested_fps': args.fps, 'negotiated_settings': cap.actual_settings,
            'observed_frames': observed,
            'ingested_frames_in_interval': last.sequence - first.sequence,
            'ingestion_fps': (last.sequence - first.sequence) / elapsed,
            'consumer_fps': (observed - 1) / elapsed,
            'superseded_frames': last.sequence - first.sequence - (observed - 1),
            'mailbox_ms_mean': float(np.mean(overhead)),
            'mailbox_ms_p99': float(np.percentile(overhead, 99)),
            'mailbox_ms_max': float(np.max(overhead)),
            'host_frame_age_ms_p99': float(np.percentile(age, 99)),
        }
        print(json.dumps(result, indent=2))
        if result['mailbox_ms_max'] >= 1:
            raise AssertionError('Measured mailbox overhead exceeded 1 ms')
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=float, default=60)
    benchmark(parser.parse_args())
