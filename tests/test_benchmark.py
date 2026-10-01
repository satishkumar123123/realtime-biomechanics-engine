"""Benchmark statistics, source pacing, CLI limits and pipeline integration."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from benchmark import (MockPose, PacedReplay, Recorder, build_parser, latency_stats,
                       main, run_benchmark)


class BenchmarkTests(unittest.TestCase):
    def test_summary_and_sequence_gap_accounting(self):
        recorder = Recorder(frames=4, warmup=1)
        sequences = [1, 10, 12, 13, 16]
        for i, sequence in enumerate(sequences):
            keep_going = recorder({'sequence': sequence, 'capture_time': i * .02,
                                   'completion_time': i * .02 + .005,
                                   'inference_ms': i + 1, 'pipeline_ms': 5,
                                   'pose_detected': True, 'reliable_metrics': 12})
        self.assertFalse(keep_going)
        report = recorder.summary()
        self.assertEqual(report['measured_frames'], 4)
        self.assertAlmostEqual(report['achieved_e2e_fps'], 50)
        self.assertEqual(report['mailbox']['superseded_frames'], 3)
        self.assertEqual(report['mailbox']['overrun_events'], 2)
        self.assertEqual(report['mailbox']['published_intervals'], 6)
        self.assertEqual(report['mailbox']['superseded_percent'], 50)
        self.assertIsNone(report['mailbox']['hardware_dropped_frames'])
        self.assertAlmostEqual(report['inference']['mean_ms'], 3.5)
        self.assertAlmostEqual(report['inference']['p95_ms'], np.percentile([2, 3, 4, 5], 95))

    def test_latency_statistics_and_insufficient_data(self):
        self.assertEqual(latency_stats([1, 2, 3])['min_ms'], 1)
        self.assertEqual(latency_stats([1, 2, 3])['max_ms'], 3)
        with self.assertRaises(ValueError):
            latency_stats([])
        with self.assertRaises(ValueError):
            Recorder(warmup=0).summary()

    def test_mock_full_numeric_pipeline_and_json_serialization(self):
        args = build_parser().parse_args(['--mock-pose', '--frames', '5', '--warmup', '2'])
        report = run_benchmark(args)
        self.assertEqual(report['measured_frames'], 5)
        self.assertEqual(report['frames_with_pose'], 5)
        self.assertEqual(report['mean_reliable_metrics'], 12)
        self.assertEqual(report['configuration']['inference'], 'analytic mock')
        self.assertGreater(report['achieved_e2e_fps'], 0)
        json.dumps(report, allow_nan=False)

    def test_duration_limit_stops_automatically(self):
        args = build_parser().parse_args(['--mock-pose', '--seconds', '.05', '--warmup', '0'])
        report = run_benchmark(args)
        self.assertGreaterEqual(report['measured_interval_seconds'], .05)
        self.assertGreaterEqual(report['measured_frames'], 2)
        self.assertLess(report['measured_interval_seconds'], 1)

    def test_recorded_source_loops_and_resizes(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'blank.avi'
            writer = cv2.VideoWriter(str(filename), cv2.VideoWriter_fourcc(*'MJPG'), 30, (320, 240))
            if not writer.isOpened():
                self.skipTest('MJPEG writer unavailable')
            try:
                for _ in range(3):
                    writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
            finally:
                writer.release()
            args = build_parser().parse_args(['--source', 'recorded', '--video', str(filename),
                                              '--frames', '6', '--warmup', '0'])
            with patch('benchmark.create_pose', side_effect=MockPose):
                report = run_benchmark(args)
            self.assertEqual(report['measured_frames'], 6)
            self.assertGreaterEqual(report['configuration']['recorded_replay_loops'], 1)
            self.assertEqual(report['configuration']['negotiated_capture']['width'], 640)

    def test_cli_validation(self):
        parser = build_parser()
        for arguments in (['--frames', '1'], ['--seconds', '-1'], ['--warmup', '-1'],
                          ['--fps', '0'], ['--source', 'recorded'],
                          ['--mock-pose', '--source', 'webcam']):
            with self.subTest(args=arguments), self.assertRaises(ValueError):
                run_benchmark(parser.parse_args(arguments))

    def test_cli_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'report.json'
            with patch('benchmark.print_summary'):
                status = main(['--mock-pose', '--frames', '3', '--warmup', '0',
                               '--output', str(filename)])
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(filename.read_text())['measured_frames'], 3)


if __name__ == '__main__':
    unittest.main()
