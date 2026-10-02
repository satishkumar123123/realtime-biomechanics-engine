"""Benchmark statistics, source pacing, CLI limits and pipeline integration."""
import json
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from benchmark import (MockPose, PacedReplay, Recorder, build_parser, latency_stats,
                       main, run_benchmark, submission_checks)


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
        self.assertEqual(report['rolling_window']['frames'], 4)
        self.assertAlmostEqual(report['rolling_window']['fps'], 50)

    def test_rolling_statistics_exclude_old_frames_and_warmup(self):
        recorder = Recorder(frames=130, warmup=2)
        for i in range(132):
            recorder({'sequence': i+1, 'capture_time': i/60,
                      'completion_time': i/60+.01, 'inference_ms': i,
                      'pipeline_ms': 10, 'pose_detected': False, 'reliable_metrics': 0})
        rolling = recorder.summary()['rolling_window']
        self.assertEqual(rolling['frames'], 120)
        self.assertAlmostEqual(rolling['inference']['mean_ms'], np.mean(np.arange(12, 132)))
        self.assertAlmostEqual(rolling['inference']['p95_ms'], np.percentile(np.arange(12, 132), 95))

    def test_latency_statistics_and_insufficient_data(self):
        self.assertEqual(latency_stats([1, 2, 3])['min_ms'], 1)
        self.assertEqual(latency_stats([1, 2, 3])['max_ms'], 3)
        with self.assertRaises(ValueError):
            latency_stats([])
        with self.assertRaises(ValueError):
            Recorder(warmup=0).summary()

    def test_mock_full_numeric_pipeline_and_json_serialization(self):
        args = build_parser().parse_args(['--mock-pose', '--frames', '5', '--warmup', '2'])
        with ExitStack() as guards:
            for name in ('namedWindow', 'imshow', 'waitKey', 'destroyWindow', 'destroyAllWindows'):
                guards.enter_context(patch(f'cv2.{name}', side_effect=AssertionError('Headless GUI call')))
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

    def test_detected_pose_subset_and_metric_coverage(self):
        recorder = Recorder(frames=4, warmup=0)
        for i in range(4):
            recorder({'sequence': i+1, 'capture_time': i*.02,
                      'completion_time': i*.02+.01, 'inference_ms': 100 if i == 0 else 10,
                      'pipeline_ms': 15, 'pose_detected': i > 0, 'reliable_metrics': int(i > 0),
                      'metrics': {'L_Elbow_Flex': 0 if i > 0 else None}})
        report = recorder.summary()
        self.assertEqual(report['pose_coverage'], .75)
        self.assertEqual(report['numeric_coverage'], .75)
        self.assertEqual(report['metric_coverage']['L_Elbow_Flex'], .75)
        self.assertEqual(report['detected_pose_latency']['inference']['mean_ms'], 10)

    def test_submission_checks_require_human_desktop_evidence(self):
        report = {'configuration': {'inference': 'MediaPipe BlazePose Full', 'source': 'webcam',
                                    'display': 'desktop'}, 'measured_interval_seconds': 30,
                  'pose_coverage': .95, 'numeric_coverage': .9, 'achieved_e2e_fps': 45}
        self.assertTrue(submission_checks(report)['passed'])
        for key, value in (('source', 'synthetic'), ('display', 'headless HUD rendering'),
                           ('inference', 'analytic mock')):
            altered = {**report, 'configuration': {**report['configuration'], key: value}}
            self.assertFalse(submission_checks(altered)['passed'])
        self.assertFalse(submission_checks({**report, 'pose_coverage': 0})['passed'])
        self.assertFalse(submission_checks({**report, 'achieved_e2e_fps': 29})['passed'])

    def test_required_human_failure_still_saves_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'mock.json'
            with patch('benchmark.print_summary'), patch('builtins.print'):
                status = main(['--mock-pose', '--frames', '3', '--warmup', '0',
                               '--require-human', '--output', str(path)])
            self.assertEqual(status, 2)
            report = json.loads(path.read_text())
            self.assertFalse(report['submission_performance']['passed'])
            self.assertEqual(report['configuration']['coordinate_frame'], 'body')

    def test_replay_preserves_aspect_ratio(self):
        source = PacedReplay(640, 480, 100000)
        video = Mock()
        video.read.return_value = (True, np.full((160, 320, 3), 255, dtype=np.uint8))
        source.video = video
        try:
            ok, frame = source.read()
            self.assertTrue(ok)
            self.assertEqual(frame.shape, (480, 640, 3))
            self.assertEqual(frame[:80].max(), 0)
            self.assertEqual(frame[80:400].min(), 255)
            self.assertEqual(frame[400:].max(), 0)
        finally:
            source.release()


if __name__ == '__main__':
    unittest.main()
