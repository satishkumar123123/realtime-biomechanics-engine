"""Reproducibility and evidence labels for generated replay/accuracy artifacts."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from benchmark import build_parser, latency_stats, run_benchmark
from benchmarks.prepare_replay import replay_frame
from benchmarks.simulated_accuracy import TARGETS, simulate, target_pose
from core.biomechanics import BiomechanicsEngine
from core.validation import analyze_csv, analyze_rows
from tests.test_validation import paired_row


class EmpiricalDataTests(unittest.TestCase):
    def test_independent_targets_preserve_all_requested_angles_across_views(self):
        for suffix, targets in TARGETS.items():
            for side in ('L', 'R'):
                for target in targets:
                    for yaw in (0, 45, 90):
                        with self.subTest(joint=suffix, side=side, target=target, yaw=yaw):
                            joint = f'{side}_{suffix}'
                            points = target_pose(joint, target, yaw)
                            actual = BiomechanicsEngine.compute_joint_metrics(
                                points, np.ones(33), coordinate_frame='body')[joint]
                            self.assertAlmostEqual(actual, target, delta=1e-5)

    def test_replay_is_cyclic_deterministic_and_changes_pixels(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[30:70, 50:150] = 255
        first = replay_frame(image, 0, 720)
        self.assertEqual(first.shape, (480, 640, 3))
        np.testing.assert_array_equal(first, replay_frame(image, 720, 720))
        self.assertFalse(np.array_equal(first, replay_frame(image, 90, 720)))

    def test_samples_recompute_report_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            samples = Path(directory)/'frames.csv'
            args = build_parser().parse_args(['--mock-pose', '--frames', '5', '--warmup', '1',
                                              '--samples-output', str(samples)])
            report = run_benchmark(args)
            self.assertNotIn(b'\r\n', samples.read_bytes())
            with samples.open(newline='') as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 5)
            self.assertEqual(report['raw_samples']['rows'], 5)
            for name in ('inference', 'pipeline'):
                self.assertEqual(latency_stats([float(row[f'{name}_ms']) for row in rows]), report[name])
            times = [float(row['completion_time']) for row in rows]
            self.assertAlmostEqual(report['achieved_e2e_fps'], 4/(times[-1]-times[0]))

    def test_bad_manifest_and_colliding_paths_fail_before_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            video, manifest = Path(directory)/'input.avi', Path(directory)/'manifest.json'
            video.write_bytes(b'test fixture')
            manifest.write_text(json.dumps({'video': {'sha256': 'mismatched'}}))
            parser = build_parser()
            with patch('benchmark.VideoCaptureAsync') as capture:
                for extra in (['--input-manifest', str(manifest)], ['--output', str(video)]):
                    with self.assertRaises(ValueError):
                        run_benchmark(parser.parse_args(['--source', 'recorded', '--video', str(video), *extra]))
                capture.assert_not_called()
            self.assertEqual(video.read_bytes(), b'test fixture')

    def test_seeded_simulation_is_repeatable_and_cannot_be_pooled_with_physical_data(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                'benchmarks.simulated_accuracy.TARGETS', {'Elbow_Flex': (90,)}):
            directory = Path(directory)
            first = simulate(directory/'first', seed=42, repetitions=1)
            second = simulate(directory/'second', seed=42, repetitions=1)
            self.assertEqual(first['cases'], second['cases'])
            self.assertEqual(first['validation_type'], 'synthetic')
            self.assertEqual(first['actual_human_participants'], 0)
            self.assertFalse(first['model_inference_executed'])
            self.assertFalse(first['required_scope_present'])
            self.assertEqual(first['frames_processed'], 362)
            self.assertEqual(first['accepted_holds'], 2)
            self.assertTrue(all(case['valid_frames'] < case['total_frames'] for case in first['cases']))
            raw_report = analyze_csv(directory/'first'/'holds.csv')
            self.assertNotIn(b'\r\n', (directory/'first'/'holds.csv').read_bytes())
            self.assertEqual(raw_report['validation_type'], 'synthetic')
            self.assertIn('not clinical accuracy', raw_report['scope'])
            with (directory/'first'/'holds.csv').open(newline='') as source:
                rows = list(csv.DictReader(source))
            with self.assertRaisesRegex(ValueError, 'separate reports'):
                analyze_rows([*rows, paired_row('physical', joint='R_Knee_Flex')])


if __name__ == '__main__':
    unittest.main()
