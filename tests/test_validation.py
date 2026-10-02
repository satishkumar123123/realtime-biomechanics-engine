"""Static-hold data quality, paired errors, rejection accounting and CLI tests."""
import csv
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from core.validation import (FIELDS, HoldCollector, analyze_csv, analyze_rows,
                             append_hold, markdown_report)
from core.capture import FrameSnapshot
from main import PoseProcessor, run_pipeline
from tests.test_pipeline import MockCapture, MockPose, pose_result
from validation import main


def paired_row(hold='h1', joint='L_Elbow_Flex', reference=90, software=92, **updates):
    row = dict.fromkeys(FIELDS, '')
    row.update(hold_id=hold, participant='P01', joint=joint, view='front',
               reference_method='manual goniometer', reference_deg=reference,
               software_deg=software, valid_frames=60, total_frames=60,
               valid_fraction=1, iqr_deg=.2, drift_deg=.1, coordinate_frame='body',
               min_cutoff=1, beta=5, d_cutoff=1, width=640, height=480)
    row.update(updates)
    return row


def hold_record(t, angle=45):
    return {'capture_time': t, 'metrics': {'L_Elbow_Flex': angle}, 'reasons': {}}


class ValidationTests(unittest.TestCase):
    def test_settling_is_excluded_and_window_ends_without_wall_clock_waits(self):
        collector = HoldCollector('L_Elbow_Flex', seconds=2, settling=1)
        for i in range(181):
            self.assertEqual(collector(hold_record(i/60, 100 if i < 60 else 45)), i < 180)
        summary = collector.summary()
        self.assertEqual(summary['total_frames'], 120)
        self.assertEqual(summary['software_deg'], 45)
        self.assertEqual(summary['rejection_reason'], '')

    def test_occluded_samples_stay_in_coverage_denominator(self):
        collector = HoldCollector('L_Elbow_Flex', seconds=1, settling=0)
        for i in range(61):
            collector(hold_record(i/60, None if i % 2 else 45))
        summary = collector.summary()
        self.assertEqual(summary['valid_fraction'], .5)
        self.assertIn('low valid-frame coverage', summary['rejection_reason'])

    def test_motion_spread_and_slow_drift_are_rejected(self):
        collector = HoldCollector('L_Elbow_Flex', seconds=1, settling=0)
        for i in range(61):
            collector(hold_record(i/60, 45 if i < 30 else 55))
        summary = collector.summary()
        self.assertIn('unstable angle spread', summary['rejection_reason'])
        self.assertIn('half-window', summary['rejection_reason'])

    def test_sparse_frames_across_a_camera_stall_are_not_a_stable_hold(self):
        collector = HoldCollector('L_Elbow_Flex', seconds=2, settling=0)
        for timestamp in (0, .01, .02, 1.90, 1.91, 2):
            collector(hold_record(timestamp, 45))
        summary = collector.summary()
        self.assertEqual(summary['valid_fraction'], 1)
        self.assertEqual(summary['valid_frames'], 5)
        self.assertIn('insufficient temporal coverage', summary['rejection_reason'])

    def test_contiguous_occlusion_and_unobserved_window_edges_are_rejected(self):
        for first_missing in (0, 30, 102):
            with self.subTest(first_missing=first_missing):
                collector = HoldCollector('L_Elbow_Flex', seconds=2, settling=0)
                for i in range(121):
                    angle = None if first_missing <= i < first_missing+18 else 45
                    collector(hold_record(i/60, angle))
                summary = collector.summary()
                self.assertGreaterEqual(summary['valid_fraction'], .8)
                self.assertIn('insufficient temporal coverage', summary['rejection_reason'])

    def test_regular_30_fps_hold_meets_temporal_coverage(self):
        collector = HoldCollector('L_Elbow_Flex', seconds=2, settling=0)
        for i in range(61):
            collector(hold_record(i/30, 45))
        self.assertEqual(collector.summary()['rejection_reason'], '')
        for max_gap in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                HoldCollector('L_Elbow_Flex', max_gap=max_gap)

    def test_early_exit_empty_nonfinite_and_sample_cap_cannot_pass(self):
        collector = HoldCollector('L_Elbow_Flex', seconds=1, settling=0, max_samples=5)
        self.assertIn('incomplete hold', collector.summary()['rejection_reason'])
        for i in range(5):
            result = collector(hold_record(i/60, float('nan')))
        self.assertFalse(result)
        self.assertEqual(collector.summary()['valid_frames'], 0)
        self.assertIn('incomplete hold', collector.summary()['rejection_reason'])

    def test_callback_timestamp_and_parameter_validation(self):
        collector = HoldCollector('L_Elbow_Flex')
        collector(hold_record(1))
        for time in (1, .9, float('inf')):
            with self.assertRaises(ValueError):
                collector(hold_record(time))
        for arguments in ({'seconds': 0}, {'settling': -1}, {'seconds': 70},
                          {'max_samples': 3}, {'min_coverage': 0}):
            with self.assertRaises(ValueError):
                HoldCollector('L_Elbow_Flex', **arguments)

    def test_error_statistics_use_holds_and_preserve_zero(self):
        report = analyze_rows([paired_row('a', reference=0, software=0),
                               paired_row('b', reference=90, software=94),
                               paired_row('c', reference=90, software=88)])
        row = report['groups'][0]
        self.assertEqual(row['holds'], 3)
        self.assertAlmostEqual(row['mae_deg'], 2)
        self.assertAlmostEqual(row['bias_deg'], 2/3)
        self.assertAlmostEqual(row['rmse_deg'], (20/3)**.5)
        self.assertEqual(row['reference_min_deg'], 0)
        self.assertFalse(report['required_scope_present'])

    def test_reports_preserve_reference_method_and_configuration(self):
        report = analyze_rows([paired_row(reference_method='assessor A: baseline goniometer')])
        self.assertEqual(report['groups'][0]['reference_method'], 'assessor A: baseline goniometer')
        markdown = markdown_report(report)
        self.assertIn('assessor A: baseline goniometer', markdown)
        self.assertIn('640x480', markdown)
        self.assertIn('1.0 / 5.0 / 1.0', markdown)

    def test_different_reference_methods_cannot_silently_share_one_error_statistic(self):
        with self.assertRaisesRegex(ValueError, 'Mixed reference methods'):
            analyze_rows([paired_row('a', reference_method='manual goniometer'),
                          paired_row('b', reference_method='another reference system')])

    def test_required_scope_and_per_view_groups_include_challenging_flexion(self):
        rows = [paired_row('a'), paired_row('b', joint='R_Knee_Flex'),
                paired_row('c', joint='L_Shoulder_Flex', view='left-side'),
                paired_row('d', view='oblique')]
        report = analyze_rows(rows)
        self.assertTrue(report['required_scope_present'])
        self.assertEqual(len(report['groups']), 4)
        self.assertIn('not a clinical pass', markdown_report(report))
        rows[2]['joint'] = 'L_Shoulder_Abd'
        self.assertFalse(analyze_rows(rows)['required_scope_present'])

    def test_rejected_and_low_coverage_holds_do_not_improve_mae(self):
        report = analyze_rows([paired_row('a'),
                               paired_row('b', rejection_reason='occluded'),
                               paired_row('c', valid_frames=30, valid_fraction=.5),
                               paired_row('d', software_deg=''),
                               paired_row('e', iqr_deg=10)])
        self.assertEqual(report['accepted_holds'], 1)
        self.assertEqual(report['rejected_holds'], 4)
        self.assertAlmostEqual(report['groups'][0]['mae_deg'], 2)

    def test_bad_reference_duplicate_ids_bad_coverage_and_mixed_configs_fail(self):
        for row in (paired_row(reference=float('nan')), paired_row(valid_fraction=.3),
                    paired_row(participant=''), paired_row(coordinate_frame='unknown')):
            with self.assertRaises(ValueError):
                analyze_rows([row])
        with self.assertRaises(ValueError):
            analyze_rows([paired_row(), paired_row()])
        with self.assertRaises(ValueError):
            analyze_rows([paired_row('a'), paired_row('b', beta=.007)])

    def test_csv_roundtrip_and_incompatible_schema_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'data.csv'
            append_hold(path, paired_row())
            self.assertEqual(analyze_csv(path)['accepted_holds'], 1)
            bad = Path(directory)/'bad.csv'
            bad.write_text('wrong,header\n')
            with self.assertRaises(ValueError):
                append_hold(bad, paired_row())
            self.assertEqual(bad.read_text(), 'wrong,header\n')

    def test_empty_template_reports_pending_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path, output = Path(directory)/'template.csv', Path(directory)/'report.json'
            with path.open('w', newline='') as stream:
                csv.writer(stream).writerow(FIELDS)
            markdown = Path(directory)/'report.md'
            with patch('builtins.print'):
                status = main(['report', '--csv', str(path), '--json', str(output),
                               '--markdown', str(markdown)])
            self.assertEqual(status, 2)
            self.assertEqual(json.loads(output.read_text())['accepted_holds'], 0)
            self.assertIn('Accuracy remains unmeasured', markdown.read_text())

    def test_collect_cli_uses_real_reference_and_shared_pipeline(self):
        def run(capture, **kwargs):
            for i in range(181):
                if not kwargs['on_frame'](hold_record(i/60, 45)):
                    break
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'pairs.csv'
            with patch('validation.run_pipeline', side_effect=run), patch('builtins.print'):
                status = main(['collect', '--participant', 'P01', '--joint', 'L_Elbow_Flex',
                               '--view', 'front', '--reference', '44', '--csv', str(path), '--headless'])
            self.assertEqual(status, 0)
            self.assertEqual(analyze_csv(path)['groups'][0]['mae_deg'], 1)

    def test_headless_collection_runs_full_loop_without_highgui(self):
        class ClockedCapture(MockCapture):
            actual_settings = {'width': 640, 'height': 480, 'fps': 60}
            def wait_for_frame(self, *args, **kwargs):
                sample = super().wait_for_frame(*args, **kwargs)
                # Deterministic acquisition clock for hold-window logic only;
                # these fixtures never become performance/clinical evidence.
                return FrameSnapshot(sample.sequence, sample.sequence * 20_000_000, sample.frame)
        capture, model = ClockedCapture(), MockPose(pose_result())
        def run(capture, **kwargs):
            return run_pipeline(capture, pose_factory=lambda: model, **kwargs)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as guards:
            for name in ('namedWindow', 'imshow', 'waitKey', 'destroyWindow', 'destroyAllWindows'):
                guards.enter_context(patch(f'cv2.{name}', side_effect=AssertionError('Headless GUI call')))
            guards.enter_context(patch('validation.VideoCaptureAsync', return_value=capture))
            guards.enter_context(patch('validation.run_pipeline', side_effect=run))
            guards.enter_context(patch('builtins.print'))
            path = Path(directory)/'synthetic_test_only.csv'
            status = main(['collect', '--participant', 'TEST_ONLY', '--joint', 'L_Elbow_Flex',
                           '--view', 'front', '--reference', '0', '--csv', str(path),
                           '--seconds', '.2', '--settling', '0', '--headless'])
            self.assertEqual(status, 0)
            self.assertGreater(model.calls, 5)
            self.assertTrue(capture.stopped and model.closed)
            self.assertEqual(analyze_csv(path)['accepted_holds'], 1)

    def test_float_camera_dimensions_roundtrip_in_csv(self):
        collector = HoldCollector('L_Elbow_Flex', seconds=1, settling=0)
        for i in range(61):
            collector(hold_record(i/60, 0))
        row = collector.row(participant='P01', view='front', reference_deg=0,
                            reference_method='manual goniometer', processor=PoseProcessor(),
                            width=640.0, height=480.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'holds.csv'
            append_hold(path, row)
            report = analyze_csv(path)
            self.assertEqual(report['accepted_holds'], 1)
            self.assertEqual(report['groups'][0]['mae_deg'], 0)

    def test_incompatible_csv_and_overlapping_outputs_fail_without_camera_or_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'holds.csv'
            path.write_text('wrong,header\n')
            with patch('validation.run_pipeline') as pipeline, patch('builtins.print'):
                status = main(['collect', '--participant', 'P01', '--joint', 'L_Elbow_Flex',
                               '--view', 'front', '--reference', '0', '--csv', str(path)])
                self.assertEqual(status, 1)
                pipeline.assert_not_called()
                status = main(['report', '--csv', str(path), '--json', str(path)])
                self.assertEqual(status, 1)
            self.assertEqual(path.read_text(), 'wrong,header\n')


if __name__ == '__main__':
    unittest.main()
