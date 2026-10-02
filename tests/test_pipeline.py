"""Headless integration tests: real rendering, mock camera/model, no GUI calls."""
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from core.capture import FrameSnapshot, VideoCaptureAsync
from main import (CameraUnavailable, Diagnostics, HUDRenderer, OpenCVDisplay, PoseProcessor,
                  create_pose, run_pipeline)
from tests.test_biomechanics import neutral_pose


def pose_result(visibility=1.0):
    points, _ = neutral_pose()
    world = [SimpleNamespace(x=x, y=y, z=z) for x, y, z in points]
    image = [SimpleNamespace(x=0.5 + x, y=0.3 + y * 0.3, z=z,
                             visibility=visibility) for x, y, z in points]
    return SimpleNamespace(pose_world_landmarks=SimpleNamespace(landmark=world),
                           pose_landmarks=SimpleNamespace(landmark=image))


BLANK = SimpleNamespace(pose_world_landmarks=None, pose_landmarks=None)


class MockCapture:
    def __init__(self, disconnect_at=None, fail_start=False, stale=False):
        self.running = False
        self.error = None
        self.sequence = 0
        self.stopped = False
        self.disconnect_at = disconnect_at
        self.fail_start = fail_start
        self.stale = stale

    def start(self):
        if self.fail_start:
            raise OSError('Cannot open mock camera')
        self.running = True
        return self

    def wait_for_frame(self, after_sequence, timeout=1, copy=True):
        if self.disconnect_at is not None and self.sequence >= self.disconnect_at:
            self.running = False
            self.error = OSError('Mock disconnect')
            return None
        if not self.stale or self.sequence == 0:
            self.sequence += 1
        return FrameSnapshot(self.sequence, time.perf_counter_ns(),
                             np.zeros((480, 640, 3), dtype=np.uint8))

    def stop(self):
        self.stopped = True
        self.running = False


class MockPose:
    def __init__(self, result=BLANK, fail=False):
        self.result = result
        self.fail = fail
        self.closed = False
        self.calls = 0

    def process(self, rgb):
        self.calls += 1
        if self.fail:
            raise RuntimeError('Mock inference failure')
        assert not rgb.flags.writeable
        assert rgb.shape == (480, 640, 3)
        return self.result

    def close(self):
        self.closed = True


class HeadlessDisplay:
    def __init__(self, keys=()):
        self.keys = iter(keys)
        self.frames = 0
        self.closed = False
        self.last = None

    def submit(self, frame):
        self.frames += 1
        self.last = frame

    def poll_key(self):
        return next(self.keys, -1)

    def close(self):
        self.closed = True


class PipelineTests(unittest.TestCase):
    def test_blank_frames_run_without_camera_model_or_gui(self):
        cap, model, display = MockCapture(), MockPose(), HeadlessDisplay()
        with patch('main.cv2.imshow', side_effect=AssertionError('No GUI allowed')):
            summary = run_pipeline(cap, lambda: model, display, max_frames=10)
        self.assertEqual(summary['processed_frames'], 10)
        self.assertEqual(display.frames, 10)
        self.assertTrue(cap.stopped and model.closed and display.closed)
        self.assertGreater(np.count_nonzero(display.last), 0)  # HUD really rendered.
        self.assertTrue(np.isfinite(list(summary['diagnostics'].values())).all())

    def test_pose_and_diagnostics_window_run_130_frames(self):
        cap, model, display = MockCapture(), MockPose(pose_result()), HeadlessDisplay()
        processor = PoseProcessor()
        summary = run_pipeline(cap, lambda: model, display, processor=processor, max_frames=130)
        self.assertEqual(model.calls, 130)
        self.assertEqual(summary['diagnostics']['window_frames'], 120)
        self.assertGreater(summary['diagnostics']['fps'], 0)
        self.assertIsNotNone(processor.filter.cutoff)

    def test_pose_loss_and_occlusion_clear_filter(self):
        processor = PoseProcessor()
        metrics, reasons = processor.process(pose_result(), 0)
        self.assertTrue(all(value is not None for value in metrics.values()))
        metrics, reasons = processor.process(pose_result(0.2), 0.1)
        self.assertTrue(all(value is None for value in metrics.values()))
        self.assertTrue(all(value == 'Occluded / Low Conf' for value in reasons.values()))
        metrics, _ = processor.process(BLANK, 0.2)
        self.assertIsNone(processor.filter.cutoff)
        metrics, _ = processor.process(pose_result(), 0.3)
        self.assertTrue(all(abs(value) < 0.5 for value in metrics.values()))

    def test_threshold_and_nonfinite_keypoint_recovery(self):
        processor = PoseProcessor()
        metrics, _ = processor.process(pose_result(.65), 0)
        self.assertTrue(all(value is None for value in metrics.values()))
        results = pose_result()
        processor.process(results, .1)
        results.pose_world_landmarks.landmark[15].z = np.nan
        metrics, _ = processor.process(results, .2)
        self.assertIsNone(metrics['L_Elbow_Flex'])
        self.assertTrue(np.isnan(processor.filter._filtered[15]).all())
        metrics, _ = processor.process(pose_result(), .3)
        self.assertAlmostEqual(metrics['L_Elbow_Flex'], 0, delta=.01)

    def test_reset_and_exit_controls(self):
        for key in (ord('q'), 27):
            with self.subTest(key=key):
                cap, model = MockCapture(), MockPose(pose_result())
                display = HeadlessDisplay((ord('r'), -1, key))
                summary = run_pipeline(cap, lambda: model, display, max_frames=10)
                self.assertEqual(summary['processed_frames'], 3)
                self.assertEqual(summary['filter_resets'], 1)
                self.assertTrue(cap.stopped and model.closed and display.closed)

    def test_disconnect_cleans_every_resource(self):
        cap, model, display = MockCapture(disconnect_at=2), MockPose(), HeadlessDisplay()
        with self.assertRaises(CameraUnavailable):
            run_pipeline(cap, lambda: model, display, max_frames=10)
        self.assertTrue(cap.stopped and model.closed and display.closed)

    def test_initialization_and_inference_errors_clean_resources(self):
        cap, display = MockCapture(fail_start=True), HeadlessDisplay()
        with self.assertRaises(OSError):
            run_pipeline(cap, display=display)
        self.assertTrue(cap.stopped and display.closed)
        cap, display = MockCapture(), HeadlessDisplay()
        def fail_model():
            raise RuntimeError('Model initialization failed')
        with self.assertRaises(RuntimeError):
            run_pipeline(cap, fail_model, display)
        self.assertTrue(cap.stopped and display.closed)
        cap, model, display = MockCapture(), MockPose(fail=True), HeadlessDisplay()
        with self.assertRaises(RuntimeError):
            run_pipeline(cap, lambda: model, display)
        self.assertTrue(cap.stopped and model.closed and display.closed)

    def test_duplicate_frames_do_not_run_inference_twice(self):
        cap, model = MockCapture(stale=True), MockPose()
        display = HeadlessDisplay((-1, ord('q')))
        summary = run_pipeline(cap, lambda: model, display)
        self.assertEqual(model.calls, 1)
        self.assertEqual(summary['processed_frames'], 1)

    def test_pose_configuration_disables_builtin_smoothing(self):
        from unittest.mock import Mock
        pose_constructor = Mock(return_value=MockPose())
        module = SimpleNamespace(solutions=SimpleNamespace(
            pose=SimpleNamespace(Pose=pose_constructor)))
        with patch.dict('sys.modules', {'mediapipe': module}):
            create_pose()
        settings = pose_constructor.call_args.kwargs
        self.assertEqual(settings['model_complexity'], 1)
        self.assertFalse(settings['smooth_landmarks'])
        self.assertFalse(settings['enable_segmentation'])

    def test_display_failure_and_shutdown_timeout_still_close_resources(self):
        cap, model, display = MockCapture(), MockPose(), HeadlessDisplay()
        def fail_display(frame):
            raise RuntimeError('Display failed')
        display.submit = fail_display
        with self.assertRaises(RuntimeError):
            run_pipeline(cap, lambda: model, display)
        self.assertTrue(cap.stopped and model.closed and display.closed)
        cap, model, display = MockCapture(), MockPose(), HeadlessDisplay()
        def blocked_stop():
            raise TimeoutError('Native read blocked')
        cap.stop = blocked_stop
        with self.assertRaises(TimeoutError):
            run_pipeline(cap, lambda: model, display, max_frames=1)
        self.assertTrue(model.closed and display.closed)

    def test_camera_stall_is_bounded(self):
        cap, display = MockCapture(), HeadlessDisplay()
        cap.wait_for_frame = lambda *args, **kwargs: None
        with self.assertRaises(CameraUnavailable):
            run_pipeline(cap, lambda: MockPose(), display, stall_timeout=0.01)
        self.assertTrue(cap.stopped and display.closed)

    def test_diagnostic_math_and_eviction(self):
        diagnostics = Diagnostics(window=120)
        for i in range(130):
            diagnostics.record_inference(i)
            diagnostics.record_display(i / 60, i / 60 + 0.01)
        stats = diagnostics.summary()
        self.assertEqual(stats['window_frames'], 120)
        self.assertAlmostEqual(stats['inference_current_ms'], 129)
        self.assertAlmostEqual(stats['inference_mean_ms'], np.mean(np.arange(10, 130)))
        self.assertAlmostEqual(stats['inference_p95_ms'], np.percentile(np.arange(10, 130), 95))
        self.assertAlmostEqual(stats['pipeline_mean_ms'], 10)
        self.assertAlmostEqual(stats['pipeline_p95_ms'], 10)
        self.assertAlmostEqual(stats['fps'], 60)

    def test_closed_native_window_is_a_clean_exit_key(self):
        display = OpenCVDisplay()
        display._opened = True
        with patch('main.cv2.waitKey', return_value=-1), patch(
                'main.cv2.getWindowProperty', side_effect=cv2.error('Window destroyed')):
            self.assertEqual(display.poll_key(), 27)
        with patch('main.cv2.destroyWindow') as destroy:
            display.close()
            display.close()
            destroy.assert_called_once()

    def test_linux_without_display_does_not_enter_native_gui(self):
        display = OpenCVDisplay()
        with patch('main.sys.platform', 'linux'), patch.dict('main.os.environ', {}, clear=True), patch(
                'main.cv2.namedWindow') as create:
            with self.assertRaises(RuntimeError):
                display.submit(np.zeros((480, 640, 3), dtype=np.uint8))
            create.assert_not_called()

    def test_native_gui_calls_are_rejected_off_main_thread(self):
        display = OpenCVDisplay()
        with ThreadPoolExecutor(max_workers=1) as workers, patch(
                'main.cv2.namedWindow') as create, patch('main.cv2.imshow') as show, patch(
                'main.cv2.waitKey') as wait, patch('main.cv2.destroyWindow') as destroy:
            with self.assertRaisesRegex(RuntimeError, 'main thread'):
                workers.submit(display.submit, np.zeros((10, 10, 3), dtype=np.uint8)).result(2)
            display._opened = True
            for operation in (display.poll_key, display.close):
                with self.assertRaisesRegex(RuntimeError, 'main thread'):
                    workers.submit(operation).result(2)
            for native in (create, show, wait, destroy):
                native.assert_not_called()
            display.close()  # Owner thread can still clean up after misuse.
            destroy.assert_called_once()

    def test_event_pump_exception_still_cleans_all_resources(self):
        cap, model, display = MockCapture(), MockPose(), OpenCVDisplay()
        display._opened = True
        with patch('main.cv2.imshow'), patch('main.cv2.waitKey', side_effect=cv2.error(
                'Event pump failed')), patch('main.cv2.destroyWindow') as destroy:
            with self.assertRaises(cv2.error):
                run_pipeline(cap, lambda: model, display)
            self.assertTrue(cap.stopped and model.closed)
            self.assertFalse(display._opened)
            destroy.assert_called_once()

    def test_small_frame_hud_has_space_for_all_rows(self):
        renderer = HUDRenderer()
        metrics, reasons = PoseProcessor().process(BLANK, 0)
        output = renderer.render(np.zeros((240, 320, 3), dtype=np.uint8),
                                 BLANK, metrics, reasons, Diagnostics().summary())
        self.assertEqual(output.shape[:2], (460, 640))

    def test_real_async_capture_with_mock_inference(self):
        class SyntheticCamera:
            def __init__(self):
                self.released = False
            def isOpened(self):
                return True
            def set(self, *args):
                return True
            def get(self, *args):
                return 60
            def read(self):
                time.sleep(0.005)
                return True, np.zeros((480, 640, 3), dtype=np.uint8)
            def release(self):
                self.released = True
        camera = SyntheticCamera()
        with patch('core.capture.cv2.VideoCapture', return_value=camera):
            summary = run_pipeline(VideoCaptureAsync(), lambda: MockPose(),
                                   HeadlessDisplay(), max_frames=10)
        self.assertEqual(summary['processed_frames'], 10)
        self.assertTrue(camera.released)

    @unittest.skipUnless(importlib.util.find_spec('mediapipe'), 'MediaPipe not installed')
    def test_real_mediapipe_blank_frame_smoke(self):
        summary = run_pipeline(MockCapture(), create_pose, HeadlessDisplay(), max_frames=3)
        self.assertEqual(summary['processed_frames'], 3)


if __name__ == '__main__':
    unittest.main()
