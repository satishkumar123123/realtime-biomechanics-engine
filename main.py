"""Desktop biomechanics pipeline with injectable headless integration seams."""
import argparse
from collections import deque
from contextlib import ExitStack
import logging
import os
import sys
import time

import cv2
import numpy as np

from core.biomechanics import BiomechanicsEngine
from core.capture import VideoCaptureAsync
from core.filter import OneEuroFilter


LOGGER = logging.getLogger(__name__)
WINDOW = 'Real-Time Biomechanics'
METRIC_ROWS = (
    ('Elbow_Flex', 'Elbow Flex', (11, 13, 15)),
    ('Knee_Flex', 'Knee Flex', (23, 25, 27)),
    ('Shoulder_Flex', 'Shoulder Flex', (23, 11, 13)),
    ('Shoulder_Abd', 'Shoulder Abd', (23, 11, 13)),
    ('Hip_Flex', 'Hip Flex', (11, 23, 25)),
    ('Ankle_Dorsi_Plantar', 'Ankle D/P', (25, 27, 31)),
)
# BlazePose wireframe edges; normalized image landmarks are used for display only.
POSE_CONNECTIONS = ((11, 12), (11, 13), (13, 15), (15, 17), (15, 19),
                    (15, 21), (17, 19), (12, 14), (14, 16), (16, 18),
                    (16, 20), (16, 22), (18, 20), (11, 23), (12, 24),
                    (23, 24), (23, 25), (25, 27), (27, 29), (29, 31),
                    (27, 31), (24, 26), (26, 28), (28, 30), (30, 32),
                    (28, 32), (0, 1), (1, 2), (2, 3), (3, 7),
                    (0, 4), (4, 5), (5, 6), (6, 8), (9, 10))


class CameraUnavailable(RuntimeError):
    """Camera disconnected, capture stopped, or no frame arrived in time."""


class Diagnostics:
    """Bounded 120-frame timing windows; no unbounded running histories."""

    def __init__(self, window=120):
        if window < 2:
            raise ValueError('Diagnostic window must be at least two frames')
        self.inference = deque(maxlen=window)
        self.pipeline = deque(maxlen=window)
        self.completed = deque(maxlen=window)

    def record_inference(self, milliseconds):
        self.inference.append(float(milliseconds))

    def record_display(self, capture_time, completion_time):
        # Host receipt -> UI submission/event pump, not exposure -> photons.
        self.pipeline.append(max(0.0, (completion_time - capture_time) * 1000))
        self.completed.append(completion_time)

    def summary(self):
        fps = 0.0
        if len(self.completed) > 1:
            duration = self.completed[-1] - self.completed[0]
            if duration > 0:
                fps = (len(self.completed) - 1) / duration
        return {
            'inference_current_ms': self.inference[-1] if self.inference else 0.0,
            'inference_mean_ms': float(np.mean(self.inference)) if self.inference else 0.0,
            'inference_p95_ms': float(np.percentile(self.inference, 95)) if self.inference else 0.0,
            'pipeline_mean_ms': float(np.mean(self.pipeline)) if self.pipeline else 0.0,
            'pipeline_p95_ms': float(np.percentile(self.pipeline, 95)) if self.pipeline else 0.0,
            'fps': fps, 'window_frames': len(self.pipeline),
        }


class PoseProcessor:
    """Convert model outputs to visibility-gated, filtered metric coordinates."""

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.filter = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.reset_count = 0

    def reset(self):
        self.filter.reset()
        self.reset_count += 1

    def process(self, results, timestamp):
        world = getattr(results, 'pose_world_landmarks', None)
        image = getattr(results, 'pose_landmarks', None)
        valid_pose = (world is not None and image is not None
                      and len(world.landmark) == 33 and len(image.landmark) == 33)
        if valid_pose:
            points = np.array([(p.x, p.y, p.z) for p in world.landmark], dtype=float)
            visibility = np.array([p.visibility for p in image.landmark], dtype=float)
            visible = (np.isfinite(visibility) & (visibility > 0.65) & (visibility <= 1)
                       & np.all(np.isfinite(points), axis=1))
            points = self.filter(points, timestamp, valid_mask=visible[:, None])
        else:
            self.filter.reset()  # Never bridge tracking loss with stale motion history.
            points = np.full((33, 3), np.nan)
            visibility = np.zeros(33)
        metrics = BiomechanicsEngine.compute_joint_metrics(points, visibility)
        reasons = {}
        for side, offset in (('L', 0), ('R', 1)):
            for suffix, _, indices in METRIC_ROWS:
                key = f'{side}_{suffix}'
                if metrics[key] is None:
                    confidence = visibility[np.array(indices) + offset]
                    reasons[key] = ('Occluded / Low Conf' if not np.all(
                        np.isfinite(confidence) & (confidence > 0.65) & (confidence <= 1))
                        else 'Undefined geometry')
        return metrics, reasons


class HUDRenderer:
    """Small OpenCV primitives and one ROI blend; no external GUI framework."""

    @staticmethod
    def _text(frame, text, position, color=(230, 230, 230), scale=0.42):
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)

    def render(self, frame, results, metrics, reasons, diagnostics):
        height, width = frame.shape[:2]
        landmarks = getattr(results, 'pose_landmarks', None)
        if landmarks is not None and len(landmarks.landmark) == 33:
            pixels = {}
            for i, point in enumerate(landmarks.landmark):
                if (np.isfinite([point.x, point.y, point.visibility]).all()
                        and 0.65 < point.visibility <= 1
                        and 0 <= point.x <= 1 and 0 <= point.y <= 1):
                    pixels[i] = (min(width - 1, int(point.x * width)),
                                 min(height - 1, int(point.y * height)))
            for first, second in POSE_CONNECTIONS:
                if first in pixels and second in pixels:
                    cv2.line(frame, pixels[first], pixels[second], (90, 210, 100), 1, cv2.LINE_AA)
            for point in pixels.values():
                cv2.circle(frame, point, 2, (120, 245, 150), -1)

        # Keep all twelve rows visible even when a driver negotiates <480 height.
        # Pad small images instead of resizing the inference input.
        if height < 460 or width < 640:
            frame = cv2.copyMakeBorder(frame, 0, max(0, 460 - height),
                                       0, max(0, 640 - width), cv2.BORDER_CONSTANT)
        height, width = frame.shape[:2]
        panel_width = 310
        x = width - panel_width
        roi = frame[:, x:]
        background = np.full_like(roi, (22, 25, 28))
        cv2.addWeighted(roi, 0.30, background, 0.70, 0, dst=roi)
        self._text(frame, 'BIOMECHANICS', (x + 10, 24), scale=0.52)
        self._text(frame, f"FPS {diagnostics['fps']:.1f} | Window {diagnostics['window_frames']}/120",
                   (x + 10, 47))
        self._text(frame, f"Inference {diagnostics['inference_current_ms']:.1f} ms",
                   (x + 10, 68))
        self._text(frame, f"Infer mean/P95 {diagnostics['inference_mean_ms']:.1f} / "
                   f"{diagnostics['inference_p95_ms']:.1f} ms", (x + 10, 89), scale=0.39)
        self._text(frame, f"Capture->UI mean {diagnostics['pipeline_mean_ms']:.1f} ms",
                   (x + 10, 110))
        self._text(frame, f"Capture->UI P95 {diagnostics['pipeline_p95_ms']:.1f} ms",
                   (x + 10, 131))
        y = 157
        for suffix, label, _ in METRIC_ROWS:
            for side in ('L', 'R'):
                key = f'{side}_{suffix}'
                value = metrics[key]
                self._text(frame, f'{side} {label}', (x + 10, y), scale=0.38)
                if value is None:
                    self._text(frame, reasons.get(key, 'Unavailable'), (x + 139, y),
                               (80, 85, 255), 0.36)
                else:
                    text = f'{value:.1f}'
                    self._text(frame, text, (x + 221, y), (135, 240, 165), 0.42)
                    text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
                    # Hershey fonts do not support Unicode degree glyphs.
                    cv2.circle(frame, (x + 224 + text_width, y - 9), 2, (135, 240, 165), 1)
                y += 22
        self._text(frame, 'q / ESC: exit   r: reset filter', (x + 10, height - 12), scale=0.38)
        return frame


class OpenCVDisplay:
    """Own the desktop window and event pump; instantiate only outside headless tests."""

    def __init__(self):
        self._opened = False

    def submit(self, frame):
        if not self._opened:
            if sys.platform.startswith('linux') and not (os.environ.get('DISPLAY')
                                                        or os.environ.get('WAYLAND_DISPLAY')):
                raise RuntimeError('No desktop display; use benchmark.py in headless mode')
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            self._opened = True
        cv2.imshow(WINDOW, frame)

    def poll_key(self):
        if not self._opened:
            return -1
        key = cv2.waitKey(1)
        try:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return 27
        except cv2.error:
            return 27  # Some HighGUI backends destroy the window before this query.
        return key & 0xFF if key >= 0 else -1

    def close(self):
        if self._opened:
            try:
                cv2.destroyWindow(WINDOW)
            except cv2.error:
                LOGGER.debug('Window was already closed', exc_info=True)
            finally:
                self._opened = False


def create_pose():
    """Lazy model import keeps mock/headless tests independent of MediaPipe."""
    import mediapipe as mp
    if not hasattr(mp, 'solutions'):
        raise RuntimeError('Legacy Pose API unavailable; install requirements.txt in a fresh venv')
    return mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=1, smooth_landmarks=False,
        enable_segmentation=False, min_detection_confidence=0.6,
        min_tracking_confidence=0.6)


def run_pipeline(capture, pose_factory=create_pose, display=None, renderer=None,
                 processor=None, max_frames=None, stall_timeout=3.0, on_frame=None):
    """Run fresh frames serially on the UI thread while camera polling continues.

    Dependencies are injectable for mock tests. Resource ownership transfers to
    this function: camera/model/display are closed on every exit path, including
    initialization failures. Native camera stop timeouts surface to the caller.
    Timing uses perf_counter throughout. Capture->UI means host read completion
    through imshow/event pumping; OpenCV cannot timestamp actual screen photons.
    HUD pipeline aggregates describe completed frames, so they lag one frame.
    Optional on_frame(record) receives completed-frame timings and returns False
    to stop; observer work is excluded from that frame's latency, but affects FPS.
    """
    if max_frames is not None and max_frames <= 0:
        raise ValueError('max_frames must be positive')
    if not np.isfinite(stall_timeout) or stall_timeout <= 0:
        raise ValueError('stall_timeout must be positive and finite')
    display = OpenCVDisplay() if display is None else display
    renderer = HUDRenderer() if renderer is None else renderer
    processor = PoseProcessor() if processor is None else processor
    diagnostics = Diagnostics()
    sequence, processed = 0, 0
    last_arrival = time.perf_counter()
    with ExitStack() as resources:
        # ExitStack runs every callback even when a cleanup itself raises.
        resources.callback(display.close)
        resources.callback(capture.stop)
        capture.start()
        model = pose_factory()
        resources.callback(model.close)
        last_arrival = time.perf_counter()  # Model initialization is not camera stall time.
        while max_frames is None or processed < max_frames:
            sample = capture.wait_for_frame(sequence, timeout=0.05)
            if sample is None:
                key = display.poll_key()
                if key in (ord('q'), 27):
                    break
                if key == ord('r'):
                    processor.reset()
                if capture.error is not None or not capture.running:
                    raise CameraUnavailable(f'Camera stopped: {capture.error}')
                if time.perf_counter() - last_arrival >= stall_timeout:
                    raise CameraUnavailable('Camera stalled; no new frames')
                continue
            if sample.sequence <= sequence:
                # Defensive check: repeated snapshots must not run inference twice.
                if time.perf_counter() - last_arrival >= stall_timeout:
                    raise CameraUnavailable('Camera returned only stale frames')
                key = display.poll_key()
                if key in (ord('q'), 27):
                    break
                if key == ord('r'):
                    processor.reset()
                continue
            sequence = sample.sequence
            last_arrival = time.perf_counter()
            frame = sample.frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            inference_start = time.perf_counter()
            results = model.process(rgb)
            inference_ms = (time.perf_counter() - inference_start) * 1000
            diagnostics.record_inference(inference_ms)
            metrics, reasons = processor.process(results, sample.timestamp_ns / 1e9)
            rendered = renderer.render(frame, results, metrics, reasons, diagnostics.summary())
            display.submit(rendered)
            key = display.poll_key()
            completion = time.perf_counter()
            diagnostics.record_display(sample.timestamp_ns / 1e9, completion)
            processed += 1
            if on_frame is not None and on_frame({
                    'sequence': sample.sequence, 'capture_time': sample.timestamp_ns / 1e9,
                    'completion_time': completion, 'inference_ms': inference_ms,
                    'pipeline_ms': max(0, (completion - sample.timestamp_ns / 1e9) * 1000),
                    'reliable_metrics': sum(value is not None for value in metrics.values()),
                    'pose_detected': getattr(results, 'pose_world_landmarks', None) is not None,
            }) is False:
                break
            if key in (ord('q'), 27):
                break
            if key == ord('r'):
                processor.reset()
    return {'processed_frames': processed, 'filter_resets': processor.reset_count,
            'diagnostics': diagnostics.summary()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=float, default=60)
    parser.add_argument('--min-cutoff', type=float, default=1)
    parser.add_argument('--beta', type=float, default=0.007)
    parser.add_argument('--d-cutoff', type=float, default=1)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        capture = VideoCaptureAsync(args.camera, args.width, args.height, args.fps)
        processor = PoseProcessor(args.min_cutoff, args.beta, args.d_cutoff)
        LOGGER.info('Starting biomechanics. q / ESC: exit, r: reset filter.')
        summary = run_pipeline(capture, processor=processor)
        LOGGER.info('Stopped cleanly: %s', summary)
        return 0
    except KeyboardInterrupt:
        LOGGER.info('Interrupted; resources closed.')
        return 0
    except (RuntimeError, OSError, ValueError, ImportError, cv2.error) as exc:
        LOGGER.error('Application stopped: %s', exc)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
