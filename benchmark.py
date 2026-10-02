"""Benchmark the same capture/inference/filter/angle/HUD loop as the desktop app."""
import argparse
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np

from core.capture import VideoCaptureAsync
from core.pose import create_pose
from main import OpenCVDisplay, PoseProcessor, run_pipeline


class PacedReplay:
    """OpenCV-compatible synthetic/recorded source; decoding runs on capture thread.

    Recorded files loop at EOF and letterbox to the requested resolution. Replay
    pacing uses host time, not recorded exposure timestamps. Late reads resume
    at the current time instead of bursting to catch up.
    """
    def __init__(self, width, height, fps, video=None):
        self.width, self.height, self.fps = width, height, fps
        self.video = None if video is None else cv2.VideoCapture(str(video))
        self.period = 1 / fps
        self.deadline = time.perf_counter()
        self.count = 0
        self.loops = 0
        self.closed = False
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)

    def isOpened(self):
        return not self.closed and (self.video is None or self.video.isOpened())

    def set(self, prop, value):
        return False  # Replay settings were explicitly configured at construction.

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: self.width, cv2.CAP_PROP_FRAME_HEIGHT: self.height,
                cv2.CAP_PROP_FPS: self.fps}.get(prop, 0)

    def read(self):
        self.deadline += self.period
        time.sleep(max(0, self.deadline - time.perf_counter()))
        self.deadline = max(self.deadline, time.perf_counter())
        self.count += 1
        if self.video is None:
            return True, self.frame
        # Loop files so duration/frame limits are reproducible for short recordings.
        ok, frame = self.video.read()
        if not ok:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.loops += 1
            ok, frame = self.video.read()
        if ok and frame is not None:
            height, width = frame.shape[:2]
            scale = min(self.width / width, self.height / height)
            resized_width = max(1, min(self.width, round(width * scale)))
            resized_height = max(1, min(self.height, round(height * scale)))
            frame = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
            left, top = (self.width-resized_width)//2, (self.height-resized_height)//2
            frame = cv2.copyMakeBorder(frame, top, self.height-resized_height-top,
                                       left, self.width-resized_width-left, cv2.BORDER_CONSTANT)
        return ok, frame

    def release(self):
        self.closed = True
        if self.video is not None:
            self.video.release()


class HeadlessDisplay:
    """Render the real HUD into memory; exclude OS display/presentation cost."""
    def submit(self, frame):
        self.last_shape = frame.shape

    def poll_key(self):
        return -1

    def close(self):
        pass


class MockPose:
    """Explicit analytic pose for numeric-path tests, never real inference results."""
    def __init__(self):
        points = np.zeros((33, 3))
        for shoulder, elbow, wrist, hip, knee, ankle, foot, x in (
                (11, 13, 15, 23, 25, 27, 31, .2),
                (12, 14, 16, 24, 26, 28, 32, -.2)):
            for index, y in ((shoulder, -.6), (elbow, -.3), (wrist, 0),
                             (hip, 0), (knee, .5), (ankle, 1)):
                points[index] = (x, y, 0)
            points[foot] = (x, 1, -.2)
        self.result = SimpleNamespace(
            pose_world_landmarks=SimpleNamespace(landmark=[
                SimpleNamespace(x=x, y=y, z=z) for x, y, z in points]),
            pose_landmarks=SimpleNamespace(landmark=[
                SimpleNamespace(x=.5 + x, y=.3 + y*.3, z=z, visibility=1.)
                for x, y, z in points]))

    def process(self, rgb):
        return self.result

    def close(self):
        pass


def latency_stats(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError('No measured frames; increase benchmark limit')
    return {'mean_ms': float(np.mean(values)), 'min_ms': float(np.min(values)),
            'max_ms': float(np.max(values)), 'p95_ms': float(np.percentile(values, 95))}


class Recorder:
    """Exclude warmup, gather all measured frames, stop at frame/time limits."""
    def __init__(self, frames=300, seconds=None, warmup=30):
        self.frames, self.seconds, self.warmup = frames, seconds, warmup
        self.seen = 0
        self.records = []

    def __call__(self, record):
        self.seen += 1
        if self.seen <= self.warmup:
            return True
        self.records.append(dict(record))
        if self.frames is not None and len(self.records) >= self.frames:
            return False
        if self.seconds is not None and len(self.records) > 1:
            return record['completion_time'] - self.records[0]['completion_time'] < self.seconds
        return True

    def summary(self):
        if len(self.records) < 2:
            raise ValueError('At least two measured frames are required for FPS')
        first, last = self.records[0], self.records[-1]
        duration = last['completion_time'] - first['completion_time']
        deltas = np.diff([r['sequence'] for r in self.records])
        if duration <= 0 or np.any(deltas <= 0):
            raise ValueError('Benchmark timestamps/sequences must increase')
        superseded = int(np.sum(deltas - 1))
        published = int(last['sequence'] - first['sequence'])
        rolling = self.records[-120:]
        rolling_duration = rolling[-1]['completion_time'] - rolling[0]['completion_time']
        pose_records = [r for r in self.records if r['pose_detected']]
        return {
            'measured_frames': len(self.records), 'warmup_frames': self.warmup,
            'measured_interval_seconds': duration,
            'achieved_e2e_fps': (len(self.records) - 1) / duration,
            'inference': latency_stats([r['inference_ms'] for r in self.records]),
            'pipeline': latency_stats([r['pipeline_ms'] for r in self.records]),
            'frames_with_pose': sum(bool(r['pose_detected']) for r in self.records),
            'mean_reliable_metrics': float(np.mean([r['reliable_metrics'] for r in self.records])),
            'pose_coverage': len(pose_records) / len(self.records),
            'numeric_coverage': sum(r['reliable_metrics'] > 0 for r in self.records) / len(self.records),
            'metric_coverage': {
                key: sum(r.get('metrics', {}).get(key) is not None for r in self.records)
                     / len(self.records)
                for key in sorted({key for r in self.records for key in r.get('metrics', {})})},
            'detected_pose_latency': None if not pose_records else {
                'frames': len(pose_records),
                'inference': latency_stats([r['inference_ms'] for r in pose_records]),
                'pipeline': latency_stats([r['pipeline_ms'] for r in pose_records]),
            },
            'rolling_window': {
                'frames': len(rolling),
                'fps': (len(rolling) - 1) / rolling_duration,
                'inference': latency_stats([r['inference_ms'] for r in rolling]),
                'pipeline': latency_stats([r['pipeline_ms'] for r in rolling]),
            },
            'mailbox': {
                'published_intervals': published,
                'consumed_intervals': len(self.records) - 1,
                'superseded_frames': superseded,
                'overrun_events': int(np.count_nonzero(deltas > 1)),
                'superseded_percent': 100 * superseded / published,
                'hardware_dropped_frames': None,
                'scope': 'Sequence gaps between first and last measured frames; excludes startup and tail',
            },
        }


def submission_checks(report):
    """Explicit evidence policy; blank/mock/headless runs cannot prove desktop FPS.

    Ten seconds, 80% pose coverage and 50% numeric coverage are this project's
    representative-run checks, not additional assignment accuracy thresholds.
    Passing performance checks does not establish angle accuracy.
    """
    config = report['configuration']
    checks = {
        'real_local_model': config['inference'] == 'MediaPipe BlazePose Full',
        'physical_webcam': config['source'] == 'webcam',
        'desktop_display': config['display'] == 'desktop',
        'at_least_10_seconds': report['measured_interval_seconds'] >= 10,
        'pose_coverage_at_least_80_percent': report['pose_coverage'] >= .8,
        'numeric_coverage_at_least_50_percent': report['numeric_coverage'] >= .5,
        'achieved_fps_at_least_30': report['achieved_e2e_fps'] >= 30,
    }
    return {'passed': all(checks.values()), 'checks': checks,
            'scope': 'Desktop performance evidence only; clinical accuracy requires paired references'}


def hardware_info():
    cpu = platform.processor() or 'unreported'
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith('model name'):
                cpu = line.split(':', 1)[1].strip()
                break
    quota = None
    quota_file = Path('/sys/fs/cgroup/cpu.max')
    if quota_file.exists():
        values = quota_file.read_text().split()
        if len(values) == 2 and values[0] != 'max':
            quota = int(values[0]) / int(values[1])
    return {'os': platform.platform(), 'cpu': cpu, 'logical_cpus': os.cpu_count(),
            'cpu_quota_cores': quota, 'python': platform.python_version(),
            'opencv': cv2.__version__, 'numpy': np.__version__}


def run_benchmark(args):
    if args.frames is not None and args.frames < 2:
        raise ValueError('--frames must be at least 2')
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        raise ValueError('--seconds must be positive and finite')
    if args.warmup < 0 or args.width <= 0 or args.height <= 0:
        raise ValueError('Warmup must be nonnegative and dimensions positive')
    if not math.isfinite(args.fps) or args.fps < 1:
        raise ValueError('--fps must be finite and at least 1')
    if args.source == 'recorded' and (args.video is None or not args.video.is_file()):
        raise ValueError('--source recorded requires an existing --video file')
    if args.mock_pose and args.source != 'synthetic':
        raise ValueError('--mock-pose is only available for synthetic numeric-path tests')
    source = None
    def replay_factory(device, backend):
        nonlocal source
        source = PacedReplay(args.width, args.height, args.fps,
                             args.video if args.source == 'recorded' else None)
        return source
    cap = VideoCaptureAsync(args.camera, args.width, args.height, args.fps,
                            capture_factory=None if args.source == 'webcam' else replay_factory)
    recorder = Recorder(frames=(300 if args.frames is None and args.seconds is None else args.frames),
                        seconds=args.seconds, warmup=args.warmup)
    run_pipeline(cap, pose_factory=MockPose if args.mock_pose else create_pose,
                 display=OpenCVDisplay() if args.display else HeadlessDisplay(),
                 processor=PoseProcessor(args.min_cutoff, args.beta, args.d_cutoff,
                                         args.coordinate_frame),
                 on_frame=recorder)
    report = recorder.summary()
    model_version = 'mock'
    if not args.mock_pose:
        model_version = version('mediapipe')
    report.update({
        'schema_version': 2, 'hardware': hardware_info(),
        'configuration': {
            'source': args.source, 'video': None if args.video is None else args.video.name,
            'input_content': 'blank' if args.source == 'synthetic' else 'user-supplied',
            'inference': 'analytic mock' if args.mock_pose else 'MediaPipe BlazePose Full',
            'mediapipe': model_version, 'model_complexity': None if args.mock_pose else 1,
            'display': 'desktop' if args.display else 'headless HUD rendering',
            'requested_resolution': [args.width, args.height], 'requested_source_fps': args.fps,
            'negotiated_capture': cap.actual_settings,
            'min_cutoff': args.min_cutoff, 'beta': args.beta, 'd_cutoff': args.d_cutoff,
            'coordinate_frame': args.coordinate_frame,
            'replay_resize': 'preserve aspect ratio with letterboxing',
            'recorded_replay_loops': source.loops if source is not None else 0,
        },
        'timing_scope': ('Host camera/replay read completion to completed HUD rendering '
                         + ('and UI event pump' if args.display else 'with headless display sink')
                         + '; excludes sensor exposure, actual screen presentation and model initialization'),
    })
    report['submission_performance'] = submission_checks(report)
    return report


def print_summary(report):
    config = report['configuration']
    print(f"\nBiomechanics performance | {config['source']} | {config['inference']} | {config['display']}")
    print(f"CPU: {report['hardware']['cpu']} | OS: {report['hardware']['os']}")
    print(f"Measured frames: {report['measured_frames']} | Warmup: {report['warmup_frames']}")
    print(f"Achieved end-to-end FPS: {report['achieved_e2e_fps']:.2f}\n")
    print('| Latency | Mean (ms) | Min (ms) | Max (ms) | P95 (ms) |')
    print('| --- | ---: | ---: | ---: | ---: |')
    for label, key in (('Model inference', 'inference'), ('End-to-end pipeline', 'pipeline')):
        values = report[key]
        print(f"| {label} | {values['mean_ms']:.3f} | {values['min_ms']:.3f} | {values['max_ms']:.3f} | {values['p95_ms']:.3f} |")
    mailbox = report['mailbox']
    print(f"\nMailbox superseded: {mailbox['superseded_frames']} ({mailbox['superseded_percent']:.2f}%)"
          f" | Overrun events: {mailbox['overrun_events']}")
    print('Hardware/driver frame drops: unknown (not exposed by OpenCV).')
    rolling = report['rolling_window']
    print(f"Final {rolling['frames']}-frame window: FPS {rolling['fps']:.2f}")
    print('| Rolling latency | Mean (ms) | P95 (ms) |')
    print('| --- | ---: | ---: |')
    for label, key in (('Inference', 'inference'), ('Pipeline', 'pipeline')):
        print(f"| {label} | {rolling[key]['mean_ms']:.3f} | {rolling[key]['p95_ms']:.3f} |")
    print(f"Frames with pose: {report['frames_with_pose']} | Mean reliable metrics: {report['mean_reliable_metrics']:.2f}/12")
    print(f"Pose coverage: {report['pose_coverage']:.1%} | Numeric coverage: {report['numeric_coverage']:.1%}")
    if report['detected_pose_latency'] is not None:
        subset = report['detected_pose_latency']
        print(f"Detected-pose subset ({subset['frames']} frames): inference mean/P95 "
              f"{subset['inference']['mean_ms']:.2f}/{subset['inference']['p95_ms']:.2f} ms; "
              f"pipeline mean/P95 {subset['pipeline']['mean_ms']:.2f}/{subset['pipeline']['p95_ms']:.2f} ms")
    evidence = report['submission_performance']
    print('Representative desktop performance checks: ' + ('PASS' if evidence['passed'] else 'NOT ESTABLISHED'))
    if not evidence['passed']:
        print('Unmet checks: ' + ', '.join(key for key, passed in evidence['checks'].items() if not passed))
    print(report['timing_scope'])
    if config['source'] == 'synthetic':
        print('Synthetic blank input is not a detected-human workload; mock timings are not model performance.')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('synthetic', 'recorded', 'webcam'), default='synthetic')
    parser.add_argument('--video', type=Path)
    parser.add_argument('--camera', type=int, default=0)
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument('--frames', type=int)
    limits.add_argument('--seconds', type=float)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=float, default=60)
    parser.add_argument('--min-cutoff', type=float, default=1)
    parser.add_argument('--beta', type=float, default=5.0)
    parser.add_argument('--d-cutoff', type=float, default=1)
    parser.add_argument('--coordinate-frame', choices=('body', 'camera'), default='body')
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--mock-pose', action='store_true')
    parser.add_argument('--require-human', action='store_true',
                        help='Exit 2 unless representative human webcam/desktop performance checks pass')
    parser.add_argument('--output', type=Path, help='Optional machine-readable JSON report')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        report = run_benchmark(args)
        print_summary(report)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            print(f'Report saved: {args.output}')
        return 2 if args.require_human and not report['submission_performance']['passed'] else 0
    except KeyboardInterrupt:
        print('Benchmark interrupted; pipeline resources closed.', file=sys.stderr)
        return 130
    except (RuntimeError, OSError, ValueError, ImportError, cv2.error) as exc:
        print(f'Benchmark failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
