"""Collect measured goniometer/software pairs and generate accuracy reports."""
import argparse
import json
from pathlib import Path
import sys

import cv2

from benchmark import HeadlessDisplay
from core.capture import VideoCaptureAsync
from core.validation import (JOINTS, VIEWS, HoldCollector, analyze_csv,
                             append_hold, check_csv_header, markdown_report, validate_metadata)
from main import HUDRenderer, OpenCVDisplay, PoseProcessor, run_pipeline


class HoldRenderer(HUDRenderer):
    """Show acquisition progress without changing inference or reference values."""
    def __init__(self, collector):
        self.collector = collector

    def render(self, frame, results, metrics, reasons, diagnostics):
        frame = super().render(frame, results, metrics, reasons, diagnostics)
        elapsed = 0 if self.collector.start is None else self.collector.last_time-self.collector.start
        phase = 'SETTLING' if elapsed < self.collector.settling else 'MEASURING'
        remaining = max(0, self.collector.settling+self.collector.seconds-elapsed)
        cv2.rectangle(frame, (0, 0), (315, 65), (25, 25, 25), -1)
        self._text(frame, f'{phase}: hold still {remaining:.1f}s', (10, 24), scale=.48)
        self._text(frame, self.collector.joint, (10, 48))
        return frame


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    collect = commands.add_parser('collect', help='One physical webcam static hold per assessor reading')
    collect.add_argument('--participant', required=True, help='Anonymous participant ID')
    collect.add_argument('--joint', choices=JOINTS, required=True)
    collect.add_argument('--view', choices=VIEWS, required=True)
    collect.add_argument('--reference', type=float, required=True, help='Measured assessor reading in signed degrees')
    collect.add_argument('--reference-method', default='manual 360-degree goniometer')
    collect.add_argument('--csv', type=Path, default=Path('validation/data/paired_holds.csv'))
    collect.add_argument('--seconds', type=float, default=2)
    collect.add_argument('--settling', type=float, default=1)
    collect.add_argument('--camera', type=int, default=0)
    collect.add_argument('--width', type=int, default=640)
    collect.add_argument('--height', type=int, default=480)
    collect.add_argument('--fps', type=float, default=60)
    collect.add_argument('--min-cutoff', type=float, default=1)
    collect.add_argument('--beta', type=float, default=5)
    collect.add_argument('--d-cutoff', type=float, default=1)
    collect.add_argument('--coordinate-frame', choices=('body', 'camera'), default='body')
    collect.add_argument('--headless', action='store_true', help='Explicitly omit the desktop window')
    report = commands.add_parser('report', help='Analyze paired holds; never fabricate reference data')
    report.add_argument('--csv', type=Path, default=Path('validation/data/paired_holds.csv'))
    report.add_argument('--json', type=Path, default=Path('validation/results/accuracy.json'))
    report.add_argument('--markdown', type=Path, default=Path('validation/results/accuracy.md'))
    simulate = commands.add_parser('simulate', help='Analytic targets with seeded noise; NOT clinical validation')
    simulate.add_argument('--output-dir', type=Path, default=Path('benchmarks/results/simulated_accuracy'))
    simulate.add_argument('--seed', type=int, default=20261003)
    simulate.add_argument('--repetitions', type=int, default=3)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == 'simulate':
            from benchmarks.simulated_accuracy import simulate
            report = simulate(args.output_dir, seed=args.seed, repetitions=args.repetitions)
            print(f"Synthetic validation: {report['accepted_holds']} accepted / "
                  f"{report['rejected_holds']} rejected holds; {report['frames_processed']} processed frames")
            print(report['scope'])
            print(f'Reports and raw holds saved: {args.output_dir}')
            return 0 if report['required_scope_present'] else 2
        elif args.command == 'collect':
            validate_metadata(args.participant, args.joint, args.view, args.reference, args.reference_method)
            check_csv_header(args.csv)
            collector = HoldCollector(args.joint, args.seconds, args.settling)
            processor = PoseProcessor(args.min_cutoff, args.beta, args.d_cutoff, args.coordinate_frame)
            capture = VideoCaptureAsync(args.camera, args.width, args.height, args.fps)
            print('Maintain the measured static position. The assessor should not view software estimates.')
            run_pipeline(capture, display=HeadlessDisplay() if args.headless else OpenCVDisplay(),
                         renderer=HoldRenderer(collector), processor=processor, on_frame=collector)
            settings = capture.actual_settings
            row = collector.row(participant=args.participant, view=args.view,
                                reference_deg=args.reference, reference_method=args.reference_method,
                                processor=processor, width=settings.get('width', args.width),
                                height=settings.get('height', args.height))
            append_hold(args.csv, row)
            print(f"Hold saved: {args.csv} | valid coverage {row['valid_fraction']:.1%}")
            if row['rejection_reason']:
                print('Rejected hold retained: ' + row['rejection_reason'])
                return 2
            print(f"Software median: {row['software_deg']:.2f} deg | assessor reference: {args.reference:.2f} deg")
        else:
            if len({args.csv.resolve(), args.json.resolve(), args.markdown.resolve()}) != 3:
                raise ValueError('Input CSV and output reports must use distinct paths')
            report = analyze_csv(args.csv)
            text = markdown_report(report)
            for path in (args.json, args.markdown):
                path.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
            args.markdown.write_text(text, encoding='utf-8')
            print(text)
            print(f'Reports saved: {args.json}, {args.markdown}')
            return 0 if report['required_scope_present'] else 2
        return 0
    except KeyboardInterrupt:
        print('Validation interrupted; pipeline resources closed.', file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError, cv2.error) as exc:
        print(f'Validation failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
