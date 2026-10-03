"""Recompute published performance and synthetic errors from checked-in raw CSVs.

No camera, GUI, image download or model inference is required for this check.
"""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

from benchmark import Recorder
from benchmarks.prepare_replay import sha256_file
from core.validation import analyze_csv


def check(actual, expected, path):
    if isinstance(expected, dict):
        for key, value in expected.items():
            check(actual[key], value, path+'.'+key)
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9):
            raise ValueError(f'Inconsistent {path}: {actual} != {expected}')
    elif actual != expected:
        raise ValueError(f'Inconsistent {path}: {actual} != {expected}')


def hold_statistics(rows):
    """Independent scalar implementation of per-hold error formulas."""
    valid = [row for row in rows if not row['rejection_reason']]
    measured = [float(row['software_deg']) for row in valid]
    errors = [value-float(row['reference_deg']) for value, row in zip(measured, valid)]
    count = len(valid)
    return {'holds': count, 'rejected_holds': len(rows)-count,
            'mean_measured_deg': math.fsum(measured)/count if count else None,
            'mae_deg': math.fsum(map(abs, errors))/count if count else None,
            'bias_deg': math.fsum(errors)/count if count else None,
            'rmse_deg': math.sqrt(math.fsum(e*e for e in errors)/count) if count else None}


def verify(performance_path, samples_path, accuracy_dir):
    report = json.loads(Path(performance_path).read_text(encoding='utf-8'))
    if sha256_file(samples_path) != report['raw_samples']['sha256']:
        raise ValueError('Raw performance samples checksum mismatch')
    with Path(samples_path).open(newline='', encoding='utf-8') as source:
        rows = list(csv.DictReader(source))
    recorder = Recorder(warmup=0)
    for row in rows:
        recorder.records.append({
            **{key: float(row[key]) for key in ('capture_time', 'completion_time', 'inference_ms', 'pipeline_ms')},
            'sequence': int(row['sequence']), 'reliable_metrics': int(row['reliable_metrics']),
            'pose_detected': row['pose_detected'] == 'True',
        })
    reconstructed = recorder.summary()
    for key in ('measured_frames', 'measured_interval_seconds', 'achieved_e2e_fps', 'inference',
                'pipeline', 'frames_with_pose', 'mean_reliable_metrics', 'pose_coverage',
                'numeric_coverage', 'rolling_window', 'mailbox', 'detected_pose_latency'):
        check(report[key], reconstructed[key], 'performance.'+key)
    accuracy_dir = Path(accuracy_dir)
    accuracy = json.loads((accuracy_dir/'accuracy.json').read_text(encoding='utf-8'))
    if sha256_file(accuracy_dir/'holds.csv') != accuracy['raw_holds']['sha256']:
        raise ValueError('Raw synthetic holds checksum mismatch')
    reconstructed = analyze_csv(accuracy_dir/'holds.csv')
    for key in ('accepted_holds', 'rejected_holds', 'validation_type', 'clinical_accuracy_established'):
        check(accuracy[key], reconstructed[key], 'accuracy.'+key)
    check(accuracy['groups'], reconstructed['groups'], 'accuracy.groups')
    targets, planes = defaultdict(list), defaultdict(list)
    with (accuracy_dir/'holds.csv').open(newline='', encoding='utf-8') as source:
        for row in csv.DictReader(source):
            suffix = row['joint'][2:]
            targets[(suffix, float(row['reference_deg']))].append(row)
            planes['out_of_plane_shoulder' if suffix == 'Shoulder_Flex' else 'in_plane_elbow_knee'].append(row)
    for row in accuracy['by_target']:
        check(row, hold_statistics(targets[(row['joint'], row['target_deg'])]), 'accuracy.by_target')
    for row in accuracy['by_plane']:
        check(row, hold_statistics(planes[row['plane']]), 'accuracy.by_plane')
    return {'performance_frames_verified': len(rows),
            'synthetic_holds_verified': accuracy['accepted_holds']+accuracy['rejected_holds'],
            'physical_validation_established': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--performance', type=Path, default=Path('benchmark_results.json'))
    parser.add_argument('--samples', type=Path, default=Path('benchmark_frame_samples.csv'))
    parser.add_argument('--accuracy-dir', type=Path, default=Path('benchmarks/results/simulated_accuracy'))
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify(args.performance, args.samples, args.accuracy_dir), indent=2))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'Evidence verification failed: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
