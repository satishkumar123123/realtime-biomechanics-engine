"""Deterministic forward-kinematic validation; no camera or pose model involved.

Targets create articulated metric landmarks independently of the angle engine.
The real PoseProcessor (visibility, One-Euro filter and geometry) and HoldCollector
then measure those landmarks. Results quantify this stated simulation only.
"""
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from core.validation import FIELDS, HoldCollector, analyze_rows, markdown_report
from main import PoseProcessor
from benchmarks.prepare_replay import sha256_file


REFERENCE_METHOD = 'synthetic: independent forward kinematics; no physical goniometer'
TARGETS = {'Elbow_Flex': (0, 90, 135), 'Knee_Flex': (0, 90), 'Shoulder_Flex': (45, 90)}
NOISE_STD = np.array((.003, .003, .010))


def target_pose(joint, target, yaw_degrees):
    """Construct 33 world points using segment lengths and commanded rotations.

    Elbow/knee targets move a distal segment around an otherwise neutral joint;
    shoulder targets move the whole straight arm anteriorly. Yaw is a rigid
    rotation of the subject relative to one fixed virtual camera.
    """
    points = np.zeros((33, 3), dtype=np.float64)
    for index in range(11):
        points[index] = (.03*((index % 3)-1), -.76, -.05)
    for side, sh, el, wr, hip, knee, ankle, foot, sign in (
            ('L', 11, 13, 15, 23, 25, 27, 31, 1),
            ('R', 12, 14, 16, 24, 26, 28, 32, -1)):
        shoulder_angle = np.radians(target if joint == f'{side}_Shoulder_Flex' else 0)
        elbow_angle = np.radians(target if joint == f'{side}_Elbow_Flex' else 0)
        knee_angle = np.radians(target if joint == f'{side}_Knee_Flex' else 0)
        points[sh] = (sign*.20, -.55, 0)
        points[hip] = (sign*.16, 0, 0)
        points[el] = points[sh] + .30*np.array((0, np.cos(shoulder_angle), -np.sin(shoulder_angle)))
        forearm = np.array((0, np.cos(shoulder_angle+elbow_angle), -np.sin(shoulder_angle+elbow_angle)))
        points[wr] = points[el] + .25*forearm
        points[knee] = points[hip] + (0, .45, 0)
        points[ankle] = points[knee] + .45*np.array((0, np.cos(knee_angle), np.sin(knee_angle)))
        points[foot] = points[ankle] + .20*np.array((0, np.sin(knee_angle), -np.cos(knee_angle)))
        for finger in (wr+2, wr+4, wr+6):
            points[finger] = points[wr] + .06*forearm
        points[ankle+2] = points[ankle] + (0, .025, .04)
    yaw = np.radians(yaw_degrees)
    rotation = np.array(((np.cos(yaw), 0, np.sin(yaw)), (0, 1, 0),
                         (-np.sin(yaw), 0, np.cos(yaw))))
    return points @ rotation.T


def pose_result(points, hidden_index=None):
    """Adapt analytic coordinates to the MediaPipe-shaped processor input."""
    return SimpleNamespace(
        pose_world_landmarks=SimpleNamespace(landmark=[
            SimpleNamespace(x=x, y=y, z=z) for x, y, z in points]),
        pose_landmarks=SimpleNamespace(landmark=[
            SimpleNamespace(x=.5+x*.3, y=.4+y*.3, z=z,
                            visibility=.2 if i == hidden_index else .99)
            for i, (x, y, z) in enumerate(points)]))


def error_summary(rows):
    """Each observation is one accepted hold median, never one video frame."""
    accepted = [row for row in rows if not row['rejection_reason']]
    errors = np.array([row['software_deg']-row['reference_deg'] for row in accepted])
    return {'holds': len(accepted), 'rejected_holds': len(rows)-len(accepted),
            'mean_measured_deg': float(np.mean([row['software_deg'] for row in accepted])) if accepted else None,
            'mae_deg': float(np.mean(abs(errors))) if accepted else None,
            'bias_deg': float(np.mean(errors)) if accepted else None,
            'rmse_deg': float(np.sqrt(np.mean(errors**2))) if accepted else None}


def simulate(output_dir, *, seed=20261003, repetitions=3):
    """Run all requested bilateral targets and export raw holds + error reports.

    Default: 54 holds, each 1 s settling + 2 s sampled at 60 Hz (181 calls).
    First 0.5 s transitions from neutral to the target. Camera-axis Gaussian
    noise is 3/3/10 mm, with 6 mm, 9 Hz depth flicker; these are chosen stress
    parameters, not measured MediaPipe error. Four occluded samples per hold
    exercise gating/reacquisition. Reset filters between independent holds.
    """
    if not isinstance(seed, int) or seed < 0 or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError('Seed must be nonnegative and repetitions a positive integer')
    rng = np.random.default_rng(seed)
    rows, cases = [], []
    by_target, by_plane = defaultdict(list), defaultdict(list)
    started = datetime.now(timezone.utc).isoformat()
    for suffix, targets in TARGETS.items():
        views = ((0, 'front'), (45, 'oblique')) if suffix == 'Shoulder_Flex' else ((90, 'left-side'),)
        plane = 'out_of_plane_shoulder' if suffix == 'Shoulder_Flex' else 'in_plane_elbow_knee'
        for target in targets:
            for side in ('L', 'R'):
                joint = f'{side}_{suffix}'
                hidden = {'Elbow_Flex': 15, 'Knee_Flex': 25, 'Shoulder_Flex': 13}[suffix] + (side == 'R')
                for yaw, view in views:
                    for repetition in range(repetitions):
                        processor = PoseProcessor()
                        collector = HoldCollector(joint, seconds=2, settling=1)
                        phase = rng.uniform(0, 2*np.pi, size=33)
                        hold_id = f'SIM-{len(rows)+1:03d}'
                        for frame in range(181):
                            timestamp = frame/60
                            angle = target*min(1, timestamp/.5)
                            points = target_pose(joint, angle, yaw)
                            points += rng.normal(size=(33, 3))*NOISE_STD
                            points[:, 2] += .006*np.sin(2*np.pi*9*timestamp+phase)
                            metrics, reasons = processor.process(
                                pose_result(points, hidden if 84 <= frame < 88 else None), timestamp)
                            if not collector({'capture_time': timestamp, 'metrics': metrics, 'reasons': reasons}):
                                break
                        row = collector.row(participant=f'SIM_SEED_{seed}', view=view,
                                            reference_deg=target, reference_method=REFERENCE_METHOD,
                                            processor=processor, width=640, height=480)
                        row.update(hold_id=hold_id, recorded_utc=started)
                        rows.append(row)
                        cases.append({'hold_id': hold_id, 'joint': joint, 'target_deg': target,
                                      'yaw_deg': yaw, 'plane_group': plane, 'repetition': repetition+1,
                                      'frames_processed': frame+1, **collector.summary()})
                        by_target[(suffix, target)].append(row)
                        by_plane[plane].append(row)
    report = analyze_rows(rows)
    report.update({
        'recorded_utc': started, 'seed': seed, 'actual_human_participants': 0,
        'model_inference_executed': False, 'frames_processed': sum(case['frames_processed'] for case in cases),
        'simulation': {'repetitions_per_side_view_target': repetitions, 'rate_hz': 60,
                       'settling_seconds': 1, 'sampled_seconds': 2, 'transition_seconds': .5,
                       'gaussian_std_xyz_m': NOISE_STD.tolist(), 'depth_flicker_amplitude_m': .006,
                       'depth_flicker_hz': 9, 'occluded_frame_indices': [84, 85, 86, 87],
                       'visibility_normal': .99, 'visibility_occluded': .2,
                       'coordinate_frame': 'body', 'min_cutoff': 1, 'beta': 5, 'd_cutoff': 1},
        'by_target': [{'joint': suffix, 'target_deg': target, **error_summary(group)}
                      for (suffix, target), group in by_target.items()],
        'by_plane': [{'plane': plane, **error_summary(group)} for plane, group in by_plane.items()],
        'cases': cases,
    })
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir/'holds.csv').open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    report['raw_holds'] = {'file': 'holds.csv', 'sha256': sha256_file(output_dir/'holds.csv'), 'rows': len(rows)}
    (output_dir/'accuracy.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = [markdown_report(report), '## Target-angle results (simulated)', '',
             '| Joint | Target | Mean measured | Holds | Rejected | MAE | Bias | RMSE |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in report['by_target']:
        numbers = ['unavailable' if row[key] is None else f'{row[key]:.4f}'
                   for key in ('mean_measured_deg', 'mae_deg', 'bias_deg', 'rmse_deg')]
        lines.append(f"| {row['joint']} | {row['target_deg']} | {numbers[0]} | {row['holds']} | "
                     f"{row['rejected_holds']} | {numbers[1]} | {numbers[2]} | {numbers[3]} |")
    lines += ['', '## In-plane versus out-of-plane groups (simulated)', '',
              '| Group | Holds | Rejected | MAE | Bias | RMSE |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in report['by_plane']:
        values = ['unavailable' if row[key] is None else f'{row[key]:.4f}'
                  for key in ('mae_deg', 'bias_deg', 'rmse_deg')]
        lines.append(f"| {row['plane']} | {row['holds']} | {row['rejected_holds']} | "
                     f"{values[0]} | {values[1]} | {values[2]} |")
    lines += ['', 'Targets come from commanded forward rotations; reported errors compare one filtered',
              'median per hold with that target. Seeds are synthetic identifiers, not participants.',
              'This dataset does not measure camera, detector or clinical anatomical error.', '']
    (output_dir/'accuracy.md').write_text('\n'.join(lines), encoding='utf-8')
    return report
