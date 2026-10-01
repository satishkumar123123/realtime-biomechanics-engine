"""Paired static-hold collection and transparent reference-error summaries.

No reference values are generated. A trained assessor supplies measured manual
readings; synthetic angle tests must be reported separately from participant data.
"""
from collections import defaultdict
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import uuid

import numpy as np


JOINTS = tuple(f'{side}_{suffix}' for side in ('L', 'R') for suffix in (
    'Elbow_Flex', 'Knee_Flex', 'Shoulder_Flex', 'Shoulder_Abd', 'Hip_Flex',
    'Ankle_Dorsi_Plantar'))
VIEWS = ('front', 'left-side', 'right-side', 'oblique')
FIELDS = ('hold_id', 'participant', 'joint', 'view', 'reference_method',
          'reference_deg', 'software_deg', 'valid_frames', 'total_frames',
          'valid_fraction', 'iqr_deg', 'drift_deg', 'rejection_reason',
          'recorded_utc', 'coordinate_frame', 'min_cutoff', 'beta', 'd_cutoff',
          'width', 'height')


def validate_metadata(participant, joint, view, reference, reference_method):
    if not participant.strip() or not reference_method.strip():
        raise ValueError('Participant ID and reference method must be nonempty')
    if joint not in JOINTS or view not in VIEWS:
        raise ValueError('Unsupported joint or view')
    if not math.isfinite(reference) or not -180 <= reference <= 180:
        raise ValueError('Reference must be finite degrees in [-180, 180]')


class HoldCollector:
    """Bounded, timestamp-driven static sample window, with settling exclusion.

    Invalid measurements stay in the denominator. Low coverage, too few valid
    frames, large interquartile spread or half-window median drift reject a hold.
    Valid observations must also cover the window without gaps over max_gap;
    a stalled camera must not turn a few sparse observations into a stable hold.
    Thresholds are data-quality heuristics, not clinical accuracy thresholds.
    Callback work is small and performs no disk I/O inside the display loop.
    """
    def __init__(self, joint, seconds=2, settling=1, min_coverage=.8,
                 max_iqr=3, max_drift=3, max_samples=10000, max_gap=.25):
        if joint not in JOINTS:
            raise ValueError('Unsupported joint')
        for name, value in (('seconds', seconds), ('max_iqr', max_iqr),
                            ('max_drift', max_drift), ('max_gap', max_gap)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if not math.isfinite(settling) or settling < 0 or seconds + settling > 60:
            raise ValueError('Settling must be nonnegative; total hold must be at most 60 seconds')
        if not math.isfinite(min_coverage) or not 0 < min_coverage <= 1:
            raise ValueError('min_coverage must be in (0, 1]')
        if not isinstance(max_samples, int) or max_samples < 5:
            raise ValueError('max_samples must be an integer >= 5')
        self.joint, self.seconds, self.settling = joint, seconds, settling
        self.min_coverage, self.max_iqr, self.max_drift = min_coverage, max_iqr, max_drift
        self.max_samples = max_samples
        self.max_gap = max_gap
        self.start = self.last_time = None
        self.samples = []
        self.completed = False

    def __call__(self, record):
        timestamp = float(record['capture_time'])
        if not math.isfinite(timestamp) or (self.last_time is not None and timestamp <= self.last_time):
            raise ValueError('Hold capture timestamps must be finite and strictly increasing')
        self.last_time = timestamp
        if self.start is None:
            self.start = timestamp
        elapsed = timestamp - self.start
        if elapsed >= self.settling + self.seconds:
            self.completed = True
            return False
        if elapsed >= self.settling:
            value = record['metrics'].get(self.joint)
            value = float(value) if value is not None else None
            if value is not None and (not math.isfinite(value) or not -180 <= value <= 180):
                value = None
            self.samples.append((timestamp, value, record.get('reasons', {}).get(self.joint, '')))
            if len(self.samples) >= self.max_samples:
                return False
        return True

    def summary(self):
        valid = [value for _, value, _ in self.samples if value is not None]
        total = len(self.samples)
        fraction = len(valid) / total if total else 0
        median = float(np.median(valid)) if valid else None
        iqr = float(np.percentile(valid, 75) - np.percentile(valid, 25)) if valid else None
        midpoint = (self.start + self.settling + self.seconds/2) if self.start is not None else 0
        first = [v for t, v, _ in self.samples if v is not None and t < midpoint]
        second = [v for t, v, _ in self.samples if v is not None and t >= midpoint]
        drift = float(abs(np.median(first) - np.median(second))) if first and second else None
        reasons = []
        if not self.completed:
            reasons.append('incomplete hold')
        if len(valid) < 5:
            reasons.append('fewer than 5 valid frames')
        if fraction < self.min_coverage:
            reasons.append('low valid-frame coverage')
        if iqr is not None and iqr > self.max_iqr:
            reasons.append('unstable angle spread')
        if drift is None or drift > self.max_drift:
            reasons.append('unstable or missing half-window medians')
        if self.start is not None:
            window_start = self.start + self.settling
            valid_times = [t for t, value, _ in self.samples if value is not None]
            boundaries = [window_start, *valid_times, window_start+self.seconds]
            gap = max(np.diff(boundaries))
            if gap > self.max_gap + 1e-9:
                reasons.append(f'insufficient temporal coverage (max gap {gap:.3f}s)')
        return {'software_deg': median, 'valid_frames': len(valid), 'total_frames': total,
                'valid_fraction': fraction, 'iqr_deg': iqr, 'drift_deg': drift,
                'rejection_reason': '; '.join(reasons)}

    def row(self, *, participant, view, reference_deg, reference_method, processor,
            width, height):
        validate_metadata(participant, self.joint, view, reference_deg, reference_method)
        return {'hold_id': uuid.uuid4().hex, 'participant': participant,
                'joint': self.joint, 'view': view, 'reference_method': reference_method,
                'reference_deg': reference_deg, **self.summary(),
                'recorded_utc': datetime.now(timezone.utc).isoformat(),
                'coordinate_frame': processor.coordinate_frame,
                'min_cutoff': processor.filter.min_cutoff, 'beta': processor.filter.beta,
                'd_cutoff': processor.filter.d_cutoff, 'width': int(width), 'height': int(height)}


def check_csv_header(path):
    """Validate an existing nonempty CSV before camera acquisition or appending."""
    path = Path(path)
    nonempty = path.exists() and path.stat().st_size > 0
    if nonempty:
        with path.open(newline='', encoding='utf-8') as source:
            if next(csv.reader(source), None) != list(FIELDS):
                raise ValueError('Existing CSV has an incompatible header')
    return nonempty


def append_hold(path, row):
    """Append one hold; reject incompatible CSV schemas without overwriting data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nonempty = check_csv_header(path)
    with path.open('a', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        if not nonempty:
            writer.writeheader()
        writer.writerow(row)


def analyze_rows(rows):
    """Report per-joint/view errors across valid holds, never across video frames.

    Rejected holds remain counted. Incomplete/low-coverage rows cannot contribute
    to MAE. No claim of statistical independence or clinical acceptance is made.
    """
    groups = defaultdict(list)
    rejected = []
    seen = set()
    configurations = {}
    reference_methods = {}
    for number, row in enumerate(rows, start=2):
        try:
            hold_id = row['hold_id'].strip()
            if not hold_id or hold_id in seen:
                raise ValueError('Hold IDs must be nonempty and unique')
            seen.add(hold_id)
            reference = float(row['reference_deg'])
            validate_metadata(row['participant'], row['joint'], row['view'], reference,
                              row['reference_method'])
            software = (None if row['software_deg'] is None or row['software_deg'] == ''
                        else float(row['software_deg']))
            valid, total = int(row['valid_frames']), int(row['total_frames'])
            if total < 0 or not 0 <= valid <= total:
                raise ValueError('Invalid frame counts')
            fraction = valid / total if total else 0
            stored_fraction = float(row['valid_fraction'])
            if not math.isfinite(stored_fraction) or abs(fraction - stored_fraction) > 1e-6:
                raise ValueError('Coverage does not match frame counts')
            reason = row['rejection_reason'].strip()
            if row['coordinate_frame'] not in ('body', 'camera'):
                raise ValueError('Invalid coordinate frame')
            parameters = tuple(float(row[key]) for key in ('min_cutoff', 'beta', 'd_cutoff'))
            if (not all(math.isfinite(p) for p in parameters)
                    or parameters[0] <= 0 or parameters[1] < 0 or parameters[2] <= 0):
                raise ValueError('Invalid filter configuration')
            if any(int(row[key]) <= 0 for key in ('width', 'height')):
                raise ValueError('Invalid image dimensions')
            for key in ('iqr_deg', 'drift_deg'):
                spread = None if row[key] is None or row[key] == '' else float(row[key])
                if spread is None or not math.isfinite(spread) or not 0 <= spread <= 3:
                    reason = reason or 'unstable or missing static-hold summary'
            if software is None or not math.isfinite(software) or not -180 <= software <= 180:
                reason = reason or 'missing/nonfinite software angle'
            if valid < 5 or fraction < .8:
                reason = reason or 'insufficient valid-frame coverage'
            if reason:
                rejected.append({'hold_id': hold_id, 'joint': row['joint'],
                                 'view': row['view'], 'reason': reason})
            else:
                config = (row['coordinate_frame'], *parameters, int(row['width']), int(row['height']))
                group = (row['joint'], row['view'])
                method = row['reference_method'].strip()
                if group in reference_methods and reference_methods[group] != method:
                    raise ValueError('Mixed reference methods within a joint/view; use separate CSV studies')
                reference_methods[group] = method
                if group in configurations and configurations[group] != config:
                    raise ValueError('Mixed configurations within a joint/view; use separate CSV studies')
                configurations[group] = config
                groups[(row['joint'], row['view'])].append(
                    (reference, software, row['participant'], fraction))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Invalid CSV row {number}: {exc}') from exc
    summaries = []
    accepted_joints = set()
    for (joint, view), pairs in sorted(groups.items()):
        reference = np.array([p[0] for p in pairs])
        software = np.array([p[1] for p in pairs])
        error = software - reference
        accepted_joints.add(joint)
        summaries.append({'joint': joint, 'view': view, 'holds': len(pairs),
                          'reference_method': reference_methods[(joint, view)],
                          'participants': len({p[2] for p in pairs}),
                          'mae_deg': float(np.mean(abs(error))),
                          'bias_deg': float(np.mean(error)),
                          'rmse_deg': float(np.sqrt(np.mean(error**2))),
                          'max_absolute_error_deg': float(np.max(abs(error))),
                          'reference_min_deg': float(np.min(reference)),
                          'reference_max_deg': float(np.max(reference)),
                          'configuration': dict(zip(('coordinate_frame', 'min_cutoff', 'beta',
                                                     'd_cutoff', 'width', 'height'),
                                                    configurations[(joint, view)])),
                          'mean_valid_fraction': float(np.mean([p[3] for p in pairs]))})
    covered = {'elbow': any(j.endswith('_Elbow_Flex') for j in accepted_joints),
               'knee': any(j.endswith('_Knee_Flex') for j in accepted_joints),
               'shoulder_or_hip_flexion': any(j.endswith(('_Shoulder_Flex', '_Hip_Flex'))
                                            for j in accepted_joints)}
    return {'schema_version': 1, 'accepted_holds': sum(s['holds'] for s in summaries),
            'rejected_holds': len(rejected), 'groups': summaries, 'rejections': rejected,
            'required_joint_categories': covered, 'required_scope_present': all(covered.values()),
            'scope': 'Static paired holds against user-supplied measured references; '
                     'category presence alone does not establish adequate clinical validation',
            'limitations': ['Manual readings have assessor/alignment uncertainty',
                            'Static holds do not establish dynamic accuracy or filter phase lag',
                            'No fixed MAE acceptance threshold; report observed errors and coverage',
                            'Reference ranges, sample counts and views must be assessed with the protocol']}


def analyze_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != list(FIELDS):
            raise ValueError('CSV header must match the supplied validation template')
        return analyze_rows(reader)


def markdown_report(report):
    lines = ['# Paired static-hold accuracy results', '', report['scope'], '',
             f"Accepted holds: {report['accepted_holds']}; rejected holds: {report['rejected_holds']}.", '',
             '| Joint | View | Holds | Participants | Reference range (deg) | MAE (deg) | Bias (deg) | RMSE (deg) | Valid frames |',
             '| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    for row in report['groups']:
        lines.append(f"| {row['joint']} | {row['view']} | {row['holds']} | {row['participants']} | "
                     f"{row['reference_min_deg']:.1f} to {row['reference_max_deg']:.1f} | "
                     f"{row['mae_deg']:.2f} | {row['bias_deg']:.2f} | {row['rmse_deg']:.2f} | "
                     f"{row['mean_valid_fraction']:.1%} |")
    if not report['groups']:
        lines += ['', 'No accepted measurements. Accuracy remains unmeasured.']
    lines += ['', 'Required joint categories: ' + ', '.join(
        f"{key}={'present' if value else 'missing'}" for key, value in report['required_joint_categories'].items()),
        '', 'Category presence is a coverage check, not a clinical pass criterion.']
    if report['groups']:
        lines += ['', '## Reference and measurement setup', '',
                  '| Joint | View | Reference method | Frame | Cutoff / beta / derivative cutoff | Resolution |',
                  '| --- | --- | --- | --- | --- | --- |']
        for row in report['groups']:
            config = row['configuration']
            method = row['reference_method'].replace('|', '/').replace('\n', ' ').replace('\r', ' ')
            lines.append(f"| {row['joint']} | {row['view']} | {method} | {config['coordinate_frame']} | "
                         f"{config['min_cutoff']} / {config['beta']} / {config['d_cutoff']} | "
                         f"{config['width']}x{config['height']} |")
    lines += ['', '## Limitations', '']
    lines.extend('- ' + text for text in report['limitations'])
    if report['rejections']:
        lines += ['', '## Rejected holds', '', '| Hold ID | Joint | View | Reason |', '| --- | --- | --- | --- |']
        for row in report['rejections']:
            reason = row['reason'].replace('|', '/').replace('\n', ' ')
            hold_id = row['hold_id'].replace('|', '/').replace('\n', ' ')
            lines.append(f"| {hold_id} | {row['joint']} | {row['view']} | {reason} |")
    return '\n'.join(lines) + '\n'
