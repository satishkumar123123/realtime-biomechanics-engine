"""Reproduce static-noise and meter-scale ramp checks; no human accuracy claims."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

# Allow `python benchmarks/filter_response.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.filter import OneEuroFilter


def experiment():
    times = np.arange(1800)/60
    noisy = np.random.default_rng(42).normal(0, .02, (1800, 33, 3))
    results = []
    for beta in (0, .007, 5):
        smoother = OneEuroFilter(beta=beta)
        filtered = np.array([smoother(x, t) for x, t in zip(noisy, times)])
        reduction = 100*(1-np.var(filtered[300:])/np.var(noisy[300:]))
        smoother.reset()
        ramp_times = np.arange(600)/60
        raw = 3*ramp_times
        filtered_ramp = np.array([smoother(x, t) for x, t in zip(raw, ramp_times)])
        lag = float(np.mean(raw[120:]-filtered_ramp[120:])/3*1000)
        results.append({'beta': beta, 'stationary_variance_reduction_percent': float(reduction),
                        'steady_ramp_coordinate_lag_ms': lag})
    return {'schema_version': 1, 'type': 'Analytical signals, not human motion or joint-angle validation',
            'rate_hz': 60, 'min_cutoff_hz': 1, 'd_cutoff_hz': 1, 'seed': 42,
            'noise_std_m': .02, 'noise_samples': 1800, 'noise_warmup_samples': 300,
            'ramp_velocity_m_per_s': 3, 'ramp_samples': 600, 'ramp_warmup_samples': 120,
            'application_default_beta': 5, 'results': results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    report = experiment()
    content = json.dumps(report, indent=2, allow_nan=False)+'\n'
    print(content)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
