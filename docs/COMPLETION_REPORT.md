# Assignment follow-up: implementation and evidence

> This records the first completion pass. See [Assignment Check](ASSIGNMENT_CHECK.md)
> for the subsequent validation fixes, separate model module and latest test count.

Date: 1 October 2026. Baseline: GitHub commit
`e7b50abd78d29b6f4dbd80a045ec23c4c091e60f`.

The software gaps identified against the original assignment have been addressed.
Actual human webcam throughput and paired manual-reference accuracy remain pending
physical measurements. This environment exposes no webcam device or participant
reference readings; software fixtures cannot supply that missing evidence.

## Changes and reasoning

| Gap | Delivered change | Verification / remaining limit |
| --- | --- | --- |
| Camera-fixed projections misclassify a turned subject | Default desktop body frame from reliable bilateral hips/shoulders, with orthonormal left/down/posterior axes | Signed flexion and separate abduction survive synthetic yaw through 90/180/270 degrees and an arbitrary rigid rotation; actual model depth/yaw accuracy remains unvalidated |
| Neutral abduction biased by different hip/shoulder widths | Body-mode coronal shoulder reference follows torso-down | Neutral and 45-degree tests use unequal shoulder/hip spans; camera mode retains the original formula for compatibility |
| Unreliable body-frame estimates | Reject missing/collapsed/contradictory/nearly collinear torso anchors | Shoulder/hip show `Torso frame unavailable`; valid elbow/knee/ankle triplets remain available |
| Default filter weak at meter-scale speeds | App/benchmark/validation beta=5, min/d cutoffs=1 Hz | Analytical ramp lag drops from 155.88 to 9.95 ms; noise variance reduction changes from 95.03% to 89.84%; human joint-angle phase lag remains unmeasured |
| Proposed validation protocol had no acquisition/report tooling | `validation.py collect/report`, bounded static-hold collector, empty CSV template, per-joint/view error summaries | Tests verify neutral zero, median/error calculations, rejection counts, duplicate IDs, configuration consistency, CSV/CLI behavior and float camera dimensions |
| Blank/model-mock FPS could be mistaken for submission evidence | Benchmark pose/numeric/per-metric coverage, detected-pose-only timings and `--require-human` checks | Blank, mock, headless, short, low-coverage and sub-30-FPS reports do not pass the performance evidence policy |
| Replay distorted aspect ratios | Letterbox recorded inputs | Nonmatching aspect-ratio regression preserves proportions and output dimensions |
| Model-download instructions missing | Explain bundled Full model/detector and offline inference; asset-check command | Installed MediaPipe 0.10.21 contains Full landmark asset (6,440,512 bytes) and detector (2,959,046 bytes) |
| README implied fixed MAE goals | Remove illustrative error ranges; state that the assignment has no fixed error threshold | Actual error tables stay explicitly unmeasured until reference holds are collected |

The camera remains fixed and singular. Body orientation comes from the same
model's inferred 3D landmarks, not a second camera, depth sensor or remote API.
The changed torso frame is a geometric anatomical proxy, not calibrated clinical
pelvic/scapular axes. Four torso anchors are an additional reliability requirement
for projected shoulder/hip values; side/oblique positioning is documented.

## Verification

| Check | Result |
| --- | --- |
| Full `python -m pytest -q` | **100 tests and 67 subtests passed**, zero failures/skips |
| Dependency consistency | `python -m pip check`: no broken requirements |
| Whitespace check | `git diff --check`: clean |
| Runtime | Isolated Python 3.12.14 environment with requirements-dev.txt |
| Native model smoke test | Real local MediaPipe on a blank image passes |
| Rendered HUD inspection | Body-mode title, all twelve rows, torso warnings and settling notice fit at 640x480 |
| Static collector I/O behavior | Disk writes occur after the shared loop closes camera/model/display |
| Data protection checks | Incompatible CSV rejected before camera acquisition; overlapping report/input paths rejected; participant data ignored by Git |

Two upstream protobuf deprecation warnings occur at model import, with no failed
tests. Physical camera/desktop interaction on Windows/macOS is not verified here.
Camera teardown remains bounded but cannot forcibly cancel a permanently wedged
native read; that driver limitation is documented in the original audit.

## Current measured software benchmarks

The runs below were executed sequentially after the final geometry changes,
without running another project benchmark/test job concurrently. They have
30 warmup and 300 measured frames, requested 640x480 at 60 FPS, body frame and
One-Euro 1 Hz / beta=5 / derivative 1 Hz.

Hardware: AMD EPYC 9V74 80-Core Processor; container reports 9 logical CPUs and an
8-core quota. Linux 6.18.44 x86_64/glibc 2.39, Python 3.12.14, MediaPipe 0.10.21,
OpenCV 4.11.0, NumPy 1.26.4. This is the execution host, not a documented physical
desktop webcam test machine.

| Workload | Achieved FPS | Inference mean / P95 (ms) | Pipeline mean / P95 (ms) | Pose / numeric coverage | Superseded frames |
| --- | ---: | ---: | ---: | --- | ---: |
| Real MediaPipe, blank input, headless HUD | 59.16 | 13.472 / 16.453 | 18.289 / 26.899 | 0% / 0% | 2 (0.66%) |
| Analytic pose mock, all twelve numeric measurements, headless HUD | 59.64 | 0.001 / 0.001, mock only | 3.764 / 5.099 | 100% / 100% | 0 |

Raw reports include min/max and final-120-frame statistics:

- [Real-model blank report](../benchmarks/results/orientation_synthetic_headless.json)
- [Mock numeric-path report](../benchmarks/results/orientation_mock_numeric.json)
- [Filter signal experiment](../benchmarks/results/orientation_filter_response.json)

The real-model run exercises the no-person path. The mock run exercises body-frame
math/filter/render integration but cannot establish landmark-model performance.
Both correctly return `submission_performance.passed=false`. These measurements
do not satisfy the detected-human desktop >=30 FPS evidence requirement.

Inference time includes only `pose.process`. Pipeline time starts at host camera
or replay read completion and ends at HUD/display sink completion. Headless runs
exclude OS window presentation. Sensor exposure and actual screen photons are not
measured. Filter signal lag is distinct from both compute timings.

## Finish the physical evidence on the target machine

1. Run the person-containing webcam with the visible desktop HUD for at least
   thirty measured seconds; repeat three times with identical settings. Save
   reports, CPU/OS, negotiated resolution, coverage, inference/pipeline mean/P95
   and mailbox stats. `--require-human` returns 2 when evidence checks fail.
2. Collect actual assessor goniometer readings against software static-hold
   medians for elbow, knee and at least shoulder or hip flexion/extension. Include
   neutral/several comfortable bends, repeat holds, both sides where reliable,
   documented views and rejected observations. Generate per-joint/view MAE,
   bias/RMSE and reference ranges; discuss assessor uncertainty and monocular limits.
3. Add anonymized resulting tables/reports to the README. Do not replace pending
   tables with mock/synthetic errors or illustrative target numbers. Category
   presence in the report is a scope check, not a clinical acceptance verdict.

See [Windows verification guide](WINDOWS_VALIDATION.md) for executable setup,
benchmark and paired-hold commands. No additional libraries or cameras are needed.
