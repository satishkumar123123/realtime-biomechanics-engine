# Final regression audit

Reviewed 2 October 2026 against GitHub baseline
`fe897d40d4e1e0518dc565f318daf89be45d95d7`.

**Result:** the strict automated regression suite passes after the fixes below.
This establishes the tested software behavior, not universal driver safety,
clinical accuracy, or a human webcam desktop throughput guarantee.

## Reproduced defects and fixes

| Finding | Reproduction / impact | Fix |
| --- | --- | --- |
| Finite-vector norm overflow | A vector with three `1.7e308` components emitted a NumPy overflow warning, failing strict execution | Reject an unrepresentable norm as unavailable with scoped arithmetic guards |
| Malformed visibility accepted through deprecated conversion | `(33, 1)` confidence arrays triggered NumPy scalar-conversion deprecations in geometry and bone checks | Require scalar confidence entries; malformed confidence returns unavailable |
| Signed branch-cut flicker | Opposite-ray perturbations of `+/-0.0001` produced alternating angles near `+/-180` | Reject observations within 0.5 degrees of the antiparallel branch cut; exact opposition retains the documented +180 tie |
| Whole-frame missing observation failed | Passing `None` after a `(33, 3)` filter sample raised a shape-change exception | Invalidate the existing components, clear their motion history and initialize on valid recovery; duplicate/backward timestamps still invalidate missing data |
| Failure retained a native backend | Keeping `capture.error` alive also retained the failed camera through exception traceback frames | Detach worker/cleanup traceback chains while preserving exception type, message and cause details; weak-reference regressions verify collection |
| Interrupted startup double-released the camera | Simulated `Thread.start()` launching and then raising lost the live thread handle and released the camera from the wrong thread | Signal shutdown, join a launched worker with a bound, retain its handle if still alive and leave release exclusively to it |
| Strict Python 3.12 runtime import failed | Baseline `pytest -W error` produced two failing tests from protobuf native-container deprecations and the partially initialized MediaPipe extension | Scope compatibility handling to the two exact known import messages; benchmark version reporting reads package metadata without importing the native model |

Additional hardening explicitly rejects HighGUI calls outside Python's main
thread. Event-pump errors still unwind camera, model and display cleanup. Headless
benchmark and validation tests now prohibit `namedWindow`, `imshow`, `waitKey`,
`destroyWindow` and `destroyAllWindows` throughout their full shared pipeline.

The branch-cut margin is an uncertainty heuristic, not temporal angle unwrapping.
Large motion/noise across that margin can still change the principal angle's sign;
the engine does not silently relabel such motion as a clinical rotation.

## Verification

| Check | Observed result |
| --- | --- |
| Baseline strict pytest | 2 failed, 103 passed, 70 subtests passed |
| Final strict pytest | **121 passed, 95 subtests passed; 0 failures, skips or reported test warnings** |
| Camera worker inventory after the full suite | **0 live `camera-capture` threads** |
| Repeated lifecycle, slow consumer, disconnect, blocked read and release failure | Passed; blocked reads retain bounded-stop behavior and exclusive release ownership |
| Bilateral geometry | Neutral, ROM endpoints, signs, yaw/roll/pitch transforms, confidence threshold, collapsed/nonfinite geometry, overflow, projection and branch-cut regressions passed |
| Filter behavior | Noise reduction, adaptive ramp/step, duplicate/backward time, per-component and whole-frame missing-data recovery passed |
| Real local model | Blank-frame inference plus fresh-process nonempty landmark-packet roundtrip passed with `-W error` |
| Benchmark/validation headless paths | Full rendering and numeric pipeline tests passed without HighGUI; recorded replay tests passed |
| Empty reference CSV CLI report | Correctly returned exit 2 and wrote JSON/Markdown stating accuracy is unmeasured |
| Dependency consistency / patch whitespace | `pip check` passed; `git diff --check` clean |

Strict execution uses:

```bash
python -m pytest -q -W error
```

The final audit also ran pytest in an interpreter with `-W error` and checked
`threading.enumerate()` immediately afterward. Its output was:

```text
121 passed, 95 subtests passed in 5.01s
Live camera workers after full suite: 0
```

**Warning policy:** MediaPipe 0.10.21 requires protobuf 4.x. On Python 3.12, two
known protobuf native map-container import deprecations are handled locally in
`load_mediapipe()`; see [upstream issue 15077](https://github.com/protocolbuffers/protobuf/issues/15077).
Tests verify that the filter is restored and unrelated deprecations still raise.
There are no blanket pytest warning ignores and inference warnings are not hidden.
Native MediaPipe/Abseil startup log messages remain visible in CLI runs; a clean
Python warning count does not mean that native libraries emit no diagnostic logs.

## Headless measurements

Host: Linux 6.18.44 x86_64 / glibc 2.39, Intel Xeon Platinum 8573C, 9 reported
logical CPUs and an 8-core cgroup quota. Python 3.12.14, NumPy 1.26.4,
OpenCV 4.11.0, MediaPipe 0.10.21 and protobuf 4.25.9. Requested input:
640x480 at 60 FPS. Both full-pipeline runs exclude 30 warmup frames and measure
300 subsequent frames; they use the actual HUD renderer and a headless sink.

| Workload | FPS | Inference mean / P95 ms | Pipeline mean / P95 ms | Mailbox superseded |
| --- | ---: | ---: | ---: | ---: |
| Real BlazePose Full, blank synthetic input | 58.13 | 12.714 / 19.787 | 17.428 / 30.911 | 8 |
| Analytic pose, all 12 numeric metrics | 59.52 | Mock; not a model result | 4.210 / 5.547 | 0 |

Separate 300-frame synthetic capture check: **60.00 FPS**, mailbox access mean
**0.00294 ms**, P99 **0.00439 ms**, maximum **0.12568 ms**, zero superseded frames.
The measured lock/mailbox maximum is below 1 ms on this run, not an OS scheduling
guarantee. It excludes camera waiting and frame copying.

Raw reports:

- [Real model on blank headless frames](../benchmarks/results/regression_blank_headless.json)
- [Mock numeric pipeline](../benchmarks/results/regression_mock_numeric.json)
- [Capture mailbox check](../benchmarks/results/regression_capture_mailbox.json)

Reproduction commands:

```bash
python -W error benchmark.py --source synthetic --frames 300 --output benchmarks/results/regression_blank_headless.json
python -W error benchmark.py --mock-pose --frames 300 --output benchmarks/results/regression_mock_numeric.json
python -W error -m tests.benchmark_capture --synthetic --frames 300
python -W error validation.py report --csv validation/template.csv --json validation/results/empty.json --markdown validation/results/empty.md
```

The last command intentionally returns 2 for missing reference data. Blank input
has zero detected poses; the mock supplies analytic landmarks. Neither run proves
human detection/landmark inference throughput, actual desktop presentation cost,
or clinical error. Both correctly report representative desktop performance as
**not established**.

## Remaining evidence and limits

- This execution host has no physical webcam or supported desktop test session.
  Windows/macOS/Fedora driver teardown and native window behavior need hardware
  runs; mocks cannot establish universal platform safety.
- Python cannot safely cancel a permanently blocked native camera read. Stop is
  bounded, reports `TimeoutError` and preserves the worker handle; the worker can
  release only after the read returns. Backend release failure is surfaced, not
  presented as guaranteed cleanup. Camera opening and native GUI/model calls can
  also depend on driver responsiveness.
- Real detected-human desktop performance and paired elbow/knee/shoulder-or-hip
  reference measurements remain the assignment's two outstanding evidence gaps.
  Follow [the Windows measurement guide](WINDOWS_VALIDATION.md). No synthetic
  benchmark or geometric test is substituted for those measurements.
