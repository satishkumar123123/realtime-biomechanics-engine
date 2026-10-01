# Executive Audit Report

**Date:** 1 October 2026

**Repository:** `satishkumar123123/realtime-biomechanics-engine`

**Baseline reviewed:** `a0bac5c99cc4b9a0bebdbd8995d9cd4b10c4acbb`

**Scope:** every production module, both dependency files, all tests, README and benchmark evidence. Findings below describe the corrected working tree delivered with this report.

## Executive assessment

**Conditional pass for the implemented software architecture; clinical accuracy and detected-human desktop performance remain unverified.** The repository is a coherent single-camera research prototype. It has a bounded latest-frame capture mailbox, real MediaPipe inference, explicit temporal filtering, twelve neutral-zero geometric measurements and a lightweight desktop HUD. It does not require depth hardware, multiple cameras, a browser wrapper or a remote inference service.

The audit corrected concrete geometry, stale-data, lifecycle, UI and diagnostic defects. The expanded suite passes in the existing runtime and a fresh isolated environment, including real MediaPipe blank-frame inference. No test-order failures were observed in three shuffled runs. These results support software correctness for exercised paths; they do not certify all physical drivers, human motion, clinical measurement accuracy or absolute leak freedom.

## Compliance matrix

| Criterion | Assessment after fixes | Evidence and qualification |
| --- | --- | --- |
| One fixed monocular webcam | Pass | Desktop opens one `VideoCaptureAsync` source. Synthetic/recorded modes are offline benchmark alternatives, not additional simultaneous cameras. |
| Modular ingestion/inference/filter/math/UI | Pass | Capture, filtering and geometry reside in separate core modules. `create_pose`, `PoseProcessor`, `HUDRenderer` and `OpenCVDisplay` have injectable boundaries in `main.py`; inference orchestration and UI share a file, not responsibilities. |
| Nonblocking capture and bounded mailbox | Pass | Camera I/O remains on one daemon worker; publication uses a short condition lock. Copying and downstream work occur outside it. Capacity is one latest frame, with intentional replacement. |
| Resource lifecycle | Conditional | Tested normal exit, initialization/model/display failure, disconnect, restart, thread-start failure and backend release failure. Twenty mock lifecycle cycles release every device once. A permanently blocked native read cannot be forcibly cancelled safely by this Python thread design. |
| Twelve bilateral measurements | Pass for specified geometry | Both sides include elbow, knee, shoulder flexion, shoulder abduction, hip flexion and signed ankle deviation. |
| Neutral-zero references | Pass for synthetic geometric conventions | Straight elbow/knee, standing hip, arm-at-side shoulder and right-angle ankle give zero. Known bends, signed motions and nominal ROM endpoints are tested. |
| Strict clinical anatomical interpretation | Partial | Camera planes, trunk reference rays and ankle-foot-index geometry are proxies. Signed elbow/knee hyperextension is unavailable. No subject anatomical calibration or paired clinical dataset exists. |
| Elbow 0-150 / knee 0-135 references | Endpoints pass; hard caps intentionally absent | Tests cover both nominal endpoints. Geometric output remains 0-180 rather than silently clipping true observations; nominal ROM varies by population and is not a universal measurement limit. |
| Visibility gate >0.65 | Pass after alignment | Engine, processor, wireframe and optional bone checker reject equality at 0.65; missing/nonfinite visibility also fails. This tightens the earlier inclusive threshold contract. |
| Zero/collinear/nonfinite safeguards | Pass for exercised cases | Epsilon length guard, finite checks, clamped cosine, projection conditioning, deterministic opposite-ray handling and overflow regression tests. |
| Temporal filtering | Pass algorithmically; motion tuning pending | Array-based filter, dynamic dt, mask/reset handling and missing-data recovery tested. Default beta is a starting value, not zero-lag tracking. |
| Separate inference and pipeline timings | Pass | `pose.process` timing is distinct from host read-completion-to-UI timing. Camera exposure and screen photons are outside the measurement boundary. |
| Rolling mean/P95 diagnostics | Pass after fixes | HUD includes inference mean/P95 and pipeline mean/P95 over bounded histories. Benchmark includes full-run statistics plus the final 120 post-warmup-frame window. |
| At least 30 FPS, target 60 | Demonstrated only for blank/headless workload | Post-fix real-MediaPipe run achieves 58.64 FPS. Detected-human webcam + visible desktop throughput remains unmeasured. |
| Test reliability | Pass within audit runs | 73 tests + 56 subtests; three shuffled stdlib runs also pass. Fresh-environment installation, dependency check and pytest pass. No physical webcam tests. |
| Submission documentation | Pass with explicit evidence gaps | Hardware, rationale, math, setup, benchmark definitions, clinical protocol and limitations are present. Target MAE/performance ranges are clearly separated from measured results. |

## Defects corrected during the audit

| Finding | Severity | Before | Correction and regression evidence |
| --- | --- | --- | --- |
| Opposite-ray signed-zero ambiguity | High | A straight-up shoulder could display -180 degrees as extension/adduction depending on signed zero. | Exact antiparallel rays now consistently use +180; bilateral shoulder regression test. Rotation direction at the opposite ray remains inherently ambiguous. |
| Ill-conditioned projected segments | High | Nearly perpendicular segments retained tiny in-plane components, allowing noise to dominate reported angles. | Reject projected/3D length ratios below 0.05 in either reference/moving ray. Tests cover near-normal shoulder and hip rays. The 5% guard is a heuristic requiring real-motion tuning. |
| Stale values on invalid data + duplicate timestamp | High | dt guard returned the old filter output before considering current invalid components. | Invalid components clear history and return NaN even when time does not advance; valid components hold. Processor now gates all coordinates of a nonfinite keypoint together. Recovery tests prevent mixed old/new coordinate histories. |
| Thread-start cleanup fault | Medium | Failed `Thread.start()` left an unstarted thread reference; subsequent `stop()` tried to join it and raised a second error. | Clear failed worker reference, release device and notify waiters. Stop/restart regression passes. |
| Backend release exception escaped daemon | Medium | `release()` failure could be an unhandled worker exception with no explicit cleanup-failure status. | Worker captures release errors, clears running/frame state, and `stop()` raises a meaningful OSError. Other ExitStack cleanup callbacks still run. A backend that refuses to release cannot be certified leak-free. |
| Native window-close query error | Medium | Some backends throw from `getWindowProperty` after the window has already been destroyed. | Treat that error as an exit key; close remains idempotent. Added Linux no-display preflight to avoid entering native GUI initialization without a desktop. |
| Incomplete P95/rolling outputs | Medium | HUD omitted inference P95; benchmark provided only aggregate run statistics. | Added HUD inference P95 and benchmark final-window mean/P95/FPS, with eviction/warmup exclusion tests. |
| Confidence-boundary mismatch | Contract alignment | Previous code accepted exactly 0.65, while the latest audit checklist asks for >0.65. | All relevant layers now enforce the strict boundary, and README/tests agree. |
| Tests could hide thread errors or scheduling delay | Low | Concurrent thread-start exceptions were not returned to assertions; slow-consumer test relied on a short timing assumption. | Futures propagate failures; progress assertion waits for the required sequence. Added repeated lifecycle, startup/release and UI regressions. |

Coordinate subtraction overflow is also guarded so invalid huge finite inputs yield unavailable angles without numerical warnings. Invalid `wait_for_frame` timeouts/sequences are rejected explicitly.

## Verification evidence

| Check | Result |
| --- | --- |
| Baseline pytest before edits | 60 tests and 56 subtests passed |
| Expanded pytest after fixes | 73 tests and 56 subtests passed |
| Shuffled complete suites, seeds 101 / 202 / 303 | 73 tests each; zero failures, errors or skips |
| Camera lifecycle stress | 20 starts/stops; every mock device released exactly once; joined worker reference cleared after each stop |
| Fresh virtual environment | `requirements-dev.txt` installed successfully |
| Fresh environment dependency consistency | `python -m pip check`: no broken requirements |
| Fresh environment complete pytest | 73 tests and 56 subtests passed, including real MediaPipe smoke test |
| HUD visual inspection | All 12 rows, strict-threshold warning, degree markers and inference/pipeline statistics remain readable at 640x480 |

Two upstream protobuf Python deprecation warnings occur during MediaPipe import. They are not failed tests; the current documented Python 3.10-3.12 setup was tested at Python 3.12.14. The dependency set is version bounded, not a fully pinned transitive lockfile or a validated Windows/macOS install matrix.

### Post-fix performance run

Command:

```bash
python benchmark.py --frames 300 --output benchmarks/results/audit_synthetic_headless.json
```

[Measured JSON report](../benchmarks/results/audit_synthetic_headless.json)

| Environment | Value |
| --- | --- |
| CPU | AMD EPYC 9V74 80-Core Processor; container reports 9 logical CPUs / 8-core quota |
| OS / Python | Linux 6.18.44 x86_64, glibc 2.39 / Python 3.12.14 |
| Model / dependencies | MediaPipe 0.10.21, complexity 1; OpenCV 4.11.0; NumPy 1.26.4 |
| Input / output | 640x480 synthetic blank frames paced at 60 FPS; headless real HUD rendering |
| Measurement | 30 warmup + 300 measured frames |
| Achieved FPS | **58.64**; final 120-frame window **59.60** |
| Mailbox | 3 superseded frames / 302 publication intervals; 3 overrun events; 0.99% |
| Detected poses | **0/300**; valid-coordinate filtering/angle path is not exercised by this real-model run |
| Driver/sensor drops | Unknown, not measurable from these OpenCV counters |

| Latency | Mean (ms) | Min (ms) | Max (ms) | P95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Inference | 13.659 | 11.747 | 43.022 | 16.080 |
| Host pipeline | 17.804 | 14.014 | 48.620 | 27.793 |

The final rolling inference mean/P95 is 13.186 / 15.124 ms; rolling pipeline mean/P95 is 16.240 / 20.048 ms. Blank input may exercise detection rather than the complete detected-person landmark workload. Mock numeric-path tests establish integration correctness but cannot establish real landmark-model speed or accuracy.

### Signal lag is separate from compute latency

[Analytical signal experiment](../benchmarks/results/audit_filter_response.json): seeded 0.02 m Gaussian noise at 60 Hz has a 95.03% stationary variance reduction with defaults. A 3 m/s analytical coordinate ramp over ten seconds, discarding the first two seconds, gives:

| Beta | Equivalent steady coordinate lag |
| ---: | ---: |
| 0 (fixed 1 Hz low pass) | 159.15 ms |
| 0.007 (default) | 155.88 ms |
| 5 (test tuning) | 9.95 ms |

This is an algorithm experiment, not an observed human trajectory or measured joint-angle phase lag. It demonstrates why a 17.8 ms compute pipeline does not imply 17.8 ms motion responsiveness. Tune beta on recorded human movement and report jitter and signal lag together; increasing beta can admit more noise.

## Residual gaps and acceptance conditions

1. **High: clinical accuracy is unverified.** Neutral offsets and synthetic tests do not establish clinical goniometry. Collect paired manual-goniometer holds with assessor blinding, per-joint/view coverage, bias, MAE and uncertainty. Requested MAE ranges remain targets. Camera-fixed axes, trunk references, signed hyperextension and the ankle proxy prevent an unconditional clinical-compliance claim.
2. **High: detected-human desktop throughput is unverified.** Run a person-containing recording and the target webcam with `--display`, using the documented warmup and three repeat runs. Record valid-pose coverage; passing a blank detector workload is insufficient evidence for >=30 FPS on real human landmark tracking.
3. **Medium: absolute thread/device leak freedom cannot be guaranteed for a wedged native backend.** The bounded join raises TimeoutError while a blocking native read may still own the device. Process isolation/driver-specific cancellation would be required for forced recovery, and should be designed/tested separately. Do not race `release()` against a native read from another thread.
4. **Medium: filter and projection reliability settings need motion validation.** Default beta=0.007 is intentionally retained but is weak at meter-scale velocities. The new 5% projection guard is conservative, not clinically calibrated. Bone-length checking is optional and not automatically enforced by the desktop application.
5. **Medium: physical UI/platform behavior is incompletely covered.** The rendered HUD was inspected, but visible interaction and device drivers were not tested on Windows/macOS or an actual webcam. Injected tests cannot prove absence of native driver or GUI failures on every platform.
6. **Scope clarification: nominal range caps are not enforced.** Elbow/knee geometric flexion spans 0-180, includes the requested 150/135-degree endpoints, and does not identify signed hyperextension. If an evaluator requires hard range caps, that requirement should be explicitly implemented as an unavailable/out-of-range quality status after appropriate calibration, not silently altering the measured angle.

CDC publishes age/sex-specific ROM references, including knee flexion values beyond 135 degrees. This supports treating nominal ranges as references rather than universal hard caps: [CDC Joint Range of Motion Study](https://archive.cdc.gov/www_cdc_gov/ncbddd/jointrom/index.html). This external study does not validate this application's accuracy.

## Recommendation

Accept the software modularity, single-camera constraint, measured host diagnostics and tested geometric/filter behavior with the stated qualifications. Require detected-human target-machine performance and a paired clinical dataset before claiming production clinical accuracy or complete satisfaction of real-world performance criteria. No tested software regression remains unresolved after the audit; the residual items above are evidence, modeling and native-backend limits rather than hidden passing claims.
