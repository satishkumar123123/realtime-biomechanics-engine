# Original assignment compliance check

Reviewed 1 October 2026 against the full problem statement. Baseline GitHub commit:
`998703001b9a2e322ae10cb2a712da2071aafc56`.

**Assessment:** the implementation covers the required software scope. Complete
assignment compliance still requires actual human desktop performance and paired
reference accuracy results. Tools and a protocol are available for both; the
execution environment has no webcam/participant reference readings from which to
produce them. Synthetic and mock measurements are not substituted for that evidence.

## Requirement-by-requirement findings

| Requirement | Status | Code / evidence |
| --- | --- | --- |
| One fixed monocular webcam, no extra/depth/multiview cameras | Implemented | `core/capture.py` owns one source; replay is an alternative benchmark source |
| Fully local pose model and justified choice | Implemented | `core/pose.py` constructs bundled BlazePose Full; README explains metric 3D landmarks, native CPU inference and model configuration |
| Decoupled camera, model, filtering, biomechanics and UI responsibilities | Implemented | Dedicated capture, pose, filter and biomechanics modules; `main.py` orchestrates model calls and owns rendering/display |
| Bilateral elbow, knee, shoulder flexion/abduction, hip and ankle | Implemented | Twelve values from the required landmark families, with explicit unreliable states |
| Anatomical neutral-zero conventions | Implemented as documented geometric proxies | Straight elbow/knee, arm at side, standing hip and right-angle ankle; signed shoulder/hip and ankle; elbow/knee hyperextension and clinical ankle axes remain documented limits |
| Subject may turn relative to the fixed camera | Implemented geometrically | Default torso frame follows modeled orientation; missing torso anchors reject affected angles; side/oblique positioning and camera compatibility mode documented |
| Explain planes, coordinates, positioning and calibration | Documented | README describes world 3D versus normalized overlay coordinates, body coronal/sagittal planes, torso references and absence of clinical anatomical calibration |
| Handle unreliable landmarks and numerical edge cases | Implemented/tested | Visibility >0.65, finite checks, epsilon guards, projection conditioning, frame validity and loss/reset behavior |
| Mitigate depth jitter/perspective error | Implemented/tested on analytical data | Vectorized One-Euro filter, body-plane projections, optional calibrated bone checks; human dynamic tuning remains unvalidated |
| Minimal desktop feed, pose overlay and metrics | Implemented | OpenCV HUD, diagnostics, keyboard controls and cleanup; hardware-independent renderer/integration tests pass |
| Inference latency measured separately | Implemented; limited measured evidence | `pose.process` timing; existing real-model blank mean/P95 13.472/16.453 ms |
| End-to-end latency, FPS, hardware, OS and configuration | Implemented; limited measured evidence | Host read-completion to rendering/display sink, full-run and rolling statistics; real desktop run pending |
| At least 30 FPS with a human on documented desktop hardware | **Not established** | Existing 59.16 FPS run is blank/headless; strict benchmark checks prevent claiming a human desktop pass |
| Validate elbow, knee and shoulder/hip against a reference | **Actual results pending** | `validation.py collect/report` and assessor protocol exist; no participant MAE is claimed |
| Install, model download, dependencies, execution, decisions and limitations | Documented | README, bundled-model check, Windows guide and audit reports |
| Optional repetition/posture/velocity extensions and demo video | Not required | Omission does not prevent satisfying the mandatory scope |

The assignment has no fixed MAE threshold. Geometric unit-test accuracy and
analytical filter experiments are not clinical landmark-accuracy results. Normal
ROM reference maxima are not used as hard clipping limits, and documented proxy
differences must accompany the measured validation results.

## Additional issues found and fixed in this pass

1. **Sparse samples could pass static-hold validation after a camera stall.**
   Five identical valid frames split between the beginning and end of a two-second
   window previously had 100% frame coverage, zero spread and zero drift despite
   a long unobserved interval. The collector now rejects gaps over 0.25 seconds
   between valid observations, including unobserved window edges. This is an
   acquisition quality heuristic, not an accuracy/FPS acceptance threshold.
2. **Accuracy reports omitted the actual reference method.** JSON and Markdown
   now preserve each joint/view's reference method; Markdown also displays frame,
   filter settings and resolution. Different reference methods cannot silently be
   pooled into one joint/view error statistic.
3. **Model construction shared the UI module.** It now resides in `core/pose.py`;
   the runner still profiles each model call separately. Full model parameters,
   lazy import and local inference behavior are preserved and tested.

## Verification

- Complete suite: **105 tests and 70 subtests passed**, zero failures/skips.
- Real MediaPipe blank-frame smoke and model-configuration tests pass.
- New regressions reject camera stalls, contiguous occlusion and missing window
  edges; a continuous 30 FPS static hold passes.
- Reference method/configuration survive report generation; mixed methods fail
  with an actionable error.
- `python -m pip check`: no broken requirements. `git diff --check`: clean.
- Two upstream protobuf deprecation warnings remain; they do not fail tests.

The existing [software benchmark evidence](COMPLETION_REPORT.md) is unchanged in
scope. It documents the execution host and synthetic/headless results, rather than
a physical-camera desktop performance claim. Native camera/display drivers and
permanently blocked driver reads cannot be exhaustively validated with mocks.

## Remaining actions requiring physical measurements

1. Follow [Windows validation steps](WINDOWS_VALIDATION.md) to run three
   person-containing webcam/visible-desktop benchmarks. Report achieved FPS,
   inference/pipeline mean and P95, coverage, resolution, hardware and OS.
2. Collect real paired reference holds for elbow, knee and shoulder or hip
   flexion/extension; generate the accuracy report and add anonymized measured
   tables with procedure, sample counts, rejected observations and limitations.

These are the two remaining mandatory evidence gaps. Adding optional metrics or
more mock tests cannot close them.
