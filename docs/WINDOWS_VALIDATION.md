# Windows: run the app and finish the measured evidence

Use Python 3.12 and a single webcam in a fixed position. Run these commands from
the repository folder in the VS Code PowerShell terminal. Calling the virtual
environment's Python directly avoids changing PowerShell execution policy.

## Setup and app

For an existing checkout, first get the pushed changes with `git pull`. Then:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe main.py
```

The Full model installs with the pinned MediaPipe package. No separate download
or inference service is needed. Allow camera access for desktop applications in
Windows settings. If camera 0 is unavailable, try `main.py --camera 1`.

Keep your whole body/feet in view. Face forward for shoulder abduction; use a
side/oblique stance for flexion. In body mode both shoulders and hips must remain
reliable. If an exact side view hides those anchors, turn slightly toward the
camera until the torso warning clears. `r` resets filter history; `q`/ESC exits.

## Actual human desktop performance

Stand in view and perform representative comfortable movements while the HUD is
visible. Start one thirty-second measured run:

```powershell
.\venv\Scripts\python.exe benchmark.py --source webcam --display --seconds 30 --require-human --output benchmarks/results/webcam_run_1.json
```

Repeat with filenames `webcam_run_2.json` and `webcam_run_3.json`. Reports include
CPU/OS, model/filter configuration, resolution, actual FPS, inference and pipeline
latencies, coverage and superseded frames. A 30 FPS camera can satisfy the minimum;
requesting 60 does not guarantee that the camera or hardware supports it.

Exit code 2 means the report was saved but the documented evidence checks failed.
Inspect unmet checks: no/low pose coverage, no visible desktop, a short run or
FPS below 30. Resolve the observed cause and repeat. Do not substitute blank
synthetic/mock reports for these results.

## Actual joint-angle validation

Use a trained assessor and a manual goniometer/reference procedure. Keep the
assessor blind to the software estimate. The assessor records the angle first;
the subject holds that same position while a separate operator collects it.
Use anonymous IDs. The example numbers below are command syntax, not ground truth:
replace them with the assessor's actual readings.

```powershell
.\venv\Scripts\python.exe validation.py collect --participant P01 --joint L_Elbow_Flex --view left-side --reference 90
.\venv\Scripts\python.exe validation.py collect --participant P01 --joint L_Knee_Flex --view left-side --reference 90
.\venv\Scripts\python.exe validation.py collect --participant P01 --joint L_Shoulder_Flex --view oblique --reference 45
```

Each command opens the camera, excludes one settling second, samples two seconds,
closes the pipeline and appends a paired hold to `validation/data/paired_holds.csv`.
The window shows settling/measurement progress. Early exit, missing landmarks,
low coverage, high spread or median drift save a rejected hold and return code 2.
Keep rejected holds and repeat the measurement under an improved setup.

Collect neutral and several comfortable bends, at least three repeats per
position, and both sides where reliable (`R_Elbow_Flex`, `R_Knee_Flex`, etc.). Keep
the same configuration within a study. For the challenging category use shoulder
or hip flexion (`L_Hip_Flex` / `R_Hip_Flex`), with the view recorded accurately.
The unit is degrees: shoulder/hip extension is negative; ankle plantarflexion is
positive and dorsiflexion negative. Elbow/knee extension reaches zero; signed
hyperextension is not distinguished by the triplet formula.

Generate the report:

```powershell
.\venv\Scripts\python.exe validation.py report
```

Open `validation/results/accuracy.md` and `accuracy.json`. They show per-joint/view
MAE, bias/RMSE, sample/participant counts, reference ranges, coverage and rejected
holds. Report exit code 2 means elbow, knee or shoulder/hip flexion evidence is
missing. Even a scope-complete report needs adequate repeated positions and a
discussion of reference error; the assignment has no fixed MAE pass threshold.

Participant data and generated accuracy files are Git-ignored by default. After
checking anonymization, publish the measured summary/table explicitly in README;
only include participant-level files if appropriate. Until collection is complete,
leave the accuracy table marked pending.
