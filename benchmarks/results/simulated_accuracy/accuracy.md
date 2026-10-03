# Simulated goniometric accuracy results

Automated synthetic landmark validation; no camera/model inference or physical goniometer. Error measures this simulation only, not clinical accuracy.

Accepted holds: 54; rejected holds: 0.

| Joint | View | Holds | Synthetic seeds | Reference range (deg) | MAE (deg) | Bias (deg) | RMSE (deg) | Valid frames |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| L_Elbow_Flex | left-side | 9 | 1 | 0.0 to 135.0 | 0.39 | 0.34 | 0.57 | 96.7% |
| L_Knee_Flex | left-side | 6 | 1 | 0.0 to 90.0 | 0.44 | 0.33 | 0.56 | 96.7% |
| L_Shoulder_Flex | front | 6 | 1 | 45.0 to 90.0 | 0.07 | -0.03 | 0.12 | 96.7% |
| L_Shoulder_Flex | oblique | 6 | 1 | 45.0 to 90.0 | 0.12 | 0.12 | 0.13 | 96.7% |
| R_Elbow_Flex | left-side | 9 | 1 | 0.0 to 135.0 | 0.48 | 0.44 | 0.78 | 96.7% |
| R_Knee_Flex | left-side | 6 | 1 | 0.0 to 90.0 | 0.38 | 0.35 | 0.50 | 96.7% |
| R_Shoulder_Flex | front | 6 | 1 | 45.0 to 90.0 | 0.17 | -0.14 | 0.19 | 96.7% |
| R_Shoulder_Flex | oblique | 6 | 1 | 45.0 to 90.0 | 0.08 | -0.02 | 0.09 | 96.7% |

Required joint categories: elbow=present, knee=present, shoulder_or_hip_flexion=present

Category presence is a coverage check, not a clinical pass criterion.

## Reference and measurement setup

| Joint | View | Reference method | Frame | Cutoff / beta / derivative cutoff | Resolution |
| --- | --- | --- | --- | --- | --- |
| L_Elbow_Flex | left-side | synthetic: independent forward kinematics; no physical goniometer | body | 1.0 / 5.0 / 1.0 | 640x480 |
| L_Knee_Flex | left-side | synthetic: independent forward kinematics; no physical goniometer | body | 1.0 / 5.0 / 1.0 | 640x480 |
| L_Shoulder_Flex | front | synthetic: independent forward kinematics; no physical goniometer | body | 1.0 / 5.0 / 1.0 | 640x480 |
| L_Shoulder_Flex | oblique | synthetic: independent forward kinematics; no physical goniometer | body | 1.0 / 5.0 / 1.0 | 640x480 |
| R_Elbow_Flex | left-side | synthetic: independent forward kinematics; no physical goniometer | body | 1.0 / 5.0 / 1.0 | 640x480 |
| R_Knee_Flex | left-side | synthetic: independent forward kinematics; no physical goniometer | body | 1.0 / 5.0 / 1.0 | 640x480 |
| R_Shoulder_Flex | front | synthetic: independent forward kinematics; no physical goniometer | body | 1.0 / 5.0 / 1.0 | 640x480 |
| R_Shoulder_Flex | oblique | synthetic: independent forward kinematics; no physical goniometer | body | 1.0 / 5.0 / 1.0 | 640x480 |

## Limitations

- Analytic targets and injected noise are artificial, not measured human/model errors
- Torso landmarks/visibility are supplied; detector errors and real occlusion are not modeled
- Reported participant identifiers denote synthetic seeds, not human participants
- Static hold medians do not quantify transient motion lag or whole-pipeline image accuracy
- Physical paired-reference validation remains necessary

## Target-angle results (simulated)

| Joint | Target | Mean measured | Holds | Rejected | MAE | Bias | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Elbow_Flex | 0 | 1.1427 | 6 | 0 | 1.1427 | 1.1427 | 1.1761 |
| Elbow_Flex | 90 | 89.9872 | 6 | 0 | 0.1116 | -0.0128 | 0.1367 |
| Elbow_Flex | 135 | 135.0457 | 6 | 0 | 0.0469 | 0.0457 | 0.0579 |
| Knee_Flex | 0 | 0.7333 | 6 | 0 | 0.7333 | 0.7333 | 0.7415 |
| Knee_Flex | 90 | 89.9495 | 6 | 0 | 0.0866 | -0.0505 | 0.1046 |
| Shoulder_Flex | 45 | 45.0171 | 12 | 0 | 0.0856 | 0.0171 | 0.1007 |
| Shoulder_Flex | 90 | 89.9461 | 12 | 0 | 0.1328 | -0.0539 | 0.1659 |

## In-plane versus out-of-plane groups (simulated)

| Group | Holds | Rejected | MAE | Bias | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| in_plane_elbow_knee | 30 | 0 | 0.4242 | 0.3717 | 0.6270 |
| out_of_plane_shoulder | 24 | 0 | 0.1092 | -0.0184 | 0.1373 |

Targets come from commanded forward rotations; reported errors compare one filtered
median per hold with that target. Seeds are synthetic identifiers, not participants.
This dataset does not measure camera, detector or clinical anatomical error.
