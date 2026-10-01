"""Rigid-transform regressions, torso reliability and metric-scale filter response."""
from types import SimpleNamespace
import unittest

import numpy as np

from core.biomechanics import BiomechanicsEngine as Engine
from core.filter import OneEuroFilter
from main import PoseProcessor
from tests.test_biomechanics import neutral_pose


class OrientationTests(unittest.TestCase):
    def test_unequal_hip_shoulder_widths_do_not_bias_neutral_abduction(self):
        points, visibility = neutral_pose()
        points[[11, 13, 15], 0] = .3
        points[[12, 14, 16], 0] = -.3
        metrics = Engine.compute_joint_metrics(points, visibility, coordinate_frame='body')
        self.assertAlmostEqual(metrics['L_Shoulder_Abd'], 0)
        self.assertAlmostEqual(metrics['R_Shoulder_Abd'], 0)
        for shoulder, elbow, side, outward in ((11, 13, 'L', 1), (12, 14, 'R', -1)):
            points[elbow] = points[shoulder] + (.3*outward, .3, 0)
            self.assertAlmostEqual(Engine.compute_joint_metrics(points, visibility,
                                   coordinate_frame='body')[f'{side}_Shoulder_Abd'], 45)

    def test_body_angles_follow_yaw_including_side_and_back_views(self):
        points, visibility = neutral_pose()
        points[13] = points[11] + (0, .3, -.3)
        points[25] = points[23] + (0, .5, -.5)
        points[14] = points[12] + (0, .3, .3)  # right extension
        points[26] = points[24] + (0, .5, .5)
        expected = Engine.compute_joint_metrics(points, visibility, coordinate_frame='body')
        self.assertAlmostEqual(expected['L_Shoulder_Flex'], 45)
        self.assertAlmostEqual(expected['L_Hip_Flex'], 45)
        self.assertAlmostEqual(expected['R_Shoulder_Flex'], -45)
        for yaw in (0, 30, 60, 90, 135, 180, 270):
            theta = np.radians(yaw)
            rotation = np.array(((np.cos(theta), 0, np.sin(theta)),
                                 (0, 1, 0), (-np.sin(theta), 0, np.cos(theta))))
            rotated = points @ rotation.T + (1, 2, 3)
            actual = Engine.compute_joint_metrics(rotated, visibility, coordinate_frame='body')
            with self.subTest(yaw=yaw):
                for key in expected:
                    self.assertAlmostEqual(actual[key], expected[key], delta=1e-5)

    def test_abduction_is_separate_from_flexion_after_rotation(self):
        points, visibility = neutral_pose()
        points[13] = points[11] + (.3, .3, 0)
        points[14] = points[12] + (-.3, .3, 0)
        rotation = np.array(((0, 0, 1), (0, 1, 0), (-1, 0, 0)))
        metrics = Engine.compute_joint_metrics(points @ rotation.T, visibility,
                                               coordinate_frame='body')
        for side in ('L', 'R'):
            self.assertAlmostEqual(metrics[f'{side}_Shoulder_Abd'], 45)
            self.assertAlmostEqual(metrics[f'{side}_Shoulder_Flex'], 0)

    def test_missing_torso_anchor_suppresses_only_frame_dependent_metrics(self):
        points, visibility = neutral_pose()
        visibility[12] = .5
        metrics = Engine.compute_joint_metrics(points, visibility, coordinate_frame='body')
        for side in ('L', 'R'):
            for suffix in ('Shoulder_Flex', 'Shoulder_Abd', 'Hip_Flex'):
                self.assertIsNone(metrics[f'{side}_{suffix}'])
        self.assertAlmostEqual(metrics['L_Elbow_Flex'], 0)
        self.assertAlmostEqual(metrics['L_Knee_Flex'], 0)
        self.assertAlmostEqual(metrics['R_Ankle_Dorsi_Plantar'], 0)

    def test_degenerate_and_contradictory_torso_frames_are_rejected(self):
        points, visibility = neutral_pose()
        for change in ('coincident', 'inverted', 'collinear', 'nonfinite'):
            altered = points.copy()
            if change == 'coincident':
                altered[11] = altered[12]
            elif change == 'inverted':
                altered[[23, 24]] = altered[[24, 23]]
            elif change == 'collinear':
                altered[11], altered[12] = (0, -.5, 0), (0, -.7, 0)
                altered[23], altered[24] = (0, .1, 0), (0, -.1, 0)
            else:
                altered[11, 2] = np.nan
            with self.subTest(change=change):
                self.assertIsNone(Engine.body_frame(altered, visibility))

    def test_frame_is_orthonormal_under_camera_roll_and_pitch(self):
        points, visibility = neutral_pose()
        axis = np.array((1, 2, 3), dtype=float)
        axis /= np.linalg.norm(axis)
        cross = np.array(((0, -axis[2], axis[1]), (axis[2], 0, -axis[0]),
                          (-axis[1], axis[0], 0)))
        rotation = np.eye(3) + np.sin(.7)*cross + (1-np.cos(.7))*(cross @ cross)
        _, basis = Engine.body_frame(points @ rotation.T, visibility)
        np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(basis), 1)
        metrics = Engine.compute_joint_metrics(points @ rotation.T, visibility,
                                               coordinate_frame='body')
        self.assertTrue(all(abs(value) < 1e-5 for value in metrics.values()))

    def test_camera_mode_retains_documented_orientation_limitation(self):
        points, visibility = neutral_pose()
        points[13] = points[11] + (0, .3, -.3)
        rotation = np.array(((0, 0, 1), (0, 1, 0), (-1, 0, 0)))
        metrics = Engine.compute_joint_metrics(points @ rotation.T, visibility)
        self.assertAlmostEqual(metrics['L_Shoulder_Flex'], 0)
        with self.assertRaises(ValueError):
            Engine.compute_joint_metrics(points, visibility, coordinate_frame='invalid')

    def test_processor_explains_missing_frame_and_exposes_body_default(self):
        points, _ = neutral_pose()
        result = SimpleNamespace(
            pose_world_landmarks=SimpleNamespace(landmark=[
                SimpleNamespace(x=x, y=y, z=z) for x, y, z in points]),
            pose_landmarks=SimpleNamespace(landmark=[
                SimpleNamespace(visibility=.5 if i == 12 else 1) for i in range(33)]))
        processor = PoseProcessor()
        self.assertEqual(processor.coordinate_frame, 'body')
        self.assertEqual(processor.filter.beta, 5)
        metrics, reasons = processor.process(result, 0)
        self.assertIsNone(metrics['L_Shoulder_Flex'])
        self.assertEqual(reasons['L_Shoulder_Flex'], 'Torso frame unavailable')
        self.assertEqual(reasons['R_Shoulder_Flex'], 'Occluded / Low Conf')
        with self.assertRaises(ValueError):
            PoseProcessor(coordinate_frame='invalid')

    def test_app_tuning_reduces_meter_scale_ramp_lag_and_static_noise(self):
        tuned = PoseProcessor().filter
        baseline = OneEuroFilter(beta=.007)
        times = np.arange(600) / 60
        raw = times * 3
        outputs = np.array([tuned(x, t) for x, t in zip(raw, times)])
        old = np.array([baseline(x, t) for x, t in zip(raw, times)])
        lag = np.mean(raw[120:] - outputs[120:]) / 3
        self.assertLess(lag, .02)
        self.assertLess(lag, np.mean(raw[120:] - old[120:]) / 30)
        tuned.reset()
        noisy = np.random.default_rng(42).normal(0, .02, len(times))
        filtered = np.array([tuned(x, t) for x, t in zip(noisy, times)])
        self.assertLess(np.var(filtered[120:]), np.var(noisy[120:]) * .2)


if __name__ == '__main__':
    unittest.main()
