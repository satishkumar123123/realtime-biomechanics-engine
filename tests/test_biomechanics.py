"""Synthetic geometry checks; no camera or model inference required."""
import unittest

import numpy as np

from core.biomechanics import BiomechanicsEngine as Engine


SIDES = (("L", 11, 13, 15, 23, 25, 27, 31, 1),
         ("R", 12, 14, 16, 24, 26, 28, 32, -1))


def neutral_pose():
    points = np.zeros((33, 3), dtype=float)
    for _, shoulder, elbow, wrist, hip, knee, ankle, foot, outward in SIDES:
        x = 0.2 * outward
        for index, y in ((shoulder, -0.6), (elbow, -0.3), (wrist, 0),
                         (hip, 0), (knee, 0.5), (ankle, 1)):
            points[index] = (x, y, 0)
        points[foot] = (x, 1, -0.2)
    return points, dict.fromkeys(range(33), 1.0)


class BiomechanicsTests(unittest.TestCase):
    def setUp(self):
        self.points, self.visibility = neutral_pose()

    def metrics(self, **kwargs):
        return Engine.compute_joint_metrics(self.points, self.visibility, **kwargs)

    def test_neutral_all_bilateral_joints(self):
        metrics = self.metrics()
        self.assertEqual(len(metrics), 12)
        for key, value in metrics.items():
            with self.subTest(joint=key):
                self.assertIsNotNone(value)
                self.assertAlmostEqual(value, 0, delta=0.5)

    def test_elbow_and_knee_right_angle(self):
        for side, _, elbow, wrist, _, knee, ankle, _, _ in SIDES:
            self.points[wrist] = self.points[elbow] + (0, 0, -0.3)
            self.points[ankle] = self.points[knee] + (0, 0, -0.5)
            for joint in ('Elbow', 'Knee'):
                self.assertAlmostEqual(self.metrics()[f'{side}_{joint}_Flex'], 90, delta=0.5)

    def test_shoulder_abduction_and_adduction_signs(self):
        for side, shoulder, elbow, _, _, _, _, _, outward in SIDES:
            self.points[elbow] = self.points[shoulder] + (0.3 * outward, 0, 0)
            self.assertAlmostEqual(self.metrics()[f'{side}_Shoulder_Abd'], 90, delta=0.5)
            self.points[elbow] = self.points[shoulder] + (-0.3 * outward, 0.3, 0)
            self.assertAlmostEqual(self.metrics()[f'{side}_Shoulder_Abd'], -45, delta=0.5)

    def test_sagittal_flexion_and_extension(self):
        for side, shoulder, elbow, _, hip, knee, _, _, _ in SIDES:
            for z, expected in ((-0.3, 90), (0.3, -90)):
                self.points[elbow] = self.points[shoulder] + (0, 0, z)
                self.points[knee] = self.points[hip] + (0, 0, z)
                metrics = self.metrics()
                self.assertAlmostEqual(metrics[f'{side}_Shoulder_Flex'], expected, delta=0.5)
                self.assertAlmostEqual(metrics[f'{side}_Hip_Flex'], expected, delta=0.5)

    def test_out_of_plane_components_are_removed(self):
        for side, shoulder, elbow, _, hip, knee, _, _, outward in SIDES:
            # 45 degrees keeps both projections nondegenerate.
            self.points[elbow] = self.points[shoulder] + (0.3 * outward, 0.3, 0)
            metrics = self.metrics()
            self.assertAlmostEqual(metrics[f'{side}_Shoulder_Abd'], 45, delta=0.5)
            self.assertAlmostEqual(metrics[f'{side}_Shoulder_Flex'], 0, delta=0.5)
            self.points[elbow] = self.points[shoulder] + (0, 0.3, -0.3)
            metrics = self.metrics()
            self.assertAlmostEqual(metrics[f'{side}_Shoulder_Flex'], 45, delta=0.5)
            self.assertAlmostEqual(metrics[f'{side}_Shoulder_Abd'], 0, delta=0.5)
            self.points[knee] = self.points[hip] + (3, 0.5, 0)
            self.assertAlmostEqual(self.metrics()[f'{side}_Hip_Flex'], 0, delta=0.5)

    def test_zero_projection_is_undefined(self):
        self.points[13] = self.points[11] + (0.3, 0, 0)
        self.assertIsNone(self.metrics()['L_Shoulder_Flex'])
        self.points[13] = self.points[11] + (0, 0, -0.3)
        self.assertIsNone(self.metrics()['L_Shoulder_Abd'])

    def test_ankle_signed_deviations(self):
        for side, _, _, _, _, _, ankle, foot, _ in SIDES:
            for y, expected in ((0.2, 45), (-0.2, -45), (0, 0)):
                self.points[foot] = self.points[ankle] + (0, y, -0.2)
                self.assertAlmostEqual(self.metrics()[f'{side}_Ankle_Dorsi_Plantar'],
                                       expected, delta=0.5)

    def test_every_triplet_landmark_gates_its_joint(self):
        for side, shoulder, elbow, wrist, hip, knee, ankle, foot, _ in SIDES:
            triplets = {'Elbow_Flex': (shoulder, elbow, wrist),
                        'Knee_Flex': (hip, knee, ankle),
                        'Shoulder_Flex': (hip, shoulder, elbow),
                        'Shoulder_Abd': (hip, shoulder, elbow),
                        'Hip_Flex': (shoulder, hip, knee),
                        'Ankle_Dorsi_Plantar': (knee, ankle, foot)}
            for joint, indices in triplets.items():
                for index in indices:
                    with self.subTest(side=side, joint=joint, landmark=index):
                        self.visibility[index] = 0.649
                        self.assertIsNone(self.metrics()[f'{side}_{joint}'])
                        self.visibility[index] = 0.65
                        self.assertIsNotNone(self.metrics()[f'{side}_{joint}'])
                        self.visibility[index] = 1

    def test_missing_and_nonfinite_visibility(self):
        for value in (np.nan, np.inf, -1, 1.1, None):
            self.visibility[15] = value
            self.assertIsNone(self.metrics()['L_Elbow_Flex'])
            self.assertIsNotNone(self.metrics()['R_Elbow_Flex'])
        del self.visibility[15]
        self.assertIsNone(self.metrics()['L_Elbow_Flex'])

    def test_degenerate_and_nonfinite_coordinates(self):
        self.points[15] = self.points[13]
        self.assertIsNone(self.metrics()['L_Elbow_Flex'])
        for value in (np.nan, np.inf):
            self.points[15] = (value, 0, 0)
            self.assertIsNone(self.metrics()['L_Elbow_Flex'])
        self.points[15] = self.points[13] + (1e-8, 0, 0)
        self.assertIsNone(self.metrics()['L_Elbow_Flex'])

    def test_vector_helpers(self):
        self.assertAlmostEqual(Engine.euclidean_norm((3, 4, 0)), 5)
        self.assertIsNone(Engine.normalize((0, 0, 0)))
        self.assertIsNone(Engine.normalized_dot_product((0, 0, 0), (1, 0, 0)))
        np.testing.assert_allclose(Engine.cross_product((1, 0, 0), (0, 1, 0)), (0, 0, 1))
        np.testing.assert_allclose(Engine.project_coronal((1, 2, 3)), (1, 2, 0))
        np.testing.assert_allclose(Engine.project_sagittal((1, 2, 3)), (0, 2, 3))
        self.assertIsNone(Engine.cross_product((np.nan, 0, 0), (0, 1, 0)))
        for scale in (1, 1e100):
            cosine = Engine.normalized_dot_product((scale, scale, scale),
                                                   (scale, scale, scale))
            self.assertLessEqual(cosine, 1)
            self.assertGreaterEqual(cosine, -1)
            self.assertAlmostEqual(cosine, 1)

    def test_translation_and_scale_invariance(self):
        baseline = self.metrics()
        for scale in (0.5, 3):
            actual = Engine.compute_joint_metrics(self.points * scale + (5, -3, 2),
                                                  self.visibility)
            for key in baseline:
                self.assertAlmostEqual(actual[key], baseline[key], delta=0.5)

    def test_coordinate_sign_configuration(self):
        self.points[13] = self.points[11] + (0.3, 0.3, -0.3)
        baseline = self.metrics()
        reversed_signs = self.metrics(anterior_z_sign=1, left_x_sign=-1)
        for joint in ('Shoulder_Flex', 'Shoulder_Abd'):
            self.assertAlmostEqual(reversed_signs[f'L_{joint}'], -baseline[f'L_{joint}'])

    def test_input_shape_validation_and_visibility_sequence(self):
        with self.assertRaises(ValueError):
            Engine.compute_joint_metrics(np.zeros((3, 3)), self.visibility)
        with self.assertRaises(ValueError):
            self.metrics(anterior_z_sign=0)
        self.assertEqual(self.metrics(), Engine.compute_joint_metrics(self.points, [1] * 33))


if __name__ == '__main__':
    unittest.main()
