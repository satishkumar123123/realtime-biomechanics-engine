"""Deterministic signal and segment tests, using only stdlib and NumPy."""
import unittest

import numpy as np

from core.filter import BoneLengthConstraintChecker, OneEuroFilter


class OneEuroTests(unittest.TestCase):
    def test_stationary_gaussian_noise_variance_reduction(self):
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 0.02, size=(1800, 33, 3))
        smoother = OneEuroFilter()
        output = np.array([smoother(sample, i / 60) for i, sample in enumerate(samples)])
        # Discard initialization transient; depth gets the same independent filter.
        ratio = np.var(output[300:]) / np.var(samples[300:])
        self.assertLess(ratio, 0.12)
        self.assertLess(abs(np.mean(output[300:])), 0.001)
        self.assertLess(np.var(output[300:, :, 2]), np.var(samples[300:, :, 2]) * 0.12)

    def test_default_beta_adapts_to_fast_step(self):
        adaptive, fixed = OneEuroFilter(), OneEuroFilter(beta=0)
        adaptive(0.0, 0.0)
        fixed(0.0, 0.0)
        # Units matter: this large step checks the actual default speed response.
        fast = adaptive(100.0, 1 / 60)
        slow = fixed(100.0, 1 / 60)
        self.assertGreater(float(adaptive.cutoff), 1.0)
        self.assertLess(abs(100 - fast), abs(100 - slow) * 0.85)

    def test_meter_step_with_tuned_beta_reduces_settling_time(self):
        adaptive, fixed = OneEuroFilter(beta=5), OneEuroFilter(beta=0)
        adaptive(0, 0)
        fixed(0, 0)
        fast, slow = [], []
        for i in range(1, 121):
            fast.append(float(adaptive(1, i / 60)))
            slow.append(float(fixed(1, i / 60)))
        self.assertGreater(fast[0], slow[0] + 0.5)
        adaptive_settle = next(i for i, x in enumerate(fast) if x >= 0.95)
        fixed_settle = next(i for i, x in enumerate(slow) if x >= 0.95)
        self.assertLess(adaptive_settle, fixed_settle)
        self.assertTrue(np.all(np.diff(fast) >= -1e-12))
        self.assertTrue(np.all(np.asarray(fast) <= 1))

    def test_meter_ramp_with_tuned_beta_reduces_lag(self):
        adaptive, fixed = OneEuroFilter(beta=5), OneEuroFilter(beta=0)
        times = np.arange(600) / 60
        truth = 3 * times
        fast = np.array([adaptive(x, t) for x, t in zip(truth, times)])
        slow = np.array([fixed(x, t) for x, t in zip(truth, times)])
        adaptive_lag = np.mean(truth[120:] - fast[120:]) / 3
        fixed_lag = np.mean(truth[120:] - slow[120:]) / 3
        self.assertLess(adaptive_lag, fixed_lag * 0.15)
        self.assertGreater(float(adaptive.cutoff), 10)

    def test_dynamic_timestamps_follow_known_coefficients(self):
        smoother = OneEuroFilter(min_cutoff=2, beta=0)
        smoother(np.zeros((2, 3)), 0)
        expected = np.zeros((2, 3))
        previous_t = 0
        for t in (0.01, 0.04, 0.06, 0.15):
            alpha = 1 / (1 + 1 / (2 * np.pi * 2 * (t - previous_t)))
            expected = alpha + (1 - alpha) * expected
            np.testing.assert_allclose(smoother(np.ones((2, 3)), t), expected)
            previous_t = t

    def test_adaptive_formula_uses_filtered_raw_derivative(self):
        smoother = OneEuroFilter(min_cutoff=1, beta=2, d_cutoff=3)
        smoother([0, 10], 0)
        dt = 0.02
        derivative_alpha = 1 / (1 + 1 / (2 * np.pi * 3 * dt))
        derivative = derivative_alpha * np.array([1, -2]) / dt
        cutoff = 1 + 2 * np.abs(derivative)
        alpha = 1 / (1 + 1 / (2 * np.pi * cutoff * dt))
        np.testing.assert_allclose(smoother([1, 8], dt),
                                   alpha * [1, 8] + (1 - alpha) * [0, 10])
        np.testing.assert_allclose(smoother.cutoff, cutoff)

    def test_zero_negative_tiny_dt_leave_state_unchanged(self):
        smoother, control = OneEuroFilter(), OneEuroFilter()
        np.testing.assert_array_equal(smoother([1, 2, 3], 1), [1, 2, 3])
        control([1, 2, 3], 1)
        for t in (1, 0.5, 1 + 1e-13):
            np.testing.assert_array_equal(smoother([999, 999, 999], t), [1, 2, 3])
        np.testing.assert_allclose(smoother([2, 3, 4], 1.1), control([2, 3, 4], 1.1))

    def test_reset_allows_new_shape_and_time_origin(self):
        smoother = OneEuroFilter()
        smoother(np.zeros((33, 3)), 100)
        smoother.reset()
        self.assertIsNone(smoother.cutoff)
        np.testing.assert_array_equal(smoother([5, 6], 0), [5, 6])
        np.testing.assert_array_equal(smoother([5, 6], 1), [5, 6])

    def test_missing_components_are_not_stale_or_cross_contaminating(self):
        smoother = OneEuroFilter()
        smoother([1, 2, 3], 0)
        output = smoother([np.nan, 2, np.inf], 0.1)
        self.assertTrue(np.isnan(output[0]))
        self.assertTrue(np.isnan(output[2]))
        self.assertEqual(output[1], 2)
        np.testing.assert_array_equal(smoother([5, 2, 7], 0.2), [5, 2, 7])
        np.testing.assert_allclose(smoother.cutoff, [1, 1, 1])

    def test_visibility_mask_broadcasts_per_landmark(self):
        smoother = OneEuroFilter()
        mask = np.ones((33, 1), dtype=bool)
        mask[15] = False
        output = smoother(np.ones((33, 3)), 0, valid_mask=mask)
        self.assertTrue(np.all(np.isnan(output[15])))
        np.testing.assert_array_equal(output[14], [1, 1, 1])

    def test_invalid_data_on_duplicate_timestamp_never_returns_stale_pose(self):
        smoother = OneEuroFilter()
        smoother([1, 2], 1)
        output = smoother([np.nan, 2], 1)
        self.assertTrue(np.isnan(output[0]))
        self.assertEqual(output[1], 2)
        np.testing.assert_array_equal(smoother([5, 2], 1.1), [5, 2])

    def test_input_and_output_ownership(self):
        smoother = OneEuroFilter()
        sample = np.array([1.0, 2.0])
        output = smoother(sample, 0)
        sample.fill(999)
        output.fill(999)
        cutoff = smoother.cutoff
        cutoff.fill(999)
        np.testing.assert_array_equal(smoother([1, 2], 0), [1, 2])
        np.testing.assert_array_equal(smoother.cutoff, [1, 1])

    def test_validation_does_not_poison_state(self):
        for params in ({'min_cutoff': 0}, {'beta': -1}, {'d_cutoff': np.inf}):
            with self.assertRaises(ValueError):
                OneEuroFilter(**params)
        smoother = OneEuroFilter()
        smoother([0, 0], 0)
        for value, t in (([0], 0.1), ([0, 0], np.nan), ([0, 0], np.inf),
                         ([1e308, 1e308], 0.01)):
            with self.assertRaises(ValueError):
                smoother(value, t)
        np.testing.assert_array_equal(smoother([0, 0], 0.1), [0, 0])


class BoneLengthTests(unittest.TestCase):
    def setUp(self):
        self.points = np.zeros((33, 3))
        for pair in BoneLengthConstraintChecker.DEFAULT_SEGMENTS.values():
            self.points[pair[1]] = (0, 0.5, 0)
        self.visibility = dict.fromkeys(range(33), 1.0)
        self.checker = BoneLengthConstraintChecker(relative_tolerance=0.2)
        self.checker.calibrate(self.points, self.visibility)

    def test_calibrated_lengths_and_small_variation_pass(self):
        verdicts = self.checker.check(self.points, self.visibility)
        self.assertTrue(all(result.consistent for result in verdicts.values()))
        self.points[13] *= 1.2
        self.assertTrue(self.checker.check(self.points)['L_UpperArm'].consistent)

    def test_length_spike_is_flagged_without_baseline_drift(self):
        self.points[13] *= 1.5
        for _ in range(3):
            result = self.checker.check(self.points)['L_UpperArm']
            self.assertFalse(result.consistent)
            self.assertAlmostEqual(result.reference_length, 0.5)
            self.assertAlmostEqual(result.relative_deviation, 0.5)

    def test_occlusion_and_nonfinite_geometry_are_unavailable(self):
        self.visibility[13] = 0.65
        self.assertIsNone(self.checker.check(self.points, self.visibility)['L_UpperArm'].consistent)
        self.points[27] = np.nan
        self.assertIsNone(self.checker.check(self.points)['L_Shin'].consistent)
        self.points[28] = self.points[26]
        self.assertIsNone(self.checker.check(self.points)['R_Shin'].consistent)

    def test_rigid_translation_rotation_preserves_verdicts(self):
        rotated = self.points @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) + (4, 3, -2)
        self.assertTrue(all(result.consistent for result in self.checker.check(rotated).values()))

    def test_calibration_reset_and_invalid_calibration(self):
        self.points[13] = self.points[11]
        with self.assertRaises(ValueError):
            self.checker.calibrate(self.points)
        self.assertEqual(self.checker.check(self.points)['L_UpperArm'].reference_length, 0.5)
        self.checker.reset()
        with self.assertRaises(RuntimeError):
            self.checker.check(self.points)

    def test_configuration_validation(self):
        for params in ({'relative_tolerance': -1}, {'visibility_threshold': 1.1},
                       {'segments': {}}, {'segments': {'bad': (11, 11)}},
                       {'segments': {'bad': (11, 99)}}):
            with self.assertRaises(ValueError):
                BoneLengthConstraintChecker(**params)


if __name__ == '__main__':
    unittest.main()
