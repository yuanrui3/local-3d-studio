"""Analytic and generated checks for the held-orientation photometric control."""
import unittest
from unittest.mock import patch

import numpy as np

from experiments import held_orientation_relief_v001 as fixture


class HeldOrientationReliefTests(unittest.TestCase):
    def test_known_normals_and_rotation(self):
        u, v = fixture.patch_grid()
        actual = fixture.truth_normals(u, v, fixture.TRUTH_MM, out_of_basis=True)
        known = fixture.analytic_normals(u, v, fixture.TRUTH_MM, out_of_basis=True)
        self.assertLess(np.max(fixture.normal_angles(actual, known)), fixture.LIMITS["max_normal_proof_angle_deg"])
        np.testing.assert_allclose(np.linalg.norm(fixture.rotate_y(known, -46.), axis=-1), 1., atol=2e-15)
        np.testing.assert_allclose(fixture.rotate_y(np.array([0., 0., 1.]), 90.), [1., 0., 0.], atol=1e-14)

    def test_lowpass_preserves_constants_and_reduces_high_frequency(self):
        original = np.ones((2, 7, 9))*.4
        np.testing.assert_allclose(fixture.smooth_lowpass(original), original, atol=1e-15)
        y, x = np.indices((7, 9))
        checker = ((x+y) % 2).astype(float)
        self.assertLess(np.std(fixture.smooth_lowpass(checker)), .1*np.std(checker))

    def test_fake_material_matches_one_view_but_has_no_profile_bump(self):
        u, v = fixture.patch_grid()
        true = fixture.observation_block(u, v, (0., 48.), "genuine_relief", held=False)["intensity"]
        fake = fixture.observation_block(u, v, (0., 48.), "fake_albedo", held=False)["intensity"]
        np.testing.assert_allclose(true[0], fake[0], atol=1e-14, rtol=0.)
        self.assertGreater(float(np.max(np.abs(true[1]-fake[1]))), .02)
        self.assertEqual(fixture.profile_shift_mm(0.), 0.)
        self.assertGreater(fixture.profile_shift_mm(fixture.TRUTH_MM, extra=True), 3.)

    def test_only_two_hypotheses_fit_and_all_confounds_abstain(self):
        with patch.object(fixture, "fit_hypotheses", wraps=fixture.fit_hypotheses) as fit:
            report = fixture.compare()
        self.assertTrue(report["passed"])
        self.assertEqual(fit.call_count, 2)
        for call in fit.call_args_list:
            self.assertEqual(call.args[2]["yaw"], fixture.FIT_YAW)
            self.assertFalse(set(call.args[2]["yaw"]).intersection(fixture.HELD_YAW))
        for case in ("light_drift", "specular", "low_yaw", "registration"):
            self.assertFalse(report["outcomes"][case]["fit_performed"])
            self.assertEqual(report["outcomes"][case]["quality"]["blocked"], [case])


if __name__ == "__main__":
    unittest.main()
