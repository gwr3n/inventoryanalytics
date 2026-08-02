import unittest
from unittest import mock

import inventoryanalytics.lotsizing.stochastic.newsvendor as newsvendor


class TestNewsvendor(unittest.TestCase):

    def setUp(self):
        self.instance = {"o": 1, "u": 5, "mean": 10, "std": 2}
        self.model = newsvendor.Newsvendor(self.instance)

    def test_constructor(self):
        self.assertEqual(self.model.mean, 10)
        self.assertEqual(self.model.std, 2)
        self.assertEqual(self.model.o, 1)
        self.assertEqual(self.model.u, 5)

    def test_critical_fractile_solution(self):
        self.assertAlmostEqual(self.model.crit_frac_solution(), 11.934843132203403)

    def test_cost_components(self):
        self.assertAlmostEqual(self.model.cfolf(10), 0.7978844538795548)
        self.assertAlmostEqual(self.model.folf(10), 0.7978844538795548)
        self.assertAlmostEqual(self.model.C(12), 2.999785005512372)

    def test_expected_cost_near_fractile_is_better_than_far_points(self):
        c10 = self.model.C(10)
        c12 = self.model.C(12)
        c14 = self.model.C(14)

        self.assertLess(c12, c10)
        self.assertLess(c12, c14)

    def test_optc_delegates_to_minimize(self):
        fake_result = object()
        with mock.patch.object(newsvendor, "minimize", return_value=fake_result) as mocked_minimize:
            result = self.model.optC()

        self.assertIs(result, fake_result)
        mocked_minimize.assert_called_once_with(self.model.C, 0, method="Nelder-Mead")


if __name__ == "__main__":
    unittest.main()
