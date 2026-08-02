import unittest
from unittest import mock

import inventoryanalytics.lotsizing.stochastic.mpnewsvendor as mpnewsvendor


class TestMultiPeriodNewsvendor(unittest.TestCase):

    def setUp(self):
        self.instance = {"o": 1, "u": 5, "mean": [10, 10, 10]}
        self.model = mpnewsvendor.MultiPeriodNewsvendor(self.instance)

    def test_constructor(self):
        self.assertEqual(self.model.mean, [10, 10, 10])
        self.assertEqual(self.model.o, 1)
        self.assertEqual(self.model.u, 5)

    def test_cost_functions(self):
        self.assertAlmostEqual(self.model.cfolf(10, 10), 1.2511429232766016)
        self.assertAlmostEqual(self.model.C(10), 157.55617020253305)
        self.assertAlmostEqual(self.model.C(30), 43.25241260071023)

    def test_verify_fractile_solution(self):
        self.assertEqual(self.model.verify_fractile_solution(30), True)
        self.assertEqual(self.model.verify_fractile_solution(40), False)

    def test_optc_delegates_to_minimize(self):
        fake_result = object()
        with mock.patch.object(mpnewsvendor, "minimize", return_value=fake_result) as mocked_minimize:
            result = self.model.optC()

        self.assertIs(result, fake_result)
        mocked_minimize.assert_called_once_with(self.model.C, 0, method="Nelder-Mead")


if __name__ == "__main__":
    unittest.main()
