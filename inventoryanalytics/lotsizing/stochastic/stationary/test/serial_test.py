import unittest

import inventoryanalytics.lotsizing.stochastic.stationary.serial as serial


class TestSerial(unittest.TestCase):

    def setUp(self):
        self.h_W = 1
        self.h_R = 1.5
        self.b = 10
        self.L_R = 5
        self.L_W = 5
        self.demand_rate = 10
        self.e_R = self.h_R - self.h_W

    def test_compute_y_r(self):
        self.assertEqual(
            serial.compute_y_R(self.h_W, self.h_R, self.b, self.L_R, self.demand_rate),
            74.0,
        )

    def test_cost_functions(self):
        self.assertAlmostEqual(
            serial.C_R(60, self.e_R, self.h_R, self.b, self.L_R, self.demand_rate),
            -24.51212434084561,
        )
        self.assertAlmostEqual(
            serial.C(100, self.e_R, self.h_R, self.b, self.L_R, self.h_W, self.L_W, self.demand_rate),
            201.41620016511033,
        )

    def test_compute_y_w(self):
        y_w = serial.compute_y_W(
            self.e_R,
            self.h_R,
            self.b,
            self.L_R,
            self.h_W,
            self.L_W,
            self.demand_rate,
            100,
        )
        self.assertEqual(y_w, 133)
        self.assertAlmostEqual(
            serial.C(y_w, self.e_R, self.h_R, self.b, self.L_R, self.h_W, self.L_W, self.demand_rate),
            26.364038065539773,
        )


if __name__ == "__main__":
    unittest.main()
