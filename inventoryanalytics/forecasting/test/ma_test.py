import unittest

import numpy as np

import inventoryanalytics.forecasting.MA as ma


class MAForecastingTest(unittest.TestCase):

    def test_sample_ma_process_is_deterministic(self):
        expected = [
            0.47143516373249306,
            -0.8138275637204699,
            0.5742134454074244,
            0.5953185397078722,
            -0.6841688565531625,
            0.24816157439738662,
        ]
        self.assertEqual(list(ma.sample_MA_process(0, [0.8, 0.2], 6)), expected)

    def test_sample_ma_process_arma_is_deterministic(self):
        expected = [
            0.47143516373249306,
            -0.8138275637204699,
            0.4799264126609257,
            0.833513678649165,
            -0.970710250238382,
        ]
        self.assertEqual(list(ma.sample_MA_process_ARMA(0, [0.8], 5)), expected)


if __name__ == "__main__":
    unittest.main()
