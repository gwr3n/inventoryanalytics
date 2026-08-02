import unittest

import numpy as np
import pandas as pd

import inventoryanalytics.forecasting.seasonal_naive as seasonal_naive


class TestSeasonalNaiveForecasting(unittest.TestCase):

    def test_sample_seasonal_random_walk_is_deterministic(self):
        expected = [
            0.47143516373249306,
            -1.1909756947064645,
            1.4327069684260973,
            0.15878326764078016,
            -1.911564428071476,
            2.319869908733836,
        ]

        self.assertEqual(list(seasonal_naive.sample_seasonal_random_walk(6, 3)), expected)

    def test_seasonal_naive_forecasts_repeat_last_season(self):
        series = pd.Series([10, 20, 30, 40, 50, 60])

        forecast = seasonal_naive.seasonal_naive(series, 3, 2)
        np.testing.assert_array_equal(np.isnan(forecast[:3]), [True, True, True])
        self.assertEqual(forecast[3:].tolist(), [10.0, 20.0, 30.0])

        rolling = seasonal_naive.seasonal_naive_rolling(series, 3)
        np.testing.assert_array_equal(rolling.tolist(), [np.nan, np.nan, np.nan, 10.0, 20.0, 30.0])

    def test_residuals_helper(self):
        residuals = seasonal_naive.residuals(pd.Series([13, 24, 35]), pd.Series([10, 20, 30]))
        self.assertEqual(residuals.tolist(), [3, 4, 5])


if __name__ == "__main__":
    unittest.main()