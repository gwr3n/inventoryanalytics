import unittest

import numpy as np
import pandas as pd

import inventoryanalytics.forecasting.naive as naive


class TestNaiveForecasting(unittest.TestCase):

    def test_sample_random_walk_is_deterministic(self):
        expected = [
            0.47143516373249306,
            -0.7195405309739714,
            0.7131664374521259,
            0.40051454136041303,
            -0.3200741920045986,
        ]

        self.assertEqual(list(naive.sample_random_walk(0, 5)), expected)

    def test_sample_random_walk_arma_is_deterministic(self):
        expected = [
            0.47143516373249306,
            -0.7195405309739714,
            0.7131664374521259,
            0.40051454136041303,
            -0.3200741920045986,
        ]

        self.assertEqual(list(naive.sample_random_walk_arma(0, 5)), expected)

    def test_naive_forecast_and_residuals(self):
        series = pd.Series([10, 20, 30, 40, 50, 60])

        forecast = naive.naive(series, 2)
        np.testing.assert_array_equal(np.isnan(forecast[:3]), [True, True, True])
        self.assertEqual(forecast[3:].tolist(), [30.0, 30.0, 30.0])

        rolling = naive.naive_rolling(series)
        np.testing.assert_array_equal(rolling.tolist(), [np.nan, 10.0, 20.0, 30.0, 40.0, 50.0])

        residuals = naive.residuals(pd.Series([11, 13, 17]), pd.Series([10, 10, 14]))
        self.assertEqual(residuals.tolist(), [1, 3, 3])


if __name__ == "__main__":
    unittest.main()