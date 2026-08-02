import unittest

import numpy as np
import pandas as pd

import inventoryanalytics.forecasting.sma as sma


class TestSimpleMovingAverageForecasting(unittest.TestCase):

    def test_sample_gaussian_process_is_deterministic(self):
        expected = [
            22.357175818662466,
            14.045121526467678,
            27.16353484213049,
            18.436740519541434,
            16.39705633317494,
        ]

        self.assertEqual(list(sma.sample_gaussian_process(20, 5, 5)), expected)

    def test_moving_average_forecasts_and_rolling(self):
        series = pd.Series([10, 20, 30, 40, 50, 60])

        forecast = sma.moving_average(series, 3, 2)
        np.testing.assert_array_equal(np.isnan(forecast[:3]), [True, True, True])
        self.assertEqual(forecast[3:].tolist(), [20.0, 20.0, 20.0])

        rolling = sma.moving_average_rolling(series, 3)
        np.testing.assert_array_equal(rolling.tolist(), [np.nan, np.nan, 20.0, 30.0, 40.0, 50.0])

    def test_residuals_helper(self):
        residuals = sma.residuals(pd.Series([11, 13, 17]), pd.Series([10, 10, 14]))
        self.assertEqual(residuals.tolist(), [1, 3, 3])


if __name__ == "__main__":
    unittest.main()