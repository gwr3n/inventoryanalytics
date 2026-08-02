import unittest

import numpy as np
import pandas as pd

import inventoryanalytics.forecasting.drift as drift


class TestDriftForecasting(unittest.TestCase):

    def test_sample_random_walk_is_deterministic(self):
        expected = [
            0.571435163732493,
            -0.5195405309739715,
            1.0131664374521259,
            0.8005145413604131,
            0.17992580799540148,
        ]

        self.assertEqual(list(drift.sample_random_walk(0, 0.1, 5)), expected)

    def test_drift_forecasts_and_rolling(self):
        series = pd.Series([1, 3, 5, 7])

        forecast = drift.drift(series, 2)
        np.testing.assert_array_equal(np.isnan(forecast[:3]), [True, True, True])
        self.assertEqual(forecast[3:].tolist(), [7.0])

        rolling = drift.drift_rolling(series)
        np.testing.assert_array_equal(rolling.tolist(), [np.nan, np.nan, 5.0, 7.0])

    def test_residuals_helper(self):
        residuals = drift.residuals(pd.Series([11, 15, 20]), pd.Series([10, 13, 18]))
        self.assertEqual(residuals.tolist(), [1, 2, 2])


if __name__ == "__main__":
    unittest.main()